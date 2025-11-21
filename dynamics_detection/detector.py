#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
import threading

# ROS 2 Messages
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2

# TF & Math
from tf2_ros import Buffer, TransformListener, TransformException
from scipy.spatial.transform import Rotation as R

class DynamicPeopleDetector(Node):
    def __init__(self):
        super().__init__('people_detector')

        # --- Parameters ---
        self.declare_parameter('map_threshold', 0.35)     # Dist > this = Dynamic
        self.declare_parameter('cluster_tol', 0.4)        # DBSCAN epsilon
        self.declare_parameter('min_cluster_size', 10)    # DBSCAN min points
        self.declare_parameter('voxel_size', 0.1)         # Voxel downsampling size
        self.declare_parameter('max_range', 15.0)         # Max detection range
        
        self.map_threshold = self.get_parameter('map_threshold').value
        self.cluster_tol = self.get_parameter('cluster_tol').value
        self.min_cluster_size = self.get_parameter('min_cluster_size').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.max_range = self.get_parameter('max_range').value

        # --- State Variables ---
        self.ref_map = None
        self.ref_map_tree = None
        self.current_pose = np.eye(4)

        self.pub_markers = self.create_publisher(MarkerArray, '/detected_people_markers', 10)
        self.pub_debug = self.create_publisher(PointCloud2, '/debug/dynamic_points', 10)
        
        self.create_subscription(PointCloud2, '/kiss/local_map', self.map_cb, 10)
        self.create_subscription(Odometry, '/kiss/odometry', self.odom_cb, 10)
        self.create_subscription(PointCloud2, '/a201_0000/sensors/lidar3d_0/points', self.scan_cb, 10)

        self.get_logger().info("Detector Started.")

    def odom_cb(self, msg):
        pos = msg.pose.pose.position
        
        ori = msg.pose.pose.orientation
        
        r = R.from_quat([ori.x, ori.y, ori.z, ori.w])
        rot_mat = r.as_matrix()
        
        self.current_pose = np.eye(4)
        self.current_pose[:3, :3] = rot_mat
        self.current_pose[:3, 3] = [pos.x, pos.y, pos.z]

    def map_cb(self, msg):
        if self.ref_map is not None:
            return
                
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        points_list = [[p[0], p[1], p[2]] for p in gen]
        
        if len(points_list) == 0:
            return

        points = np.array(points_list, dtype=np.float64)

        self.ref_map = o3d.geometry.PointCloud()
        self.ref_map.points = o3d.utility.Vector3dVector(points)
        
        print("map size", len(self.ref_map.points))
        self.ref_map_tree = o3d.geometry.KDTreeFlann(self.ref_map)
            
    def scan_cb(self, msg):
        if self.ref_map is None:
            return

        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        points_list = [[p[0], p[1], p[2]] for p in gen]
        
        if len(points_list) == 0:
            return

        points = np.array(points_list, dtype=np.float64)

        current_scan = o3d.geometry.PointCloud()
        current_scan.points = o3d.utility.Vector3dVector(points)
        # current_scan = current_scan.voxel_down_sample(voxel_size=self.voxel_size)

        print("scan size", len(current_scan.points))
        
        scan_in_map = current_scan.transform(self.current_pose)

        dists = scan_in_map.compute_point_cloud_distance(self.ref_map)
        
        dists = np.asarray(dists)
        
        ind = np.where(dists > self.map_threshold)[0]
        
        if len(ind) < self.min_cluster_size:
            return
            
        dynamic_cloud = scan_in_map.select_by_index(ind)

        # Optional: 
        # dynamic_cloud = self.filter_by_range(dynamic_cloud, max_range=self.max_range, center=[tx, ty, tz])

        labels = np.array(dynamic_cloud.cluster_dbscan(eps=self.cluster_tol, min_points=self.min_cluster_size, print_progress=False))

        if len(labels) == 0:
            return

        marker_array = MarkerArray()
        max_label = labels.max()
        
        valid_id = 0
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            cluster = dynamic_cloud.select_by_index(cluster_indices)
            
            aabb = cluster.get_axis_aligned_bounding_box()
            extent = aabb.get_extent()
            wx, wy, h = extent[0], extent[1], extent[2]
            
            if (0.5 < h < 2.3) and (wx < 1.2) and (wy < 1.2):
                self.add_marker(valid_id, aabb.get_center(), marker_array)
                valid_id += 1

        self.pub_markers.publish(marker_array)
        
        self.publish_debug_cloud(dynamic_cloud, msg.header)

    def publish_debug_cloud(self, o3d_cloud, header):

        points = np.asarray(o3d_cloud.points)
        if len(points) == 0: return
        
        #print("debug point cloud size: ", points.shape)
        
        header.frame_id = "odom_lidar" 
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        self.pub_debug.publish(pc2_msg)

    def add_marker(self, id, center, array):
        print("Found object at: ", center[0], center[1])
        
        marker = Marker()
        marker.header.frame_id = "odom_lidar"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 1.0
        marker.color.a = 0.8
        marker.color.r = 1.0; marker.color.g = 0.2; marker.color.b = 0.2
        array.markers.append(marker)

    def filter_by_range(self, pcd, max_range, center):

        points = np.asarray(pcd.points)
        dists = np.linalg.norm(points - np.array(center), axis=1)
        ind = np.where(dists < max_range)[0]
        return pcd.select_by_index(ind)

def main(args=None):
    rclpy.init(args=args)
    node = DynamicPeopleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()