# dynamicPointDetector
A simple package to detect dynamic points


1. Build the packages with colcon.
```
colcon build --symlink-install
```
2. Run the detector node.
```
ros2 run dynamics_detection detector
```
3. In new terminal start your SLAM node.