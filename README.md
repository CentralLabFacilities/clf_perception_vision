### CLF 2D OBJECT DETECT

Author: Florian Lier [flier AT techfak.uni-bielefeld DOT de]

Detect any kind of objects using 2d object detection and GPU acceleration
using OpenCV and CUDA

![CLF GPU DETECT](https://github.com/CentralLabFacilities/clf_2d_gpu_detect/blob/master/clf_gpu_detect_screenshot.png "")

### Installation

Install the latest nvidia cuda toolkit: https://developer.nvidia.com/cuda-toolkit
This version (master) has been tested with version 8.0

Install OpenCV minimum 3.2.0 with CUDA support

Invoke OpenCV's cmake with:

    -DWITH_CUDA=ON

Then:

    git clone https://github.com/CentralLabFacilities/clf_2d_gpu_detect.git
    cd clf_2d_gpu_detect
    git checkout ros_support_cv3
    mkdir build
    cd build
    source /opt/ros/indigo/setup.bash
    cmake -DCMAKE_INSTALL_PREFIX={YOUR DECISION} ..
    make
    make install

### Usage

NOTE: Images larger than 1000px will be resized before displaying. The actual
feature detection will be done on the original input size.

An exemplary config file can be found in the data folder.

    source /opt/ros/indigo/setup.bash

    ./clf_2d_detect_ros /path/to/configfile/example.yaml /camera/input/topic

    source /opt/ros/indigo/setup.bash

    rostopic echo /clf_2d_detect/objects

### License

See README.md
