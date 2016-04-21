### CLF 2D OBJECT DETECT

Author: Florian Lier [flier AT techfak.uni-bielefeld DOT de]

Detect any kind of objects using 2d object detection and GPU acceleration
using OpenCV and CUDA

![CLF GPU DETECT](https://github.com/CentralLabFacilities/clf_2d_gpu_detect/blob/master/clf_gpu_detect_screenshot.png "")

### Installation

Install the latest nvidia cuda toolkit: https://developer.nvidia.com/cuda-toolkit
This version (master) has been tested with version 7.5

    HowTo: http://www.r-tutor.com/gpu-computing/cuda-installation/cuda7.5-ubuntu

Install OpenCV minimum 2.4.12 with CUDA support

    HowTo: http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html

Invoke OpenCV's cmake with:

    -DWITH_CUDA=ON

Then:

    git clone https://github.com/CentralLabFacilities/clf_2d_gpu_detect.git
    cd clf_2d_gpu_detect
    git checkout ros_support
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

    ./clf_2d_detect /path/to/configfile/example.yaml /camera/input/topic

    source /opt/ros/indigo/setup.bash

    rostopic echo /clf_2d_detect/objects

### License

See README.md
