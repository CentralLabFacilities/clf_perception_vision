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

    ./clf_detect_ros /path/to/configfile/example.yaml /camera/input/topic /output/topic

    source /opt/ros/indigo/setup.bash

    rostopic echo /clf_detect/objects


### Usage

The config values.

<pre>
    %YAML:1.0
    # Image locations
    targets:
     - "/home/fl/coke.png"
     - "/home/fl/cup.png"
     - "/home/fl/deo.png"
     - "/home/fl/gel.png"
     - "/home/fl/limo.png"
     - "/home/fl/milk.png"
     - "/home/fl/paste.png"
     - "/home/fl/pizza.png"
     - "/home/fl/soap.png"
     - "/home/fl/tea.png"
     - "/home/fl/water.png"

    # Corresponding labels
    labels:
     - "coke"
     - "cup"
     - "deo"
     - "gel"
     - "limo"
     - "milk"
     - "paste"
     - "pizza"
     - "soap"
     - "tea"
     - "water"

    # ORB ONLY, number of maximum keypoints applied to an image
    # Higher values may result in better detection but slow down
    # the process
    maxkeypoints_orb: 1500

    # Number of minimum matches between query image and current camera image
    # to accept a valid identification
    minmatches: 10

    # Number of maximum matches. This may affect fitting speed if set too high
    maxmatches: 20

    # Distance of two consecutive keypoints in an image. Setting this too
    # high will result in inaccurate results
    detectionthreshold: 0.5

    # Algorithm to extract keypoints
    keypointalgo: SURF

    # Matching algorithm. Valid combinations are:
    # SURF+KNN, SURF+BF (evil slow), ORB+BF
    matcher: KNN

    # Draw window
    silent: false

    # Scale up camera image, makes things more robust but slower
    # Scale factor is 2
    pyr_up: true

</pre>
