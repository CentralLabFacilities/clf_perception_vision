### CLF 2D OBJECT DETECT

Author: Florian Lier [flier AT techfak.uni-bielefeld DOT de]

Detect any kind of objects and faces using 2d object detection, fisher faces and Haar classifier plus GPU acceleration using OpenCV/CUDA

![CLF GPU DETECT](https://github.com/CentralLabFacilities/clf_2d_gpu_detect/blob/master/clf_gpu_detect_screenshot.png "")

### Installation

Install the latest nvidia cuda toolkit: https://developer.nvidia.com/cuda-toolkit
This version (master) has been tested with version 8.0

Install OpenCV minimum 3.2.0 with CUDA support and OpenCV contrib modules

Invoke OpenCV's cmake with:

<pre>
    -D WITH_CUDA=ON
    -D BUILD_opencv_face=ON
</pre>

Here's our exact CMAKE invocation

<pre>
    cmake  \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D WITH_QT=ON \
        -D WITH_GSTREAMER=OFF \
        -D WITH_OPENGL=ON \
        -D WITH_OPENCL=OFF \
        -D WITH_V4L=ON \
        -D WITH_TBB=ON \
        -D WITH_XIMEA=OFF \
        -D WITH_IPP=ON \
        -D BUILD_opencv_adas=OFF \
        -D BUILD_opencv_bgsegm=OFF \
        -D BUILD_opencv_bioinspired=OFF \
        -D BUILD_opencv_ccalib=OFF \
        -D BUILD_opencv_datasets=OFF \
        -D BUILD_opencv_datasettools=OFF \
        -D BUILD_opencv_face=ON \
        -D BUILD_opencv_latentsvm=OFF \
        -D BUILD_opencv_line_descriptor=OFF \
        -D BUILD_opencv_matlab=OFF \
        -D BUILD_opencv_optflow=OFF \
        -D BUILD_opencv_reg=OFF \
        -D BUILD_opencv_saliency=OFF \
        -D BUILD_opencv_surface_matching=OFF \
        -D BUILD_opencv_text=OFF \
        -D BUILD_opencv_tracking=OFF \
        -D BUILD_opencv_xobjdetect=OFF \
        -D BUILD_opencv_xphoto=OFF \
        -D BUILD_opencv_stereo=OFF \
        -D BUILD_opencv_hdf=OFF \
        -D BUILD_opencv_cvv=OFF \
        -D BUILD_opencv_fuzzy=OFF \
        -D WITH_CUDA=ON \
        -D CUDA_GENERATION=Auto \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D WITH_CUBLAS=1 \
        -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
        -D CMAKE_INSTALL_PREFIX="/vol/pepper/systems/pepper-robocup-nightly" \
        -D CMAKE_BUILD_TYPE=RelWithDebInfo \
        -D CMAKE_CXX_FLAGS="${CXXFLAGS}" \
        -D CMAKE_C_FLAGS="${CFLAGS}" \
        -D CMAKE_SKIP_BUILD_RPATH=FALSE \
        -D CMAKE_BUILD_WITH_INSTALL_RPATH=FALSE \
        -D CMAKE_INSTALL_RPATH="/vol/pepper/systems/pepper-robocup-nightly/lib" \
        -D CMAKE_INSTALL_LIBDIR=lib \
     ..
</pre>

Then:

<pre>
    git clone https://github.com/CentralLabFacilities/clf_2d_gpu_detect.git
    cd clf_2d_gpu_detect
    git checkout ros_support_cv3
    mkdir build
    cd build
    source /opt/ros/indigo/setup.bash
    cmake -DCMAKE_INSTALL_PREFIX={YOUR DECISION} ..
    make
    make install
</pre>

### Usage

NOTE: Images larger than 1000px will be resized before displaying. The actual
feature detection will be done on the original input size.

An exemplary config file can be found in the data folder.

<pre>
    source /opt/ros/$ROSVERSION/setup.bash

    ./clf_detect_objects_ros /path/to/configfile/example.yaml /camera/input/topic /output/topic

    ./clf_detect_faces_ros --cascade /path/to/cascade.xml --cascade-profile /path/to/cascade.xml --topic /webcam/image_raw

    Face detection with gender classification

    ./clf_detect_faces_ros --cascade /path/to/cascade --cascade-profile /path/to/cascade.xml --gender /path/to/fisher_gender.xml --topic /webcam/image_raw
</pre>

### Usage Object Detection

The config values. Examples can be found in the data folder.

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
