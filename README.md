### clf_2d_gpu_detect

Author: Florian Lier [flier AT techfak.uni-bielefeld DOT de]

Detect any kind of objects using 2d object detection and GPU acceleration

![CLF GPU DETECT](https://github.com/CentralLabFacilities/clf_2d_gpu_detect.git/raw/master/clf_gpu_detect_screenshot.png "")

### Installation

Install the latest nvidia cuda toolkit: https://developer.nvidia.com/cuda-toolkit
This version (master) has been tested with version 7.5

    HowTo: http://www.r-tutor.com/gpu-computing/cuda-installation/cuda7.5-ubuntu

Then:

    git clone https://github.com/CentralLabFacilities/clf_2d_gpu_detect.git
    cd clf_2d_gpu_detect
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX={your decision} ..
    make
    make install

### Usage

An exemplary config file can be found in the data folder.

    ./clf_2d_detect /path/to/configfile/example.yaml

### License

See README.md
