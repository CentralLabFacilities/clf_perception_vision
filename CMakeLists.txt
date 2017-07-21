CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)

PROJECT(clf_2d_gpu_detect)

SET(ENV{ROS_LANG_DISABLE} "genjava" "genlisp")

SET(OpenCV_MIN_VERSION "3.2.0")

SET(ROS_VERSION "kinetic")

SET(DEFAULT_CXX_FLAGS "" CACHE STRING "-O2 -march=native -Wno-deprecated-declarations")

SET(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

MESSAGE(STATUS "CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

FIND_PACKAGE(dlib REQUIRED)

FIND_PACKAGE(Caffe REQUIRED)

ADD_DEFINITIONS(${Caffe_DEFINITIONS})

ADD_DEFINITIONS(-DENABLE_ASSERTS)

FIND_PACKAGE(CUDA REQUIRED)

FIND_PACKAGE(Boost REQUIRED COMPONENTS system program_options)

FIND_PACKAGE(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs image_transport cv_bridge)

FIND_PACKAGE(OpenCV REQUIRED COMPONENTS face
                                        highgui
                                        core
                                        imgproc
                                        features2d
                                        xfeatures2d
                                        imgcodecs
                                        calib3d
                                        cudafeatures2d
                                        cudaarithm
                                        cudabgsegm
                                        cudafilters
                                        cudaimgproc
                                        cudalegacy
                                        cudaobjdetect
                                        cudaoptflow
                                        cudastereo
                                        cudawarping
                                        cudacodec
)

IF(OpenCV_FOUND)
        IF(${OpenCV_VERSION} VERSION_LESS ${OpenCV_MIN_VERSION} AND (SUPPORT_VIDEO OR SUPPORT_IMAGE) )
                MESSAGE(FATAL_ERROR "OpenCV version (${OpenCV_VERSION} is present, while ${OpenCV_MIN_VERSION}+ is required")
        ENDIF()

        MESSAGE(STATUS "OpenCV_VERSION: " ${OpenCV_VERSION})
        MESSAGE(STATUS "OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
        MESSAGE(STATUS "OpenCV_LIBRARIES: " ${OpenCV_LIBS})
ELSE()
        MESSAGE(FATAL_ERROR "OpenCV not found.")
ENDIF()

IF(OpenCV_CUDA_VERSION)
	MESSAGE("Found OpenCV_CUDA_VERSION: ${OpenCV_CUDA_VERSION}")
ELSE()
	MESSAGE(FATAL_ERROR "OpenCV_CUDA_VERSION NOT FOUND")
ENDIF()

SET(ROS_BUILD_TYPE RELEASE)
SET(CMAKE_BUILD_TYPE RELEASE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

SET(CUSTOM_ROS_LIBS /opt/ros/${ROS_VERSION}/lib/libroscpp.so
                    /opt/ros/${ROS_VERSION}/lib/libroscpp.so
                    /opt/ros/${ROS_VERSION}/lib/libcv_bridge.so
                    /opt/ros/${ROS_VERSION}/lib/librosconsole.so
                    /opt/ros/${ROS_VERSION}/lib/libroscpp_serialization.so
                    /opt/ros/${ROS_VERSION}/lib/librostime.so
                    /opt/ros/${ROS_VERSION}/lib/libmessage_filters.so
                    /opt/ros/${ROS_VERSION}/lib/libclass_loader.so
                    /opt/ros/${ROS_VERSION}/lib/libroslib.so
                    /opt/ros/${ROS_VERSION}/lib/libxmlrpcpp.so
                    /opt/ros/${ROS_VERSION}/lib/libcpp_common.so
                    /opt/ros/${ROS_VERSION}/lib/librosconsole_log4cxx.so
                    /opt/ros/${ROS_VERSION}/lib/librosconsole_backend_interface.so
                    /opt/ros/${ROS_VERSION}/lib/librospack.so
                    /opt/ros/${ROS_VERSION}/lib/libimage_transport.so
)

SET(CV_LIBS  ${CMAKE_INSTALL_PREFIX}/lib/libopencv_imgcodecs.so
             ${CMAKE_INSTALL_PREFIX}/lib/libopencv_face.so
             ${CMAKE_INSTALL_PREFIX}/lib/libopencv_imgproc.so
             ${CMAKE_INSTALL_PREFIX}/lib/libopencv_core.so
             ${CMAKE_INSTALL_PREFIX}/lib/libopencv_highgui.so
             ${CMAKE_INSTALL_PREFIX}/lib/libopencv_xfeatures2d.so
             ${CMAKE_INSTALL_PREFIX}/lib/libopencv_features2d.so
             ${CMAKE_INSTALL_PREFIX}/lib/libopencv_calib3d.so
             ${CMAKE_INSTALL_PREFIX}/lib/libopencv_objdetect.so
)

SET(CUDA_LIBS  ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudaarithm.so
               ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudabgsegm.so
               ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudacodec.so
               ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudafeatures2d.so
               ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudafilters.so
               ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudaimgproc.so
               ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudalegacy.so
               ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudaobjdetect.so
               ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudaoptflow.so
               ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudastereo.so
               ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudawarping.so
)

# MESSAGE(STATUS "CUSTOM_ROS_LIBS: " ${CUSTOM_ROS_LIBS})
# MESSAGE(STATUS "CV_LIBS: " ${CV_LIBS})
# MESSAGE(STATUS "CUDA_LIBS: " ${CUDA_LIBS})

CATKIN_PACKAGE(
  INCLUDE_DIRS include
  CATKIN_DEPENDS message_runtime
)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    ${catkin_INCLUDE_DIRS}
                    ${Caffe_INCLUDE_DIRS}
                    ${dlib_INCLUDE_DIRS}
                    include)

FILE(GLOB CLF_HEADER_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/*.hpp include/*.h)

ADD_EXECUTABLE(clf_detect_objects_surb_ros
               src/clf_2d_detect_surb.cpp
               src/ros_grabber.cpp
               src/main_surb.cpp
               ${CLF_HEADER_LIST}
               ${INCLUDE_DIRS}
)

ADD_EXECUTABLE(clf_collect_data_ros
               src/clf_2d_collect.cpp
               src/ros_grabber.cpp
               ${CLF_HEADER_LIST}
               ${INCLUDE_DIRS}
)


ADD_EXECUTABLE(clf_detect_people_ros
               src/ros_grabber.cpp
               src/clf_2d_cv_dlib_people.cpp
               src/clf_2d_faceprocessing.cpp
               src/clf_2d_caffee_classification.cpp
               src/clf_2d_body.cpp
               ${CLF_HEADER_LIST}
               ${INCLUDE_DIRS}
)

ADD_EXECUTABLE(clf_detect_faces_dlib_ros
               src/ros_grabber.cpp
               src/clf_2d_dlib_faces.cpp
               src/main_dlib_faces.cpp
               ${CLF_HEADER_LIST}
               ${INCLUDE_DIRS}
)

ADD_EXECUTABLE(clf_cmt
               src/cmt/common.cpp
               src/cmt/gui.cpp
               src/cmt/main.cpp
               src/cmt/CMT.cpp
               src/cmt/Consensus.cpp
               src/cmt/Fusion.cpp
               src/cmt/Matcher.cpp
               src/cmt/Tracker.cpp
               src/cmt/fastcluster.cpp
               src/ros_grabber.cpp
               ${CLF_HEADER_LIST}
               ${INCLUDE_DIRS}
)

# Deprecated
# ADD_EXECUTABLE(clf_detect_faces_ros
#               src/clf_2d_faces.cpp
#               src/ros_grabber.cpp
#               ${CLF_HEADER_LIST}
#               ${INCLUDE_DIRS}
# )

# Deprecated
# ADD_EXECUTABLE(clf_train_gender_fisher
#                src/clf_2d_train_gender_fisher.cpp
#                ${INCLUDE_DIRS}
# )

TARGET_LINK_LIBRARIES(clf_detect_objects_surb_ros ${CUDA_LIBS} ${CV_LIBS} ${CUSTOM_ROS_LIBS} ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(clf_detect_people_ros ${CUDA_LIBS} ${CV_LIBS} ${CUSTOM_ROS_LIBS} ${Boost_LIBRARIES} ${dlib_LIBRARIES} ${Caffe_LIBRARIES})
TARGET_LINK_LIBRARIES(clf_detect_faces_dlib_ros ${CV_LIBS} ${CUSTOM_ROS_LIBS} ${Boost_LIBRARIES} ${dlib_LIBRARIES})
TARGET_LINK_LIBRARIES(clf_collect_data_ros ${CV_LIBS} ${CUSTOM_ROS_LIBS} ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(clf_cmt ${OpenCV_LIBS} ${CUSTOM_ROS_LIBS} ${Boost_LIBRARIES})

# TARGET_LINK_LIBRARIES(clf_detect_faces_ros ${CUDA_LIBS} ${CUSTOM_ROS_LIBS} ${CV_LIBS} ${Boost_LIBRARIES})
# TARGET_LINK_LIBRARIES(clf_train_gender_fisher ${CV_LIBS})

INSTALL(TARGETS clf_detect_objects_surb_ros RUNTIME DESTINATION bin)
INSTALL(TARGETS clf_collect_data_ros RUNTIME DESTINATION bin)
INSTALL(TARGETS clf_detect_people_ros RUNTIME DESTINATION bin)
INSTALL(TARGETS clf_detect_faces_dlib_ros RUNTIME DESTINATION bin)
INSTALL(TARGETS clf_cmt RUNTIME DESTINATION bin)

# INSTALL(TARGETS clf_detect_faces_ros RUNTIME DESTINATION bin)
# INSTALL(TARGETS clf_train_gender_fisher RUNTIME DESTINATION bin)

INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data" DESTINATION ${CMAKE_INSTALL_PREFIX}/share/clf_detect)