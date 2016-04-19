CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
PROJECT(clf_2d_detect)

FIND_PACKAGE(CUDA)

FIND_PACKAGE(Boost REQUIRED COMPONENTS system program_options)

SET(OpenCV_MIN_VERSION "3.1.0")

FIND_PATH(OpenCV_DIR NAMES opencv-config.cmake OpenCVConfig.cmake OpenCVModules.cmake PATH_SUFFIXES lib/cmake lib)

FIND_PACKAGE( OpenCV ${OpenCV_MIN_VERSION} REQUIRED COMPONENTS core features2d imgproc objdetect video highgui flann
              cudafeatures2d cudev cudaarithm cudabgsegm cudafilters cudaimgproc cudawarping cudacodec cudalegacy
              cudaobjdetect cudaoptflow cudastereo)
IF(OpenCV_FOUND)
        IF(${OpenCV_VERSION} VERSION_LESS ${OpenCV_MIN_VERSION} AND (SUPPORT_VIDEO OR SUPPORT_IMAGE) )
                MESSAGE(FATAL_ERROR "OpenCV version (${OpenCV_VERSION} is present, while ${OpenCV_MIN_VERSION}+ is required")
        ENDIF()

        INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

        MESSAGE("Found OpenCV ${OpenCV_VERSION} libraries ${OpenCV_LIBRARIES} and headers ${OpenCV_INCLUDE_DIRS}")
ELSE()
        MESSAGE(FATAL_ERROR "OpenCV not found.")
ENDIF()

SET(CMAKE_BUILD_TYPE DEBUG)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -march=core-avx2 -O3")

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} include)

FILE(GLOB CLF_HEADER_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/*.hpp)

ADD_EXECUTABLE(clf_2d_detect
               src/clf_2d_detect.cpp
               src/native_grabber.cpp
               src/main.cpp
               ${CLF_HEADER_LIST}
               ${INCLUDE_DIRS}

)

TARGET_LINK_LIBRARIES(clf_2d_detect ${OpenCV_LIBRARIES})

SET_PROPERTY(TARGET clf_2d_detect PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
SET_TARGET_PROPERTIES(clf_2d_detect PROPERTIES COMPILE_FLAGS -DCMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")

INSTALL(TARGETS clf_2d_detect RUNTIME DESTINATION bin)
INSTALL(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/tools/clf_2d_collect" DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
