cmake_minimum_required(VERSION 3.5)

project(ClassificadorEdgeTpu LANGUAGES CXX)

set(SRC_FILES src/main.cpp src/detector.cpp src/detector.h src/bbox.h)



# OpenCV Path
if(NOT DEFINED OPENCV_STATIC_PATH)
        message(WARNING "OPENCV_STATIC_PATH not set, using default value.")
        set (OPENCV_STATIC_PATH /home/dev-ia/coral/opencv/build_minimal/out/install/lib/)
endif()


set(OpenCV_DIR /home/dev-ia/coral/opencv/build_minimal/out/install/lib/cmake/opencv4/)
find_package(OpenCV REQUIRED PATHS ${OPENCV_STATIC_PATH} NO_DEFAULT_PATH)

message(STATUS "    libraries: ${OpenCV_LIBS}")

# Build Tensorflow Lite

set(TENSORFLOW_SOURCE_DIR "/home/dev-ia/coral/tensorflow_src/" CACHE PATH
  "Directory that contains the TensorFlow project" )
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

set(CMAKE_CXX_STANDARD 14)

add_executable(classificador ${SRC_FILES})
target_link_libraries(classificador LINK_PRIVATE tensorflow-lite ${CMAKE_DL_LIBS} /usr/local/lib/libedgetpu.so.1)
target_link_libraries(classificador LINK_PRIVATE ${OpenCV_LIBS})


add_library(_classificador SHARED ${SRC_FILES})
target_link_libraries(_classificador LINK_PRIVATE tensorflow-lite ${CMAKE_DL_LIBS} ${OpenCV_LIBS} /usr/local/lib/libedgetpu.so.1)
target_link_libraries(classificador LINK_PRIVATE ${OpenCV_LIBS})
