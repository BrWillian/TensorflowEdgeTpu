cmake_minimum_required(VERSION 3.5)

project(ClassificadorEdgeTpu LANGUAGES CXX)

set(SRC_FILES src/main.cpp src/detector.cpp src/detector.h src/serializer.cpp src/serializer.h src/bbox.h)

# OpenCV Path
if(NOT DEFINED OPENCV_STATIC_PATH)
        message(WARNING "OPENCV_STATIC_PATH not set, using default value.")
        set (OPENCV_STATIC_PATH ../../opencv/build_minimal/out/install/lib/)
endif()

find_package(OpenCV REQUIRED PATHS ${OPENCV_STATIC_PATH} NO_DEFAULT_PATH)

message(STATUS "    libraries: ${OpenCV_LIBS}")

# Build Tensorflow Lite

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project" )
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

set(CMAKE_CXX_STANDARD 17)  

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
        message(WARNING "CMAKE_INSTALL_PREFIX not set, using default value.")
        set (CMAKE_INSTALL_PREFIX ./release/)
endif()

add_executable(classificador ${SRC_FILES})
target_link_libraries(classificador LINK_PRIVATE tensorflow-lite ${CMAKE_DL_LIBS} /usr/lib/x86_64-linux-gnu/libedgetpu.so.1 stdc++fs)
target_link_libraries(classificador LINK_PRIVATE ${OpenCV_LIBS})


add_library(_classificador SHARED ${SRC_FILES})
target_link_libraries(_classificador LINK_PRIVATE tensorflow-lite ${CMAKE_DL_LIBS} /usr/lib/x86_64-linux-gnu/libedgetpu.so.1)
target_link_libraries(_classificador LINK_PRIVATE ${OpenCV_LIBS})


install(TARGETS classificador DESTINATION ${CMAKE_INSTALL_PREFIX} CONFIGURATIONS Release)
install(TARGETS _classificador DESTINATION ${CMAKE_INSTALL_PREFIX} CONFIGURATIONS Release)


install(TARGETS classificador DESTINATION ${CMAKE_INSTALL_PREFIX})
install(TARGETS _classificador DESTINATION ${CMAKE_INSTALL_PREFIX})
