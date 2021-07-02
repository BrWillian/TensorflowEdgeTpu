cmake_minimum_required(VERSION 3.5)

project(ClassificadorEdgeTpu LANGUAGES CXX)

set(SRC_FILES src/main.cpp src/detector.cpp src/detector.h src/bbox.cpp src/bbox.h)

set(TENSORFLOW_SOURCE_DIR "/home/dev-ia/coral/tensorflow_src/" CACHE PATH
  "Directory that contains the TensorFlow project" )
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

set(CMAKE_CXX_STANDARD 11)

add_executable(classificador ${SRC_FILES})
target_link_libraries(classificador LINK_PRIVATE tensorflow-lite ${CMAKE_DL_LIBS} /usr/local/lib/libedgetpu.so.1)

add_library(_classificador SHARED ${SRC_FILES})
target_link_libraries(_classificador LINK_PRIVATE tensorflow-lite ${CMAKE_DL_LIBS} /usr/local/lib/libedgetpu.so.1)
