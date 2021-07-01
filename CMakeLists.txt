cmake_minimum_required(VERSION 3.5)

project(ClassificadorEdgeTpu LANGUAGES CXX)

set(SRC_FILES src/main.cpp src/detector.cpp)

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
target_link_libraries(classificador
  tensorflow-lite
  ${CMAKE_DL_LIBS}
)


add_library(libclassificador SHARED ${SRC_FILES})
target_link_libraries(libclassificador
  tensorflow-lite
  ${CMAKE_DL_LIBS}
)
