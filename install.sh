#!/bin/bash 

###################################################################
#Script Name	: install.sh                                                                                          
#Description	: Step by step Classificador Vizentec                                                                                
#Args           :                                                                                           
#Author       	: Willian Antunes                                               
#Email         	: willianmoreira@vizentec.com.br                                           
###################################################################

# Variables

DPKG_PACKAGES="git wget g++ openjdk-8-jdk libudev-dev libusb-1.0-0-dev build-essential libssl-dev"
BASE_PATH=$PWD
USE_LOCAL_CMAKE=true

# Opencv Variables
OPENCV_PATH=$BASE_PATH/opencv
OPENCVCONTRIB_PATH=$BASE_PATH/opencv_contrib
OPENCV_INSTALL_PATH=$OPENCV_PATH/build/release

# Tensorflow Variables
TENSORFLOW_PATH=$PWD/tensorflow_src

# Classificador
CLASSIFICADOR_PATH=$PWD/classificador_edge_tpu
CLASSIFICADOR_INSTALL_PATH=$CLASSIFICADOR_PATH/build/release

# install necessary dpkg packages

for pkg in $DPKG_PACKAGES; do
    if ! (dpkg --get-selections | grep -q "^$pkg[:[:space:]*].*[[:space:]]*install$" >/dev/null); then
        if sudo apt-get -qq --yes install $pkg; then
            echo "Successfully installed $pkg"
        else
            echo "Error installing $pkg"
            exit 1
        fi
    fi
done


# install cmake if necessary

CMAKE_VERSION="$(cmake --version)"
echo ${CMAKE_VERSION}
read -ra version_array <<< ${CMAKE_VERSION}
version=${version_array[2]}

readarray -d . -t version_array <<< "$version"

if [[ "${version_array[1]}" -gt 5 ]]; then
    USE_LOCAL_CMAKE=false
  else
    USE_LOCAL_CMAKE=true
  	dpkg --purge cmake*
fi


if [[ "$USE_LOCAL_CMAKE" = true ]]; then
	git clone -b v3.20.0 https://github.com/Kitware/CMake cmake
	cd cmake
	CMAKE_PATH=$PWD
	./bootstrap && make && sudo make install
else
	echo "Cmake is installed..."
fi


# Git clone and install opencv

cd $BASE_PATH

if [[ ! -d "${OPENCV_PATH}" ]]
then
    git clone -b 4.5.0 https://github.com/opencv/opencv.git ${OPENCV_PATH} || (rm -rf ${OPENCV_PATH} && exit 1) 
fi
if [[ ! -d "${OPENCVCONTRIB_PATH}" ]]
then
    git clone -b 4.5.0 https://github.com/opencv/opencv_contrib.git ${OPENCVCONTRIB_PATH} || (rm -rf ${OPENCVCONTRIB_PATH} && exit 1) 
fi

cd ${OPENCV_PATH}

mkdir -p build
mkdir -p ${OPENCV_INSTALL_PATH}

cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=${OPENCVCONTRIB_PATH}/modules -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DOPENCV_FORCE_3RDPARTY_BUILD=ON -DWITH_1394=OFF -DWITH_VTK=OFF -DWITH_GTK=OFF -DWITH_NGRAPH=OFF -DWITH_CUDA=OFF -DWITH_V4L=OFF -DWITH_PROTOBUF=OFF -DWITH_IMGCODEC_HDR=OFF -DWITH_IMGCODEC_SUNRASTER=OFF -DWITH_IMGCODEC_PXM=OFF -DWITH_IMGCODEC_PFM=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_PATH} -DOPENCV_GENERATE_PKGCONFIG=YES .. || echo 'failed'
make -j3
make install

OPENCV_STATIC_PATH=${OPENCV_INSTALL_PATH}/lib/cmake/opencv4/

# Git clone tensorflow lite

cd $BASE_PATH

if [[ ! -d "${TENSORFLOW_PATH}" ]]
then
    git clone -b v2.5.0 https://github.com/tensorflow/tensorflow.git ${TENSORFLOW_PATH} || (rm -rf ${TENSORFLOW_PATH} && exit 1)
fi


# Git clone classificador edgetpu

cd $BASE_PATH

if [[ ! -d "${CLASSIFICADOR_PATH}" ]]
then
    git clone -b v1.0.0-rc0 https://bitbucket.org/vizentecpdi/classificador_edge_tpu ${CLASSIFICADOR_PATH} || (rm -rf ${CLASSIFICADOR_PATH} && exit 1)
fi
 
cd $CLASSIFICADOR_PATH

mkdir -p build
mkdir -p ${CLASSIFICADOR_INSTALL_PATH}

cd build

cmake -DTENSORFLOW_SOURCE_DIR=${TENSORFLOW_PATH} -DOPENCV_STATIC_PATH=${OPENCV_STATIC_PATH} -DCMAKE_INSTALL_PREFIX=${CLASSIFICADOR_INSTALL_PATH} .. || echo 'failed'
cmake --build . -j3