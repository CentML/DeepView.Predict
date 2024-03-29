#! /bin/bash

MODULE_NAME=habitat_cuda
SO_NAME=${MODULE_NAME}$(python3-config --extension-suffix)
PACKAGE_NAME="habitat-predict"
CUPTI_PATH="/usr/local/cuda/extras/CUPTI"

export CUPTI_INCLUDE_DIR="/usr/local/cuda/extras/CUPTI/include"

# Operate out of the script directory
SCRIPT_PATH=$(cd $(dirname $0) && pwd -P)
cd $SCRIPT_PATH

# Abort if an error occurs
set -e

function pushd() {
  command pushd "$@" > /dev/null
}

function popd() {
  command popd "$@" > /dev/null
}

function compile_habitat_cuda() {
  echo "Compiling the Habitat C++ extension..."
  pushd ../cpp
  mkdir -p build
  pushd build

  cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$(which python) ..
  make -j habitat_cuda

  if [ ! -f $SO_NAME ]; then
    echo "ERROR: Could not find $SO_NAME after compilation. Please double "
    echo "check that compilation completed successfully."
    exit 1
  fi

  popd
  popd
  echo ""
}

function symlink_habitat_cuda() {
  echo "Adding a symbolic link to the Habitat C++ extension..."
  if [ ! -h habitat/$SO_NAME ]; then
    ln -s ../../cpp/build/$SO_NAME habitat
  fi
  echo ""
}

function install_habitat() {
  echo "Install an editable version of the Habitat package..."
  pip3 install --editable .
  echo ""
}

function uninstall_habitat() {
  pip3 uninstall $PACKAGE_NAME
}

function check_prereqs() {
  if [ -z $(which cmake) ]; then
    echo "Please ensure cmake 3.17+ is installed."
    exit 1
  fi
  if [ -z $(which make) ]; then
    echo "Please ensure make is installed."
  fi
  if [ -z $(which pip3) ]; then
    echo "Please ensure pip3 is installed."
    exit 1
  fi
}

function install_cupti_sample() {
  echo "Copying CUPTI examples from" $CUPTI_PATH
  cp -r ${CUPTI_PATH}/samples/extensions/src ../cpp/external/cupti_profilerhost_util/
  cp -r ${CUPTI_PATH}/samples/extensions/include ../cpp/external/cupti_profilerhost_util/
}

function main() {
  if [ "$1" = "--uninstall" ]; then
    uninstall_habitat
  elif [ "$1" = "--build" ]; then
    install_cupti_sample
    check_prereqs
    compile_habitat_cuda
    symlink_habitat_cuda
  else
	  install_cupti_sample
    check_prereqs
    compile_habitat_cuda
    symlink_habitat_cuda
    install_habitat
  fi
}

main $@
