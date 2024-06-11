#!/bin/bash

PYTHON_VERSION=$1
VENV_NAME=venv_${PYTHON_VERSION}

python3 -m virtualenv ${VENV_NAME} -p ${PYTHON_VERSION}
ln -s /usr/bin/${PYTHON_VERSION}-config ${VENV_NAME}/bin/python3-config

source $VENV_NAME/bin/activate

rm -r cpp/build analyzer/habitat/*.so
git submodule update --init --recursive
git lfs pull

pushd analyzer
./install-dev.sh
popd

pushd experiments

device_pairs_list=()
IFS=';' read -ra input_devices <<< "${DEVICE_PAIRS}"

for i in "${input_devices[@]}"; do
    IFS=',' read -ra pair <<< "${i}"
    orig=${pair[0]}
    dest=${pair[1]}
    device_pairs_list+=("${orig},${dest}" "${dest},${orig}")
done

for j in "${device_pairs_list[@]}"; do
    IFS=',' read -ra pair <<< "${j}"
    orig=${pair[0]}
    dest=${pair[1]}
    if [ ${orig} == ${LOCAL_DEVICE} ]; then
        python model_eval_per_device.py ${orig} ${dest}
    fi
done

popd

pushd analyzer/habitat/data
find -iname "model.pth" | xargs sha256sum

popd 

deactivate
rm -r ${VENV_NAME}