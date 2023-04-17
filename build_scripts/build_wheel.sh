PYTHON_VERSION=$1
CUDA_TAG=$2
VENV_NAME=venv_${PYTHON_VERSION}

python3 -m virtualenv ${VENV_NAME} -p ${PYTHON_VERSION}
ln -s /usr/bin/${PYTHON_VERSION}-config ${VENV_NAME}/bin/python3-config

source $VENV_NAME/bin/activate

rm -r cpp/build analyzer/habitat/*.so
git submodule update --init --recursive
git lfs pull

pushd analyzer
./install-dev.sh --build
python3 setup.py sdist bdist_wheel
popd

deactivate

rm -r ${VENV_NAME}
