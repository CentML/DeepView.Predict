PYTHON_VERSION=$1
PYTHON_TAG=$2
VENV_NAME=venv_${PYTHON_TAG}

python3 -m virtualenv ${VENV_NAME} -p ${PYTHON_VERSION}
ln -s /usr/bin/${PYTHON_VERSION}-config ${VENV_NAME}/bin/python3-config

source $VENV_NAME/bin/activate

# git clean -d --force
rm -r cpp/build analyzer/habitat/*.so
git submodule update --init --recursive
git lfs pull

pushd analyzer
./install-dev.sh
python setup.py sdist bdist_wheel --python-tag ${PYTHON_TAG} --build-number 0+${CUDA_TAG}
popd

deactivate

rm -r ${VENV_NAME}
