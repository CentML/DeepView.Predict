PYTHON_VERSION=$1
VENV_NAME=venv_${PYTHON_VERSION}

python3 -m virtualenv ${VENV_NAME} -p ${PYTHON_VERSION}
ln -s /usr/bin/${PYTHON_VERSION}-config ${VENV_NAME}/bin/python3-config

source $VENV_NAME/bin/activate

pip install ${TORCH_VERSION}
pip install numpy tqdm pandas

rm -r cpp/build analyzer/habitat/*.so
git submodule update --init --recursive
git lfs pull

pushd analyzer
./install-dev.sh --build
python3 setup.py bdist_wheel
popd

pip list

deactivate

rm -r ${VENV_NAME}
