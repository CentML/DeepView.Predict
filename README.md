# DeepView.Predict

[![License](https://img.shields.io/badge/license-Apache--2.0-green?style=flat)](https://github.com/CentML/habitat/blob/main/LICENSE)
[![Maintainability](https://api.codeclimate.com/v1/badges/fbb68badd0c0599f1843/maintainability)](https://codeclimate.com/github/CentML/DeepView.Predict/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/fbb68badd0c0599f1843/test_coverage)](https://codeclimate.com/github/CentML/DeepView.Predict/test_coverage)


A Runtime-Based Computational Performance Predictor for Deep Neural Network Training

- [Installation](#installation)
- [Building from source](#build)
- [Usage example](#getting-started)
- [Development Environment Setup](#dev-setup)
- [Release process](#release-process)
- [Release history](#release-history)
- [License](#license)
- [Research paper](#paper)
- [Contributing](#contributing)

DeepView.Predict is a tool that predicts a deep neural network's training iteration execution time on a given GPU. It currently supports PyTorch. To learn more about how DeepView.Predict works, please see our [research paper](https://arxiv.org/abs/2102.00527).

<h2 id="installation">Installation</h2>

To run DeepView.Predict, you need:
- [Python 3.6+](https://www.python.org/)
- [Pytorch 1.1.0+](https://pytorch.org/)
- A system equiped with an Nvidia GPU with properly configured CUDA

Currently, we have predictors for the following Nvidia GPUs:

| GPU        | Generation  | Memory | Mem. Type | SMs |
| ---------- |:-----------:| ------:| :-------: | :-: |
| P4000      | Pascal      | 8 GB   | GDDR5     | 14  |
| P100       | Pascal      | 16 GB  | HBM2      | 56  |
| V100       | Volta       | 16 GB  | HBM2      | 80  |
| 2070       | Turing      | 8 GB   | GDDR6     | 36  |
| 2080Ti     | Turing      | 11 GB  | GDDR6     | 68  |
| T4         | Turing      | 16 GB  | GDDR6     | 40  |
| 3090       | Ampere      | 24 GB  | GDDR6X    | 82  |
| A100       | Ampere      | 40 GB  | HBM2      | 108 |
| A40        | Ampere      | 48 GB  | GDDR6     | 84  |
| A4000      | Ampere      | 16 GB  | GDDR6     | 48  |
| 4000       | Turing      | 8 GB   | GDDR6     | 36  |


<h2 id="building-locally">Building locally</h2>

### 1. Install CUPTI

CUPTI is a profiling interface required by DeepView.Predict. Select your version of CUDA [here](https://developer.nvidia.com/cuda-toolkit-archive) and follow the instructions to add NVIDIA's repository. Then, install CUPTI with:
  ```bash
  sudo apt-get install cuda-cupti-xx-x
  ```
where `xx-x` represents the version of CUDA you have installed.

Alternatively, if you do not have root access on your machine, you can use `conda` to install CUPTI. Select your version of CUDA [here](https://anaconda.org/nvidia/cuda-cupti) and follow the instructions. For example if you have CUDA 11.6.0, you can install CUPTI with:
  ```bash
  conda install -c "nvidia/label/cuda-11.6.0" cuda-cupti
  ```
After installing CUPTI, add `$CONDA_HOME/extras/CUPTI/lib64/` to `LD_LIBRARY_PATH` to ensure the library is linked.

### 2. Install DeepView.Predict

You can install via pip if you have the following versions of CUDA and Python

- CUDA: 10.2, 11.1, 11.3, 11.6, 11.7
- Python: 3.7 - 3.10

### Installing from pip

Install via pip with the following command

```bash
pip install http://centml-releases.s3-website.us-east-2.amazonaws.com/habitat/wheels/habitat_predict-1.0.0-20221123+cuYYY-pyZZ-none-any.whl
```

where YYY is your CUDA version and ZZ is your Python version. 

For example, if you are using CUDA 10.2 and Python 3.7): 

```bash
pip install http://centml-releases.s3-website.us-east-2.amazonaws.com/habitat/wheels/habitat_predict-1.0.0-20221123+cu102-py37-none-any.whl
```

If you do not find matching version of CUDA and Python above, you need to build DeepView.Predict from source with the following instructions

### Installing from source

1. Install CMake 3.17+.
    - Note that CMake 3.24.0 and 3.24.1 has a bug that breaks DeepView.Predict as it is not able to find the CUPTI directory and you should not use those versions
        - [https://gitlab.kitware.com/cmake/cmake/-/merge_requests/7608/diffs](https://gitlab.kitware.com/cmake/cmake/-/merge_requests/7608/diffs)
    - Run the following commands to download and install a precompiled version of CMake 3.24.2
        
        ```bash
        wget https://github.com/Kitware/CMake/releases/download/v3.24.2/cmake-3.24.2-linux-x86_64.sh
        chmod +x cmake-3.24.2-linux-x86_64.sh
        mkdir /opt/cmake
        sh cmake-3.24.2-linux-x86_64.sh --prefix=/opt/cmake --skip-license
        ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
        ```
        
    - You can verify the version of CMake you installed with the following command
        
        ```bash
        cmake --version
        ```
        
2. Install [Git Large File Storage](https://git-lfs.github.com/)
3. Clone the DeepView.Predict package
    
    ```bash
    git clone https://github.com/CentML/DeepView.Predict
    ```
    
4. Get the pre-trained models used by DeepView.Predict
    
    ```bash
    git submodule init && git submodule update
    git lfs pull
    ```
    
5. Finally build DeepView.Predict with the following command
    
    ```bash
    ./analyzer/install-dev.sh
    ```

<h2 id="building-with-docker">Building with Docker</h2>

DeepView.Predict has been tested to work on the latest version of [NVIDIA NGC PyTorch containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

1. To build DeepView.Predict with Docker, first run the NGC container where
```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:XX.XX-py3
```
2. Inside the container, clone the repository then build and install DeepView.Predict Python package:
```bash
git clone --recursive https://github.com/CentML/DeepView.Predict
./habitat/analyzer/install-dev.sh
```

**Note:** DeepView.Predict needs access to your GPU's performance counters, which requires special permissions if you are running with a recent driver (418.43 or later). If you encounter a `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` error when running DeepView.Predict, please follow the instructions [here](https://developer.nvidia.com/ERR_NVGPUCTRPERM) and in [issue #5](https://github.com/geoffxy/habitat/issues/5).

<h2 id="usage-example">Usage example</h2>

You can verify your DeepView.Predict installation by running the simple usage example:
```python
# example.py
import habitat
import torch
import torchvision.models as models

# Define model and sample inputs
model = models.resnet50().cuda()
image = torch.rand(8, 3, 224, 224).cuda()

# Measure a single inference
tracker = habitat.OperationTracker(device=habitat.Device.RTX2080Ti)
with tracker.track():
    out = model(image)

trace = tracker.get_tracked_trace()
print("Run time on source:", trace.run_time_ms)

# Perform prediction to a single target device
pred = trace.to_device(habitat.Device.V100)
print("Predicted time on V100:", pred.run_time_ms)
```

```bash
python3 example.py
```

See [experiments/run_experiment.py](https://github.com/CentML/DeepView.Predict/tree/main/experiments) for other examples of DeepView.Predict usage.

<h2 id="release-history">Release History</h2>

See [Releases](https://github.com/UofT-EcoSystem/habitat/releases)

<h2 id="license">License</h2>

The code in this repository is licensed under the Apache 2.0 license (see
`LICENSE` and `NOTICE`), with the exception of the files mentioned below.

This software contains source code provided by NVIDIA Corporation. These files
are:

- The code under `cpp/external/cupti_profilerhost_util/` (CUPTI sample code)
- `cpp/src/cuda/cuda_occupancy.h`

The code mentioned above is licensed under the [NVIDIA Software Development
Kit End User License Agreement](https://docs.nvidia.com/cuda/eula/index.html).

We include the implementations of several deep neural networks under
`experiments/` for our evaluation. These implementations are copyrighted by
their original authors and carry their original licenses. Please see the
corresponding `README` files and license files inside the subdirectories for
more information.


<h2 id="paper">Research Paper</h2>

DeepView.Predict began as a research project in the [EcoSystem Group](https://www.cs.toronto.edu/ecosystem) at the [University of Toronto](https://cs.toronto.edu). The accompanying research paper appeared in the proceedings of [USENIX
ATC'21](https://www.usenix.org/conference/atc21/presentation/yu). If you are
interested, you can read a preprint of the paper [here](https://arxiv.org/abs/2102.00527).

If you use DeepView.Predict in your research, please consider citing our paper:

```bibtex
@inproceedings{habitat-yu21,
  author = {Yu, Geoffrey X. and Gao, Yubo and Golikov, Pavel and Pekhimenko,
    Gennady},
  title = {{Habitat: A Runtime-Based Computational Performance Predictor for
    Deep Neural Network Training}},
  booktitle = {{Proceedings of the 2021 USENIX Annual Technical Conference
    (USENIX ATC'21)}},
  year = {2021},
}
```
<h2 id="contributing">Contributing</h2>

Check out [CONTRIBUTING.md](https://github.com/CentML/habitat/blob/main/CONTRIBUTING.md) for more information on how to help with Habitat.

