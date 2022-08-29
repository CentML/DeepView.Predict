# Habitat

[![License](https://img.shields.io/badge/license-Apache--2.0-green?style=flat)](https://github.com/CentML/habitat/blob/main/LICENSE)



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

Habitat is a tool that predicts a deep neural network's training iteration execution time on a given GPU. It currently supports PyTorch. To learn more about how Habitat works, please see our [research paper](https://arxiv.org/abs/2102.00527).

<h2 id="installation">Installation</h2>

To run Habitat, you need:
- [Python 3.6+](https://www.python.org/)
- [Pytorch 1.1.0+](https://pytorch.org/)
- A system equiped with an Nvidia GPU.

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

**NOTE:** Not implmented yet
```zsh
python3 -m pip install habitat
python3 -c "import habitat"
```

<h2 id="build">Building from source</h2>

Prerequsites:
- A system equiped with an Nvidia GPU with properly configured CUDA
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
- [cmake v3.17+](https://github.com/Kitware/CMake/releases)
- [Habitat pre-trained models](https://zenodo.org/record/4876277)

```zsh
git clone https://github.com/CentML/habitat.git && cd habitat
git submodule init && git submodule update

# Download the pre-trained models
cd analyzer
curl -O https://zenodo.org/record/4876277/files/habitat-models.tar.gz\?download\=1

# Install the models
./extract-models.sh
```

**Note:** Habitat needs access to your GPU's performance counters, which requires special permissions if you are running with a recent driver (418.43 or later). If you encounter a `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` error when running Habitat, please follow the instructions [here](https://developer.nvidia.com/ERR_NVGPUCTRPERM) and in [issue #5](https://github.com/geoffxy/habitat/issues/5).

### Building with Docker

Habitat has been tested to work on the latest version of [NVIDIA NGC PyTorch containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). 

1. To build Habitat with Docker, first run the NGC container.
```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:22.08-py3
```
2. Inside the container, clone the repository then build and install the Habitat Python package:
```bash
git clone -r https://github.com/centml/habitat
./habitat/analyzer/install-dev.sh
```
3. Download and extract the pretrained models by following the steps in the previous section. 

### Building without Docker

1. Install CUPTI

CUPTI is a profiling interface required by Habitat. Select the correct version of CUDA [here](https://developer.nvidia.com/cuda-toolkit-archive) and following the instructions to add NVIDIA's repository. Then, install CUPTI with:
```bash
sudo apt-get install cuda-cupti-11-x
```  
where `11-x` represents the version of CUDA you have installed.

2. Install `CMake` 3.17+. 

Follow these steps to download and install a precompiled version of CMake:
```bash
wget https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0-linux-x86_64.sh
chmod +x cmake-3.24.0-linux-x86_64.sh
mkdir /opt/cmake
sh cmake-3.24.0-linux-x86_64.sh --prefix=/opt/cmake --skip-license
ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
```
You can verify the version of CMake you installed with the following:
```bash
cmake --version
```
3. Build and install the Habitat Python package:
```bash
git clone https://github.com/centml/habitat
./habitat/analyzer/install-dev.sh
```
4. Download and extract the pretrained models by following the steps in the previous section. 

<h2 id="getting-started">Usage example</h2>

You can verify your Habitat installation by running the simple usage example:
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

```zsh
python3 example.py
```

See [experiments/run_experiment.py](https://github.com/CentML/habitat/tree/main/experiments) for other examples of Habitat usage.

<h2 id="dev-setup">Development Environment Setup</h2>

<h2 id="release-process">Release Process</h2>

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

Habitat began as a research project in the [EcoSystem Group](https://www.cs.toronto.edu/ecosystem) at the [University of Toronto](https://cs.toronto.edu). The accompanying research paper appeared in the proceedings of [USENIX
ATC'21](https://www.usenix.org/conference/atc21/presentation/yu). If you are
interested, you can read a preprint of the paper [here](https://arxiv.org/abs/2102.00527).

If you use Habitat in your research, please consider citing our paper:

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

