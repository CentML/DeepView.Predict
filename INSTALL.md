# Installation
## CUDA Toolkit
Habitat depends on the CUDA toolkit, including the CUPTI examples. You can find a list of installers [here](https://developer.nvidia.com/cuda-toolkit-archive) from NVIDIA's website.

After installation, verify that the folder `/usr/local/cuda/extras/CUPTI/samples` exists. On other distributions such as Arch Linux, this could also be located at `/opt/cuda/extras/CUPTI/samples`.

## CMake
Habitat requires `cmake` versions 3.17 or above. To do so, consult `docker/Dockerfile` or run the following commands:
```sh
wget "https://github.com/Kitware/CMake/releases/download/v3.17.0-rc1/cmake-3.17.0-rc1.tar.gz" -O cmake-3.17.0-rc1.tar.gz
tar xzf cmake-3.17.0-rc1.tar.gz

cd cmake-3.17.0-rc1 && \
    ./bootstrap && \
    make -j && \
    sudo make install
```

## Building Habitat
Change directory to `analyzer` and ensure that:
* the Python version in `SO_PATH` is set correctly (e.g. `habitat_cuda.cpython-39-x86_64-linux-gnu.so` for Python 3.9)
* the `CUPTI_PATH` variable is pointed to the CUPTI directory for your distribution

Then, to begin building, run `./install-dev.sh`.

## Download pretrained models
The MLP component of Habitat requires pretrained models that are not included in the main repository. To download them, run:
```sh
wget https://zenodo.org/record/4876277/files/habitat-models.tar.gz?download=1 -O habitat-models.tar.gz
./extract-models.sh habitat-models.tar.gz
```

## Verify installation
You can verify your Habitat installation by running the simple usage example:
```py
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