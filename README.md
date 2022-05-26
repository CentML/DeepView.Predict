# Habitat
Habitat is a tool that predicts a deep neural network's training iteration
execution time on a given GPU. It currently supports PyTorch. To learn more
about how Habitat works, please see our [research
paper](https://arxiv.org/abs/2102.00527).
## Installation
You can install Habitat using the prebuilt wheel files. To install, download the whl files from the [releases page](https://github.com/CentML/habitat/releases) then run:
```sh
pip install habitat*.whl
```

## Usage example
You can verify your Habitat installation by running the simple usage example. This example measures a single inference iteration of Resnet50 on the RTX2080Ti and extrapolates the runtime to the V100. 
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
## Development Environment Setup
Habitat requires both the native component and the Python binding to function. For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Release process
Run the `Build Habitat` GitHub action. This will build the wheel files for each platform.

## Release history
See [Releases](https://github.com/CentML/habitat/releases).
## Meta

Habitat began as a research project in the [EcoSystem
Group](https://www.cs.toronto.edu/ecosystem) at the [University of
Toronto](https://cs.toronto.edu). The accompanying research paper will appear
in the proceedings of [USENIX
ATC'21](https://www.usenix.org/conference/atc21/presentation/yu). If you are
interested, you can read a preprint of the paper
[here](https://arxiv.org/abs/2102.00527).

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
## Contributing
 - Guidelines on how to contribute to the project