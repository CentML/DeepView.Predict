import argparse
import random
import torch
import numpy

from habitat.analysis.mlp.mlp import RuntimePredictor
from habitat.analysis.mlp.dataset_process import get_devices

def main():
    parser = argparse.ArgumentParser(description="MLP Training Script")
    parser.add_argument("operation", type=str)
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--layer_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1337)

    args = parser.parse_args()

    devices = get_devices(args.dataset_path)
    print("devices", devices)

    # Ensure reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)

    predictor = RuntimePredictor(args.operation, devices, args.layers, args.layer_size)
    # def __init__(self, model_name, devices, layers, layer_size, model_path=None):
    predictor.train_with_dataset(args.dataset_path, epochs=args.epochs)


if __name__ == "__main__":
    main()
