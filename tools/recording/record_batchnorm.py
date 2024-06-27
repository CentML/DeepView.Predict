import argparse
import logging
import math
import torch
from record_common import Measurer
import features as f

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = False

def index_to_config(args, index):
    batch = (index % args.batches) + 1
    index //= args.batches

    channels = (index % args.channels) + 1
    index //= args.image_size

    image_size = (index % args.image_size) + 1

    return (
        batch,
        image_size,
        channels,
    )

def index_filter(args, index):
    config = index_to_config(args, index) # (batch, channels, image_size)
    # NOTE: We multiply because the dimensions have different ranges; we want
    #       them to each "contribute equally". We weigh the image size more to
    #       select smaller image sizes.
    # image_size (1-dim) * channels
    batchnorm_size = math.pow(config[1], 1.15) * config[2]

    # NOTE: This value was chosen arbitrarily: we don't want the 
    #       channels and image size to all be too large. This way, large values
    #       for the channels would lead to a smaller image size (and
    #       vice versa).

    # NOTE: batch size can't be 1. in _verify_batch_size
    # raise ValueError(f"Expected more than 1 value per channel when training, got input size {size}")
    return batchnorm_size <= 35000000 and config[0] > 1

def config_to_profiler_args(config):
    (
        batch,
        image_size,
        channels,
     ) = config
    
    device = torch.device('cuda')
    batchnorm = torch.nn.BatchNorm2d(channels).to(device)
    inp = torch.randn((batch, channels, image_size, image_size), device=device)
    inp = inp.requires_grad_()

    return {
        'func': batchnorm,
        'args': (inp, ),
        'kwargs': {},
    }

def main():
    measurer = Measurer(
        op_name = 'batch_norm',
        recorder_config=f.batch_norm,
        index_to_config=index_to_config,
        index_filter=index_filter,
        config_to_profiler_args=config_to_profiler_args
    )

    parser = argparse.ArgumentParser()
    measurer.add_args(parser)
    parser.add_argument('--batches', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--channels', type=int, default=2048)

    args = parser.parse_args()

    num_configs = (
        args.batches *
        args.image_size * 
        args.channels
    )

    measurer.measure_configurations(args, num_configs)

if __name__ == '__main__':
    kwargs = {
        "format": "%(asctime)s %(levelname)-8s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M",
        "level": logging.INFO,
    }
    logging.basicConfig(**kwargs)
    main()