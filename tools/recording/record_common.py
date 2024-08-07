import logging
import gc
import random
import signal

import torch

import habitat
from habitat.profiling.operation import OperationProfiler
from database import Recorder
from parameter_generator import main_generator
import sys


logger = logging.getLogger(__name__)

"""
Some operators such as conv2d and linear need to be sampled from a different distribution (gaussian + uniform)
main_generator generates these new samples
"""
SPECIAL_SAMPLING_OPS = ['conv2d','linear', 'batch_norm', 'bmm']

class Measurer:
    def __init__(
        self,
        op_name,
        recorder_config,
        index_to_config,
        config_to_profiler_args,
        index_filter=None,
    ):
        self._op_name = op_name
        self._recorder_config = recorder_config
        self._index_to_config = index_to_config
        self._config_to_profiler_args = config_to_profiler_args
        self._index_filter = index_filter
        self._shutdown_early = False
        self._initialize()

    def _initialize(self):
        def signal_handler(signal, frame):
            logger.info(
                "Received shutdown command. Will shutdown after "
                "completing current measurement."
            )
            self._shutdown_early = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def add_args(self, parser):
        parser.add_argument("device", type=str)
        parser.add_argument("--seed", type=int, default=1337)
        parser.add_argument("--num-points", type=int, default=200000)
        parser.add_argument("--rank", type=int, default=0)
        parser.add_argument("--world-size", type=int, default=1)
        parser.add_argument("--no-kernels", action="store_true")
        parser.add_argument("--skip", type=int)

    def measure_configurations(self, args, num_configs):
        # Store the arguments for future use
        self._args = args

        if args.rank >= args.world_size:
            raise ValueError("Rank must be less than world size.")
        if args.num_points % args.world_size != 0:
            raise ValueError("Number of points must be divisible by the world size.")

        # Want to ensure we measure the same configurations across each device
        random.seed(args.seed)

        logger.info("Total configurations: %d", num_configs)

        to_record = random.sample(range(num_configs), args.num_points)
        print(f"before filter :{len(to_record)}")
        if self._index_filter is not None:
            to_record = list(
                filter(
                    lambda idx: self._index_filter(args, idx),
                    to_record,
                )
            )
            slice_size = len(to_record) // args.world_size
        else:
            slice_size = args.num_points // args.world_size

        logger.info("Total configurations after filtering: %d", len(to_record))
        logger.info("Slice size: %d", slice_size)

        if args.world_size != 1:
            # If we split the sample set across multiple workers, we
            # want to increase the number of overlapping samples between
            # a machine with just one worker if this recording script is
            # stopped early. This is because the workers process the
            # configurations sequentially.
            random.shuffle(to_record)
            offset = slice_size * args.rank
            to_record = to_record[offset : offset + slice_size]

        file_name = "{}-{}-{}.sqlite".format(
            self._op_name,
            args.device,
            args.rank,
        )
        self._recorder = Recorder(file_name, self._recorder_config)
        num_recordings = self._recorder.get_num_recordings()

        # We make 2 recordings per configuration
        num_configs_measured = num_recordings // 2
        logger.info(
            "--- Found %d recordings in %s, so skipping the first %d configurations ---",
            num_recordings,
            file_name,
            num_configs_measured,
        )

        # A device doesn't need to be passed in here
        self._profiler = OperationProfiler(device=None, measure_for=3)

        logger.info("Warming up...")
        self._measure(self._index_to_config(args, to_record[0]))
        self._measure(self._index_to_config(args, to_record[1]))
        self._measure(self._index_to_config(args, to_record[2]))

        logger.info(
            "Starting to record. This process records slice %d of %d.",
            args.rank + 1,
            args.world_size,
        )

        last_count = self._recorder.get_num_recordings()

        # For conv2d and linear, obtain a new generator combining random and gaussian samples
        if self._op_name in SPECIAL_SAMPLING_OPS:
            params_generator = main_generator(self._op_name)

        try:
            for idx, config_id in enumerate(to_record):
                if idx < num_configs_measured:
                    continue
                if args.skip is not None and idx < args.skip:
                    continue

                if self._op_name in SPECIAL_SAMPLING_OPS:
                    # only for conv2d and linear replace the features with the ones obtained from main_generator
                    sample = params_generator.generate_sample()
                    config = list(self._index_to_config(args, config_id))
                    config[len(config) - len(sample) :] = sample
                    config = tuple(config)
                else:
                    config = self._index_to_config(args, config_id)

                logger.info(f"config: {config}")

                fw, bw = self._measure(config)

                self._record(config, fw, bw)

                if (idx + 1) % 100 == 0:
                    logger.info("[{}/{}] Processed".format(idx + 1, slice_size))

                if idx % 100 == 0:
                    self._recorder.commit()
                    cur_count = self._recorder.get_num_recordings()
                    logger.info(
                        f"commit. num_recordings: {cur_count}, new: {cur_count-last_count}"
                    )
                    last_count = cur_count

                if self._shutdown_early:
                    break
        finally:
            self._recorder.commit()

    def _measure(self, config):
        config_with_problems = [
            # include here the configs that throw errors and we need to skip
            # eg conv2d: (False, 50, 182, 1706, 205, 2, 2, 1)
        ]
        try:
            kwargs = self._config_to_profiler_args(config)
            if kwargs is None or config in config_with_problems:
                return None, None
            return self._profiler.measure_operation(
                record_kernels=not self._args.no_kernels,
                **kwargs,
            )

        except RuntimeError as e:
            msg = str(e)

            if "out of memory" in msg:
                allocated = torch.cuda.memory_allocated()
                torch.cuda.empty_cache()
                gc.collect()
                logger.info(
                    f"Cleared memory: {allocated} -> {torch.cuda.memory_allocated()}"
                )

            if (
                "out of memory" not in msg
                and "cuDNN error" not in msg
                and "Calculated padded" not in msg
            ):
                logger.exception("Unexpected error during measurement.")
                logger.info(f"config: {config}")
                sys.exit(
                    "error with the current configuration, please add it to config_with_problems array"
                )
            return None, None

    def _record(self, config, forward_result, backward_result):
        if forward_result is not None:
            self._recorder.record(
                config=config,
                is_forward=True,
                ktime_ns=forward_result.ktime_ns,
                run_time_ms=forward_result.run_time_ms,
                recorded_kernels=forward_result.kernels,
            )
        if backward_result is not None:
            self._recorder.record(
                config=config,
                is_forward=False,
                ktime_ns=backward_result.ktime_ns,
                run_time_ms=backward_result.run_time_ms,
                recorded_kernels=backward_result.kernels,
            )
