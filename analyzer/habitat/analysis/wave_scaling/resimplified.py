import math

from habitat.analysis.kernels import PredictedKernel
from habitat.analysis.wave_scaling.common import calculate_wave_info
import logging
logger = logging.getLogger(__name__)

def resimplified_wave_scaling(
    kernel,
    origin_device,
    dest_device,
    metadata_manager,
):
    origin_wave_size, dest_wave_size, origin_occupancy, dest_occupancy = (
        calculate_wave_info(
            kernel,
            origin_device,
            dest_device,
            metadata_manager,
        )
    )
    # print(f"{origin_wave_size}, {dest_wave_size}, {origin_occupancy}, {dest_occupancy}")

    # Check if the kernel is too "small" - if it doesn't fill a single wave
    # on the current device AND if it doesn't fill a single wave on the
    # destination device
    if (origin_wave_size == 0 or dest_wave_size == 0):
        logger.warn(f"One or more invalid wave sizes: kernel: {kernel.name} origin: {origin_wave_size}, dest: {dest_wave_size}")
    if ((origin_wave_size == 0 or dest_wave_size == 0) or (kernel.num_blocks // origin_wave_size == 0 and kernel.num_blocks // dest_wave_size == 0)):
        # We scale the run time by the compute factor only
        origin_max_occupancy = math.ceil(
            kernel.num_blocks / origin_device.num_sms
        )
        dest_max_occupancy = math.ceil(
            kernel.num_blocks / dest_device.num_sms
        )
        return PredictedKernel(kernel, kernel.run_time_ns)

    bandwidth_ratio = (
        origin_device.mem_bandwidth_gb / dest_device.mem_bandwidth_gb
    )

    return PredictedKernel(kernel, kernel.run_time_ns * bandwidth_ratio)
