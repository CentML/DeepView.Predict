#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>

namespace {

// We don't want any symbols from this header to be visible outside of this file
#include "cuda_occupancy.h"

cudaOccDeviceProp occDeviceProps(const habitat::cuda::DeviceProperties& properties) {
  cudaOccDeviceProp device_properties;
  device_properties.computeMajor = properties.compute_major;
  device_properties.computeMinor = properties.compute_minor;
  device_properties.maxThreadsPerBlock = properties.max_threads_per_block;
  device_properties.maxThreadsPerMultiprocessor = properties.max_threads_per_multiprocessor;
  device_properties.regsPerBlock = properties.regs_per_block;
  device_properties.regsPerMultiprocessor = properties.regs_per_multiprocessor;
  device_properties.warpSize = properties.warp_size;
  device_properties.sharedMemPerBlock = properties.shared_mem_per_block;
  device_properties.sharedMemPerMultiprocessor = properties.shared_mem_per_multiprocessor;
  device_properties.numSms = properties.num_sms;
  device_properties.sharedMemPerBlockOptin = properties.shared_mem_per_block_optin;
  return device_properties;
}

}

namespace habitat {
namespace cuda {

uint32_t KernelMetadata::threadBlockOccupancy(const DeviceProperties& device) const {
  return threadBlockOccupancy(device, registers_per_thread_);
}

uint32_t guess_shared_memory_size(size_t mem_used) {
  if (mem_used <= 48*1024) return 48*1024;
  fprintf(stderr, "Guessing larger shared memory size. Used: %ld\n", mem_used);
  if (mem_used <= 64*1024) return 64*1024;
  if (mem_used <= 100*1024) return 100*1024;
  return 0;
}

uint32_t guess_shared_memory_carveout_perc(size_t mem_used) {
  if (mem_used <= 48*1024) return SHAREDMEM_CARVEOUT_DEFAULT;
  if (mem_used <= 64*1024) return 64;
  if (mem_used <= 96*1024) return 96;
  if (mem_used <= 100*1024) return 100;

  // otherwise we return invalid output
  return 101;
}

uint32_t KernelMetadata::threadBlockOccupancy(
    const DeviceProperties& device, uint16_t registers_per_thread) const {
  cudaOccDeviceProp device_properties(occDeviceProps(device));
  cudaOccDeviceState device_state;
  cudaOccFuncAttributes attributes;
  attributes.maxThreadsPerBlock = INT_MAX;
  attributes.maxDynamicSharedSizeBytes = INT_MAX;
  attributes.numRegs = registers_per_thread;
  attributes.sharedSizeBytes = static_shared_memory_;

  size_t shmem_used = dynamic_shared_memory_ + static_shared_memory_, shmem_alloc = shmem_used;
  size_t shared_memory_max = dynamic_shared_memory_;

  // If we're using more than the default slice for shared memory, then
  // update the carveoutConfig so that cudaOccMaxActiveBlocksPerMultiprocessor 
  // return 0.
  if (shmem_used > 48*1024) {
      cudaOccAlignUpShmemSizeVoltaPlus(&shmem_alloc, &device_properties);
      if (shmem_alloc > device_properties.sharedMemPerMultiprocessor) {
        shared_memory_max = device_properties.sharedMemPerMultiprocessor;
        shmem_alloc = device_properties.sharedMemPerMultiprocessor;
      }
      device_state.carveoutConfig = shmem_alloc * 100 / device_properties.sharedMemPerMultiprocessor;
      device_properties.sharedMemPerBlock = std::max(device_properties.sharedMemPerBlock, shmem_alloc);
  }

  int res;
  cudaOccResult result;
  if ((res = cudaOccMaxActiveBlocksPerMultiprocessor(
        &result,
        &device_properties,
        &attributes,
        &device_state,
        block_size_,
        shared_memory_max)) != CUDA_OCC_SUCCESS) {
    fprintf(stderr, "for kernel: %s\n", this->name().c_str());
    fprintf(stderr, "KernelMetadata::threadBlockOccupancy: cudaOccMaxActiveBlocksPerMultiprocessor failed and returned %d.\n", res);

    return 0;
  }

  if (result.activeBlocksPerMultiprocessor == 0) {
      fprintf(stderr, "for kernel: %s\n", this->name().c_str());
      fprintf(stderr, "KernelMetadata::threadBlockOccupancy: succeeded, returning %d.\n", result.activeBlocksPerMultiprocessor);
      fprintf(stderr, "static_shmem: %d, dynamic_shmem: %d\n", static_shared_memory_, dynamic_shared_memory_);
  }
  return result.activeBlocksPerMultiprocessor;
}

bool operator==(const KernelMetadata& lhs, const KernelMetadata& rhs) {
  return lhs.num_blocks_ == rhs.num_blocks_ &&
    lhs.block_size_ == rhs.block_size_ &&
    lhs.dynamic_shared_memory_ == rhs.dynamic_shared_memory_ &&
    lhs.static_shared_memory_ == rhs.static_shared_memory_ &&
    lhs.registers_per_thread_ == rhs.registers_per_thread_ &&
    lhs.name_ == rhs.name_;
}

void KernelInstance::addMetric(std::string name, double value) {
  metrics_.push_back(std::make_pair(std::move(name), value));
}

}
}
