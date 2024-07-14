import pickle
import numpy as np
from scipy.stats import gaussian_kde
import sys
import random
from typing import Dict, List

# SET CEIL FOR AVAILABLE RAM (avoid running-out-mem for sampling bmm)
CURR_MEM = 16000000000 # cap at 16GB
BMM_MEM_CEIL = int(0.9 * CURR_MEM)

class main_generator:
    "Special distribution for conv2d, bmm, batch_norm, and linear"

    def __init__(self, ops):

        self._distribution: gaussian_kde = None
        self._ops: str = ops

        if ops == "conv2d" or ops == "batch_norm":
            filename = "conv2d_sampled_params.pkl"

        elif ops == "bmm":
            filename = "bmm_sampled_params.pkl"

        elif ops == "linear":
            filename = "linear_sampled_params.pkl"

        with open(filename, "rb") as f:
            data = pickle.load(f)

        param_dict: Dict[str, int] = dict()
        dist_arr: List[List[int, int]] = []

        if ops in ["conv2d", "bmm", "batch_norm"]:
            # weight by model count
            model_counts: Dict[str, int] = dict()
            for row in data:
                model_name = row[0]
                if model_name not in model_counts:
                    model_counts[model_name] = 0
                model_counts[model_name] += 1

            rows = []
            weights = []
            for row in data:
                rows.append([float(i) for i in row[1:]])
                weights.append(np.log(1 / model_counts[row[0]]))
            rows = np.array(rows).astype("int32").transpose()
            self._distribution = gaussian_kde(rows, weights=weights)

        elif ops == "linear":
            param_dict: Dict[str, List[int]] = {
                "linear_in_features": list(),
                "linear_out_features": list(),
            }
            for item in data:
                param_dict["linear_in_features"].append(item[0])
                param_dict["linear_out_features"].append(item[1])
                dist_arr.append([item[0], item[1]])

            np_dist_array = np.array(dist_arr).transpose()
            self._distribution = gaussian_kde(np_dist_array)

    def generate_sample(self):
        if self._distribution is None:
            sys.exit("Error generating a new distribution")

        round_sample = []
        while True:
            # keep sampling until valid configuration is found
            sample = self._distribution.resample(1)
            if self._ops == "conv2d":
                round_sample = [
                    self.round(sample[0][0]),  # in_channels
                    self.round(sample[1][0]),  # out_channels
                    self.round(sample[2][0]),  # kernel_size
                    self.round(sample[3][0]),  # stride
                    self.round(sample[4][0]),  # padding
                ]
                if round_sample[2] != 0 and round_sample[3] != 0:
                    return round_sample

            elif self._ops == "batch_norm":
                round_sample = [
                    self.round(sample[0][0]),  # in_channels
                    self.round(sample[1][0]),  # out_channels
                    self.round(sample[2][0]),  # kernel_size
                    self.round(sample[3][0]),  # stride
                    self.round(sample[4][0]),  # padding
                ]
                if round_sample[1] != 0:
                    return [round_sample[1]]

            elif self._ops == "bmm":
                round_sample = [
                    self.round(sample[0][0]),  # bs
                    self.round(sample[1][0]),  # left
                    self.round(sample[2][0]),  # middle
                    self.round(sample[3][0]),  # right
                ]
                # validate non-zeros
                # check if available memory (RuntimeError DefaultCPUAllocator: can't allocate memory)
                # 4 for FP32, leaving as default since in matmul the accumulation can be stored in fp32
                matrix_a_size = 4 * round_sample[0] * round_sample[1] * round_sample[2]
                matrix_b_size = 4 * round_sample[0] * round_sample[2] * round_sample[3]
                
                if (
                    np.all(round_sample)
                    and matrix_a_size + matrix_b_size < BMM_MEM_CEIL
                ):
                    return round_sample

            elif self._ops == "linear":
                in_features = self.round(sample[0][0])
                out_features = self.round(sample[1][0])
                if in_features != 0 and out_features != 0:
                    return [
                        in_features,  # in_features
                        out_features,  # out_features
                    ]

    def round(self, n: float) -> int:
        "randomly round up or down"
        n = abs(n)
        frac = n - int(n)
        r = random.random()  # [0,1)
        if r < frac:
            return int(n) + 1
        return int(n)
