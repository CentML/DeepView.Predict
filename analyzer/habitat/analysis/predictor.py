import functools
import logging
import operator
import numpy as np
import math
import pickle

from habitat.analysis import SPECIAL_OPERATIONS
from habitat.analysis.operation import PredictedOperation
from habitat.analysis.run_time import RunTimePrediction, RunTimePurePrediction
from habitat.analysis.wave_scaling.metadata import MetadataManager
from habitat.analysis.wave_scaling.unified import unified_wave_scaling
from habitat.data import path_to_data
from habitat.utils import ns_to_ms, ms_to_ns, name_all_arguments

from habitat.analysis.mlp.mlp import RuntimePredictor

logger = logging.getLogger(__name__)

CONV2D_PARAMS = [
    "input",
    "weight",
    "bias",
    "stride",
    "padding",
    "dilation",
    "groups",
]

CONVTRANSPOSE2D_PARAMS = [
    "input",
    "weight",
    "bias",
    "stride",
    "padding",
    "dilation",
    "groups",
]

LINEAR_PARAMS = ["input", "weight", "bias"]

BMM_PARAMS = ["input", "mat2", "out"]

LSTM_PARAMS_NO_BATCH_SIZES = [
    "input",
    "hx",
    "flat_weights",
    "bias",
    "num_layers",
    "dropout",
    "training",
    "bidirectional",
    "batch_first",
]

LSTM_PARAMS = [
    "input",
    "batch_sizes",
    "hx",
    "flat_weights",
    "bias",
    "num_layers",
    "dropout",
    "training",
    "bidirectional",
]

MATMUL_PARAMS = ["input", "other", "out"]

BATCH_NORM = [
    "input",
    "running_mean",
    "running_var",
    "weight",
    "bias",
    "training",
    "momentum",
    "eps",
]


class Predictor:
    def __init__(
        self, kernel_metadata_file=None, wave_scaling_strategy=unified_wave_scaling
    ):
        self._kernel_metadata = MetadataManager(
            kernel_metadata_file
            if kernel_metadata_file is not None
            else path_to_data("kernels.sqlite")
        )
        self._wave_scaling_strategy = wave_scaling_strategy

        # Load MLP predictor from saved models
        self.linear_pred = {
            "fp16": RuntimePredictor(
                "linear", 8, 1024, path_to_data("linear/model_fp16.pth")
            ),
            "fp32": RuntimePredictor(
                "linear",
                8,
                1024,
                path_to_data("linear/model_fp32.pth"),
            ),
            "knames_fp32": self._load_kernel_names(
                path_to_data("linear/knames_fp32.pkl")
            ),
            "knames_fp16": self._load_kernel_names(
                path_to_data("linear/knames_fp16.pkl")
            ),
        }

        self.lstm_pred = {
            "fp32": RuntimePredictor(
                "lstm",
                8,
                1024,
                path_to_data("lstm/model_fp32.pth"),
            )
        }

        self.conv2d_pred = {
            "fp16": RuntimePredictor(
                "conv2d",
                8,
                1024,
                path_to_data("conv2d/model_fp16.pth"),
            ),
            "fp32": RuntimePredictor(
                "conv2d",
                8,
                1024,
                path_to_data("conv2d/model_fp32.pth"),
            ),
            "knames_fp32": self._load_kernel_names(
                path_to_data("conv2d/knames_fp32.pkl")
            ),
            "knames_fp16": self._load_kernel_names(
                path_to_data("conv2d/knames_fp16.pkl")
            ),
        }

        self.bmm_pred = {
            "fp16": RuntimePredictor(
                "bmm",
                8,
                1024,
                path_to_data("bmm/model_fp16.pth"),
            ),
            "fp32": RuntimePredictor(
                "bmm",
                8,
                1024,
                path_to_data("bmm/model_fp32.pth"),
            ),
            "knames_fp32": self._load_kernel_names(path_to_data("bmm/knames_fp32.pkl")),
            "knames_fp16": self._load_kernel_names(path_to_data("bmm/knames_fp16.pkl")),
        }

        self.conv_transpose2d_pred = {
            "fp16": RuntimePredictor(
                "conv_transpose2d",
                8,
                1024,
                path_to_data("conv_transpose2d/model_fp16.pth"),
            ),
            "fp32": RuntimePredictor(
                "conv_transpose2d",
                8,
                1024,
                path_to_data("conv_transpose2d/model_fp32.pth"),
            ),
            "knames_fp32": self._load_kernel_names(
                path_to_data("conv_transpose2d/knames_fp32.pkl")
            ),
            "knames_fp16": self._load_kernel_names(
                path_to_data("conv_transpose2d/knames_fp16.pkl")
            ),
        }

        self.batch_norm_pred = {
            "fp16": RuntimePredictor(
                "batch_norm",
                8,
                1024,
                path_to_data("batch_norm/model_fp16.pth"),
            ),
            "fp32": RuntimePredictor(
                "batch_norm",
                8,
                1024,
                path_to_data("batch_norm/model_fp32.pth"),
            ),
            "knames_fp32": self._load_kernel_names(
                path_to_data("batch_norm/knames_fp32.pkl")
            ),
            "knames_fp16": self._load_kernel_names(
                path_to_data("batch_norm/knames_fp16.pkl")
            ),
        }

    def predict_operation(
        self, operation, dest_device, unscaled=False, to_precision=None
    ):
        if operation.name not in SPECIAL_OPERATIONS:
            return PredictedOperation(
                operation,
                self._wave_scale(operation.forward, dest_device),
                (
                    self._wave_scale(operation.backward, dest_device)
                    if operation.backward is not None
                    else None
                ),
                dest_device,
            )

        if operation.name == "conv2d":
            return self._special_scale(
                operation, dest_device, self._conv2d_scale, unscaled, to_precision
            )
        elif operation.name == "lstm":
            return self._special_scale(
                operation, dest_device, self._lstm_scale, unscaled, to_precision
            )
        elif operation.name == "linear":
            return self._special_scale(
                operation, dest_device, self._linear_scale, unscaled, to_precision
            )
        elif operation.name in ["bmm", "__matmul__"]:
            return self._special_scale(
                operation, dest_device, self._bmm_scale, unscaled, to_precision
            )
        elif operation.name == "conv_transpose2d":
            return self._special_scale(
                operation,
                dest_device,
                self._conv_transpose2d_scale,
                unscaled,
                to_precision,
            )
        elif operation.name == "batch_norm":
            return self._special_scale(
                operation, dest_device, self._batch_norm_scale, unscaled, to_precision
            )

        logger.warn("Unhandled special operation: %s", operation.name)
        return PredictedOperation(
            operation,
            operation.forward,
            operation.backward,
            dest_device,
        )

    def _wave_scale(self, run_time, dest_device):
        run_time_ns = ms_to_ns(run_time.run_time_ms)
        total_ktime_ns = sum(map(lambda k: k.run_time_ns, run_time.kernels))
        overhead_ns = run_time_ns - total_ktime_ns

        predicted_kernels = list(
            map(
                lambda kernel: self._wave_scaling_strategy(
                    kernel,
                    run_time.device,
                    dest_device,
                    self._kernel_metadata,
                ),
                run_time.kernels,
            )
        )

        return RunTimePrediction(
            overhead_ns=0 if overhead_ns < 0 else overhead_ns,
            predicted_kernels=predicted_kernels,
            device=dest_device,
        )

    def _special_scale(
        self, operation, dest_device, scaler, unscaled=False, to_precision=None
    ):
        predicted_ms = scaler(operation, dest_device, unscaled, to_precision)

        if predicted_ms < 0:
            logger.warn(
                "Operation %s predicted run time %.2f ms",
                operation.name,
                predicted_ms,
            )
            predicted_ms = 0.0

        return PredictedOperation(
            operation,
            RunTimePurePrediction(predicted_ms, dest_device),
            None,
            dest_device,
        )

    def _conv2d_scale(self, operation, dest_device, unscaled=False, to_precision=None):
        # 1. Merge arguments (give them all names)
        merged = name_all_arguments(
            CONV2D_PARAMS,
            operation.arguments.args,
            operation.arguments.kwargs,
        )

        # 2. Construct arguments that the predictor expects
        arguments = dict(
            batch=merged["input"][0],
            image_size=merged["input"][2],
            in_channels=merged["input"][1],
            out_channels=merged["weight"][0],
            kernel_size=merged["weight"][2],
            stride=(
                merged["stride"][0]
                if isinstance(merged["stride"], tuple)
                else merged["stride"]
            ),
            padding=(
                merged["padding"][0]
                if isinstance(merged["padding"], tuple)
                else merged["padding"]
            ),
            bias=(1 if merged.get("bias", None) != None else 0),
        )

        fw_kernel = operation.forward.kernels if operation.forward else []
        bw_kernel = operation.backward.kernels if operation.backward else []
        kernels = fw_kernel + bw_kernel

        # features are the same independent of the precision
        arguments = [arguments[x] for x in self.conv2d_pred["fp32"].model.features]

        # 3. Call model to make prediction
        pred_dest = self._calculate_dest_runtime(
            self.conv2d_pred, kernels, operation, arguments, dest_device
        )
        pred_orig = self._calculate_dest_runtime(
            self.conv2d_pred, kernels, operation, arguments, operation.device
        )

        if unscaled:
            return pred_dest

        if dest_device.name == operation.device.name:  # local prediction
            return pred_orig

        return operation.run_time_ms * pred_dest / pred_orig

    def _conv_transpose2d_scale(
        self, operation, dest_device, unscaled=False, to_precision=None
    ):
        # 1. Merge arguments (give them all names)
        merged = name_all_arguments(
            CONVTRANSPOSE2D_PARAMS,
            operation.arguments.args,
            operation.arguments.kwargs,
        )

        # 2. Construct arguments that the predictor expects
        arguments = dict(
            batch=merged["input"][0],
            image_size=merged["input"][2],
            in_channels=merged["input"][1],
            out_channels=merged["weight"][0],
            kernel_size=merged["weight"][2],
            stride=(
                merged["stride"][0]
                if isinstance(merged["stride"], tuple)
                else merged["stride"]
            ),
            padding=(
                merged["padding"][0]
                if isinstance(merged["padding"], tuple)
                else merged["padding"]
            ),
            bias=(1 if merged.get("bias", None) != None else 0),
        )

        fw_kernel = operation.forward.kernels if operation.forward else []
        bw_kernel = operation.backward.kernels if operation.backward else []
        kernels = fw_kernel + bw_kernel

        arguments = [
            arguments[x] for x in self.conv_transpose2d_pred["fp32"].model.features
        ]

        # 3. Call model to make prediction
        pred_dest = self._calculate_dest_runtime(
            self.conv_transpose2d_pred, kernels, operation, arguments, dest_device
        )
        pred_orig = self._calculate_dest_runtime(
            self.conv_transpose2d_pred, kernels, operation, arguments, operation.device
        )

        if unscaled:
            return pred_dest

        if dest_device.name == operation.device.name:  # local prediction
            return pred_orig

        return operation.run_time_ms * pred_dest / pred_orig

    def _linear_scale(self, operation, dest_device, unscaled=False, to_precision=None):
        merged = name_all_arguments(
            LINEAR_PARAMS,
            operation.arguments.args,
            operation.arguments.kwargs,
        )

        # The input to the linear function in PyTorch can contain an arbitrary
        # number of dimensions between the batch size and the input feature
        # dimensions.
        #
        #   e.g., The input can have size (32, 50, 512), where 32 is the batch
        #         size and 512 is the input feature dimension.
        #
        # This means that the effective batch size is the product of all the
        # dimensions before the input feature dimension (e.g., 32 * 50 = 1600).
        # We need to take this into account when making a prediction.
        effective_batch = functools.reduce(
            operator.mul,
            merged["input"][:-1],
        )

        arguments = dict(
            batch=effective_batch,
            in_features=merged["weight"][1],
            out_features=merged["weight"][0],
            bias=(1 if merged.get("bias", None) != None else 0),
        )

        fw_kernel = operation.forward.kernels if operation.forward else []
        bw_kernel = operation.backward.kernels if operation.backward else []
        kernels = fw_kernel + bw_kernel

        arguments = [arguments[x] for x in self.linear_pred["fp32"].model.features]

        pred_dest = self._calculate_dest_runtime(
            self.linear_pred, kernels, operation, arguments, dest_device
        )

        pred_orig = self._calculate_dest_runtime(
            self.linear_pred, kernels, operation, arguments, operation.device
        )

        if unscaled:
            return pred_dest

        if dest_device.name == operation.device.name:  # local prediction
            return pred_orig

        return operation.run_time_ms * pred_dest / pred_orig

    def _bmm_scale(self, operation, dest_device, unscaled=False, to_precision=None):
        # nn.Linear may call __matmul__ which in turn calls bmm
        # but the shape of the arguments may be [a,b,c,d].
        # So we need to reshape them into [a*b,c,d]
        reshape_args = []
        for arg in operation.arguments.args:
            if len(arg) > 3:
                reshape_args.append([math.prod(arg[:-2]), arg[-2], arg[-1]])
            else:
                reshape_args.append(arg)
        operation.arguments.args = reshape_args

        merged = name_all_arguments(
            BMM_PARAMS,
            operation.arguments.args,
            operation.arguments.kwargs,
        )

        arguments = dict(
            batch=merged["input"][0],
            left=merged["input"][1],
            middle=merged["input"][2],
            right=merged["mat2"][2],
        )

        fw_kernel = operation.forward.kernels if operation.forward else []
        bw_kernel = operation.backward.kernels if operation.backward else []
        kernels = fw_kernel + bw_kernel

        arguments = [arguments[x] for x in self.bmm_pred["fp32"].model.features]

        pred_dest = self._calculate_dest_runtime(
            self.bmm_pred, kernels, operation, arguments, dest_device
        )
        pred_orig = self._calculate_dest_runtime(
            self.bmm_pred, kernels, operation, arguments, operation.device
        )

        if unscaled:
            return pred_dest

        if dest_device.name == operation.device.name:  # local prediction
            return pred_orig

        return operation.run_time_ms * pred_dest / pred_orig

    def _lstm_scale(self, operation, dest_device, unscaled=False, to_precision=None):
        # This is hacky, but unfortunately the only way to differentiate these
        # overloaded LSTM calls.
        has_batch_sizes = isinstance(operation.arguments.args[4], bool)

        if not has_batch_sizes:
            merged = name_all_arguments(
                LSTM_PARAMS_NO_BATCH_SIZES,
                operation.arguments.args,
                operation.arguments.kwargs,
            )
            arguments = dict(
                bias=(1 if merged.get("bias", None) != None else 0),
                bidirectional=(1 if merged["bidirectional"] else 0),
                batch=merged["input"][1],  # We require the batch to be in position 1
                seq_len=merged["input"][0],
                input_size=merged["input"][2],
                hidden_size=merged["hx"][0][2],
                num_layers=merged["num_layers"],
            )

        else:
            merged = name_all_arguments(
                LSTM_PARAMS,
                operation.arguments.args,
                operation.arguments.kwargs,
            )
            max_batch_size = max(operation.arguments.special["batch_sizes"])
            arguments = dict(
                bias=(1 if merged.get("bias", None) != None else 0),
                bidirectional=(1 if merged["bidirectional"] else 0),
                batch=max_batch_size,
                seq_len=merged["input"][0] // max_batch_size,
                input_size=merged["input"][1],
                hidden_size=merged["hx"][0][2],
                num_layers=merged["num_layers"],
            )

        arguments = [arguments[x] for x in self.lstm_pred["fp32"].model.features]

        pred_dest = self.lstm_pred["fp32"].predict(arguments, dest_device.name)
        pred_orig = self.lstm_pred["fp32"].predict(arguments, operation.device.name)

        if unscaled:
            return pred_dest

        if dest_device.name == operation.device.name:  # local prediction
            return pred_orig

        return operation.run_time_ms * pred_dest / pred_orig

    def _batch_norm_scale(
        self, operation, dest_device, unscaled=False, to_precision=None
    ):
        merged = name_all_arguments(
            BATCH_NORM,
            operation.arguments.args,
            operation.arguments.kwargs,
        )

        # 2. Construct arguments that the predictor expects
        arguments = dict(
            batch=merged["input"][0],
            channels=merged["input"][1],
            # batch_norm can be called by BatchNorm1d, BatchNorm2d, BatchNorm3d
            # so we need to collapse all features after channels into a single int
            image_size=np.mean(merged["input"][2:]),
        )

        fw_kernel = operation.forward.kernels if operation.forward else []
        bw_kernel = operation.backward.kernels if operation.backward else []
        kernels = fw_kernel + bw_kernel

        arguments = [arguments[x] for x in self.batch_norm_pred["fp32"].model.features]

        # 3. Call model to make prediction
        pred_dest = self._calculate_dest_runtime(
            self.batch_norm_pred, kernels, operation, arguments, dest_device
        )
        pred_orig = self._calculate_dest_runtime(
            self.batch_norm_pred, kernels, operation, arguments, operation.device
        )

        if unscaled:
            return pred_dest

        if dest_device.name == operation.device.name:  # local prediction
            return pred_orig

        return operation.run_time_ms * pred_dest / pred_orig

    def _calculate_dest_runtime(
        self, mlp_dict, kernels, operation, arguments, dest_device
    ):
        kernels_not_in_common = [k for k in kernels if k.name not in mlp_dict["knames_fp32"] and k.name not in mlp_dict["knames_fp16"]]
        #logger.warning(f"{len(kernels)}, {len(kernel_not_in_common)}")
        run_time_acc = 0
        
        if len(kernels_not_in_common) == len(kernels) and operation.name != "batch_norm":
            ## EXTREME CASE: none of the kernels was found in list of recorded kernels
            ## we use MLP-FP32 to scale the longest running kernel and wave scale the rest
            logger.warning("not kernels in common exists.\n")
            logger.warning(f"{operation.name}")
            for k in kernels_not_in_common: logger.warning(k.name)
            time_ns = 0
            longest_kernel = sorted(kernels, key=lambda x: x.run_time_ns, reverse=True)[0]
            to_wave_scale = list(
                filter(lambda x: x.name != longest_kernel.name, kernels)
            )

            for kernel in to_wave_scale:
                time_ns += self._wave_scaling_strategy(
                    kernel, operation.device, dest_device, self._kernel_metadata
                ).run_time_ns

            run_time_acc = mlp_dict["fp32"].predict(
                arguments, dest_device.name
            ) + ns_to_ms(time_ns)
            
            return run_time_acc

        if operation.name != "batch_norm":
            for kernel in kernels_not_in_common:
                run_time_acc += ns_to_ms(
                    self._wave_scaling_strategy(
                        kernel, operation.device, dest_device, self._kernel_metadata
                    ).run_time_ns
                )

        # choose which mlp to use, obtaine dtype of arguments
        dtypes = []
        for arg in operation.arguments.debug_args:
            if isinstance(arg, tuple) and isinstance(arg[1], str):
                dtypes.append(arg[1])
        
        if "torch.float16" in dtypes:
            #logger.warning(f"found fp16 op, {dtypes}")
            run_time_acc += mlp_dict["fp16"].predict(arguments, dest_device.name)
        else:
            run_time_acc += mlp_dict["fp32"].predict(arguments, dest_device.name) 
        
        return run_time_acc

    def _load_kernel_names(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        # returns a set of kernel names
        return data
