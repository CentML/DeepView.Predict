import argparse
import collections
import csv
import os
from pathlib import Path
import numpy as np
import torch
import gc
import habitat
from habitat.analysis import SPECIAL_OPERATIONS
from habitat.profiling.run_time import RunTimeProfiler
from habitat.analysis.operation import MeasuredOperation
from habitat.analysis.predictor import Predictor
###############################################################################

# Experiment configuration

RESNET50_BATCHES = [16, 32, 64]#[16, 32, 64]
GNMT_BATCHES = [16, 32, 48]
NANOGPT_BATCHES = [32, 48, 64]
DCGAN_BATCHES = [64, 96, 128]

###############################################################################

Context = collections.namedtuple(
    "Context",
    ["origin_device", "destination_device", "profiler", "percentile", "storage_folder"],
)

torch.backends.cudnn.benchmark = True

## RE-RUN OPS ####

def re_run_operations(tracker, num_shuffles, origin_device, config_name, storage_folder):
    predictor = Predictor()
    ops = [(op,[]) for op in tracker._operations] # op, run_times
    for _ in range(num_shuffles):
        np.random.shuffle(ops)
        for random_op, run_times_arr in ops:
            fw, bw = tracker._profiler.measure_operation(random_op.func, random_op.args, random_op.kwargs)
            fw_time = fw.run_time_ms
            bw_time = bw.run_time_ms if bw else 0
            run_times_arr.append(fw_time + bw_time)

    for random_op, run_times_arr in ops:
        pred_time = random_op.to_device(origin_device, predictor).unscaled_predicted
        run_times_arr.append(pred_time)

    file_name = "{}-{}-predicted_local.csv".format(config_name, origin_device.name)
    run_names = [f"run_{i}" for i in range(num_shuffles)]
    with open(os.path.join(storage_folder, file_name), "w") as file:
        writer = csv.writer(file)
        writer.writerow(["operation"]+run_names+["predicted_local"])
        for op,times in ops:
            writer.writerow([op.name] + times)

def record_e2e(config_name, origin_device, data, storage_folder):
    file_name = "{}-{}-e2e.csv".format(config_name, origin_device.name)
    file_path = os.path.join(storage_folder, file_name)
    exists = os.path.exists(file_path)
    with open(file_path, "a") as file:
        writer = csv.writer(file)
        if not exists: writer.writerow(["device", "run_time_ms"])
        #writer.writerow(["device", "run_time_ms"])
        for device, run_time_ms in data:
            writer.writerow([device.name, run_time_ms])


def record_breakdown(config_name, origin_device, dest_device, trace, storage_folder):
    file_name = "{}-{}-{}-breakdown.csv".format(
        config_name,
        origin_device.name,
        dest_device.name,
    )
    ops_sum = 0
    with open(os.path.join(storage_folder, file_name), "w") as file:
        writer = csv.writer(file)
        writer.writerow(["operation", "run_time_ms", "measured_local", "predicted_local", "unscaled_predicted"])
        for op in trace.operations:
            measured_local = None
            predicted_local = None
            unscaled_predicted = None
            if hasattr(op,'measured_local'):
                measured_local = op.measured_local
            if hasattr(op, 'predicted_local'):
                predicted_local = op.predicted_local
            if hasattr(op, 'unscaled_predicted'):
                unscaled_predicted = op.unscaled_predicted
            ops_sum += op.run_time_ms
            writer.writerow([op.name, op.run_time_ms, measured_local, predicted_local, unscaled_predicted])
    print(f"ops sum: {ops_sum}")

def compute_threshold(runnable, context):
    tracker = habitat.OperationTracker(context.origin_device)
    with tracker.track():
        runnable()

    run_times = []
    trace = tracker.get_tracked_trace()
    for op in trace.operations:
        if op.name in SPECIAL_OPERATIONS:
            continue
        run_times.append(op.forward.run_time_ms)
        if op.backward is not None:
            run_times.append(op.backward.run_time_ms)

    return np.percentile(run_times, context.percentile)


def run_experiment_config(config_name, runnable, context):
    print("Processing:", config_name)
    origin_run_time_ms = context.profiler.measure_ms(runnable)
    print(f"time from context.profiler.measure_ms: {origin_run_time_ms}")
    # e2e_results = [(context.origin_device, origin_run_time_ms)]

    threshold = compute_threshold(runnable, context)
    tracker = habitat.OperationTracker(
        device=context.origin_device,
        metrics=[
            habitat.Metric.SinglePrecisionFLOPEfficiency,
            habitat.Metric.DRAMReadBytes,
            habitat.Metric.DRAMWriteBytes,
        ],
        metrics_threshold_ms=threshold,
    )
    with tracker.track():
        runnable()

    trace = tracker.get_tracked_trace()
    re_run_operations(tracker,5, context.origin_device, config_name, context.storage_folder)

    record_breakdown(
        config_name,
        context.origin_device,
        context.origin_device,
        trace,
        context.storage_folder,
    )
    print(f"time from trace.run_time_ms : {trace.run_time_ms}")
    e2e_results = [(context.origin_device, trace.run_time_ms)]

    predicted_trace = trace.to_device(context.destination_device)
    record_breakdown(
        config_name,
        context.origin_device,
        context.destination_device,
        predicted_trace,
        context.storage_folder,
    )
    print(f"e2e: {config_name} | run_time_ms : {predicted_trace.run_time_ms}")
    e2e_results.append((context.destination_device, predicted_trace.run_time_ms))

    record_e2e(config_name, context.origin_device, e2e_results, context.storage_folder)


def run_resnet50_experiments(context):
    import resnet.entry_point as rep

    model = rep.skyline_model_provider()
    iteration = rep.skyline_iteration_provider(model)

    for batch_size in RESNET50_BATCHES:
        inputs = rep.skyline_input_provider(batch_size=batch_size)

        def runnable():
            iteration(*inputs)

        run_experiment_config(
            "resnet50+{}".format(batch_size),
            runnable,
            context,
        )


def run_dcgan_experiments(context):
    import dcgan.entry_point as dep

    models = dep.skyline_model_provider()
    iteration = dep.skyline_iteration_provider(*models)

    for batch_size in DCGAN_BATCHES:
        inputs = dep.skyline_input_provider(batch_size=batch_size)

        def runnable():
            iteration(*inputs)

        run_experiment_config(
            "dcgan+{}".format(batch_size),
            runnable,
            context,
        )


def run_inception_experiments(context):
    import inception.entry_point as iep

    model = iep.skyline_model_provider()
    iteration = iep.skyline_iteration_provider(model)

    # N.B. We use the same batch sizes as resnet
    for batch_size in RESNET50_BATCHES:
        inputs = iep.skyline_input_provider(batch_size=batch_size)

        def runnable():
            iteration(*inputs)

        run_experiment_config(
            "inception+{}".format(batch_size),
            runnable,
            context,
        )


def run_gnmt_experiments(context):
    import gnmt.entry_point as gep

    model = gep.skyline_model_provider()
    iteration = gep.skyline_iteration_provider(model)

    for batch_size in GNMT_BATCHES:
        inputs = gep.skyline_input_provider(batch_size=batch_size)

        def runnable():
            iteration(*inputs)

        run_experiment_config(
            "gnmt+{}".format(batch_size),
            runnable,
            context,
        )


def run_nanogpt_experiments(context):
    import nanogpt.entry_point as tep

    model = tep.deepview_model_provider()
    iteration = tep.deepview_iteration_provider(model)

    for batch_size in NANOGPT_BATCHES:
        inputs = tep.deepview_input_provider(batch_size=batch_size)

        def runnable():
            iteration(*inputs)

        run_experiment_config(
            "nanogpt+{}".format(batch_size),
            runnable,
            context,
        )


def main():
    import habitat.habitat_cuda as hc

    parser = argparse.ArgumentParser()
    parser.add_argument("org", type=str)
    parser.add_argument("dest", type=str)
    parser.add_argument("--percentile", type=float, default=99.5)
    args = parser.parse_args()

    # create folder to store csv files
    curr_path = os.path.dirname(os.path.abspath(__file__))
    storage_folder = f"{curr_path}/results"
    Path(f"{storage_folder}").mkdir(parents=True, exist_ok=True)

    # Ask the profiler to cache metrics for kernels that share the same name
    # and launch configuration.
    hc.set_cache_metrics(True)
    origin_device = getattr(habitat.Device, args.org)
    destination_device = getattr(habitat.Device, args.dest)
    profiler = RunTimeProfiler()

    context = Context(
        origin_device=origin_device,
        destination_device=destination_device,
        profiler=profiler,
        percentile=args.percentile,
        storage_folder=storage_folder,
    )

    # run_dcgan_experiments(context)
    # run_inception_experiments(context)
    run_resnet50_experiments(context)
    # run_gnmt_experiments(context)
    # run_nanogpt_experiments(context)


if __name__ == "__main__":
    main()
