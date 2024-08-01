import torch
import nanogpt.entry_point as nep
import habitat
from habitat.analysis.predictor import Predictor
import time

DEFAULT_PREDICTOR = Predictor()

# model = nep.deepview_model_provider()
# interator = nep.deepview_iteration_provider(model)
# inp = nep.deepview_input_provider()

def generate_tracker():
    return habitat.OperationTracker(device=habitat.Device.T4,
                                   metrics=[
            habitat.Metric.SinglePrecisionFLOPEfficiency,
            habitat.Metric.DRAMReadBytes,
            habitat.Metric.DRAMWriteBytes,
        ])

# with tracker.track():
#     interator(*inp)

# trace = tracker.get_tracked_trace()

# pow_ops = []

# for op in trace.operations:
#     if op.name == "pow":
#         pow_ops.append(op)

# profile_op = pow_ops[1]
# profile_op.to_device(habitat.Device.L4, DEFAULT_PREDICTOR)
# #print(dest.run_time_ms)


n_embd = 1024
device = torch.device('cuda')


def profile_fp16():
    tracker = generate_tracker()
    A = torch.rand((n_embd, 4*n_embd)).to(torch.float16).to(device)
    for i in range(30):
        B = torch.pow(A,3) 
    torch.cuda.synchronize()
    with tracker.track():
        B = torch.pow(A,3)

    traces = tracker.get_tracked_trace()

    for op in traces.operations:
        op.to_device(habitat.Device.RTX4090, DEFAULT_PREDICTOR)

def profile_fp32():
    tracker = generate_tracker()
    A = torch.rand((n_embd, 4*n_embd)).to(torch.float32).to(device)
    for i in range(30):
        B = torch.pow(A, 3) 
    torch.cuda.synchronize()
    with tracker.track():
        B = torch.pow(A, 3)

    traces = tracker.get_tracked_trace()

    for op in traces.operations:
        op.to_device(habitat.Device.RTX4090, DEFAULT_PREDICTOR)

profile_fp16()
profile_fp32()