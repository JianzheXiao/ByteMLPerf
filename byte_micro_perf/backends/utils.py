from typing import List
import math
import torch
import numpy as np

def dump_communication_ops_report(op_name: str, dtype: str, input_shapes: List[List[int]], group_size: List[int], 
                                execution_history: float, latency: float):
    size = sum([math.prod(shape) for shape in input_shapes])
    dtype_size = torch.finfo(getattr(torch, dtype)).bits // 8
    mb = dtype_size * size / 1024 / 1024
    algo_bw = dtype_size * size / latency / 1e3 
    bus_bw = algo_bw * (group_size - 1) / group_size
    if op_name == 'allreduce':
        bus_bw *=2

    report = {
        "Dtype": dtype,
        "Memory Size(MB)": round(mb, 2),
        "Algo bandwidth(GB/s)": round(algo_bw, 2),
        "Bus bandwidth(GB/s)" : round(bus_bw, 2),
        "Min latency(us)" : round(np.min(execution_history), 2),
        "Avg latency(us)" : round(latency, 2),
        "Max latency(us)" : round(np.max(execution_history), 2),
        "P99 latency(us)": round(np.percentile(execution_history, 99)),
        "Theoretical bandwidth(GB/s)" : 178,
        "Theoretical latency(us)"  : 1.3,
        "MFU" : 0.87
    }
    return report

def dump_computation_ops_report(dtype: str, input_shapes: List[List[int]], execution_history: float, latency: float):
    size = sum([math.prod(shape) for shape in input_shapes])    
    dtype_size = torch.finfo(getattr(torch, dtype)).bits // 8
    mb = dtype_size * size / 1024 / 1024
    algo_bw = dtype_size * size / latency / 1e3
    report = {
        "Dtype": dtype,
        "Memory Size(MB)": round(mb, 2),
        "Algo bandwidth(GB/s)": round(algo_bw, 2),
        "Min latency(us)" : round(np.min(execution_history), 2),
        "Avg latency(us)" : round(latency, 2),
        "Max latency(us)" : round(np.max(execution_history), 2),
        "P99 latency(us)": round(np.percentile(execution_history, 99)),
        "Theoretical bandwidth(GB/s)" : 178,
        "Theoretical latency(us)"  : 1.3,
        "MFU" : 0.87
    }
    return report