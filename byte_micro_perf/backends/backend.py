from abc import ABC, abstractmethod
from typing import Any, Dict, List
import time
import os
from backends.utils import dump_communication_ops_report, dump_computation_ops_report

class Backend(ABC):
    def __init__(self, workload_dict: Dict[str, Any]):
        self.op_name = workload_dict['operator']
        self.iterations = workload_dict['iterations']
        self.warmup = int(0.1 * workload_dict['iterations'])
        self.op = None
        self.execution_history = []
        self.dtype = workload_dict['dtype']
        # communication params
        self.rank = None
        self.world_size = None
        self.group = None
        self.device = workload_dict['device']

    def setup_2d_group(self):
        pass    

    def gemm(self):
        pass

    def axpy(self):
        pass # 返回具体op实现

    def softmax(self):
        pass     

    def allreduce(self):
        pass

    def allgather(self):
        pass

    def reducescatter(self):
        pass

    def alltoall(self):
        pass

    def host2device(self):
        pass

    def device2host(self):
        pass                           

    @abstractmethod
    def build_tensor(self, input_shapes: List[List[int]], dtype):
        pass

    @abstractmethod
    def _run_operation(self, operation, inputs):
        pass

    def perf(self, input_shapes: List[List[int]]):
        # warmup
        for _ in range(20):
            inputs = self.build_tensor(input_shapes, self.dtype)
            self._run_operation(self.op, inputs)
        
        self.execution_history = []
        total_time = 0
        for _ in range(self.iterations):
            inputs = self.build_tensor(input_shapes, self.dtype)
            start_time = time.time()
            result = self._run_operation(self.op, inputs)
            execution_time = time.time() - start_time
            self.execution_history.append(execution_time*1e6)
            total_time +=execution_time

        latency = round(total_time *1e6 / self.iterations, 2)
        if self.op_name in ['allreduce', 'allgather', 'reducescatter', 'alltoall']:
            report = dump_communication_ops_report(self.op_name, self.dtype, input_shapes, 
                        self.group.size(), self.execution_history, latency)
        else:
            report = dump_computation_ops_report(self.dtype, input_shapes, self.execution_history, latency)    
        return report