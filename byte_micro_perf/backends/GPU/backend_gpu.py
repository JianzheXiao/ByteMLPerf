import os

from backends.backend import Backend
from backends.module_store import *

import torch
import torch.distributed as dist

class BackendGPU(Backend):
    def gemm(self):
        self.op = GemmOp()

    def softmax(self):
        self.op = SoftmaxOp()

    def allreduce(self):
        self.setup_2d_group()
        self.op = AllReduceOp(self.group)

    def allgather(self):
        self.setup_2d_group()
        self.op = AllGatherOp(self.group)

    def reducescatter(self):
        self.setup_2d_group()
        self.op = ReduceScatterOp(self.group)

    def alltoall(self):
        self.setup_2d_group()
        self.op = AllToAllOp(self.group)

    def host2device(self):
        self.op = Host2DeviceOp(torch.device('cuda'))

    def device2host(self):
        self.op = Device2HostOp()            

    def build_tensor(self, input_shapes, dtype):
        tensors = [torch.randn(shape).type(getattr(torch, self.dtype)).to(torch.device(self.device)) for shape in input_shapes]
        return tensors

    def _run_operation(self, operation, inputs):
        return operation(*inputs)

    def setup_2d_group(self):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        ranks = range(0, self.world_size)
        group = dist.new_group(ranks)
        if self.rank in ranks:
            self.group = group
        
