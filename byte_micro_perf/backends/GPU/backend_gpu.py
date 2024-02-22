import os
from datetime import timedelta
from backends.backend import Backend
from backends.module_store import *

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d
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
        self.op = Host2DeviceOp(torch.device(self.device))

    def device2host(self):
        self.op = Device2HostOp()            

    def build_tensor(self, input_shapes, dtype):
        input_tensors = [torch.randn(shape).type(getattr(torch, self.dtype)).to(torch.device(self.device)) for shape in input_shapes]
        if self.op_name in ["allreduce", "allgather", "reducescatter", "alltoall"]:
            self.setup_2d_group()
            if self.op_name == "allreduce":
                #allreduce dont need to chunk
                return input_tensors
            input_tensor_list = list(torch.chunk(input_tensors[0], dist.get_world_size(self.group)))
            if self.op_name == "all2all":
                #all2all needs two tensor lists
                out_tensors_list = list(torch.chunk(torch.empty_like(input_tensors[0]), dist.get_world_size(self.group)))
                return input_tensor_list, out_tensors_list
            else:
                return input_tensor_list
        else:    
            return input_tensors

    def _run_operation(self, operation, inputs):
        return operation(*inputs)
    
    def initialize_ccl(self, rank, world_size, dist_backend):
        torch.manual_seed(1)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '49373'
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)

        torch.cuda.set_device(rank)
        # Call the init process
        timeout_seconds = int(os.environ.get("MEGATRON_NCCL_TIMEOUT_SECOND", 30))
        torch.distributed.init_process_group(
            backend=dist_backend,
            world_size=world_size,
            rank=rank,
            store=None,
            timeout=timedelta(seconds=timeout_seconds))
        self.setup_2d_group()
        print(f'DIST INFO: rank {rank}, world_size {world_size}', flush=True)

    def setup_2d_group(self):
        self.rank = dist.get_rank()
        torch.cuda.set_device(self.rank)
        origin_store_based_barrier = dist_c10d._store_based_barrier
        dist_c10d._store_based_barrier = lambda *a, **kw: None
        self.world_size = dist.get_world_size()
        ranks = range(0, self.world_size)
        group = dist.new_group(ranks)
        if self.rank in ranks:
            self.group = group
        dist_c10d._store_based_barrier = origin_store_based_barrier
        # wait for all ranks finish group initializing
        torch.distributed.barrier()
        
