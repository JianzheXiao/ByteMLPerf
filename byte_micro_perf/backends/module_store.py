import torch
import torch.distributed as dist

class AddMulOp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_tensor_a, input_tensor_b, input_tensor_c):
        result = (input_tensor_a + input_tensor_b) * input_tensor_c
        return result


class GemmOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor_a, input_tensor_b):
        logits = torch.matmul(input_tensor_a, input_tensor_b)
        return logits


class SoftmaxOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        logits = torch.nn.functional.softmax(hidden_states, dim=-1)
        return logits

class AllReduceOp(torch.nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group

    def forward(self, input_tensors):
        dist.all_reduce(input_tensors[0], group = self.group)
        return True

class AllGatherOp(torch.nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group
    
    def process_tensor(self, input_tensor):
        input_tensor_list = list(torch.chunk(input_tensor, dist.get_world_size(self.group)))
        return [input_tensor_list]
    
    def forward(self, input_tensor_list):
        dist.all_gather(input_tensor_list, input_tensor_list[dist.get_rank(self.group)], group = self.group)
        return True

class ReduceScatterOp(torch.nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group
    
    def process_tensor(self, input_tensor):
        input_tensor_list = list(torch.chunk(input_tensor, dist.get_world_size(self.group)))
        return [input_tensor_list]
    
    def forward(self, input_tensor_list):
        dist.reduce_scatter(input_tensor_list[dist.get_rank(self.group)], input_tensor_list, group = self.group)
        return True

class AllToAllOp(torch.nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group
    
    def process_tensor(self, input_tensor):
        type(input_tensor)
        input_tensor_list = list(torch.chunk(input_tensor, dist.get_world_size(self.group)))
        output_tensor_list = list(torch.chunk(input_tensor, dist.get_world_size(self.group)))
        return [input_tensor_list, output_tensor_list]
    
    def forward(self, in_tensors_list, out_tensors_list):
        dist.all_to_all(out_tensors_list, in_tensors_list, group = self.group)
        return True

class Device2HostOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        assert input_tensors[0].device.type != 'cpu'
        output_cpu = input_tensors[0].cpu()
        return output_cpu

class Host2DeviceOp(torch.nn.Module):
    def __init__(self, xpu_device):
        super().__init__()
        self.xpu_device = xpu_device
    def forward(self, input_tensors):
        assert input_tensors[0].device.type == 'cpu'
        output_xpu = input_tensors[0].to(self.xpu_device)
        return output_xpu