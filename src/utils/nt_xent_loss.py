#Inspired from https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py

import math
import torch

class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        
        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op = torch.distributed.ReduceOp.SUM, async_op = False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

class NTXentLoss:
    def __init__(self,
                 temperature = 0.9,
                 eps: float = 1e-5):
        self.temperature = temperature
        self.eps = eps
        
    def __call__(self, img_a, img_b):

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(img_a)
            out_2_dist = SyncFunction.apply(img_b)
        else:
            out_1_dist = img_a
            out_2_dist = img_b

        out = torch.cat([img_a, img_b], dim = 0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim = 0)

        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / self.temperature)
        neg = sim.sum(dim = -1)

        row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / self.temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min = self.eps) 

        pos = torch.exp(torch.sum(img_a * img_b, dim = -1) / self.temperature)
        pos = torch.cat([pos, pos], dim = 0)

        loss = -torch.log(pos / (neg + self.eps)).mean()

        return loss
	