# use gumbel-softmax tricks to reduce variance in (pi+D)/2

import torch
import torch.nn.functional as F
from reinmax_v2 import rao_gumbel_v2, rao_gumbel_v3
class ReinMaxCore_v2_jacobian(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            logits: torch.Tensor,
            tau: torch.Tensor,
            model_ref,
            jacobian_method
    ):
        if jacobian_method in ['reinmax', 'st', 'rao_gumbel', 'reinmax_v2', 'reinmax_v3']:
            y_soft = logits.view(-1, logits.size()[-1]).softmax(dim=-1)
            sample = torch.multinomial(
                y_soft,
                num_samples=1,
                replacement=True,
            )
            one_hot_sample = torch.zeros_like(
                y_soft,
                memory_format=torch.legacy_contiguous_format
            ).scatter_(-1, sample, 1.0)
            ctx.save_for_backward(one_hot_sample, logits, y_soft, tau)
            ctx.model_ref = model_ref
            ctx.jacobian_method = jacobian_method
            return one_hot_sample, y_soft

        elif jacobian_method in ['gumbel']:
            y_soft = F.softmax(logits, dim=-1).view(-1, logits.size()[-1])
            gumbel_logits = F.gumbel_softmax(logits, tau=tau)
            # construct one hot sample
            dim=-1
            index = gumbel_logits.max(dim, keepdim=True)[1]
            one_hot_sample = torch.zeros_like(
                logits, memory_format=torch.legacy_contiguous_format
            ).scatter_(dim, index, 1.0).view(-1, logits.size()[-1])
            ctx.save_for_backward(one_hot_sample, gumbel_logits, y_soft, tau)
            ctx.model_ref = model_ref
            ctx.jacobian_method = jacobian_method
            return one_hot_sample, y_soft
        else:
            print('method not found')
    @staticmethod
    def backward(
            ctx,
            grad_at_sample: torch.Tensor, # (BL, C)
            grad_at_p: torch.Tensor,
    ):
        one_hot_sample, logits, y_soft, tau = ctx.saved_tensors

        if ctx.jacobian_method == 'reinmax':
            #print(one_hot_sample.shape, logits.shape)
            shifted_y_soft = .5 * ((logits.view(-1, logits.size()[-1]) / tau).softmax(dim=-1) + one_hot_sample)
            grad_at_input_1 = (2 * grad_at_sample) * shifted_y_soft
            grad_at_input_1 = grad_at_input_1 - shifted_y_soft * grad_at_input_1.sum(dim=-1, keepdim=True)

            grad_at_input_0 = (-0.5 * grad_at_sample + grad_at_p) * y_soft
            grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
            grad_at_input = grad_at_input_0 + grad_at_input_1
            jacobian = softmax_jacobian(logits, shifted_y_soft)

        elif ctx.jacobian_method == 'st':
            # print(logits.shape)
            #grad_at_input_0 = (grad_at_sample + grad_at_p) * y_soft
            #grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
            jacobian = softmax_jacobian(logits/tau) # BL, C, C
            upstream_grad = grad_at_sample+grad_at_p
            grad_at_input = torch.matmul(jacobian, upstream_grad.unsqueeze(-1)).squeeze(-1)
            #print((grad2-grad_at_input).abs().max())
            #print(grad_at_input_0.shape, logits.shape, grad_at_sample.shape)

        elif ctx.jacobian_method == 'gumbel':
            # here, logits are pi=softmax(gumbel_logits) = softmax(theta + G/tau)
            jacobian = softmax_jacobian(logits, logits)/tau # BL, C, C
            upstream_grad = grad_at_sample+grad_at_p
            grad_at_input = torch.matmul(jacobian, upstream_grad.unsqueeze(-1)).squeeze(-1)

        elif ctx.jacobian_method == 'rao_gumbel':
            # here, logits are pi=softmax(gumbel_logits) = softmax(theta + G/tau)
            jacobian = rao_gumbel_v3(logits, one_hot_sample.reshape(logits.shape), tau) # BL, C, C
            upstream_grad = grad_at_sample+grad_at_p
            grad_at_input = torch.matmul(jacobian, upstream_grad.unsqueeze(-1)).squeeze(-1)

        elif ctx.jacobian_method == 'reinmax_v2':
            jacobian = rao_gumbel_v2(logits, one_hot_sample.reshape(logits.shape), tau) # BL, C, C
            grad_at_input_1 = 2 * torch.matmul(jacobian, grad_at_sample.unsqueeze(-1)).squeeze(-1)
            grad_at_input_0 = (-0.5 * grad_at_sample + grad_at_p) * y_soft
            grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
            grad_at_input = grad_at_input_0 + grad_at_input_1

        elif ctx.jacobian_method == 'reinmax_v3':
            jacobian = rao_gumbel_v3(logits, one_hot_sample.reshape(logits.shape), tau) # BL, C, C
            grad_at_input_1 = 2 * torch.matmul(jacobian, grad_at_sample.unsqueeze(-1)).squeeze(-1)
            grad_at_input_0 = (-0.5 * grad_at_sample + grad_at_p) * y_soft
            grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
            grad_at_input = grad_at_input_0 + grad_at_input_1
        # save jacobian tensor
        #print(ctx.model_ref)
        ctx.model_ref.jacobian = jacobian
        return (grad_at_input - grad_at_input.mean(dim=-1, keepdim=True)).reshape(logits.shape), None, None, None

def softmax_jacobian(logits, pi=None):
    B, L, C = logits.shape
    #print(pi)
    if pi == None:
        pi = logits.softmax(dim=-1)
    jacobian = torch.diag_embed(pi)-pi.unsqueeze(-1)*pi.unsqueeze(-2)
    return jacobian.reshape((B*L, C, C))

def reinmax_jacobian(
        logits: torch.Tensor,
        model_ref,
        tau: float,
        jacobian_method
):
    shape = logits.size()
    #logits = logits.view(-1, shape[-1])
    grad_sample, y_soft = ReinMaxCore_v2_jacobian.apply(logits, logits.new_empty(1).fill_(tau), model_ref, jacobian_method)
    return grad_sample.view(shape), y_soft.view(shape)