# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
print('using pip install version of reinmax')
class ReinMax_Auto(torch.autograd.Function):
    """
    `torch.autograd.Function` implementation of the ReinMax gradient estimator.
    """
    
    @staticmethod
    def forward(
        ctx, 
        logits: torch.Tensor, 
        tau: torch.Tensor,
        alpha: torch.Tensor,
    ):
        y_soft = logits.softmax(dim=-1)
        sample = torch.multinomial(
            y_soft,
            num_samples=1,
            replacement=True,
        )
        one_hot_sample = torch.zeros_like(
            y_soft, 
            memory_format=torch.legacy_contiguous_format
        ).scatter_(-1, sample, 1.0)
        ctx.save_for_backward(one_hot_sample, logits, y_soft, tau, torch.tensor(alpha, dtype=logits.dtype, device=logits.device))
        return one_hot_sample, y_soft

    @staticmethod
    def backward(
        ctx, 
        grad_at_sample: torch.Tensor, 
        grad_at_p: torch.Tensor,
    ):
        #if grad_at_p.abs().sum()>0:
        #    print('mysterious term > 0 in reinmax.py')
        one_hot_sample, logits, y_soft, tau, alpha = ctx.saved_tensors
        pi_alpha = (1-1/(2*alpha))*(logits / tau).softmax(dim=-1) +1/(2*alpha)*one_hot_sample
        shifted_y_soft = .5 * ((logits / tau).softmax(dim=-1) + one_hot_sample)
        grad_at_input_1 = (2 * grad_at_sample) * pi_alpha
        grad_at_input_1 = grad_at_input_1 - shifted_y_soft * grad_at_input_1.sum(dim=-1, keepdim=True)
        
        grad_at_input_0 = (-1/(2*alpha) * grad_at_sample + grad_at_p) * y_soft
        grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
        
        grad_at_input = grad_at_input_0 + grad_at_input_1
        return grad_at_input - grad_at_input.mean(dim=-1, keepdim=True), None, None
    '''
    @staticmethod
    def backward(
            ctx,
            grad_at_sample: torch.Tensor,
            grad_at_p: torch.Tensor,
    ):

        beta=0.5
        one_hot_sample, logits, y_soft, tau, alpha = ctx.saved_tensors
        pi_alpha = -beta * (logits / tau).softmax(dim=-1) + beta * one_hot_sample

        shifted_y_soft = .5 * ((logits / tau).softmax(dim=-1) + one_hot_sample)
        grad_at_input_1 = (2 * grad_at_sample) * pi_alpha
        grad_at_input_1 = grad_at_input_1 - shifted_y_soft * grad_at_input_1.sum(dim=-1, keepdim=True)

        grad_at_input_0 = (-beta * grad_at_sample + grad_at_p) * y_soft
        grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)

        grad_at_input = grad_at_input_0 + grad_at_input_1
        #print(grad_at_input - grad_at_input.mean(dim=-1, keepdim=True))
        #print(logits)
        #print(one_hot_sample, logits)
        return grad_at_input - grad_at_input.mean(dim=-1, keepdim=True), None, None
    '''

def reinmax(
        logits: torch.Tensor, 
        tau: float,
        alpha: float,
        hard: bool = True,
    ):
    r"""
    ReinMax Gradient Approximation.

    Parameters
    ---------- 
    logits: ``torch.Tensor``, required.
        The input Tensor for the softmax. Note that the softmax operation would be conducted along the last dimension. 
    tau: ``float``, required. 
        The temperature hyper-parameter. 

    Returns
    -------
    sample: ``torch.Tensor``.
        The one-hot sample generated from ``multinomial(softmax(logits))``. 
    p: ``torch.Tensor``.
        The output of the softmax function, i.e., ``softmax(logits)``. 
    """
    shape = logits.size()
    logits = logits.view(-1, shape[-1])
    grad_sample, y_soft = ReinMax_Auto.apply(logits, logits.new_empty(1).fill_(tau), alpha)
    return grad_sample.view(shape), y_soft.view(shape)