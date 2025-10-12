# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F

print('using pip install version of reinmax')

def argmax_onehot(logits: torch.Tensor) -> torch.Tensor:
    """
    Given logits of shape (N, M), return a one-hot tensor of the same shape
    indicating the argmax class for each categorical variable.
    """
    # indices of maximum along last dimension
    idx = torch.argmax(logits, dim=-1, keepdim=True)  # shape (N, 1)
    # one-hot encode
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, idx, 1.0)
    return one_hot

class ReinMax_Auto_test(torch.autograd.Function):
    """
    `torch.autograd.Function` implementation of the ReinMax gradient estimator.
    """

    @staticmethod
    def forward(
            ctx,
            logits: torch.Tensor,
            tau: torch.Tensor,
            alpha: torch.Tensor,
            model_ref
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
        ctx.save_for_backward(one_hot_sample, logits, y_soft, tau,
                              torch.tensor(alpha, dtype=logits.dtype, device=logits.device))
        ctx.model_ref = model_ref
        return one_hot_sample, y_soft

    @staticmethod
    def backward(
            ctx,
            grad_at_sample: torch.Tensor,
            grad_at_p: torch.Tensor,
    ):
        one_hot_sample, logits, y_soft, tau, alpha = ctx.saved_tensors
        one_hot_fixed = torch.zeros_like(one_hot_sample)
        one_hot_fixed[:, 4] = 1.0
        one_hot_argmax = argmax_onehot(logits)
        one_hot_logits = F.softmax(logits/tau, dim=-1)
        D = one_hot_argmax
        #print('test')
        # compute modified second term
        shifted_y_soft = .5 * ((logits/tau).softmax(dim=-1) + D)
        grad_at_input_1 = (2 * grad_at_sample) * shifted_y_soft
        grad_at_input_1 = grad_at_input_1 - shifted_y_soft * grad_at_input_1.sum(dim=-1, keepdim=True)
        # compute first term
        grad_at_input_0 = (-1 / (2 * alpha) * grad_at_sample + grad_at_p) * y_soft
        grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
        # save reinmax terms into the model
        if ctx.model_ref is not None:
            ctx.model_ref.reinmax_term1 = (-2) * grad_at_input_0.detach().clone()
            ctx.model_ref.reinmax_term2 = 0.5 * grad_at_input_1.detach().clone()
        # return true reinmax gradient
        #shifted_y_soft = .5 * ((logits / tau).softmax(dim=-1) + one_hot_sample)
        #grad_at_input_1 = (2 * grad_at_sample) * shifted_y_soft
        #grad_at_input_1 = grad_at_input_1 - shifted_y_soft * grad_at_input_1.sum(dim=-1, keepdim=True)

        grad_at_input = grad_at_input_0 + grad_at_input_1

        return grad_at_input - grad_at_input.mean(dim=-1, keepdim=True), None, None, None

def reinmax_test(
        logits: torch.Tensor,
        model_ref,
        tau: float,
        alpha: float,
        # hard: bool = True,
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
    grad_sample, y_soft = ReinMax_Auto_test.apply(logits, logits.new_empty(1).fill_(tau), alpha, model_ref)
    return grad_sample.view(shape), y_soft.view(shape)


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
        one_hot_sample, logits, y_soft, tau, alpha = ctx.saved_tensors

        pi_alpha = (1-1/(2*alpha))*(logits / tau).softmax(dim=-1) +1/(2*alpha)*one_hot_sample
        shifted_y_soft = .5 * ((logits / tau).softmax(dim=-1) + one_hot_sample)
        grad_at_input_1 = (2 * grad_at_sample) * shifted_y_soft
        grad_at_input_1 = grad_at_input_1 - shifted_y_soft * ((2*grad_at_sample)*pi_alpha).sum(dim=-1, keepdim=True)
        
        grad_at_input_0 = (-1/(2*alpha) * grad_at_sample + grad_at_p) * y_soft
        grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
        
        grad_at_input = grad_at_input_0 + grad_at_input_1
        # shape is [batch size * latent dim, categorical dim]
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
        #hard: bool = True,
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

# The code has been modified from https://github.com/chijames/GST
import torch

def st(logits, tau):
    shape, dtype = logits.size(), logits.dtype
    logits = logits.view(-1, shape[-1]).float()

    y_soft = logits.softmax(dim=-1)
    sample = torch.multinomial(
        y_soft,
        num_samples=1,
        replacement=True,
    )
    one_hot_sample = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(-1, sample, 1.0)

    prob = (logits / tau).softmax(dim=-1)
    one_hot_sample = one_hot_sample - prob.detach() + prob
    return one_hot_sample.view(shape), y_soft.view(shape)


def exact(logits, tau=1.0):
    m = torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)
    action = m.sample()
    prob = logits.softmax(dim=-1)
    return action, prob


class Reinmax_CORE(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            logit: torch.Tensor,
            p_sample0: torch.Tensor,
    ):
        ctx.save_for_backward(p_sample0)
        return torch.zeros_like(logit)

    @staticmethod
    def backward(ctx, grad_at_output):
        p_sample0 = ctx.saved_tensors[0]
        grad_at_input = grad_at_output * p_sample0
        grad_at_input = grad_at_input - p_sample0 * grad_at_input.sum(dim=-1, keepdim=True)
        return grad_at_input - grad_at_input.mean(dim=-1, keepdim=True), None


def reinmax2(logits, tau=1.0, alpha=1.0):
    shape, _ = logits.size(), logits.dtype
    logits = logits.view(-1, shape[-1])
    y_soft = logits.softmax(dim=-1)
    sample = torch.multinomial(
        y_soft,
        num_samples=1,
        replacement=True,
    )
    one_hot_sample = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(-1, sample, 1.0)

    shifted_logits_0 = logits / tau
    shifted_softmax_0 = shifted_logits_0.softmax(dim=-1)
    adapted_softmax_0 = .5 * (shifted_softmax_0 + one_hot_sample)

    core_sample = 2 * Reinmax_CORE.apply(
        logits,
        adapted_softmax_0,
    )
    grad_sample = .5 * (y_soft.detach() - y_soft) + core_sample + one_hot_sample
    return grad_sample.view(shape), y_soft.view(shape)


class ReinmaxMean_KERNEL(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            logit: torch.Tensor,
            p: torch.Tensor,
            p_0: torch.Tensor,
            sample: torch.Tensor,
    ):
        assert logit.dim() == 2
        ctx.save_for_backward(p, p_0, sample)
        return torch.zeros_like(p, memory_format=torch.legacy_contiguous_format).scatter_(-1, sample, 1.0)

    @staticmethod
    def backward(ctx, grad_at_output):
        p, p_0, sample = ctx.saved_tensors
        one_hot_sample = torch.zeros_like(p, memory_format=torch.legacy_contiguous_format).scatter_(-1, sample, 1.0)
        grad_fo_0 = grad_at_output.gather(dim=-1, index=sample) - grad_at_output.mean(dim=-1, keepdim=True)
        grad_fo_1 = one_hot_sample - p
        grad_fo = grad_fo_0 * grad_fo_1

        grad_st_0 = grad_at_output * p
        grad_st_1 = grad_st_0 * one_hot_sample - grad_st_0.sum(dim=-1, keepdim=True) * p
        N = p_0.size(-1)
        grad_st = grad_st_1 / (N * p_0.detach().gather(dim=-1, index=sample) + 1e-12)

        grad_at_input = .5 * (grad_fo + grad_st)
        return grad_at_input, None, None, None


def reinmax_mean_baseline(logits, tau=1.0):
    shape, _ = logits.size(), logits.dtype
    logits = logits.view(-1, shape[-1])
    y_soft = logits.softmax(dim=-1)

    sample = torch.multinomial(
        y_soft,
        num_samples=1,
        replacement=True,
    )

    y_soft_tau = (logits / tau).softmax(dim=-1)

    onehot_sample = ReinmaxMean_KERNEL.apply(logits, y_soft, y_soft_tau, sample)
    return onehot_sample.view(shape), y_soft.view(shape)
