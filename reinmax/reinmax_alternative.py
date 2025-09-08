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


def reinmax(logits, tau=1.0):
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
