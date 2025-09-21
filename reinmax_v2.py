# use gumbel-softmax tricks to reduce variance in (pi+D)/2

import torch

class ReinMaxCore_v2(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            logits: torch.Tensor,
            tau: torch.Tensor,
            model_ref,
            jacobian_method
    ):
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

    @staticmethod
    def backward(
            ctx,
            grad_at_sample: torch.Tensor, # (BL, C)
            grad_at_p: torch.Tensor,
    ):
        one_hot_sample, logits, y_soft, tau = ctx.saved_tensors

        #shifted_y_soft = .5 * ((logits / tau).softmax(dim=-1) + one_hot_sample)
        #grad_at_input_1 = (2 * grad_at_sample) * shifted_y_soft
        #grad_at_input_1 = grad_at_input_1 - shifted_y_soft * grad_at_input_1.sum(dim=-1, keepdim=True)
        if ctx.jacobian_method == 'gumbel_D':
            jacobian_avg = rao_gumbel_v2(logits, one_hot_sample.reshape(logits.shape), tau) # (B*L, C, C)
        elif ctx.jacobian_method == 'gumbel_ST':
            jacobian_avg = rao_gumbel_v3(logits, one_hot_sample.reshape(logits.shape), tau)  # (B*L, C, C)
        else:
             print('jacobian method not found')
        grad_at_input_1 = 2*torch.matmul(grad_at_sample.unsqueeze(-2), jacobian_avg).squeeze()

        grad_at_input_0 = (-0.5 * grad_at_sample + grad_at_p) * y_soft
        grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)

        grad_at_input = grad_at_input_0 + grad_at_input_1
        # save first and second terms for analysis
        if ctx.model_ref is not None:
            ctx.model_ref.reinmax_term1 = (-2) * grad_at_input_0.detach().clone()
            ctx.model_ref.reinmax_term2 = 0.5 * grad_at_input_1.detach().clone()
        return (grad_at_input - grad_at_input.mean(dim=-1, keepdim=True)).reshape(logits.shape), None, None, None

def reinmax_v2(
        logits: torch.Tensor,
        model_ref,
        tau: float,
        jacobian_method
):
    shape = logits.size()
    #logits = logits.view(-1, shape[-1])
    grad_sample, y_soft = ReinMaxCore_v2.apply(logits, logits.new_empty(1).fill_(tau), model_ref, jacobian_method)
    return grad_sample.view(shape), y_soft.view(shape)

def rao_gumbel_v2(logits, D, tau=1.0, repeats=100, hard=True):
    '''
    :param logits:
    :param D:
    :param tau:
    :param repeats:
    :param hard:
    :return: jacobian of E[softmax evaluated at: prob=0.5(pi+softmax_tau(theta+G|D))]
    (bs*latdim, catdim, catdim)
    '''

    logits_cpy = logits.detach()
    #probs = logits_cpy.softmax(dim=-1)
    #m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=probs)
    action = D#m.sample()#D
    action_bool = action.bool() # this is D

    logits_shape = logits.shape
    bs = logits_shape[0]

    E = torch.empty(logits_shape + (repeats,),
                    dtype=logits.dtype,
                    layout=logits.layout,
                    device=logits.device,
                    memory_format=torch.legacy_contiguous_format).exponential_()
    # sample exponential using .exponential_()
    Ei = E[action_bool].view(logits_shape[:-1] + (repeats,))  # rv. for the sampled location

    wei = logits_cpy.exp()
    Z = wei.sum(dim=-1, keepdim=True)  # (bs, latdim, 1)
    EiZ = (Ei / Z).unsqueeze(-2)  # (bs, latdim, 1, repeats)
    new_logits = E / (wei.unsqueeze(-1))  # (bs, latdim, catdim, repeats)
    new_logits[action_bool] = 0.0
    new_logits = -(new_logits + EiZ + 1e-20).log()

    new_pi = 0.5*(
            logits.unsqueeze(-1).softmax(dim=-2)+
            (new_logits/tau).softmax(dim=-2))

    # now compute the softmax jacobian at pi', then average
    # jacobian = diag(pi) - pi @ pi.T
    B, L, C, R = new_pi.shape
    new_pi = new_pi.reshape(B * L, C, R).permute((0, 2, 1))  # (BL, R, C)
    pi_diag = torch.diag_embed(new_pi)
    jacobian = pi_diag - torch.matmul(new_pi.unsqueeze(-1), new_pi.unsqueeze(-2))
    jacobian_avg = jacobian.mean(dim=1)
    #logits_diff = new_logits - logits_cpy.unsqueeze(-1)
    #prob = ((logits.unsqueeze(-1) + logits_diff) / tau).softmax(dim=-2).mean(dim=-1)
    #action = action - prob.detach() + prob if hard else prob
    return jacobian_avg #action.view(logits_shape), distribution_original

def rao_gumbel_v3(logits, D, tau=1.0, repeats=100, hard=True):
    '''
    :param logits:
    :param D:
    :param tau:
    :param repeats:
    :param hard:
    :return: jacobian of E[softmax_t(x) evaluated at x = theta'+G|D]
    where theta'=softmax^-1((pi+D)/2)
    (bs*latdim, catdim, catdim)
    '''
    logits_cpy = logits.detach()
    logits_cpy = (0.5*(logits_cpy.softmax(dim=-1)+D)).log()
    #probs = logits_cpy.softmax(dim=-1)
    #m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=probs)
    action = D
    action_bool = action.bool() # this is D

    logits_shape = logits.shape
    bs = logits_shape[0]

    E = torch.empty(logits_shape + (repeats,),
                    dtype=logits.dtype,
                    layout=logits.layout,
                    device=logits.device,
                    memory_format=torch.legacy_contiguous_format).exponential_()
    # sample exponential using .exponential_()
    Ei = E[action_bool].view(logits_shape[:-1] + (repeats,))  # rv. for the sampled location

    wei = logits_cpy.exp()
    Z = wei.sum(dim=-1, keepdim=True)  # (bs, latdim, 1)
    EiZ = (Ei / Z).unsqueeze(-2)  # (bs, latdim, 1, repeats)
    new_logits = E / (wei.unsqueeze(-1))  # (bs, latdim, catdim, repeats)
    new_logits[action_bool] = 0.0
    new_logits = -(new_logits + EiZ + 1e-20).log()

    new_pi = (new_logits/tau).softmax(dim=-2)

    # now compute the softmax jacobian at pi', then average
    # jacobian = diag(pi) - pi @ pi.T
    B, L, C, R = new_pi.shape
    new_pi = new_pi.reshape(B * L, C, R).permute((0, 2, 1))  # (BL, R, C)
    pi_diag = torch.diag_embed(new_pi)
    jacobian = pi_diag - torch.matmul(new_pi.unsqueeze(-1), new_pi.unsqueeze(-2))
    jacobian_avg = jacobian.mean(dim=1)/tau
    #logits_diff = new_logits - logits_cpy.unsqueeze(-1)
    #prob = ((logits.unsqueeze(-1) + logits_diff) / tau).softmax(dim=-2).mean(dim=-1)
    #action = action - prob.detach() + prob if hard else prob
    return jacobian_avg #action.view(logits_shape), distribution_original
