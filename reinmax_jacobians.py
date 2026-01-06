# use gumbel-softmax tricks to reduce variance in (pi+D)/2

import torch
import torch.nn.functional as F
from reinmax_v2 import rao_gumbel_v2, rao_gumbel_v3
import os
import matplotlib.pyplot as plt

manualSeed = 52
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(manualSeed)

class ReinMaxCore_v2_jacobian(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            logits: torch.Tensor,
            tau: torch.Tensor,
            model_ref,
            jacobian_method
    ):
        if jacobian_method in ['reinmax', 'st', 'rao_gumbel', 'reinmax_v2', 'reinmax_v3', 'st_v2']:
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

        elif jacobian_method in ['gumbel', 'gaussian', 'reinmax_cv']:
            y_soft = F.softmax(logits, dim=-1).view(-1, logits.size()[-1])
            if jacobian_method == 'gaussian':
                theta = gaussian_softmax(logits, tau=tau)
                y_soft = empirical_y_soft(theta).view(-1, logits.size()[-1])
                #print(y_soft.shape)
                #one_hot_sample = F.gumbel_softmax(logits, tau=tau, hard=True).view(-1, logits.size()[-1])
                #print(theta.shape, one_hot_sample.shape)
                #ctx.save_for_backward(one_hot_sample, theta, y_soft, tau)
                #ctx.model_ref = model_ref
                #ctx.jacobian_method = jacobian_method
                #return one_hot_sample, y_soft
            elif jacobian_method in ['gumbel', 'reinmax_cv']:
                theta = F.gumbel_softmax(logits, tau=tau)
            #plot_softmaxes(logits)
            # construct one hot sample
            dim=-1
            index = theta.max(dim, keepdim=True)[1]
            one_hot_sample = torch.zeros_like(
                logits, memory_format=torch.legacy_contiguous_format
            ).scatter_(dim, index, 1.0).view(-1, logits.size()[-1])
            if jacobian_method == 'reinmax_cv':
                ctx.save_for_backward(one_hot_sample, logits, y_soft, tau, theta)
            else:
                ctx.save_for_backward(one_hot_sample, theta, y_soft, tau)
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
        if ctx.jacobian_method == 'reinmax_cv':
            one_hot_sample, logits, y_soft, tau, pi_G = ctx.saved_tensors
            # pi_G is softmax_tau(theta+G)
        else:
            one_hot_sample, logits, y_soft, tau = ctx.saved_tensors
        B, L, C = logits.shape
        if ctx.jacobian_method == 'reinmax':
            #print(one_hot_sample.shape, logits.shape)
            shifted_y_soft = .5 * ((logits.view(-1, logits.size()[-1]) / tau).softmax(dim=-1) + one_hot_sample)
            grad_at_input_1 = (2 * grad_at_sample) * shifted_y_soft
            grad_at_input_1 = grad_at_input_1 - shifted_y_soft * grad_at_input_1.sum(dim=-1, keepdim=True)

            grad_at_input_0 = (-0.5 * grad_at_sample + grad_at_p) * y_soft
            grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
            grad_at_input = grad_at_input_0 + grad_at_input_1
            jacobian = softmax_jacobian(logits, shifted_y_soft)
            #modified_jacobian = modify_jacobian(jacobian)
            ctx.model_ref.jacobian = jacobian

        elif ctx.jacobian_method == 'st':
            # print(logits.shape)
            #grad_at_input_0 = (grad_at_sample + grad_at_p) * y_soft
            #grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
            jacobian = softmax_jacobian(logits/tau) # BL, C, C
            upstream_grad = grad_at_sample+grad_at_p
            #print(upstream_grad.shape)# (latent * batch size, categorical)
            grad_at_input = torch.matmul(jacobian, upstream_grad.unsqueeze(-1)).squeeze(-1)
            #print((grad2-grad_at_input).abs().max())
            #print(grad_at_input_0.shape, logits.shape, grad_at_sample.shape)
            ctx.model_ref.jacobian = jacobian

        elif ctx.jacobian_method == 'st_v2':
            # D is BL, C
            I = torch.eye(C).unsqueeze(0).repeat((B*L, 1, 1))
            D = one_hot_sample.unsqueeze(1).repeat((1, C, 1)) # BL, C, C check this ?
            f_D = ctx.model_ref.loss.unsqueeze(-1).repeat(1, L).reshape((B*L)) # (B) -> BL
            jacobian = softmax_jacobian(logits/tau) # BL, C, C
            upstream_grad = grad_at_sample+grad_at_p # BL, C
            #print(f_D.shape, I.shape, upstream_grad.unsqueeze(-1).shape)
            left_matrix = f_D.reshape((B*L, 1, 1)) + torch.bmm((I-D), upstream_grad.unsqueeze(-1))
            #print(left_matrix.shape, jacobian.shape, upstream_grad.shape)
            grad_at_input = torch.matmul(jacobian, left_matrix).squeeze(-1)
            ctx.model_ref.jacobian = jacobian

        elif ctx.jacobian_method == 'gumbel':
            # here, logits are pi=softmax(gumbel_logits) = softmax(theta + G/tau)
            jacobian = softmax_jacobian(logits, logits)/tau # BL, C, C
            upstream_grad = grad_at_sample+grad_at_p
            grad_at_input = torch.matmul(jacobian, upstream_grad.unsqueeze(-1)).squeeze(-1)
            ctx.model_ref.jacobian = jacobian*tau

        elif ctx.jacobian_method == 'gaussian':
            jacobian = softmax_jacobian(logits, logits)/tau # BL, C, C
            upstream_grad = grad_at_sample+grad_at_p
            grad_at_input = torch.matmul(jacobian, upstream_grad.unsqueeze(-1)).squeeze(-1)
            ctx.model_ref.jacobian = jacobian*tau

        elif ctx.jacobian_method == 'rao_gumbel':
            # here, logits are pi=softmax(gumbel_logits) = softmax(theta + G/tau)
            jacobian = rao_gumbel_v3(logits, one_hot_sample.reshape(logits.shape), tau) # BL, C, C
            upstream_grad = grad_at_sample+grad_at_p
            grad_at_input = torch.matmul(jacobian, upstream_grad.unsqueeze(-1)).squeeze(-1)
            ctx.model_ref.jacobian = jacobian*tau

        elif ctx.jacobian_method == 'reinmax_v2':
            jacobian = rao_gumbel_v2(logits, one_hot_sample.reshape(logits.shape), tau) # BL, C, C
            grad_at_input_1 = 2 * torch.matmul(jacobian, grad_at_sample.unsqueeze(-1)).squeeze(-1)
            grad_at_input_0 = (-0.5 * grad_at_sample + grad_at_p) * y_soft
            grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
            grad_at_input = grad_at_input_0 + grad_at_input_1
            ctx.model_ref.jacobian = jacobian

        elif ctx.jacobian_method == 'reinmax_v3':
            jacobian = rao_gumbel_v3(logits, one_hot_sample.reshape(logits.shape), tau) # BL, C, C
            grad_at_input_1 = 2 * torch.matmul(jacobian, grad_at_sample.unsqueeze(-1)).squeeze(-1)
            grad_at_input_0 = (-0.5 * grad_at_sample + grad_at_p) * y_soft
            grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
            grad_at_input = grad_at_input_0 + grad_at_input_1
            ctx.model_ref.jacobian = jacobian*tau

        elif ctx.jacobian_method == 'reinmax_cv':
            # Reinmax
            shifted_y_soft = .5 * ((logits.view(-1, logits.size()[-1]) / tau).softmax(dim=-1) + one_hot_sample)
            grad_at_input_1 = (2 * grad_at_sample) * shifted_y_soft
            grad_at_input_1 = grad_at_input_1 - shifted_y_soft * grad_at_input_1.sum(dim=-1, keepdim=True)

            grad_at_input_0 = (-0.5 * grad_at_sample + grad_at_p) * y_soft
            grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
            grad_reinmax = grad_at_input_0 + grad_at_input_1

            # Gumbel rao evaluated at pi+D/2
            eta =1.0
            tau2 = 1.3
            jacobian_GR = rao_gumbel_v3(logits, one_hot_sample.reshape(logits.shape), tau2) # BL, C, C
            grad_GR = torch.matmul(jacobian_GR, grad_at_sample.unsqueeze(-1)).squeeze(-1)

            # Gumbel softmax evaluated at pi+D/2
            new_pi = 0.5*(logits.softmax(dim=-1)+one_hot_sample.reshape(logits.shape))
            jacobian_GS = softmax_jacobian(new_pi.log(), new_pi)/tau2 # BL, C, C
            grad_GS = torch.matmul(jacobian_GS, grad_at_sample.unsqueeze(-1)).squeeze(-1)

            # put terms together
            grad_at_input = grad_reinmax - eta * grad_GS + eta * grad_GR
            # cv equation: E[reinmax - GS] + E[GR]
        return (grad_at_input - grad_at_input.mean(dim=-1, keepdim=True)).reshape(logits.shape), None, None, None

def modify_jacobian(jacobian):
    # jacobian is batched
    rank = 5
    try:
        U, S, Vh = torch.linalg.svd(jacobian, full_matrices=False)
        U, S, Vh = U[:, :, :rank], S[:, :rank], Vh[:, :rank, :]
        modified_jacobian = torch.bmm(U, Vh)
        return modified_jacobian
    except:
        print('svd failed')
        return jacobian

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

def gaussian_softmax(
        logits,
        tau: float = 1,
        hard: bool = False,
        eps: float = 1e-10,
        dim: int = -1,
):
    gaussians = torch.randn_like(logits)

    theta = (logits + gaussians) / tau  # ~Gumbel(logits,tau)
    y_soft = theta.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def plot_softmaxes(logits, tau=0.01, dim=-1, K=1):
    y_soft_gaussian = torch.zeros_like(logits)
    y_soft_gumbel = torch.zeros_like(logits)
    for k in range(K):
        gaussians = torch.randn_like(logits)
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )

        theta_gaussian = (logits + gaussians) / tau
        theta_gumbel = (logits + gumbels) / tau
        y_soft_gaussian +=theta_gaussian.softmax(dim)/K
        y_soft_gumbel +=theta_gumbel.softmax(dim)/K

    y_soft_gumbel = y_soft_gumbel[0]
    y_soft_gaussian = y_soft_gaussian[0]
    y_soft = logits.softmax(dim)[0] # 0th batch

    B, L, C = logits.shape

    fig, axes = plt.subplots(L, 1, figsize=(8, 2.5 * L), sharex=True)

    for i in range(L):
        ax = axes[i]

        # Sort by descending true softmax probability
        sort_idx = torch.argsort(y_soft[i], descending=True)
        y_true_sorted = y_soft[i][sort_idx]
        y_gumbel_sorted = y_soft_gumbel[i][sort_idx]
        y_gaussian_sorted = y_soft_gaussian[i][sort_idx]

        x = torch.arange(C)

        # Plot each curve
        ax.plot(x, y_true_sorted.cpu().numpy(), 'o-', label='True', color='C0')
        ax.plot(x, y_gumbel_sorted.cpu().numpy(), 's-', label=f'Gumbel (τ={tau})', color='C1')
        ax.plot(x, y_gaussian_sorted.cpu().numpy(), '^-', label=f'Normal (τ={tau})', color='C2')

        ax.set_ylabel("Probability")
        ax.set_title(f"Latent dim {i}")
        ax.set_ylim(0, max(y_true_sorted.max(), y_gumbel_sorted.max(), y_gaussian_sorted.max()) * 1.1)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Category Index")
    plt.tight_layout()
    plt.savefig(f'saved_figs/softmaxes', dpi=300)
    plt.close(fig)
    print('saved softmaxes')
