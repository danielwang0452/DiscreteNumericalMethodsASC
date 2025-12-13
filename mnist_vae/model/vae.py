# The code has been modified from https://github.com/chijames/GST

import torch
import torch.nn.functional as F
from torch import nn
from .categorical_beta import categorical_repara, categorical_repara_jacobian
import torch.distributions as dists

import math

activation_map = {
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
}

class Encoder(nn.Module):
    def __init__(self, latent_dim, categorical_dim, activation='relu'):
        super(Encoder, self).__init__()
        self.categorical_dim = categorical_dim
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)
        self.activation = activation_map[activation.lower()]()
        
    def forward(self, x):
        h1 = self.activation(self.fc1(x))
        h2 = self.activation(self.fc2(h1))
        # Note that no activation function is applied to the output of encoder 
        # this is consistent with the original categorical MNIST VAE as in
        # https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
        # Intuitively, applying activation function like ReLU on encoder output is not
        # recommended, since:
        # 1. softmax function is a non-linear transformation itself;
        # 2. as the output of softmax always sums up to one, the gradient on its input 
        #    would sum up to zero. applying additional activation functions like ReLU woul
        #    break this structure, leading to a sub-optimal performance. 
        return self.fc3(h2).view(-1, self.latent_dim, self.categorical_dim)

class Decoder(nn.Module):
    def __init__(self, latent_dim, categorical_dim, activation='relu'):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 784)
        self.activation = activation_map[activation.lower()]()
        self.sigmoid = nn.Sigmoid()
    
    def decode(self, logits, sigmoid=True):
        h1 = self.activation(self.fc1(logits))
        h2 = self.activation(self.fc2(h1))
        if sigmoid:
            return self.sigmoid(self.fc3(h2))
        else:
            return self.fc3(h2)
        
    def forward(self, logits, target):
        return torch.nn.functional.binary_cross_entropy(
            self.decode(logits),
            target,
            reduction='none',
        )

class VAE(nn.Module):
    def __init__(
        self, 
        latent_dim=4, 
        categorical_dim=2, 
        temperature=1., 
        method='reinmax', 
        activation='relu'
    ):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(latent_dim, categorical_dim, activation=activation)
        self.decoder = Decoder(latent_dim, categorical_dim, activation=activation)
        
        self.categorical_dim = categorical_dim
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.method = method
        self.alpha=1.0
        
        if 'exact' == self.method:
            self.forward = self.forward_exact
        else:
            self.forward = self.forward_approx
        self.itensor = None
        #self.compute_code = self.compute_code_track#regular

    def compute_code_jacobian(self, data, with_log_p=False):
        theta = self.encoder(data)
        def theta_gradient_save(gradient):
            self.theta_gradient = gradient
            return gradient
        theta.register_hook(theta_gradient_save)
        z, qy = categorical_repara_jacobian(theta, self.temperature, self.method, self.alpha, self)
        qy = qy.view(data.size(0), self.latent_dim, self.categorical_dim)
        z = z.view(data.size(0), self.latent_dim, self.categorical_dim)
        if with_log_p:
            log_y = (z * theta).sum(dim=-1) - torch.logsumexp(theta, dim=-1)
            return z, qy, log_y
        else:
            return z, qy

    def compute_code_regular(self, data, with_log_p=False):
        theta = self.encoder(data)
        z, qy = categorical_repara(theta, self.temperature, self.method, self.alpha, model_ref=self)
        qy = qy.view(data.size(0), self.latent_dim, self.categorical_dim)
        z = z.view(data.size(0), self.latent_dim, self.categorical_dim)
        self.z = z
        if with_log_p:
            log_y = (z * theta).sum(dim=-1) - torch.logsumexp(theta, dim=-1)
            return z, qy, log_y
        else:
            return z, qy
        
    def compute_code_track(self, data, with_log_p=False):
        theta = self.encoder(data)
        #print(theta)
        def theta_gradient_save(gradient):
            self.theta_gradient = gradient
            #print(self.theta_gradient)
            return gradient 
        theta.register_hook(theta_gradient_save)
        z, qy = categorical_repara(theta, self.temperature, self.method, self.alpha, self)
        qy = qy.view(data.size(0), self.latent_dim, self.categorical_dim)
        z = z.view(data.size(0), self.latent_dim, self.categorical_dim)
        if with_log_p:
            log_y = (z * theta).sum(dim=-1) - torch.logsumexp(theta, dim=-1)
            return z, qy, log_y
        else:
            return z, qy
    
    def compute_bce_loss(self, data):
        z, _ = self.compute_code(data)
        loss = self.decoder(z.view(data.size(0), -1), data)
        return loss
    
    def forward_approx(self, data):
        batch_size = data.size(0)
        z, qy = self.compute_code(data)
        r_d = z.view(batch_size, -1)
        #print(z.shape, r_d.shape) torch.Size([100, 128, 10]) torch.Size([100, 1280])
        BCE = self.decoder(r_d, data)

        BCE_reduced = BCE.sum() / batch_size
        
        qy = qy.view(batch_size, -1)
        log_ratio = torch.log(qy + 1e-10)
        #print(qy.shape, log_ratio.shape)
        KLD = torch.sum(qy * log_ratio, dim=-1) + math.log(self.categorical_dim) * self.latent_dim
        KLD_reduced = torch.sum(qy * log_ratio, dim=-1).mean() + math.log(self.categorical_dim) * self.latent_dim
        #print(BCE.shape, KLD.shape)
        unreduced_loss = BCE.sum(dim=-1) + KLD
        #print(unreduced_loss.mean(), BCE_reduced+KLD_reduced)
        self.loss = unreduced_loss
        return BCE_reduced, KLD_reduced, (z, r_d), qy
    
    def exact_bce_loss(self, data):
        def convert_to_i_base(i):
            i_list = list()
            while i > 0:
                i_list.append(i % self.categorical_dim)
                i = i // self.categorical_dim
            i_list = [0] * (self.latent_dim - len(i_list)) + i_list[::-1]
            return i_list
        
        search_size = self.categorical_dim ** self.latent_dim
        assert search_size < 16384, "categorical_dim ** latent_dim too large"
        with torch.no_grad():
            if self.itensor is None:
                i_tensor_list = list()
                for i in range(search_size):
                    i_list = convert_to_i_base(i)
                    i_tensor_list.append(
                        torch.Tensor(
                            [
                                [i_i == j for j in range(self.categorical_dim)]
                                for i_i in i_list
                            ]
                        ).view(-1)
                    )
                self.itensor = torch.stack(i_tensor_list).view(search_size, -1).to(data.device)
                
            i_tensor = self.itensor.unsqueeze(0).expand(data.size(0), -1, -1) # batchsize, sample, z_logits
            target = data.unsqueeze(1).expand(-1, search_size, -1) # batchsize, sample, i_logits
            i_loss = self.decoder(i_tensor, target).detach()
        
        _, qy = self.compute_code(data)
        qy = qy.view(-1, self.latent_dim * self.categorical_dim).unsqueeze(1).expand(-1, search_size, -1)
        qy = (i_tensor * qy).view(-1, search_size, self.latent_dim, self.categorical_dim).sum(dim=-1)
        loss = (torch.prod(qy, dim=-1) * i_loss.sum(dim=-1)).sum() / data.size(0)
        
        return loss 

    def forward_exact(self, data):
        def convert_to_i_base(i):
            i_list = list()
            while i > 0:
                i_list.append(i % self.categorical_dim)
                i = i // self.categorical_dim
            i_list = [0] * (self.latent_dim - len(i_list)) + i_list[::-1]
            return i_list
        
        batch_size = data.size(0)
        search_size = self.categorical_dim ** self.latent_dim
        with torch.no_grad():
            if self.itensor is None:
                i_tensor_list = list()
                for i in range(search_size):
                    i_list = convert_to_i_base(i)
                    i_tensor_list.append(
                        torch.Tensor(
                            [
                                [i_i == j for j in range(self.categorical_dim)]
                                for i_i in i_list
                            ]
                        ).view(-1)
                    )
                self.itensor = torch.stack(i_tensor_list).view(search_size, -1).to(data.device)
                
            i_tensor = self.itensor.unsqueeze(0).expand(batch_size, -1, -1) # batchsize, sample, z_logits
            target = data.unsqueeze(1).expand(-1, search_size, -1) # batchsize, sample, i_logits
            i_loss = self.decoder(i_tensor, target).detach()
        
        z, qy = self.compute_code(data)
        i_qy = qy.view(-1, self.latent_dim * self.categorical_dim).unsqueeze(1).expand(-1, search_size, -1)
        i_qy = (i_tensor * i_qy).view(-1, search_size, self.latent_dim, self.categorical_dim).sum(dim=-1)
        BCE_ENC = (torch.prod(i_qy, dim=-1) * i_loss.sum(dim=-1)).sum() / batch_size
        
        qy = qy.view(batch_size, -1)
        log_ratio = torch.log(qy + 1e-10)
        KLD = torch.sum(qy * log_ratio, dim=-1).mean() + math.log(self.categorical_dim) * self.latent_dim
        
        r_d = z.detach().view(batch_size, -1)
        BCE_DEC = self.decoder(r_d, data).sum() / batch_size
        return BCE_ENC - BCE_ENC.detach() + BCE_DEC, KLD, (z, r_d), qy
    
    def exact_bce_gradient(self, data):
        self.train()
        
        loss = self.exact_bce_loss(data)
        
        self.zero_grad()
        loss.backward()
        return self.theta_gradient

    def approx_bce_gradient(self, data):
        self.train()
        
        if self.method == 'reinforce':
            z, _, log_y = self.compute_code(data, with_log_p=True)
            loss = self.decoder(z.view(data.size(0), -1), data)
            log_p = log_y.sum(-1)
            loss = torch.sum(log_p * loss.detach()) / data.size(0)
        else:
            loss = self.compute_bce_loss(data).sum() / data.size(0)
        self.zero_grad()
        loss.backward()
        return self.theta_gradient

    def analyze_gradient(self, data, ct):
        exact_grad = self.exact_bce_gradient(data)
        mean_grad = torch.zeros_like(exact_grad).to(torch.float32)
        std_grad = torch.zeros_like(exact_grad).to(torch.float32)
        if self.method in ['reinmax_test', 'reinmax_v2', 'reinmax_v3']:
            mean_t1_grad = torch.zeros_like(mean_grad.view(-1, exact_grad.size()[-1])).double()
            mean_t2_grad = torch.zeros_like(mean_grad.view(-1, exact_grad.size()[-1])).double()
            std_t1_grad = torch.zeros_like(mean_grad.view(-1, exact_grad.size()[-1])).double()
            std_t2_grad = torch.zeros_like(mean_grad.view(-1, exact_grad.size()[-1])).double()

        for i in range(ct):
            if self.method == 'exact':
                grad = exact_grad
            else:
                grad = self.approx_bce_gradient(data)
            mean_grad += grad 
            std_grad += grad ** 2
            '''
            if self.method in ['reinmax_test', 'reinmax_v2', 'reinmax_v3']:
                t1_grad = self.reinmax_term1#.reshape((100, 4 ,8))
                t2_grad = self.reinmax_term2#.reshape((100, 4 ,8))
                mean_t1_grad += t1_grad
                mean_t2_grad += t2_grad
                std_t1_grad += t1_grad ** 2
                std_t2_grad += t2_grad ** 2
            '''

        mean_grad = mean_grad / ct 
        std_grad = (std_grad / ct - mean_grad ** 2).abs() ** 0.5
        diff = (exact_grad - mean_grad).norm()  # this norm is taken over the batch dimension?

        if self.method in ['reinmax_test']:
            mean_t1_grad = mean_t1_grad / ct
            std_t1_grad = (std_t1_grad / ct - mean_t1_grad ** 2).abs() ** 0.5
            mean_t2_grad = mean_t2_grad / ct
            std_t2_grad = (std_t2_grad / ct - mean_t2_grad ** 2).abs() ** 0.5
            return (
                diff / exact_grad.norm(),
                diff / mean_grad.norm(),
                (std_grad.reshape((100, 4 ,8)).norm(dim=(1, 2)) / mean_grad.reshape((100, 4 ,8)).norm(dim=(1, 2))).mean(),#.norm() / mean_grad.norm(),
                (exact_grad * mean_grad).sum() / (exact_grad.norm() * mean_grad.norm()),
                mean_grad.norm(dim=(1, 2)).mean(),
                (std_t1_grad.reshape((100, 4 ,8)).norm(dim=(1, 2)) / mean_t1_grad.reshape((100, 4 ,8)).norm(dim=(1, 2))).mean(),
                (std_t2_grad.reshape((100, 4 ,8)).norm(dim=(1, 2)) / mean_t2_grad.reshape((100, 4 ,8)).norm(dim=(1, 2))).mean()
            )
            # for t1_std, either
            # std_t1_grad.norm() / mean_t1_grad.norm(),
            # or (std_t1_grad.reshape((100, 4 ,8)).norm(dim=(1, 2)) / mean_t1_grad.reshape((100, 4 ,8)).norm(dim=(1, 2))).mean()

        return (
            #diff.norm() / exact_grad.norm(),
            #diff.norm() / mean_grad.norm(),#(diff.reshape((100, 4 ,8)).norm(dim=(1, 2))/mean_grad.reshape((100, 4 ,8)).norm(dim=(1, 2))).mean(),
            #(std_grad.reshape((100, 4 ,8)).norm(dim=(1, 2))/mean_grad.reshape((100, 4 ,8)).norm(dim=(1, 2))).mean(),#std_grad.norm(dim=(1, 2)).mean(),#std_grad.norm() / mean_grad.norm(),
            (exact_grad * mean_grad).sum() / (exact_grad.norm() * mean_grad.norm())
            #mean_grad.norm(dim=(1, 2)).mean()
        )

       # return torch.tensor(1),torch.tensor(1), torch.tensor(1), torch.tensor(1), torch.tensor(1)

    def get_sample_variance(self, data, ct):
        mean_grad = None
        for i in range(ct):
            grad = self.approx_bce_gradient(data)
            if mean_grad == None:
                mean_grad = torch.zeros_like(grad).to(torch.float32)
                std_grad = torch.zeros_like(grad).to(torch.float32)
            mean_grad += grad
            std_grad += grad ** 2

        mean_grad = mean_grad / ct
        std_grad = (std_grad / ct - mean_grad ** 2).abs() ** 0.5
        return (
            # (diff.reshape((100, 4 ,8)).norm(dim=(1, 2))/mean_grad.reshape((100, 4 ,8)).norm(dim=(1, 2))).mean(),
            (std_grad.reshape((100, self.latent_dim, self.categorical_dim)).norm(dim=(1, 2)) / mean_grad.reshape((100, self.latent_dim, self.categorical_dim)).norm(
                dim=(1, 2))).mean(),  # std_grad.norm(dim=(1, 2)).mean(),#std_grad.norm() / mean_grad.norm(),
            mean_grad.norm(dim=(1, 2)).mean()
        )

    def compute_marginal_log_likelihood(self, data, k=100):
        """
        Importance-weighted estimate of the marginal log-likelihood:
            log p(x) ≈ log(1/K * sum_k [p(x|z_k)p(z_k)/q(z_k|x) ])

        Args:
            data: [batch, 784]
            k: number of samples per datapoint (default: 1)
        Returns:
            log_marginal_likelihood: scalar tensor (mean across batch)
            log_w: [batch, k] tensor of log-weights
        """
        batch_size = data.size(0)
        device = data.device

        # Expand input for K samples
        data_expanded = data.unsqueeze(1).expand(-1, k, -1).reshape(-1, data.size(-1))  # [batch*k, 784]

        # Sample z ~ q(z|x)
        z_list = []
        log_q_list = []
        qy_list = []

        for _ in range(k):
            z, qy = self.compute_code(data, with_log_p=False)
            z_list.append(z)
            qy_list.append(qy)
            # log q(z|x) = sum_i sum_c z_ic * log qy_ic
            log_q = (z * torch.log(qy + 1e-10)).sum(dim=(1, 2))

            #qy = qy.view(batch_size, -1)
            #log_ratio = torch.log(qy + 1e-10)
            #log_q = torch.sum(qy * log_ratio, dim=-1) + math.log(self.categorical_dim) * self.latent_dim
            #print(log_q.mean())

            log_q_list.append(log_q)


        z_samples = torch.stack(z_list, dim=1)  # [batch, k, latent_dim, categorical_dim]
        log_q_z_given_x = torch.stack(log_q_list, dim=1)  # [batch, k]
        # test IWAE vs ELBO for k=1
        #qy2 = qy.view(batch_size, -1)
        #log_ratio = torch.log(qy2 + 1e-10)
        #KLD = torch.sum(qy2 * log_ratio, dim=-1)
        #print(log_q_z_given_x.mean(dim=1), KLD)
        # Flatten z for decoding
        z_flat = z_samples.view(batch_size * k, -1)  # [batch*k, latent_dim*categorical_dim]

        # Compute reconstruction log-likelihood log p(x|z)
        x_recon = self.decoder.decode(z_flat)#, sigmoid=False)
        #print(x_recon)
        #x_recon = torch.clamp(x_recon, 1e-8, 1 - 1e-8)
        #log_p_x_given_z2 = data_expanded * torch.log(x_recon) + (1 - data_expanded) * torch.log(1 - x_recon)
        log_p_x_given_z = -F.binary_cross_entropy(x_recon, data_expanded, reduction='none')
        log_p_x_given_z = log_p_x_given_z.sum(dim=1).view(batch_size, k)
        # Prior: uniform categorical, so log p(z) = -log(C) * L
        log_p_z = -math.log(self.categorical_dim) * self.latent_dim
        log_p_z = torch.full_like(log_p_x_given_z, log_p_z)

        # Compute unnormalized log-weights

        log_w = log_p_x_given_z + log_p_z - log_q_z_given_x  # [batch, k]
        #print(2,(log_p_z - log_q_z_given_x).mean())
        # IWAE marginal log-likelihood estimate (mean over batch)
        log_marginal_likelihood = (torch.logsumexp(log_w, dim=1) - torch.log(torch.tensor(k))).mean()

        return log_marginal_likelihood, log_w
