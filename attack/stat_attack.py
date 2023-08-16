import torch
import torch.nn as nn
from torchattacks.attack import Attack
from attack.mmd import MMDLoss
from math import inf
import torch.nn.functional as F
import torch.autograd as ag
from attack.blur import AdaptiveGaussianFilter


class StatAttack(Attack):
    def __init__(
            self,
            model: nn.Module,
            step: int = 10,
            attack_mode: str = 'all',
            # noise attack
            noise_mode: str = 'add',

            # exposure attack
            bias_mode: str = 'same',
            bias_lr: float = 1e-1,
            spatial_lr: float = 1e-3,
            degree: int = 11,
            tune_scale: int = 8,
            lambda_mmd: float = 10,
            lambda_n: float = 0.0,
            lambda_b: float = 10,
            lambda_s: float = 10,
            momentum_decay: float = 1.0,

            # exposure
            epsilon: float = 60/255,

            # noise
            epsilon_n: float = 4. / 255.,
            noise_lr: float = 1. / 255.,
            noise_ti: bool = False,
            ti_size: int = 21,

            # blur
            blur_attack: str = 'add',
            blur_lr: float = 1e-1,
            radius: int = 3

    ):
        super().__init__("BiasFieldAttack", model)
        # same assert
        assert step >= 0, f'step should be non-negative integer, got {step}'
        assert attack_mode in ('first', 'all')
        assert noise_mode in ('none', 'add')

        assert bias_mode in ('none', 'rgb', 'same')
        assert noise_lr >= 0, f'noise_lr should be non-negative floats, got {noise_lr}'
        assert bias_lr >= 0, f'bias_lr should be non-negative floats, got {bias_lr}'
        assert spatial_lr >= 0, f'spatial_lr should be non-negative floats, got {spatial_lr}'
        assert lambda_mmd >= 0, f'lambda_l should be non-negative float, got {lambda_mmd}'
        assert lambda_n >= 0, f'lambda_b should be non-negative float, got {lambda_n}'
        assert lambda_b >= 0, f'lambda_b should be non-negative float, got {lambda_b}'
        assert lambda_s >= 0, f'lambda_s should be non-negative float, got {lambda_s}'
        assert momentum_decay >= 0, f'momentum_decay should be non-negative float, got {momentum_decay}'
        assert epsilon >= 0, f'epsilon should be non-negative float, got {epsilon}'
        assert epsilon_n >= 0, f'epsilon_n should be non-negative float, got {epsilon_n}'
        assert degree > 0, f'degree should be positive integer, got {degree}'
        assert tune_scale > 0, f'tune_scale should be positive integer, got {tune_scale}'
        assert ti_size > 0, f'ti_size should be positive integer, got {ti_size}'
        if noise_ti:
            assert noise_mode == 'add'

        self.step = step
        self.attack_mode = attack_mode
        self.noise_mode = noise_mode
        self.bias_mode = bias_mode
        self.noise_lr = noise_lr
        self.bias_lr = bias_lr
        self.spatial_lr = spatial_lr
        self.lambda_mmd = lambda_mmd
        self.lambda_n = lambda_n
        self.lambda_b = lambda_b
        self.lambda_s = lambda_s
        self.momentum_decay = momentum_decay
        self.epsilon = epsilon
        self.epsilon_n = epsilon_n
        self.noise_ti = noise_ti
        self.degree = degree
        self.tune_scale = tune_scale
        self.ti_size = ti_size
        self.blur_attack = blur_attack
        self.blur_lr = blur_lr
        self.radius = radius

    def __call__(self, guidedImage: torch.Tensor, tensor: torch.Tensor):
        # identity bias field

        n, c, h, w = tensor.size()

        assert c == 3, f'tensor should be batched RGB images, got {c} channels'
        params = []
        lrs = []

        guidedImage = guidedImage.clone().detach().to(self.device)
        ############################################################################################
        # add noise
        ############################################################################################
        if self.noise_mode == 'add':
            noise = torch.zeros_like(tensor).requires_grad_()
            params.append(noise)
            lrs.append(self.noise_lr)
        elif self.noise_mode == 'none':
            noise = None

        ############################################################################################
        # add blur
        ############################################################################################
        if self.blur_attack == 'add':
            new_norm = 100.0
            if h == w:
                gaussian = AdaptiveGaussianFilter(filter_size=self.radius, img_dim=h).to(tensor)
            sigmas = torch.full_like(tensor, 1).requires_grad_()
            norm_value = torch.norm(sigmas, p=2., dim=[1, 2, 3], keepdim=True)
            sigmas = sigmas / norm_value * new_norm
            params.append(sigmas)
            lrs.append(self.blur_lr)

        ############################################################################################
        # add exposure
        ############################################################################################
        # init coef
        if self.bias_mode in ('rgb', 'same'):
            a = 5
            num_coef = (self.degree + 1) * (self.degree + 2) // 2
            coef = torch.zeros(n, 1 if self.bias_mode == 'same' else 3, num_coef).to(tensor).requires_grad_()
            params.append(coef)
            lrs.append(self.bias_lr)
            # init optical_flow
            optical_flow = torch.zeros(n, h // self.tune_scale, w // self.tune_scale, 2).to(tensor).requires_grad_()
            params.append(optical_flow)
            lrs.append(self.spatial_lr)

            # create coord base
            coord_x = torch.linspace(-1, 1, w).to(tensor)[None, :]
            coord_y = torch.linspace(-1, 1, h).to(tensor)[:, None]
            coord = torch.stack((coord_x.expand(h, -1), coord_y.expand(-1, w)), dim=-1)

            base = torch.zeros(num_coef, h, w).to(tensor)
            i = 0
            for t in range(self.degree + 1):
                for l in range(self.degree - t + 1):
                    base[i, :, :].add_(coord_x ** t).mul_(coord_y ** l)
                    i += 1
            del i


        for n_iter in range(self.step + 1):
            pert = tensor.clone()
            if self.bias_mode in ('rgb', 'same'):
                bias_field = (base[None, None, :, :, :] * coef[:, :, :, None, None]).sum(dim=2)
                upsampled_optical_flow = F.interpolate(
                    optical_flow.permute(0, 3, 1, 2),
                    align_corners=False,
                    mode='bilinear',
                    size=(h, w),
                ).permute(0, 2, 3, 1)

                spatial_tuning = (coord + upsampled_optical_flow).clamp_(-1, 1)
                bias_field = F.grid_sample(bias_field, spatial_tuning, align_corners=True)
                pert = pert.log_().add_(bias_field).exp_()

            # add blur
            if self.blur_attack == 'add':
                pert = gaussian(pert, sigmas)

            if self.noise_mode == 'add':
                pert = pert + noise

            # optimized clamp
            if self.epsilon != inf:
                pert = torch.min(pert, tensor + self.epsilon)
                pert = torch.max(pert, tensor - self.epsilon)
                pert = pert.clamp(0, 1)

            # extract guideImage & pertImage feature
            _ = self.model(guidedImage)
            guided_output_feature = self.model.features['global_pool'].flatten(1)
            _ = self.model(pert)
            Output_feature = self.model.features['global_pool'].flatten(1)
            MMD = MMDLoss()

            # calculate MMD loss
            mmd_loss = MMD(source=Output_feature, target=guided_output_feature)
            print("mmd_loss:",mmd_loss)

            loss = torch.zeros(1).to(tensor)

            if self.lambda_mmd > 0:
                loss.add_(mmd_loss, alpha=self.lambda_mmd)


            if self.lambda_n > 0 and self.noise_mode != 'none':
                sparsity_n = noise.pow(2).sum()
                loss.add_(sparsity_n, alpha=self.lambda_n)

            if self.lambda_b > 0 and self.bias_mode != 'none':
                sparsity_b = bias_field.pow(2).sum().div_(h * w)
                loss.add_(sparsity_b, alpha=self.lambda_b)

            if self.lambda_s > 0 and self.bias_mode != 'none':
                diff_s_h = (optical_flow[:, :, 1:, :] - optical_flow[:, :, :-1, :])
                diff_s_v = (optical_flow[:, 1:, :, :] - optical_flow[:, :-1, :, :])
                sparsity_s = diff_s_h.pow(2).sum() + diff_s_v.pow(2).sum()
                loss.add_(sparsity_s, alpha=self.lambda_s / (self.tune_scale * self.tune_scale))



            if n_iter < self.step:
                with torch.no_grad():

                    grads = ag.grad(loss, params)

                    if self.momentum_decay > 0:
                        grad_norms = [grad.flatten(1).norm(p=1, dim=1) for grad in grads]
                        grad_norms = [norm.where(norm > 0, torch.ones(1).to(norm)) for norm in grad_norms]
                        grads = [
                            grad.div_(norm.view((-1, *((1,) * (grad.dim() - 1)))))
                            for grad, norm in zip(grads, grad_norms)
                        ]
                        if n_iter == 0:
                            cum_grads = grads
                        else:
                            cum_grads = [
                                cum_grad.mul_(self.momentum_decay).add_(grad)
                                for cum_grad, grad in zip(cum_grads, grads)
                            ]
                    else:
                        cum_grads = grads

                    for param, cum_grad, lr in zip(params, cum_grads, lrs):

                        if self.attack_mode == 'first':
                            param[0].sub_(cum_grad[0].sign(), alpha=lr)
                        elif self.attack_mode == 'all':
                            param.sub_(cum_grad.sign(), alpha=lr)
                    if self.noise_mode != 'none' and self.epsilon_n != inf:
                        noise = noise.clamp_(-self.epsilon_n, self.epsilon_n)

        return pert
