import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal, kaiming_uniform

class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (1 + gammas) * x + betas

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, with_residual, with_batchnorm):
        super(ResidualBlock, self).__init__()

        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1)

        if with_batchnorm:
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.bn = lambda x:x

        if with_residual:
            self.residual = lambda x,result : x+result
        else:
            self.residual = lambda x,result: result
        self.conv1 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        result = F.relu(self.proj(x))
        result = self.bn(self.conv1(result))
        result = F.relu(self.residual(x,result))
        return result

class FiLMedResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, with_residual, with_batchnorm):

        super(FiLMedResBlock, self).__init__()

        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1)

        if with_batchnorm:
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.bn = lambda x:x

        if with_residual:
            self.residual = lambda x,result : x+result
        else:
            self.residual = lambda x,result: result

        self.conv1 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.film = FiLM()

    def forward(self, x, gammas, betas):

        after_proj = F.relu(self.proj(x))
        output = self.bn(self.conv1(after_proj))
        output = self.film(output, gammas, betas)
        output = F.relu(self.residual(after_proj,output))

        return output

def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = kaiming_normal
    elif init.lower() == 'uniform':
        init_params = kaiming_uniform
    else:
        return
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init_params(m.weight)