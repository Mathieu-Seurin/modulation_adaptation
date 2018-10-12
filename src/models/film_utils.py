import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal, kaiming_uniform
from torch.autograd import Variable
from models.gpu_utils import FloatTensor

class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (1 + gammas) * x + betas
        #return gammas * x + betas

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, with_residual, with_batchnorm):
        super(ResidualBlock, self).__init__()

        self.conv_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1)

        if with_batchnorm:
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.bn = lambda x:x

        if with_residual:
            self.residual = lambda x,result : x+result
        else:
            self.residual = lambda x,result: result
        self.conv_main = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        # Adding coord maps
        w, h, batch_size = x.size(2), x.size(3), x.size(0)
        coord = coord_map((w,h)).expand(batch_size, -1, -1, -1).type_as(x)
        x = torch.cat([x, coord], 1)

        #Before residual connection
        after_proj = self.conv_proj(x)
        after_proj = F.relu(after_proj)

        # Second convolution, relu batch norm
        output = self.conv_main(after_proj)
        output = self.bn(output)
        output = F.relu(output)

        # Add residual connection
        output = self.residual(after_proj,output)

        return output

class FiLMedResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, with_residual, with_batchnorm):

        super(FiLMedResBlock, self).__init__()

        self.conv_proj = nn.Conv2d(in_dim + 2, out_dim, kernel_size=1, stride=1)

        if with_batchnorm:
            self.bn = nn.BatchNorm2d(out_dim, affine=False)
        else:
            self.bn = lambda x:x

        if with_residual:
            self.residual = lambda x,result : x+result
        else:
            self.residual = lambda x,result: result

        self.conv_main = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.film = FiLM()

    def forward(self, x, gammas, betas):

        # Adding coord maps
        w, h, batch_size = x.size(2), x.size(3), x.size(0)
        coord = coord_map((w,h)).expand(batch_size, -1, -1, -1).type_as(x)
        x = torch.cat([x, coord], 1)

        #Before residual connection
        after_proj = self.conv_proj(x)
        after_proj = F.relu(after_proj)

        # Conv before film
        output = self.conv_main(after_proj)
        output = self.bn(output)

        # Film
        output = self.film(output, gammas, betas)
        output = F.relu(output)

        # Add residual connection
        output = self.residual(after_proj,output)

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
        elif isinstance(m, nn.GRU):
            init_params(m.weight_hh_l0)
            init_params(m.weight_ih_l0)


def coord_map(shape, start=-1, end=1):
    """
    Gives, a 2d shape tuple, returns two mxn coordinate maps,
    Ranging min-max in the x and y directions, respectively.
    """
    m, n = shape
    x_coord_row = torch.linspace(start, end, steps=n)
    y_coord_row = torch.linspace(start, end, steps=m)
    x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n))).unsqueeze(0)
    y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n))).unsqueeze(0)
    return Variable(torch.cat([x_coords, y_coords], 0))