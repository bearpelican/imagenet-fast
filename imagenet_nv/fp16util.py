import torch
import torch.nn as nn


class FP16(nn.Module):
    def __init__(self, module): 
        super(FP16, self).__init__()
        self.module = batchnorm_to_fp32(module.half())
        
    def forward(self, input): 
        return self.module(input.half())

def batchnorm_to_fp32(module):
    '''
    BatchNorm layers to have parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_fp32(child)
    return module

def copy_fp32_to_model(m, fp32_params):
    m_params = list(m.parameters())
    for fp32_param, m_param in zip(fp32_params, m_params):
        m_param.data.copy_(fp32_param.data)

def update_fp32_grads(fp32_params, m):
    m_params = list(m.parameters())
    for fp32_param, m_param in zip(fp32_params, m_params):
        if fp32_param.grad is None:
            fp32_param.grad = torch.nn.Parameter(fp32_param.data.new().resize_(*fp32_param.data.size()))
        fp32_param.grad.data.copy_(m_param.grad.data)

