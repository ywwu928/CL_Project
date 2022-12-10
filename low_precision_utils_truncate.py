import torch 
from torch.autograd import Function 
import torch.nn as nn 
from torch.nn import Parameter 
import torch.nn.functional as F 

eps = 1e-05

def truncate_with_none_check(t):
    if t is None: return
    else: t.mf_truncate_(t)

class SConvFunc(Function): 
    @staticmethod 
    def forward(ctx, input, weight, bias): 
        truncate_with_none_check(input)
        truncate_with_none_check(weight)
        truncate_with_none_check(bias)

        stride=1
        padding=1 
        dilation=1 
        groups=1
        ctx.save_for_backward(input, weight, bias) 
        ctx.stride = stride 
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        result = F.conv2d(input = input, weight = weight, bias = bias, stride = stride, padding = padding, dilation = dilation, groups = groups)
        truncate_with_none_check(result)
        return result
            
    @staticmethod 
    def backward(ctx, grad_output): 
        truncate_with_none_check(grad_output)

        input, weight, bias = ctx.saved_tensors 
        stride = ctx.stride 
        padding = ctx.padding 
        dilation = ctx.dilation 
        groups = ctx.groups 
        grad_input = grad_weight = grad_bias = None 

        if ctx.needs_input_grad[0]: 
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups) 
        if ctx.needs_input_grad[1]: 
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups) 
        if bias is not None and ctx.needs_input_grad[2]: 
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0) 

        truncate_with_none_check(grad_input)
        truncate_with_none_check(grad_weight)
        truncate_with_none_check(grad_bias)
        return grad_input, grad_weight, grad_bias 


class SConv2d(nn.Module): 
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1): 
        super(SConv2d, self).__init__() 
        self.weight = Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size)) 
        self.bias = Parameter(torch.randn(out_channel)) 
        self.conv_function = SConvFunc.apply 
        self.stride = stride 
        self.padding = padding 
        self.dilation = dilation 
        self.groups = groups 
    
    def forward(self, input):
        return self.conv_function(input, self.weight, self.bias) 

# Inherit from Function
class SLinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        truncate_with_none_check(input)
        truncate_with_none_check(weight)
        truncate_with_none_check(bias)

        ctx.save_for_backward(input, weight, bias) 

        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        truncate_with_none_check(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        truncate_with_none_check(grad_output)

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        grad_output = grad_output.float()

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        truncate_with_none_check(grad_input)
        truncate_with_none_check(grad_weight)
        truncate_with_none_check(grad_bias)
        return grad_input, grad_weight, grad_bias


class SLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(SLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = Parameter(torch.randn(output_features, input_features))
        if bias:
            self.bias = Parameter(torch.randn(output_features)) 
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return SLinearFunction.apply(input, self.weight, self.bias) 

class SBatchNormFunc(Function): 
    
    @staticmethod 
    def forward(ctx, input, gamma, beta): 
        truncate_with_none_check(input)
        truncate_with_none_check(gamma)
        truncate_with_none_check(beta)

        # gamma = gamma.view(1, -1, 1, 1) # 1 * C * 1 * 1 
        mean = input.mean(dim = (0,2,3), keepdim = True) 
        mean.requires_grad_(False) 
        variance = input.var(dim = (0,2,3), unbiased=False, keepdim = True) 
        variance.requires_grad_(False) 
        x_hat = (input - mean)/(torch.sqrt(variance + eps)) # N * C * S * S 
        x_hat.requires_grad_(False) 

        ctx.save_for_backward(input, gamma) 
        result = x_hat * gamma + beta
        truncate_with_none_check(result)
        return result
    
    @staticmethod 
    def backward(ctx, grad_output): 
        truncate_with_none_check(grad_output)
        input, gamma = ctx.saved_tensors 
        mean = input.mean(dim = (0,2,3), keepdim = True) 
        # mean.requires_grad_(False) 
        variance = input.var(dim = (0,2,3), unbiased=False, keepdim = True) 
        # variance.requires_grad_(False) 
        x_hat = (input - mean)/(torch.sqrt(variance + eps)) # N * C * S * S 
        # x_hat.requires_grad_(False) 
        B = input.shape[0] * input.shape[2] * input.shape[3] 
        dL_dxi_hat = grad_output * gamma
        # dL_dvar = (-0.5 * dL_dxi_hat * (input - avg) / ((var + eps) ** 1.5)).sum((0, 2, 3), keepdim=True) 
        # dL_davg = (-1.0 / torch.sqrt(var + eps) * dL_dxi_hat).sum((0, 2, 3), keepdim=True) + dL_dvar * (-2.0 * (input - avg)).sum((0, 2, 3), keepdim=True) / B
        dL_dvar = (-0.5 * dL_dxi_hat * (input - mean)).sum((0, 2, 3), keepdim=True)  * ((variance + eps) ** -1.5) # edit
        dL_davg = (-1.0 / torch.sqrt(variance + eps) * dL_dxi_hat).sum((0, 2, 3), keepdim=True) + (dL_dvar * (-2.0 * (input - mean)).sum((0, 2, 3), keepdim=True) / B) #edit

        dL_dxi = (dL_dxi_hat / torch.sqrt(variance + eps)) + (2.0 * dL_dvar * (input - mean) / B) + (dL_davg / B) # dL_dxi_hat / sqrt()
        # dL_dgamma = (grad_output * output).sum((0, 2, 3), keepdim=True) 
        dL_dgamma = (grad_output * x_hat).sum((0, 2, 3), keepdim=True) # edit
        dL_dbeta = (grad_output).sum((0, 2, 3), keepdim=True)

        truncate_with_none_check(dL_dxi)
        truncate_with_none_check(dL_dgamma)
        truncate_with_none_check(dL_dbeta)
        return dL_dxi, dL_dgamma, dL_dbeta 

class SBatchNorm(nn.Module): 
    def __init__(self, dimension, eps = 1e-05): 
        super(SBatchNorm, self).__init__() 
        self.gamma = Parameter(torch.ones((1, dimension, 1, 1), requires_grad = True)) 
        self.beta = Parameter(torch.zeros((1, dimension, 1, 1), requires_grad = True)) 
        self.eps = eps 

    def forward(self, input): 
        return SBatchNormFunc.apply(input, self.gamma, self.beta) 
