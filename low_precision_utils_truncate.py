import torch 
from torch.autograd import Function 
import torch.nn as nn 
import torch.nn.functional as F 

class SConvFunc(Function): 
    @staticmethod 
    def forward(ctx, weight, bias, stride=1, padding=0, dilation=1, groups=1, module=None): 
        ctx.save_for_backend(input, weight, bias) 
        ctx.module = module
        ctx.stride = stride 
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        return F.conv2d(input = input, weight = weight, bias = bias, stride = stride, padding = padding, dilation = dilation, groups = groups) 
    
    @staticmethod 
    def backward(ctx, grad_output): 
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

        return grad_input, grad_weight, grad_bias 


class SConv2d(nn.Module): 
    def __init__(self, in_channel, out_channel, h_and_w, stride, padding, dilation, groups): 
        self.weight = torch.randn(out_channel, in_channel, h_and_w, h_and_w) 
        self.bias = torch.randn(out_channel) 
        self.conv_function = SConvFunc.apply 
    
    def forward(self, input): 
        return self.conv_function(self.weight, self.bias) 

# Inherit from Function
class SLinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        if bias is not None:
            bias = bias.float()

        ctx.save_for_backward(input, weight, bias) 

        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        output = output.float()
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        grad_output = grad_output.float()

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        grad_input = grad_input.float()
        grad_weight = grad_weight.float()
        grad_bias = grad_bias.float()

        return grad_input, grad_weight, grad_bias


class SLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(SLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = torch.randn(output_features, input_features)
        if bias:
            self.bias = torch.randn(output_features)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return SLinearFunction.apply(input, self.weight, self.bias) 

class SBatchNormFunc(Function): 
    
    @staticmethod 
    def forward(ctx, input, gamma, beta, eps): 
        gamma = gamma.view(1, -1, 1, 1) # 1 * C * 1 * 1 
        B = input.shape[0] * input.shape[2] * input.shape[3] 
        mean = input.mean(dim = (0,2,3), keepdim = True)
        variance = input.var(dim = (0,2,3), unbiased=False, keepdim = True)
        x_hat = (input - mean)/(torch.sqrt(variance + eps)) # N * C * S * S 

        ctx.save_for_backward(B, mean, variance, x_hat, gamma, beta, eps) 
        return x_hat * gamma + beta 
    
    @staticmethod 
    def backward(ctx, grad_output): 
        B, mean, variance, x_hat, gamma, beta, eps = ctx.saved_tensors 
        dL_dxi_hat = grad_output * gamma
        # dL_dvar = (-0.5 * dL_dxi_hat * (input - avg) / ((var + eps) ** 1.5)).sum((0, 2, 3), keepdim=True) 
        # dL_davg = (-1.0 / torch.sqrt(var + eps) * dL_dxi_hat).sum((0, 2, 3), keepdim=True) + dL_dvar * (-2.0 * (input - avg)).sum((0, 2, 3), keepdim=True) / B
        dL_dvar = (-0.5 * dL_dxi_hat * (input - mean)).sum((0, 2, 3), keepdim=True)  * ((variance + eps) ** -1.5) # edit
        dL_davg = (-1.0 / torch.sqrt(variance + eps) * dL_dxi_hat).sum((0, 2, 3), keepdim=True) + (dL_dvar * (-2.0 * (input - mean)).sum((0, 2, 3), keepdim=True) / B) #edit

        dL_dxi = (dL_dxi_hat / torch.sqrt(variance + eps)) + (2.0 * dL_dvar * (input - mean) / B) + (dL_davg / B) # dL_dxi_hat / sqrt()
        # dL_dgamma = (grad_output * output).sum((0, 2, 3), keepdim=True) 
        dL_dgamma = (grad_output * x_hat).sum((0, 2, 3), keepdim=True) # edit
        dL_dbeta = (grad_output).sum((0, 2, 3), keepdim=True)
        return dL_dxi, dL_dgamma, dL_dbeta 

class SBatchNorm(nn.Module): 
    def __init__(self, dimension, eps = 1e-05): 
        super(SBatchNorm, self).__init__() 
        self.gamma = torch.ones((1, dimension, 1, 1)) 
        self.beta = torch.zeros((1, dimension, 1, 1)) 

    def forward(self, input, eps): 
        return SBatchNormFunc.apply(input, self.gamma, self.beta, eps) 
