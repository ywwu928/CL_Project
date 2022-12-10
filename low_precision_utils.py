import torch 
from torch.autograd import Function 
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


class SConv2d(torch.nn.Module): 
    def __init__(self, in_channel, out_channel, h_and_w, input, weight, bias, stride, padding, dilation, groups): 
        self.weight = torch.randn(out_channel, in_channel, h_and_w, h_and_w) 
        self.bias = torch.randn(out_channel) 
        self.conv_function = SConvFunc.apply 
    
    def forward(self): 
        return self.conv_function(self.weight, self.bias) 

# Inherit from Function
class LinearFunction(Function):

    @staticmethod
    def forward(ctx, mf, input, weight, bias=None):
        input = mf.truncate_floats(input)
        weight = mf.truncate_floats(weight)
        if bias is not None:
            bias = bias.float()
            bias = mf.truncate_floats(bias)

        ctx.save_for_backward(input, weight, bias, mf)

        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        output = output.float()
        output = mf.truncate_floats(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mf = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        grad_output = grad_output.float()
        grad_output = mf.truncate_floats(grad_output)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        grad_input = grad_input.float()
        grad_weight = grad_weight.float()
        grad_bias = grad_bias.float()
        grad_input = mf.truncate_floats(grad_input)
        grad_weight = mf.truncate_floats(grad_weight)
        grad_bias = mf.truncate_floats(grad_bias)

        return grad_input, grad_weight, grad_bias


class SLinear(nn.Module):
    def __init__(self, mf, input_features, output_features, bias=True):
        super(SLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = torch.randn(output_features, input_features)
        self.mf = mf
        if bias:
            self.bias = torch.randn(output_features)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return LinearFunction.apply(self.mf, input, self.weight, self.bias) 
