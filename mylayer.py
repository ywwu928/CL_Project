import torch
import torch.nn as nn
from torch.autograd import Function
import converter

# Inherit from Function
class LinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        mf  = converter.MyFloat(5, 10)
        input = input.float()
        weight = weight.float()
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
        return LinearFunction.apply(input, self.weight, self.bias)


if __name__ == '__main__':
    # Test
    input = torch.randn(1, 3)
    linear = SLinear(3, 4)
    output = linear(input)
    print(output)
    