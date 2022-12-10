import struct
import torch
class MyFloat():
    '''
    Usage: (example, bit format for fp16)
    'device' is 'cpu' or 'cuda'
    ------
    mf MyFloat(5, 10, 'cpu')
    f = mf.truncate_float(f)
    '''

    def __init__(self, exp_bits, mant_bits, device) -> None:
        self.exp_bits = exp_bits
        self.mant_bits_tensor = torch.Tensor([mant_bits+1]).to(device)
        self.mant_bits_tensor_neg = torch.Tensor([-(mant_bits+1)]).to(device)
        self.mant_bits = mant_bits

        self.exp_min_raw = (-(2**(exp_bits-1))+1)+1
        self.exp_max_raw = (2**(exp_bits-1))+1

        self.mmask = int((12+mant_bits)*'1'+(52-mant_bits)*'0', 2)
        self.emask = int('1'+11*'0'+52*'1', 2)

        self.exp_min = self.exp_max_raw + 1023 - 1
        self.exp_max = self.exp_max_raw + 1023 - 1

    def truncate_float(self, f : float) -> float:
        uint64, = struct.unpack('Q', struct.pack('d', f))
        exp = max(self.exp_min, min((uint64 >> 52) & 0x7FF, self.exp_max))
        uint64 = uint64 & self.mmask & self.emask | (exp << 52)
        return struct.unpack('d', struct.pack('Q', uint64))[0]

    def truncate_tensor(self, t : torch.Tensor) -> None:
        self.__truncate_tensor(t.detach())
        t.requires_grad_()
        
    def __truncate_tensor(self, t : torch.Tensor) -> None:
        man, exp = torch.frexp(t)
        man.ldexp_(self.mant_bits_tensor)
        man.trunc_()
        man.ldexp_(self.mant_bits_tensor_neg)
        exp.clamp_(min=self.exp_min_raw, max=self.exp_max_raw)
        t.copy_(torch.ldexp(man, exp))
