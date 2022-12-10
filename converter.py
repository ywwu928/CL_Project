 import struct
import torch
class MyFloat():
    '''
    Usage: (example, bit format for fp16)
    ------
    mf MyFloat(5, 10)
    f = mf.truncate_float(f)
    '''
    EXPECTED_BITWIDTH = 16

    def init(self, exp_bits, mant_bits) -> None:
        if ((exp_bits + mant_bits + 1) != self.EXPECTED_BITWIDTH): raise Exception('invalid width')
        self.exp_bits = exp_bits
        self.mant_bits = mant_bits
        self.conversion_buffer = bytearray(8)
        self.mmask = int((12+mant_bits)'1'+(52-mant_bits)'0', 2)
        self.emask = int('1'+11'0'+52'1', 2)
        self.exp_min = (-(2(exp_bits-1))+1) + 1023
        self.exp_max = (2(exp_bits-1)) + 1023

    def truncate_float(self, f : float) -> float:
        f = float(f)
        struct.pack_into('d', self.conversion_buffer, 0, f)
        uint64, = struct.unpack_from('Q', self.conversion_buffer)
        exp = max(self.exp_min, min((uint64 >> 52) & 0x7FF, self.exp_max))
        uint64 = uint64 & self.mmask & self.emask | (exp << 52)
        struct.pack_into('Q', self.conversion_buffer, 0, uint64)
        return struct.unpack_from('d', self.conversion_buffer)[0]

    def truncate_floats(self, fs : torch.Tensor) -> torch.Tensor:
        ffs = fs.float()
        ffs = ffs.view(-1)
        for i in range(len(ffs)):
            ffs[i] = self.truncate_float(ffs[i])
        return fs.view(fs.shape)
