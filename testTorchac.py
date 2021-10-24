'''
LastEditors: fcy
'''
import torchac
import torch
import numpy as np
# Encode to bytestream.

seed=6
torch.manual_seed(seed)
np.random.seed(seed)

dim = 500
symsNum = 40000
pdf = np.random.rand(symsNum,dim)
pdf = pdf / (np.sum(pdf,1,keepdims=True))
sym = torch.ShortTensor(np.random.randint(0,dim,symsNum,dtype=np.int16))

def pdf_convert_to_cdf_and_normalize(pdf):
    assert pdf.ndim==2
    pdf = pdf / (np.sum(pdf,1,keepdims=True))/(1+10**(-10))
    cdfF = np.cumsum( pdf, axis=1)
    cdfF = np.hstack((np.zeros((pdf.shape[0],1)),cdfF))
    return cdfF


output_cdf = torch.Tensor(pdf_convert_to_cdf_and_normalize(pdf)) # Get CDF from your model, shape B, C, H, W, Lp

byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)


# pdf = np.diff(cdfF)
# print( -np.log2(pdf[range(0,oct_len),sym]).sum())

# Number of bits taken by the stream
real_bits = len(byte_stream) * 8
print(real_bits)
# Write to a file.
with open('outfile.b', 'wb') as fout:
    fout.write(byte_stream)

# Read from a file.
with open('outfile.b', 'rb') as fin:
    byte_stream = fin.read()

# Decode from bytestream.
sym_out = torchac.decode_float_cdf(output_cdf, byte_stream)

# Output will be equal to the input.
assert sym_out.equal(sym)
