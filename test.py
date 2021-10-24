import numpyAc
import numpy as np

# Generate random symbols and pdf.
dim = 128
symsNum = 2000
pdf = np.random.rand(symsNum,dim)
pdf = pdf / (np.sum(pdf,1,keepdims=True))
sym = np.random.randint(0,dim,symsNum,dtype=np.int16)
output_pdf = pdf

# Encode to bytestream.
codec = numpyAc.arithmeticCoding()
byte_stream,real_bits = codec.encode(pdf, sym,'out.b')

# Number of bits taken by the stream.
print('real_bits',real_bits)

# Theoretical bits number
print('shannon entropy',-int(np.log2(pdf[range(0,symsNum),sym]).sum()))

# Decode from bytestream.
decodec = numpyAc.arithmeticDeCoding(None,symsNum,dim,'out.b')

# Autoregressive decoding and output will be equal to the input.
for i,s in enumerate(sym):
    assert decodec.decode(output_pdf[i:i+1,:]) == s