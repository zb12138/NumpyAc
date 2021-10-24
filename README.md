# NumpyAc: Fast Autoregressive Arithmetic Coding

## About

This is a modified version of the [torchac](https://github.com/fab-jul/torchac). NumpyAc takes numpy array as input and can decode in an autoregressive mode.

The backend is written in C++, the API is for PyTorch tensors. It will compile in the first run with ninja.

The implementation is based on [this blog post](https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html),
meaning that we implement _arithmetic coding_.
While it could be further optimized, it is already much faster than doing the equivalent thing in pure-Python (because of all the
 bit-shifts etc.).

### Set up conda environment

This library has been tested with

- PyTorch 1.5, 1.6, 1.7
- Python 3.8
And that's all you need. Other versions of Python may also work,
but on-the-fly ninja compilation only works for PyTorch 1.5+.

### Example

```python
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
```


## Important Implementation Details

### How we represent probability distributions

The probabilities are specified as [PDFs](https://en.wikipedia.org/wiki/Probability_density_function).
For each possible symbol, we need one PDF. This means that if there are `symsNum` possible symbols, and the values of them are distributed in `{0, ..., dim-1}`. The PDF ( shape (`symsNum,dim`) ) must specified the value for `symsNum` symbols.

**Example**:

```
For a symsNum = 1 particular symbol, let's say we have dim = 3 possible values. 
We can draw 4 CDF from 3 PDF to specify the symbols distribution:

symbol:        0     1     2
pdf:          P(0)  P(1)  P(2)
cdf:       C_0   C_1   C_2   C_3

This corresponds to the 3 probabilities

P(0) = C_1 - C_0
P(1) = C_2 - C_1
P(2) = C_3 - C_2

where PDF =[[ P(0), P(1) ,P(2) ]]
NOTE: The arithmetic coder assumes that P(0) + P(1) + P(2) = 1, C_0 = 0, C_3 = 1
```
The theoretical bits number can be estimated by Shannon’s source coding theorem:
![](https://latex.codecogs.com/svg.image?\\sum_{s}-log_2P(s))
## Citation
Reference from [torchac](https://github.com/fab-jul/torchac), thanks!
