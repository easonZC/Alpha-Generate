import numpy as np
np.seterr(divide='ignore', invalid='ignore', over='ignore')
import numpy as np
from gplearn.functions import make_function

# Protected operators
def _pdiv(x1, x2):
    return np.where(np.abs(x2) > 1e-12, x1 / x2, 0.0)

def _plog(x):
    return np.log(np.abs(x) + 1e-12)

def _psqrt(x):
    return np.sqrt(np.abs(x))

def _tanh(x):
    return np.tanh(x)

def _clip(x):
    return np.clip(x, -10.0, 10.0)

def _sign(x):
    return np.sign(x)

def _inv(x):
    return np.where(np.abs(x)>1e-12, 1.0/x, 0.0)

def _min2(x1, x2):
    return np.minimum(x1, x2)

def _max2(x1, x2):
    return np.maximum(x1, x2)

pdiv = make_function(function=_pdiv, name="pdiv", arity=2)
plog = make_function(function=_plog, name="plog", arity=1)
psqrt = make_function(function=_psqrt, name="psqrt", arity=1)
tanhf = make_function(function=_tanh, name="tanh", arity=1)
clipf = make_function(function=_clip, name="clip", arity=1)
signf = make_function(function=_sign, name="sign", arity=1)
invf = make_function(function=_inv, name="inv", arity=1)
minf = make_function(function=_min2, name="min", arity=2)
maxf = make_function(function=_max2, name="max", arity=2)

def default_function_set():
    # Basic arithmetic are built-in: add, sub, mul, div (we'll use pdiv instead of div)
    return ['add', 'sub', 'mul', pdiv, plog, psqrt, tanhf, signf] #,clipf,  invf, minf, maxf]
