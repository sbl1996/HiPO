import numpy as np
from scipy.stats import loguniform

__all__ = ["sample"]


def cast(v, dtype):
    if dtype == 'int':
        return int(v)
    else:
        return float(v)


def sample_linear(min_val, max_val):
    return np.random.uniform(min_val, max_val)


def sample_log(min_val, max_val):
    return loguniform.rvs(min_val, max_val)


def _get_val(s, k, hparams):
    if isinstance(s[k], str):
        k = s[k]
        if k in hparams:
            return hparams[k]
        else:
            raise ValueError("error in getting %s" % k)
    else:
        return s[k]


def sample(search_space):
    hparams = {}
    for k, s in search_space.items():
        v_min = _get_val(s, "min", hparams)
        v_max = _get_val(s, "max", hparams)
        scale = s['scale']
        if scale == 'log':
            v = sample_log(v_min, v_max)
        elif scale == 'linear':
            v = sample_linear(v_min, v_max)
        else:
            raise ValueError("Unsupported scale: %s" % scale)
        hparams[k] = cast(v, s['dtype'])
    return hparams
