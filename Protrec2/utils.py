import numpy as np
from scipy.special import logit, expit

def safe_logit(p, eps=1e-6):
    return logit(np.clip(p, eps, 1 - eps))

def safe_expit(x):
    return expit(x)
