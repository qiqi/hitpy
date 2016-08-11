from __future__ import division
from numpy import *

# n = 16
# x = 2 * pi * arange(n) / n
# x, y, z = meshgrid(x, x, x, indexing='ij')
# u = sin(x)

def pad_32rule(uhat):
    n = uhat.shape[0]
    assert uhat.shape == (n, n, n//2+1)
    assert n % 4 == 0
    uhat_padded = zeros((n*3//2, n*3//2, n*3//4+1), complex)
    uhat_padded[:n//2,:n//2,:n//2] = uhat[:n//2,:n//2,:n//2]
    uhat_padded[-n//2+1:,:n//2,:n//2] = uhat[-n//2+1:,:n//2,:n//2]
    uhat_padded[:n//2,-n//2+1:,:n//2] = uhat[:n//2,-n//2+1:,:n//2]
    uhat_padded[-n//2+1:,-n//2+1:,:n//2] = uhat[-n//2+1:,-n//2+1:,:n//2]
    return uhat_padded * (3/2)**3

def unpad_32rule(uhat_padded):
    n = uhat_padded.shape[0]
    assert uhat_padded.shape == (n, n, n//2+1)
    assert n % 6 == 0
    uhat = zeros((n*2//3, n*2//3, n//3+1), complex)
    uhat[:n//3,:n//3,:n//3] = uhat_padded[:n//3,:n//3,:n//3]
    uhat[-n//3+1:,:n//3,:n//3] = uhat_padded[-n//3+1:,:n//3,:n//3]
    uhat[:n//3,-n//3+1:,:n//3] = uhat_padded[:n//3,-n//3+1:,:n//3]
    uhat[-n//3+1:,-n//3+1:,:n//3] = uhat_padded[-n//3+1:,-n//3+1:,:n//3]
    return uhat * (2/3)**3

def convection(uhat, vhat, what):
    pass
