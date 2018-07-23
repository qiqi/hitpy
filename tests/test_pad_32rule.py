import os
import sys
from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from navierstokes import pad_32rule, unpad_32rule

def test_pad():
    n = 16
    x = 2 * pi * arange(n) / n
    x, y, z = meshgrid(x, x, x, indexing='ij')
    u = sin(x)
    v = sin(y)
    w = sin(z)

    u = fft.irfftn(pad_32rule(fft.rfftn(u)))
    v = fft.irfftn(pad_32rule(fft.rfftn(v)))
    w = fft.irfftn(pad_32rule(fft.rfftn(w)))
    assert abs(u.max() - 1) < 1E-12 and abs(u.min() + 1) < 1E-12
    assert abs(v.max() - 1) < 1E-12 and abs(v.min() + 1) < 1E-12
    assert abs(w.max() - 1) < 1E-12 and abs(w.min() + 1) < 1E-12
    assert (u.max(1).max(1) - u.min(1).min(1)).max() < 1E-16
    assert (v.max(0).max(1) - v.min(0).min(1)).max() < 1E-16
    assert (w.max(0).max(0) - w.min(0).min(0)).max() < 1E-16

    u, v, w = meshgrid(random.rand(n), random.rand(n), random.rand(n),
                       indexing='ij')
    u = fft.irfftn(pad_32rule(fft.rfftn(u)))
    v = fft.irfftn(pad_32rule(fft.rfftn(v)))
    w = fft.irfftn(pad_32rule(fft.rfftn(w)))
    assert (u.max(1).max(1) - u.min(1).min(1)).max() < 1E-16
    assert (v.max(0).max(1) - v.min(0).min(1)).max() < 1E-16
    assert (w.max(0).max(0) - w.min(0).min(0)).max() < 1E-16

def test_unpad():
    n = 12
    x = 2 * pi * arange(n) / n
    x, y, z = meshgrid(x, x, x, indexing='ij')
    u = sin(x)
    v = sin(y)
    w = sin(z)

    u = fft.irfftn(unpad_32rule(fft.rfftn(u)))
    v = fft.irfftn(unpad_32rule(fft.rfftn(v)))
    w = fft.irfftn(unpad_32rule(fft.rfftn(w)))
    assert abs(u.max() - 1) < 1E-12 and abs(u.min() + 1) < 1E-12
    assert abs(v.max() - 1) < 1E-12 and abs(v.min() + 1) < 1E-12
    assert abs(w.max() - 1) < 1E-12 and abs(w.min() + 1) < 1E-12
    assert (u.max(1).max(1) - u.min(1).min(1)).max() < 1E-16
    assert (v.max(0).max(1) - v.min(0).min(1)).max() < 1E-16
    assert (w.max(0).max(0) - w.min(0).min(0)).max() < 1E-16

    u, v, w = meshgrid(random.rand(n), random.rand(n), random.rand(n),
                       indexing='ij')
    u = fft.irfftn(pad_32rule(fft.rfftn(u)))
    v = fft.irfftn(pad_32rule(fft.rfftn(v)))
    w = fft.irfftn(pad_32rule(fft.rfftn(w)))
    assert (u.max(1).max(1) - u.min(1).min(1)).max() < 1E-16
    assert (v.max(0).max(1) - v.min(0).min(1)).max() < 1E-16
    assert (w.max(0).max(0) - w.min(0).min(0)).max() < 1E-16
