import os
import sys
from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from hit import diffx, diffy, diffz

def test_sin():
    n = 16
    x = 2 * pi * arange(n) / n
    x, y, z = meshgrid(x, x, x, indexing='ij')
    u = sin(x)
    v = sin(y)
    w = sin(z)
    ux = fft.irfftn(diffx(fft.rfftn(u)))
    vx = fft.irfftn(diffx(fft.rfftn(v)))
    wx = fft.irfftn(diffx(fft.rfftn(w)))
    assert abs(ux - cos(x)).max() < 1E-12
    assert abs(vx).max() < 1E-12
    assert abs(wx).max() < 1E-12
    uy = fft.irfftn(diffy(fft.rfftn(u)))
    vy = fft.irfftn(diffy(fft.rfftn(v)))
    wy = fft.irfftn(diffy(fft.rfftn(w)))
    assert abs(uy).max() < 1E-12
    assert abs(vy - cos(y)).max() < 1E-12
    assert abs(wy).max() < 1E-12
    uz = fft.irfftn(diffz(fft.rfftn(u)))
    vz = fft.irfftn(diffz(fft.rfftn(v)))
    wz = fft.irfftn(diffz(fft.rfftn(w)))
    assert abs(uz).max() < 1E-12
    assert abs(vz).max() < 1E-12
    assert abs(wz - cos(z)).max() < 1E-12

def test_sin2():
    n = 16
    x = 2 * pi * arange(n) / n
    x, y, z = meshgrid(x, x, x, indexing='ij')
    u = sin(x * 2)
    v = sin(y * 2)
    w = sin(z * 2)
    ux = fft.irfftn(diffx(fft.rfftn(u)))
    vx = fft.irfftn(diffx(fft.rfftn(v)))
    wx = fft.irfftn(diffx(fft.rfftn(w)))
    assert abs(ux - 2 * cos(x * 2)).max() < 1E-12
    assert abs(vx).max() < 1E-12
    assert abs(wx).max() < 1E-12
    uy = fft.irfftn(diffy(fft.rfftn(u)))
    vy = fft.irfftn(diffy(fft.rfftn(v)))
    wy = fft.irfftn(diffy(fft.rfftn(w)))
    assert abs(uy).max() < 1E-12
    assert abs(vy - 2 * cos(y * 2)).max() < 1E-12
    assert abs(wy).max() < 1E-12
    uz = fft.irfftn(diffz(fft.rfftn(u)))
    vz = fft.irfftn(diffz(fft.rfftn(v)))
    wz = fft.irfftn(diffz(fft.rfftn(w)))
    assert abs(uz).max() < 1E-12
    assert abs(vz).max() < 1E-12
    assert abs(wz - 2 * cos(z * 2)).max() < 1E-12
