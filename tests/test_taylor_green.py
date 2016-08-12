import os
import sys
from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from hit import step

def step_uvw(u, v, w, mu, dt, n):
    uvwhat = fft.rfftn([u, v, w], axes=(1,2,3))
    for i in range(n):
        uvwhat = step(uvwhat, mu, dt)
    return fft.irfftn(uvwhat, axes=(1,2,3))

def test_xy():
    n = 16
    x = 2 * pi * arange(n) / n
    x, y, z = meshgrid(x, x, x, indexing='ij')

    u0 = sin(x) * cos(y)
    v0 = -cos(x) * sin(y)
    w0 = ones([n,n,n])

    u, v, w = step_uvw(u0, v0, w0, 0, 0.1, 1)
    assert abs(u - u0).max() < 1E-12
    assert abs(v - v0).max() < 1E-12
    assert abs(w - w0).max() < 1E-12

    mu = 5 * log(2)
    u, v, w = step_uvw(u0, v0, w0, mu, 0.1, 1)
    assert abs(u - u0 / 2).max() < 1E-12
    assert abs(v - v0 / 2).max() < 1E-12
    assert abs(w - w0).max() < 1E-12

    u0 += 1
    v0 += 1

    u, v, w = step_uvw(u0, v0, w0, 0, pi/10, 10)
    assert abs(u - u0).max() < 5E-2
    assert abs(v - v0).max() < 5E-2
    u, v, w = step_uvw(u0, v0, w0, 0, pi/20, 20)
    assert abs(u - u0).max() < 1E-2
    assert abs(v - v0).max() < 1E-2
    u, v, w = step_uvw(u0, v0, w0, 0, pi/40, 40)
    assert abs(u - u0).max() < 1E-3
    assert abs(v - v0).max() < 1E-3

    mu = log(2) / pi / 2
    u, v, w = step_uvw(u0, v0, w0, mu, pi/10, 10)
    assert abs(u - (u0 + 1) / 2).max() < 5E-2
    assert abs(v - (v0 + 1) / 2).max() < 5E-2
    u, v, w = step_uvw(u0, v0, w0, mu, pi/20, 20)
    assert abs(u - (u0 + 1) / 2).max() < 1E-2
    assert abs(v - (v0 + 1) / 2).max() < 1E-2
    u, v, w = step_uvw(u0, v0, w0, mu, pi/40, 40)
    assert abs(u - (u0 + 1) / 2).max() < 1E-3
    assert abs(v - (v0 + 1) / 2).max() < 1E-3

def test_yz():
    n = 16
    x = 2 * pi * arange(n) / n
    x, y, z = meshgrid(x, x, x, indexing='ij')

    u0 = ones([n,n,n])
    v0 = -cos(z) * sin(y)
    w0 = sin(z) * cos(y)

    u, v, w = step_uvw(u0, v0, w0, 0, 0.1, 1)
    assert abs(u - u0).max() < 1E-12
    assert abs(v - v0).max() < 1E-12
    assert abs(w - w0).max() < 1E-12

    mu = 5 * log(2)
    u, v, w = step_uvw(u0, v0, w0, mu, 0.1, 1)
    assert abs(u - u0).max() < 1E-12
    assert abs(v - v0 / 2).max() < 1E-12
    assert abs(w - w0 / 2).max() < 1E-12

    w0 += 1
    v0 += 1

    u, v, w = step_uvw(u0, v0, w0, 0, pi/10, 10)
    assert abs(w - w0).max() < 5E-2
    assert abs(v - v0).max() < 5E-2
    u, v, w = step_uvw(u0, v0, w0, 0, pi/20, 20)
    assert abs(w - w0).max() < 1E-2
    assert abs(v - v0).max() < 1E-2
    u, v, w = step_uvw(u0, v0, w0, 0, pi/40, 40)
    assert abs(w - w0).max() < 1E-3
    assert abs(v - v0).max() < 1E-3

    mu = log(2) / pi / 2
    u, v, w = step_uvw(u0, v0, w0, mu, pi/10, 10)
    assert abs(w - (w0 + 1) / 2).max() < 5E-2
    assert abs(v - (v0 + 1) / 2).max() < 5E-2
    u, v, w = step_uvw(u0, v0, w0, mu, pi/20, 20)
    assert abs(w - (w0 + 1) / 2).max() < 1E-2
    assert abs(v - (v0 + 1) / 2).max() < 1E-2
    u, v, w = step_uvw(u0, v0, w0, mu, pi/40, 40)
    assert abs(w - (w0 + 1) / 2).max() < 1E-3
    assert abs(v - (v0 + 1) / 2).max() < 1E-3

def test_xz():
    n = 16
    x = 2 * pi * arange(n) / n
    x, y, z = meshgrid(x, x, x, indexing='ij')

    u0 = -cos(z) * sin(x)
    v0 = ones([n,n,n])
    w0 = sin(z) * cos(x)

    u, v, w = step_uvw(u0, v0, w0, 0, 0.1, 1)
    assert abs(u - u0).max() < 1E-12
    assert abs(v - v0).max() < 1E-12
    assert abs(w - w0).max() < 1E-12

    mu = 5 * log(2)
    u, v, w = step_uvw(u0, v0, w0, mu, 0.1, 1)
    assert abs(u - u0 / 2).max() < 1E-12
    assert abs(v - v0).max() < 1E-12
    assert abs(w - w0 / 2).max() < 1E-12

    w0 += 1
    u0 += 1

    u, v, w = step_uvw(u0, v0, w0, 0, pi/10, 10)
    assert abs(u - u0).max() < 5E-2
    assert abs(w - w0).max() < 5E-2
    u, v, w = step_uvw(u0, v0, w0, 0, pi/20, 20)
    assert abs(u - u0).max() < 1E-2
    assert abs(w - w0).max() < 1E-2
    u, v, w = step_uvw(u0, v0, w0, 0, pi/40, 40)
    assert abs(u - u0).max() < 1E-3
    assert abs(w - w0).max() < 1E-3

    mu = log(2) / pi / 2
    u, v, w = step_uvw(u0, v0, w0, mu, pi/10, 10)
    assert abs(w - (w0 + 1) / 2).max() < 5E-2
    assert abs(u - (u0 + 1) / 2).max() < 5E-2
    u, v, w = step_uvw(u0, v0, w0, mu, pi/20, 20)
    assert abs(w - (w0 + 1) / 2).max() < 1E-2
    assert abs(u - (u0 + 1) / 2).max() < 1E-2
    u, v, w = step_uvw(u0, v0, w0, mu, pi/40, 40)
    assert abs(w - (w0 + 1) / 2).max() < 1E-3
    assert abs(u - (u0 + 1) / 2).max() < 1E-3
