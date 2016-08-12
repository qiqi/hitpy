import os
import sys
from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from hit import pressure

def test_pressure():
    n = 16
    x = 2 * pi * arange(n) / n
    x, y, z = meshgrid(x, x, x, indexing='ij')
    u = fft.rfftn(random.rand(n,n,n))
    v = fft.rfftn(random.rand(n,n,n))
    w = fft.rfftn(random.rand(n,n,n))
    p = pressure(u,v,w)
    assert abs(pressure(u,v,w)).max() < 1E-6
    assert abs(pressure(u,v,w)).max() < 1E-10
    assert abs(pressure(u,v,w)).max() < 1E-12
