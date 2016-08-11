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

def diffx(uhat):
    n = uhat.shape[0]
    jk = 1j * hstack([arange(n//2), 0, arange(-n//2+1, 0)])
    return uhat * jk[:,newaxis,newaxis]

def diffy(uhat):
    n = uhat.shape[0]
    jk = 1j * hstack([arange(n//2), 0, arange(-n//2+1, 0)])
    return uhat * jk[newaxis,:,newaxis]

def diffz(uhat):
    n = uhat.shape[0]
    jk = 1j * hstack([arange(n//2), 0])
    return uhat * jk[newaxis,newaxis,:]

def convection(uhat, vhat, what):
    pass

if __name__ == '__main__':
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
