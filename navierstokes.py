from __future__ import division
from numpy import *

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

def jkx(n):
    return 1j * hstack([arange(n//2), 0, arange(-n//2+1, 0)])[:,newaxis,newaxis]

def jky(n):
    return 1j * hstack([arange(n//2), 0, arange(-n//2+1, 0)])[newaxis,:,newaxis]

def jkz(n):
    return 1j * hstack([arange(n//2), 0])[newaxis,newaxis,:]

def diffx(uhat):
    return uhat * jkx(uhat.shape[0])

def diffy(uhat):
    return uhat * jky(uhat.shape[0])

def diffz(uhat):
    return uhat * jkz(uhat.shape[0])

def convection(uhat, vhat, what, body_force):
    u = fft.irfftn(pad_32rule(uhat))
    v = fft.irfftn(pad_32rule(vhat))
    w = fft.irfftn(pad_32rule(what))
    convection_x = diffx(unpad_32rule(fft.rfftn(u * u))) \
                 + diffy(unpad_32rule(fft.rfftn(u * v))) \
                 + diffz(unpad_32rule(fft.rfftn(u * w)))
    convection_y = diffx(unpad_32rule(fft.rfftn(v * u))) \
                 + diffy(unpad_32rule(fft.rfftn(v * v))) \
                 + diffz(unpad_32rule(fft.rfftn(v * w)))
    convection_z = diffx(unpad_32rule(fft.rfftn(w * u))) \
                 + diffy(unpad_32rule(fft.rfftn(w * v))) \
                 + diffz(unpad_32rule(fft.rfftn(w * w)))
    if body_force:
        fx, fy, fz = body_force(u, v, w)
        fx = unpad_32rule(fft.rfftn(fx))
        fy = unpad_32rule(fft.rfftn(fy))
        fz = unpad_32rule(fft.rfftn(fz))
    else:
        fx, fy, fz = 0, 0, 0
    return fx - convection_x, fy - convection_y, fz - convection_z

def pressure(uhat, vhat, what):
    n = uhat.shape[0]
    div_hat = uhat * jkx(n) + vhat * jky(n) + what * jkz(n)
    jk2 = abs(jkx(n)**2 + jky(n)**2 + jkz(n)**2)
    phat = -div_hat / maximum(jk2, 1)
    uhat -= phat * jkx(n)
    vhat -= phat * jky(n)
    what -= phat * jkz(n)
    return phat

def conv_press(uvwhat, body_force):
    uhat, vhat, what = uvwhat
    conv_x, conv_y, conv_z = convection(uhat, vhat, what, body_force)
    pressure(conv_x, conv_y, conv_z)
    return array([conv_x, conv_y, conv_z])

def viscosity(uvwhat, mu_dt):
    n = uvwhat.shape[1]
    jk2 = abs(jkx(n)**2 + jky(n)**2 + jkz(n)**2)
    decay = exp(-mu_dt * jk2)
    return uvwhat * decay

def conv_press_mu_dt(uvwhat_exp, mu_dt, body_force):
    uvwhat = viscosity(uvwhat_exp, mu_dt)
    return viscosity(conv_press(uvwhat, body_force), -mu_dt)

def step(uvwhat, mu, dt, body_force=None):
    uvwhat_exp = array(uvwhat)
    f0 = conv_press_mu_dt(uvwhat_exp, 0, body_force) * dt
    f1 = conv_press_mu_dt(uvwhat_exp + f0 / 2, dt / 2 * mu, body_force) * dt
    f2 = conv_press_mu_dt(uvwhat_exp + f1 / 2, dt / 2 * mu, body_force) * dt
    f3 = conv_press_mu_dt(uvwhat_exp + f2, dt * mu, body_force) * dt
    return viscosity(uvwhat_exp + (f0 + f3) / 6 + (f1 + f2) / 3, dt * mu)

