from __future__ import division
import matplotlib
matplotlib.use('Agg')
from pylab import *
from numpy import *

from navierstokes import step, jkx, jky, jkz

def statistics(uvwhat, mu):
    n = uvwhat.shape[1]
    jk2 = abs(jkx(n)**2 + jky(n)**2 + jkz(n)**2)
    uvw = fft.irfftn(uvwhat, axes=(1,2,3))
    duvw = fft.irfftn(mu * jk2 * uvwhat, axes=(1,2,3))
    eps = (uvw * duvw).sum(0).mean() # energy dissipation rate per unit mass
    U = sqrt((uvw**2).mean()) # mean fluctuating velocity
    lam = sqrt(15 * mu * U*U / eps) # taylor microscale
    Rlam = U * lam / mu # Reynolds number
    Ek = (abs(uvwhat)**2).sum(0) / 2
    Ek[:,:,1:-1] *= 2 # because of using real fft in Z-direction
    k = sqrt(maximum(1, abs(jk2)))
    L_integral = (Ek / k).sum() / Ek.sum() * pi * 3 / 4
    L_komogorov = mu**(3/4) / eps**(1/4)
    return eps, U, lam, Rlam, L_integral, L_komogorov

def energy_spectrum(uvwhat):
    n = uvwhat.shape[1]
    Ek = (abs(uvwhat)**2).sum(0) / n**6 / 2
    Ek[:,:,1:-1] *= 2 # because of using real fft in Z-direction
    jk2 = abs(jkx(n)**2 + jky(n)**2 + jkz(n)**2)
    k = sqrt(maximum(1, abs(jk2)))
    bins = 0.5 + arange(n//2)
    i = digitize(k, bins)
    E = zeros(n//2 + 1)
    add.at(E, ravel(i), ravel(Ek))
    return arange(n//2), E[:-1]

def body_force(u, v, w):
    c = 1 / (u**2 + v**2 + w**2).mean()
    return c * u, c * v, c * w

def uvw_unravel(uvwhat_real):
    def find_n(size):
        n = int((size * 2)**(1/3)) - 2
        while n * n * (n // 2 + 1) < size:
            n += 1
        if n * n * (n // 2 + 1) == size:
            return n
        else:
            return None
    uvwhat = (uvwhat_real[:uvwhat_real.size//2] +
              uvwhat_real[uvwhat_real.size//2:] * 1j)
    n = find_n(uvwhat.size // 3)
    return uvwhat.reshape([3, n, n, n // 2 + 1])

def uvw_ravel(uvwhat):
    uvwhat = ravel(uvwhat)
    return hstack([uvwhat.real, uvwhat.imag])

def run(uvwhat_real, mu, nsteps):
    uvwhat = uvw_unravel(uvwhat_real)
    uvwhat[:,0,0,0] = 0
    J = []
    for i in range(nsteps):
        uvwhat = step(uvwhat, mu, 0.01, body_force)
        k, E = energy_spectrum(uvwhat)
        J.append(hstack([statistics(uvwhat, mu), E]))
    return uvw_ravel(uvwhat), array(J)

if __name__ == '__main__':
    n = 16
    uvwhat_real = random.rand(n * n * (n//2+1) * 6)
    uvwhat = uvw_unravel(uvwhat_real)
    mu = 0.05
    uvwhat[:,0,0,0] = 0
    for i in range(5000):
        uvwhat = step(uvwhat, mu, 0.01, body_force)
        print(uvwhat[:,0,0,0])
        if i % 100 == 0:
            print(statistics(uvwhat, mu))
            eps, U, lam, Rlam, L_integral, L_komogorov = statistics(uvwhat, mu)
            k, E = energy_spectrum(uvwhat)
            cla()
            semilogx(k * L_komogorov, E * k**(5/3) / eps**(2/3), 'o-')
            savefig('spectrum{0:06d}'.format(i))
    uvwhat_real = uvw_ravel(uvwhat)
    save('state_0.05.npy', uvwhat_real)
