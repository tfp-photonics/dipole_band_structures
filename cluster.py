import matplotlib.pyplot as plt
import numpy as np

import treams
treams.config.POLTYPE="helicity"

def chain( N : int, lc : float = 1, y : float = 0.0 ):
    return np.stack( (np.arange(N)*lc, np.ones(N)*y, np.zeros(N)), axis = 1 )

def alpha( w ):
    pol = 6*np.pi*c**3/w0**3 * (-g_0/2) * 1./(w - w0 + 1j*g_0/2)
    mc = w**3/(c**3*6*np.pi*1j) * pol
    return mc

N = 40
F = np.kron( np.eye(2), (1/np.sqrt(2))*np.array([[1,0,-1],[-1j,0,-1j],[0,np.sqrt(2),0]]) )

c = 299792458
w0 = 1
g_0 = w0 * 1e-2
l0 = 2*np.pi*c/w0
k0 = 2*np.pi/l0
lc = 0.5*l0
w = w0 + np.linspace(-1,1,100)*g_0
w = w[-1]

tmat = -np.eye(6) * alpha(w)
positions = chain(N) * lc
positions[:,0] -= 0*(lc-1) * int(N/2)
positions[:,1] -= 1*lc
dipoles = [treams.TMatrix(tmat, k0 = k0) for p in positions]
chain = treams.TMatrix.cluster(dipoles, positions).interacted()
dip = treams.spherical_wave( 1, -1, 1, 1, k0 = w/c )

# N = int(N/4)
M = int(N/2)
grid = np.mgrid[-1:2*M:201j, 0:1, -2:2:40j].squeeze().transpose((1, 2, 0)) * lc
field = np.zeros_like(grid, complex)

sca = chain @ dip.expand(chain.basis) @ dip
valid = chain.valid_points(grid, [0.1*lc for p in positions])
field[valid, :] = (sca.efield(grid[valid, :]) * sca[:, None]).sum(axis=-2)

# valid = chain.valid_points(grid, [0.1*lc for p in positions])
# field[valid, :] += (dip.efield(grid[valid, :]) * dip[:, None]).sum(axis=-2)

fig, ax = plt.subplots()
pcm = ax.pcolormesh(
    grid[:, 0, 0] / lc ,
    grid[0, :, 2] / lc ,
    np.log(np.sum(np.power(np.abs(field), 2), axis=-1).T),
    shading="nearest"
)
ax.plot( positions[:N,0] / lc, np.arange(N)*0, '.' )
# ax.plot(
#     np.linspace(-250, 250, 200),
#     np.sqrt(250 * 250 - np.linspace(-250, 250, 200) ** 2),
#     c="r",
#     linestyle=":",
# )
# ax.plot(
#     np.linspace(-250, 250, 200),
#     -np.sqrt(250 * 250 - np.linspace(-250, 250, 200) ** 2),
#     c="r",
#     linestyle=":",
# )
cb = plt.colorbar(pcm)
cb.set_label(r"$|E|^2$")
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_aspect("equal", "box")
# plt.show()
plt.savefig('foo.pdf')
