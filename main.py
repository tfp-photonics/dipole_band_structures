"""Contains main code to compute interaction matrices for Lorentzian scatterers
"""

from functools import reduce
import itertools as it

import jax
from jax import Array
import jax.numpy as jnp
from jax import lax
from jax.config import config
config.update("jax_enable_x64", True)


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar

from mpmath import clcos, clsin

import pdb
import numpy as np

# globals, treat read only, these variables have no influence on the simulations as long as spatial quantities are measured in units of LAMBDA_0
LAMBDA_0 =  1
K_0      = 2*jnp.pi / LAMBDA_0
SCALE    = -6*jnp.pi/K_0**3 * K_0**2

## geometry
def chain( N : int, lc : float, y : float ):
    """
    Constructs a Nx3 array containing the coordinates of a chain of length N with lattice constant lc in the xy-plane along the x-axis, moved by an offset y in y-direction
    """
    return jnp.stack( (jnp.arange(N)*lc, jnp.ones(N)*y, jnp.zeros(N)), axis = 1 )

def stacked_chains( args : list[tuple[int, float, float]] ):
    """
    Constructs a Nx3 array containing the coordinates of a stack of chains from the list given in args by passing them to chain and then concatenating the results.
    """
    return jnp.concatenate( [chain(*arg) for arg in args]  )

def twisted_chains(a : int, b : int, l : float, o : float, N : int):
    """
    Returns a stack representing two chains along the x-axis separated by an offset o in y-direction. 
    The stack is comprised of N unit cells and characterized by a Moiré parameter theta = a/b - 1.  
    The larger subchain periodicity is given by l.
    """
    chains = stacked_chains( [(a*N, l, 0), (b*N, a/b*l, o)] )
    return chains[jnp.argsort(chains[:,0]),:]

def lattice( N : int, lc : float ):
    return jnp.concatenate(  [chain(N, lc, lc * i) for i in range(N)] )

## em functions
@jax.jit
def dyad( vec1, vec2 ):
    """
    Computes the 6x6 dyadic Green's tensor at k = k_0 connecting vec1 and vec2
    """    
    def inner( r ):
        factor = jnp.exp(1j*K_0*r)/(4*jnp.pi*r)
        within =  jnp.eye(3)*(1 + (1j*K_0*r-1)/(K_0*r)**2 ) + (-1 + (3 - 3*1j*K_0*r)/(K_0*r)**2) * jnp.outer(r_vec,r_vec)/r**2
        between = -eps @ r_vec * (1j * K_0/r - 1/r**2)
        return factor*(jnp.kron(mat1, within) + jnp.kron(mat2, between)/K_0)
    
    mat1 = jnp.eye(2)
    # mat2 = jnp.ones((2,2)) - mat1
    mat2 = jnp.array( [[0,1], [1,0] ] )
    r_vec = vec1 - vec2
    r = jnp.linalg.norm(r_vec)
    return lax.cond( r == 0, lambda r : jnp.zeros((6,6), dtype = complex), lambda r : inner(r), r)

@jax.jit
def dyad_electric( vec1, vec2 ):
    """
    Computes the 3x3 dyadic Green's tensor at k = k_0 connecting vec1 and vec2
    """    
    r_vec = vec1 - vec2
    r = jnp.linalg.norm(r_vec)
    return lax.cond( r == 0, lambda r : jnp.zeros((3,3), dtype = complex), lambda r : jnp.exp(1j*K_0*r)/(4*jnp.pi*r)*( jnp.eye(3)*(1 + (1j*K_0*r-1)/(K_0*r)**2 ) + (-1 + (3 - 3*1j*K_0*r)/(K_0*r)**2) * jnp.outer(r_vec,r_vec)/r**2 ), r )


def reorder( mat, n_m ):
    """
    Reorders a NxNx3x3 matrix produced by vmapping dyad over an array returned by chain, stacked_chains to a 3Nx3N matrix, which is proportional to the offdiagonal part of the interaction matrix
    """
    def inner( i,j ):
        k, l = i % n_m, j % n_m
        r, s = (i/n_m).astype(int), (j/n_m).astype(int)
        return mat[r,s,l,k]
    return jax.vmap( jax.vmap( jax.jit(inner), (0, None), 0), (None,0), 0 )( jnp.arange(mat.shape[0]*n_m), jnp.arange(mat.shape[0]*n_m) )

def int_mat( pos, dyad_func = dyad, n_m = 6 ):
    """
    Maps a Nx3 position array to a 3Nx3N interaction matrix at k = k_0. 
    """
    return -SCALE * reorder( jax.vmap( jax.vmap( dyad_func, (0, None), 0 ), (None, 0), 0 )(pos, pos), n_m ) + 1j * jnp.eye( pos.shape[0] * n_m )

## stuff
def disorder( pos, sd ):
    key = jax.random.PRNGKey(171)
    return pos + sd * jax.random.uniform( key, pos.shape )

def epsilon():
    def el(ijk):
        i,j,k = ijk
        return (-i + j) * (-i + k) * (-j + k) / 2 
    return jnp.array(list( map(el, it.product(range(3), range(3), range(3)) ) )).reshape(3,3,3)
eps = epsilon()

def loc( vec ):
    key = jnp.argsort(jnp.std(jnp.abs(vec), axis = 0))[::-1]
    # key = jnp.argsort(jnp.median(jnp.abs(vec), axis = 0) - jnp.abs(vec))
    return vec[:, key], key

## pictures
def show( pos ):
    fig, ax = plt.subplots(1,1)
    ax.scatter( *pos[:,:2].T )
    # for i,p in enumerate(pos[:,:2].T):
    #     plt.annotate( str(i), *p)
    plt.show()

def show_int_mat( pos ):
    plt.matshow(int_mat(pos).real)
    plt.show()

def band_structure( vecs, N = 1, n_m = 6 ):
    """
    Assigns each right eigenvector a normalized crystal momentum k
    """
    def inner( vec ):
        return ks[ jnp.argmax(jnp.abs(jnp.fft.fft(vec, n = L)[:max_ind])) ]
    
    # array containing all "possible" normalized ks 
    L = 2**16
    max_ind = int(L/2)
    ks = jnp.arange(max_ind) / L * 2

    # for each vector, identify the oscillating component, so the component of the particle with largest abs values
    comp = jnp.argmax( jnp.stack([ jnp.abs(vecs[i::n_m*N,:]).sum(axis = 0) for i in range(n_m*N)]), axis = 0)
    vecs = jnp.stack( [ vecs[comp[i]::n_m*N,i] for i in range(comp.size) ] )
    
    # return maximum k vector for each eigenvalue
    return jax.vmap( jax.jit(inner), (0,), 0)( vecs )

def show_eigenstate( pos, selection : Array, component = 0, particle = 0, n = 1 ):
    _,_,vecs = spectrum(pos)
    fig, ax = plt.subplots(1,1)
    ax.plot( vecs[(component+particle)::3*n,selection].real, '.' )
    plt.show()

def show_eigenstate_lattice( pos, vecs ):
    N, _ = pos.shape()    
    plt.plot( vecs[(component+particle)::3*n,selection].real, '.' )
    plt.show()

    
def quickshow( vec ):
    plt.plot( jnp.arange(vec.size), vec )
    plt.show()
    
def show_comp( vecs, i, comp, N ):
    plt.matshow( jnp.abs(vecs[comp::6,i].reshape(N,N)) )
    plt.colorbar()
    plt.show()
    
def projector( vecs, thresh = 1e-10 ):
    vecs = np.array(vecs)
    vecs_r = vecs.real
    vecs_i = vecs.imag
    vecs_r[vecs_r < 1e-10] = 0
    vecs_i[vecs_i < 1e-10] = 0
    vecs = vecs_r + 1j*vecs_i
    res = np.abs(vecs.conj().T @ vecs)
    return [np.array(x) for x in set(map(tuple, [np.nonzero( v > thresh )[0] for v in res ]))]

def plot_projected( proj, shift, linewidth ):    
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    for p in proj:
        ax1.plot( jnp.arange(p.size), jnp.sort(shift[p]), '.' )
        ax2.plot( jnp.arange(p.size), jnp.sort(linewidth[p]), '.' )
    plt.show()

def ssh_chain( N : int, lc : float, d : float ):
    ch1 = jnp.stack( (jnp.arange(N)*lc, jnp.zeros(N), jnp.zeros(N)), axis = 1 )
    ch2 = jnp.stack( (jnp.arange(N)*lc+d, jnp.zeros(N), jnp.zeros(N)), axis = 1 )
    return jnp.concatenate((ch1, ch2))

if __name__ == '__main__':    
    N = 40
    vals, vecs = jnp.linalg.eig( int_mat(chain(N, 0.2, 0) ) )
    shift, linewidth =  -jnp.real( vals/2  ), jnp.imag( vals )
    vecs, key = loc( vecs )    
    ks = band_structure( vecs, 1 )
    fig, axs = plt.subplots(2,1)
    axs[0].plot( ks, shift[key], '.' )
    axs[1].plot( ks, linewidth[key], '.' )
    # plt.savefig('foo.pdf')
    plt.show()

    # vals, vecs = jnp.linalg.eig( int_mat( ssh_chain(20, 0.2, 0.11), dyad_electric, n_m = 3) )
    vals, vecs = jnp.linalg.eig( int_mat( chain(20, 0.2, 0)))
    vals, vecs = jnp.linalg.eig( int_mat( lattice(20, 0.2),  dyad_electric, n_m = 3) )
    shift, linewidth =  -jnp.real( vals/2  ), jnp.imag( vals )
    plot_projected( projector(vecs), shift, linewidth )
    
    # # weird
    # i = 1
    # wh = lambda x : np.argwhere( np.abs( shift - shift[x]) < 0.01 ) 
    # foo = np.zeros(N)
    # f1 = np.real
    # f2 = lambda x : np.real(x[::-1,:])
    # for j in wh(12):# range(shift.shape[0]): 
    #     tot = vecs[:,j].reshape(2*N,3)
    #     hel1 = tot[::2,1:] + 1j*tot[1::2,1:]
    #     hel2 = tot[::2,1:] - 1j*tot[1::2,1:]
    #     # el = jnp.linalg.norm(tot[::2,:], axis = 1)
    #     # mag = jnp.linalg.norm(tot[1::2,:], axis = 1)
    #     # foo += mag + el[::-1]
    #     # plt.plot(f1(hel1))
    #     # plt.plot(f2(hel2), '--')
    #     # plt.plot(f1(hel1))
    #     # plt.plot(f1(hel1)-f2(hel2), '--')
    #     plt.plot(np.abs(f1(hel1)+f2(hel2)))

    # plt.show()
    # plt.close()
    # # plt.plot(foo)
    # plt.show()
    
    # N = 20
    # vals, vecs = jnp.linalg.eig( int_mat(lattice(N, 0.2) ) )
    # vecs = loc(vecs)
    # for comp in range(3):
    #     show_comp(vecs, 11, comp, N)
