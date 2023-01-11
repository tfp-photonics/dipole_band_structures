import matplotlib.pyplot as plt
import jax
from jax import Array
import jax.numpy as jnp
from jax import lax
import numpy as np

import pdb
from jax.config import config
config.update("jax_enable_x64", True)

def chain( N, lc, y ):
    """
    Constructs a Nx3 array containing the coordinates of a chain of length N with lattice constant lc in the xy-plane along the x-axis, moved by an offset y in y-direction
    """
    return jnp.stack( (jnp.arange(N)*lc, jnp.ones(N)*y, jnp.zeros(N)), axis = 1 )

def stacked_chains( args ):
    """
    Constructs a Nx3 array containing the coordinates of a stack of chains from the list given in args by passing them to chain and then concatenating the results.
    """
    return jnp.concatenate( [chain(*arg) for arg in args]  )

def twisted_chains(a, b, l, o, N):
    """
    Returns a stack representing two chains along the x-axis separated by an offset o in y-direction. 
    The stack is comprised of N unit cells and characterized by a Moiré parameter theta = a/b - 1.  
    The larger subchain periodicity is given by l.
    """
    chains = stacked_chains( [(a*N, l, 0), (b*N, a/b*l, o)] )
    return chains[jnp.argsort(chains[:,0]),:]
    
@jax.jit
def dyad( vec1, vec2 ):
    """
    Computes the 3x3 dyadic Green's tensor at k = k_0 connecting vec1 and vec2
    """    
    k = 2*jnp.pi
    r_vec = vec1 - vec2
    r = jnp.linalg.norm(r_vec)
    return lax.cond( r == 0, lambda r : jnp.zeros((3,3), dtype = complex), lambda r : jnp.exp(1j*k*r)/r*( jnp.eye(3)*(1 + (1j*k*r-1)/(k*r)**2 ) + (-1 + (3 - 3*1j*k*r)/(k*r)**2) * jnp.outer(r_vec,r_vec)/r**2 ), r )

def reorder( mat, n_m = 3 ):
    """
    Reorders a NxNx3x3 matrix produced by vmapping dyad over an array returned by chain, stacked_chains to a 3Nx3N matrix, which is proportional to the offdiagonal part of the interaction matrix
    """
    def inner( i,j ):
        k, l = i % n_m, j % n_m
        r, s = (i/n_m).astype(int), (j/n_m).astype(int)
        return mat[r,s,l,k]
    return jax.vmap( jax.vmap( jax.jit(inner), (0, None), 0), (None,0), 0 )( jnp.arange(mat.shape[0]*3), jnp.arange(mat.shape[0]*3) )

def int_mat( pos ):
    """
    Maps a Nx3 position array to a 3Nx3N interaction matrix at k = k_0 (scaled by prefactors and without the diagonal part)
    """
    return -1/(4*jnp.pi) * reorder( jax.vmap( jax.vmap( dyad, (0, None), 0 ), (None, 0), 0 )(pos, pos) )

def spectrum( pos ):
    """
    Returns shift, linewidth and eigenvectors. Shift and linewidth are computed according to the expressions in https://link.springer.com/article/10.1140/epjb/e2020-100473-3
    """
    vals, vecs = jnp.linalg.eig( int_mat(pos) )
    return -jnp.real( 2*vals  ), -jnp.imag( vals ), vecs

def band_structure( vecs, N = 1, n_m = 3 ):
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

def dos( w, wm, gm ):
    """
    Computes the DOS according to the expression in https://link.springer.com/article/10.1140/epjb/e2020-100473-3
    """
    return jnp.sum( gm / ( (jnp.expand_dims(w,1) - wm)**2 + gm**2), axis = 1 )

def show( pos ):
    fig, ax = plt.subplots(1,1)
    ax.scatter( *pos[:,:2].T )
    plt.show()

def show_int_mat( pos ):
    plt.matshow(int_mat(pos).real)
    plt.show()

def show_band_structure( pos, n = 1 ):
    shift, linewidth, vecs = spectrum(pos)
    ks = band_structure( vecs, n )
    fig, axs = plt.subplots(2,1)
    axs[0].plot( ks, -shift, '.' )
    axs[1].plot( ks, linewidth, '.' )
    plt.show()

def show_eigenstate( pos, selection : Array, component = 0, particle = 0, n = 1 ):
    _,_,vecs = spectrum(pos)
    fig, ax = plt.subplots(1,1)
    ax.plot( vecs[(component+particle)::3*n,selection].real, '.' )
    plt.show()

def quickshow( vec ):
    plt.plot( jnp.arange(vec.size), vec )
    plt.show()
    
# show_band_structure(chain(20, 0.2, 0))
# show_band_structure( twisted_chains(1,1,0.3,0.1,30), 2 )
# show( twisted_chains(1,2,0.2,0.05,10) )
