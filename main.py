"""Contains main code to compute interaction matrices for Lorentzian scatterers
"""

import jax
from jax import Array
import jax.numpy as jnp
from jax import lax
from jax.config import config
config.update("jax_enable_x64", True)

from mpmath import clcos, clsin


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar

import pdb

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
    
@jax.jit
def dyad( vec1, vec2 ):
    """
    Computes the 3x3 dyadic Green's tensor at k = k_0 connecting vec1 and vec2
    """    
    k = 2*jnp.pi
    r_vec = vec1 - vec2
    r = jnp.linalg.norm(r_vec)
    return lax.cond( r == 0, lambda r : jnp.zeros((3,3), dtype = complex), lambda r : jnp.exp(1j*k*r)/(4*jnp.pi*r)*( jnp.eye(3)*(1 + (1j*k*r-1)/(k*r)**2 ) + (-1 + (3 - 3*1j*k*r)/(k*r)**2) * jnp.outer(r_vec,r_vec)/r**2 ), r )

def reorder( mat, n_m = 3 ):
    """
    Reorders a NxNx3x3 matrix produced by vmapping dyad over an array returned by chain, stacked_chains to a 3Nx3N matrix, which is proportional to the offdiagonal part of the interaction matrix
    """
    def inner( i,j ):
        k, l = i % n_m, j % n_m
        r, s = (i/n_m).astype(int), (j/n_m).astype(int)
        return mat[r,s,l,k]
    return jax.vmap( jax.vmap( jax.jit(inner), (0, None), 0), (None,0), 0 )( jnp.arange(mat.shape[0]*3), jnp.arange(mat.shape[0]*3) )

def int_mat( pos, n_m = 3 ):
    """
    Maps a Nx3 position array to a 3Nx3N interaction matrix at k = k_0. The diagonal part, which is just 1j times the identity, is omitted.
    """
    return 3 * reorder( jax.vmap( jax.vmap( dyad, (0, None), 0 ), (None, 0), 0 )(pos, pos) ) + 1j * jnp.eye( pos.shape[0] * n_m )

def spectrum( pos ):
    """
    Returns shift, linewidth and eigenvectors. Shift and linewidth are computed according to the expressions in https://link.springer.com/article/10.1140/epjb/e2020-100473-3.    
    """
    vals, vecs = jnp.linalg.eig( int_mat(pos) )
    return -jnp.real( vals/2  ), jnp.imag( vals ), vecs

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
    return 1/(wm.size*jnp.pi)*jnp.sum( gm / ( (jnp.expand_dims(w,1) - wm)**2 + gm**2), axis = 1 )

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
    axs[0].plot( ks, shift, '.' )
    axs[1].plot( ks, linewidth, '.' )
    return axs

def show_eigenstate( pos, selection : Array, component = 0, particle = 0, n = 1 ):
    _,_,vecs = spectrum(pos)
    fig, ax = plt.subplots(1,1)
    ax.plot( vecs[(component+particle)::3*n,selection].real, '.' )
    plt.show()

def quickshow( vec ):
    plt.plot( jnp.arange(vec.size), vec )
    plt.show()

def analytic_chain( l : float, kzs ):
    """
    Analytic expression for dispersion relation and collective decay rates of 1D chain with lattice constant l, evaluated at k-vectors given by the array kzs
    Input:
    l : Lattice constant
    kzs : N-dim Array containing k vectors
    Output:
    res : Nx4 array where res[:,:2] contains delta and res[:,2:] contains gamma
    """
    def inner( kz : float ) -> tuple[float, float, float, float] :
        arg_plus, arg_minus = klc + kz, klc - kz
        F1 = 1j*(clsin( 1, arg_plus ) + clsin( 1, arg_minus )) + clcos( 1, arg_plus ) + clcos( 1, arg_minus )
        F2 = 1j*(clsin( 2, arg_plus ) + clsin( 2, arg_minus )) + clcos( 2, arg_plus ) + clcos( 2, arg_minus )
        F3 = 1j*(clsin( 3, arg_plus ) + clsin( 3, arg_minus )) + clcos( 3, arg_plus ) + clcos( 3, arg_minus )
        G_11 = complex( F1/klc + 1j*F2/klc**2 - F3/klc**3 )
        G_33 = 2*complex( -1j*F2/klc**2 + F3/klc**3 )
        return [-3/4*jnp.real(G_11), -3/4*jnp.real(G_33), 3/2*jnp.imag(G_11) + 1, 3/2*jnp.imag(G_33) + 1]
    klc = 2*jnp.pi*l
    return jnp.array( [ inner(float(x)) for x in kzs ]  )

def analytic_stack( l : float, o : float, kzs ):
    chain = 2*analytic_chain( l, kzs )
    bands = analytic_flat_bands( 1, 1, l, o ) * jnp.ones_like( kzs )
    return jnp.stack( (chain[:,0] - bands[0,0], chain[:,1] - bands[0,0], chain[:,0] - bands[1,0], bands[0,:], bands[1,:], bands[1,:],
                             chain[:,2], chain[:,3], chain[:,2], bands[2,:], bands[3,:], bands[2,:]) ).T                             

def analytic_flat_bands( a : int, b : int, l : float, o : float ):
    pos = twisted_chains( a, b, l, o, 1)
    kr  = 2*jnp.pi*jnp.min(jnp.linalg.norm(pos[ pos[:,1] != 0 ] - jnp.expand_dims(pos[ pos[:,1] == 0 ],1), axis = 2), axis = 1)
    E1 = 3/(4*kr) * ( jnp.cos(kr) - jnp.cos(kr)/kr**2 - jnp.sin(kr)/kr )
    E2 = 3/(2*kr) * ( jnp.cos(kr)/kr**2 + jnp.sin(kr)/kr )
    return jnp.stack( [E1, E2, jnp.zeros_like(E1), jnp.zeros_like(E2)] )

k = jnp.linspace(0, 1, 200)
X_LABEL = r'$k \,\,\,\left( \dfrac{\pi}{\Lambda} \right)$'
D_LABEL = r'$D(k)$'
G_LABEL = r'$G(k)$'
LW = 2.1
PS = None
LL_COLOR = 'grey'
LL_STYLE = '--'
LL_ALPHA = 0.8
LL_SIZE = 2.1
LL_LABEL = r'$k = k_0$'

## single chain
# lc = 0.2
# shift, linewidth, vecs = spectrum( chain(40,lc,0) )
# ks = band_structure( vecs, 1 )
# arr = analytic_chain( lc, k*jnp.pi )

# fig, (ax1, ax2) = plt.subplots(ncols=2)
# ax1.set_box_aspect(1)
# ax1.plot( k, arr[:,:2], lw = LW )
# ax1.plot( ks, shift, '.', lw = PS )
# ax1.set_xlabel( X_LABEL )
# ax1.set_ylabel( D_LABEL )
# ax1.axvline( 2*lc, c = LL_COLOR, ls = LL_STYLE, alpha = LL_ALPHA, lw = LL_SIZE )
# ax1.legend( ax1.lines, (r'$D_{\perp}(k)$', r'$D_{\parallel}$(k)', 'finite', LL_LABEL ) )

# ax2.set_box_aspect(1)
# ax2.plot( k, arr[:,2:], lw = LW )
# ax2.plot( ks, linewidth, '.', lw = PS )
# ax2.set_xlabel( X_LABEL )
# ax2.set_ylabel( G_LABEL )
# ax2.axvline( 2*lc, c = LL_COLOR, ls = LL_STYLE, alpha = LL_ALPHA, lw = LL_SIZE )
# ax2.legend( ax2.lines, (r'$G_{\perp}(k)$', r'$G_{\parallel}(k)$', 'finite', LL_LABEL) )
# plt.savefig('single_chain.pdf')
# plt.show()
# plt.close()

## stack with no Moiré
lc, o = 0.3, 0.1
arr = analytic_stack( lc, o, k * jnp.pi)
pos = twisted_chains(1,1,lc,o,30)
shift, linewidth, vecs = spectrum( pos )
w = jnp.linspace(jnp.min(shift)-1,jnp.max(shift)+1,400)
ks = band_structure( vecs, 2 )
d = dos(w, shift, linewidth  ) 

fig = plt.figure()
canvas = gridspec.GridSpec(2,
                          1,
                          wspace=0,
                          hspace=0.05,
                          height_ratios=[0.05,1])
inlet = gridspec.GridSpecFromSubplotSpec(1,
                                         2,
                                         subplot_spec=canvas[1],
                                         wspace=0,
                                         hspace=0)

ax1 = plt.subplot( inlet[0] )
sc = ax1.scatter( ks, shift, c = linewidth )
ax1.plot( k, arr[:,:6] )
ax1.set_xlabel( X_LABEL )
ax1.set_ylabel( 'D' )
ax1.axvline( 2*lc, c = LL_COLOR, ls = LL_STYLE, alpha = LL_ALPHA, lw = LL_SIZE )

ax2 = plt.subplot( inlet[1] )
ax2.sharey(ax1)
ax2.plot( d, w, '--' )
ax2.set_xlabel( r'DOS' )
ax2.legend( ax1.lines[:-1] + ax2.lines, (r'$D_{1,+}(k)$', r'$D_{3,+}(k)$', r'$D_{2,+}(k)$', r'$D_{1,-}(k)$', r'$D_{3,-}(k)$', r'$D_{2,-}(k)$', r'$DOS(D)$' ) )
plt.setp(ax2.get_yticklabels(), visible=False)

cbax = plt.subplot(canvas[0])
cb = Colorbar(ax = cbax, mappable = sc, orientation = 'horizontal', ticklocation = 'top', label = r'$G$' )
# plt.show()
# plt.close()
plt.savefig('dos.pdf')

## twisted chains
# a, b, o, lc = 3, 4, 0.05, 0.3
# arr = analytic_flat_bands( a, b, lc, o)
# shift, linewidth, vecs = spectrum( twisted_chains(a,b,lc,o,30) )
# ks = band_structure( vecs, a+b )

# fig, (ax1, ax2) = plt.subplots(ncols=2)
# ax1.plot( ks, shift, '.' )
# for el in arr[:int(arr.shape[0]/2),:].flatten():
#     ax1.axhline( el )
# plt.show()
# plt.close()
