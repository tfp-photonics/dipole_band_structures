from main import *

def lattice( N : int, lc : float ):
    return jnp.concatenate(  [chain(N, lc, lc * i) for i in range(N)] )

def band_structure( vecs, N = 1, n_m = 3, n_dim = 1 ):
    """
    Assigns each right eigenvector a normalized crystal momentum k
    """
    def inner( vec ):
        return jnp.unravel_index( jnp.argmax(jnp.abs(jnp.fft.fft2(vec.reshape(dim,dim), s = (L,L))[:max_ind, :max_ind])), (L,L) )

    # array containing all "possible" normalized ks 
    L = 2**10
    max_ind = int(L/2)
    # ks = jnp.arange(max_ind) / L * 2
    dim = int( jnp.sqrt( vecs.shape[0] / (N * n_m) ) )

    
    # for each vector, identify the oscillating component, so the component of the particle with largest abs values
    comp = jnp.argmax( jnp.stack([ jnp.abs(vecs[i::n_m*N,:]).sum(axis = 0) for i in range(n_m*N)]), axis = 0)
    vecs = jnp.stack( [ vecs[comp[i]::n_m*N,i] for i in range(comp.size) ] )
    
    # return maximum k vector for each eigenvalue
    return jax.vmap( jax.jit(inner), (0,), 0)( vecs )

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

# pdb.set_trace()

# mat = int_mat( lattice(3,0.3), n_m = 3 )
# plt.imshow( jnp.abs(mat) )
# plt.show()

## single chain
lc = 0.3* LAMBDA_0
light_line = K_0 / (jnp.pi/lc) 
shift, linewidth, vecs = spectrum( lattice(10,lc) )
ks = band_structure( vecs, 1 )
arr = analytic_chain( lc, k*jnp.pi/lc )
