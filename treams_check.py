import treams
treams.config.POLTYPE="helicity"
import matplotlib.pyplot as plt
import numpy as np
import pdb

## geometry
def chain( N : int, lc : float = 1, y : float = 0.0 ):
    return np.stack( (np.arange(N)*lc, np.ones(N)*y, np.zeros(N)), axis = 1 )

def band_structure( vecs, N = 1, n_m = 6 ):
    def inner( vec ):
        return ks[ np.argmax(np.abs(np.fft.fft(vec, n = L)[:max_ind])) ]
    
    # array containing all "possible" normalized ks 
    L = 2**16
    max_ind = int(L/2)
    ks = np.arange(max_ind) / L * 2

    # for each vector, identify the oscillating component, so the component of the particle with largest abs values
    comp = np.argmax( np.stack([ np.abs(vecs[i::n_m*N,:]).sum(axis = 0) for i in range(n_m*N)]), axis = 0)
    vecs = np.stack( [ vecs[comp[i]::n_m*N,i] for i in range(comp.size) ] )

    # return maximum k vector for each eigenvalue
    return np.array([inner(v) for v in vecs])



N = 80
F = np.kron( np.eye(2), (1/np.sqrt(2))*np.array([[1,0,-1],[-1j,0,-1j],[0,np.sqrt(2),0]]) )
tmat = -np.eye(6) 
tmat = F.conj().T @ tmat @ F
k0 = 2 * np.pi / 5
positions = chain(N)
dipoles = [treams.TMatrix(tmat, k0 = k0) for p in positions]
int_mat = np.array(treams.TMatrix.cluster(dipoles, positions).interacted())
vals, vecs = np.linalg.eig(int_mat)
ks = band_structure( vecs, 1 )
shift, linewidth =  -0.5*np.imag( 1/vals  ), -np.real( 1/vals )
fig, axs = plt.subplots(2,1)
axs[0].plot( ks, shift, '.' )
axs[1].plot( ks, linewidth, '.' )
plt.show()


# weird
i = 1
wh = lambda x : np.argwhere( np.abs( shift - shift[x]) < 0.01 ) 
foo1 = np.zeros(N, dtype = complex)
foo2 = np.zeros(N, dtype = complex)
for j in wh(50)[::2]: #range(shift.shape[0]):
    tot1 = vecs[::2,j].reshape(N,3)
    tot2 = vecs[1::2,j].reshape(N,3)

    plt.plot( np.abs(tot2) )
    plt.plot( np.abs(tot1), '--' )

    # foo1 += np.linalg.norm(tot1, axis = 1)
    # foo2 += np.linalg.norm(tot2, axis = 1)
#plt.plot(foo1 + foo2)
# plt.plot(foo1 + foo2[::-1])
plt.show()
    # foo1 += tot1
    # foo2 += tot2
    # el = f1(tot1[:,i])
    # mag = f2(tot[:,i+1])
    # el = np.linalg.norm(tot1, axis = 1)
    # mag = np.linalg.norm(tot2, axis = 1)
    # plt.plot(el)
    # plt.plot(mag, '--')

    # el = f1(tot[::2,i] - 1j*tot[1::2,i+1])
    # mag = f2(tot[1::2,i] - 1j*tot[::2,i+1])
    # plt.plot(el)
    # plt.plot(mag, '--')

    # foo += mag + el
    # pdb.set_trace()

    # el = f1(tot1[:,i])
    # mag = f2(tot2[:,i+1])
    # foo += mag + el[::-1]
    # plt.plot(el)
    # plt.plot(mag, '--')

# plt.plot( np.linalg.norm(foo, axis = 1) )
# plt.plot( np.linalg.norm(foo1, axis = 1) )
# plt.plot( np.linalg.norm(foo2, axis = 1) )
plt.plot( foo1[:,0] )
plt.plot( foo2[:,0] )
plt.show()
