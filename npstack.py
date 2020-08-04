from . import config
import warnings

if config.STACK == 0:
  import numpy as np
  import numpy.random
  RNG = numpy.random.default_rng()
  from scipy import linalg,sparse

numsvd = 0
def safesvd(matrix):
  """Perform SVD with fallbacks"""
  for i in range(config.svd_retry):
    try:
      U, s, Vd = linalg.svd(matrix, full_matrices=False)
    except linalg.LinAlgError:
      if i+1<config.svd_retry and config.linalg_verbose:
        print('SVD did not converge, retrying (#%d)'%(i+1))
    else:
      return U, s, Vd
  global numsvd
  numsvd += 1
  if config.linalg_verbose:
    print('SVD did not converge, manually computing SVD (#%d)' % numsvd)
  try:
    d0 = matrix.shape[0]
    d1 = matrix.shape[1]
    d = min(d0,d1)
    H = np.zeros((d0+d1,d0+d1),dtype=complex)
    H[d1:,:d1] = matrix
    H[:d1,d1:] = matrix.conj().T
    w, v = linalg.eigh(H)
    s = np.abs(w[:d])
    U = np.sqrt(2)*v[d1:,:d]
    V = -np.sqrt(2)*v[:d1,:d]
    # Make columns of U and of V orthonormal with Gram-Schmidt
    # Start with largest singular values, so that there is most alteration
    #  in places which least affect result
    for k in range(d):
      U[:,k] -= U[:,:k].dot(U[:,:k].T.conj().dot(U[:,k]))
      V[:,k] -= V[:,:k].dot(V[:,:k].T.conj().dot(V[:,k]))
      U[:,k] = U[:,k] / linalg.norm(U[:,k])
      V[:,k] = V[:,k] / linalg.norm(V[:,k])
    return U, s, V.conj().T
  except:
    if config.linalg_verbose:
      print('Stable method failed; using unstable method')
    flip = (matrix.shape[0] < matrix.shape[1])
    if flip:
      matrix = matrix.T
    N0,N1 = matrix.shape
    # A = U.s.V^H
    # A.A^H = U.s^2.U^H
    U, d = linalg.eigh(matrix.dot(matrix.T.conj()))
    # Return to descending order
    U = U[:,:-N1-1:-1]
    # u left singular value: u^H.A = s*v^H
    V0 = U.T.conj().dot(matrix)
    Vd = np.zeros(V0.shape,dtype=V0.dtype)
    s = []
    for i in range(N1):
      v = V0[i,:]
      if i:
        # Gram-Schmidt
        v -= Vd.T.conj().dot(V).dot(v)
      norm = linalg.norm(v)
      s.append(norm)
      Vd[i,:] = v/norm

    if flip:
      U,Vd = Vd.T,U.T
    if not (np.all(np.isfinite(U)) and np.all(np.isfinite(Vd)) and \
        np.all(np.isfinite(s))):
      raise linalg.LinAlgError('Built-in and manual SVD have failed')
  return U,s,Vd
