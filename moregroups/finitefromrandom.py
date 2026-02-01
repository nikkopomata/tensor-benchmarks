from finitegroups import *
import sanity
import numpy as np
from scipy import linalg
import itertools
import pickle

def inv_from_compose(table):
  order = len(table)
  indarr, invarr = np.nonzero(table == 0)
  assert len(indarr) == order
  indsort = np.argsort(indarr)
  assert all(indarr[indsort] == range(order))
  invmap = invarr[indsort]
  return invmap

def reps_from_compose(compose,tol=1e-7,tollow=1e-16):
  N = compose.shape[0]
  # Get inverses
  invmap = inv_from_compose(compose)
  # Get conjugacy classes
  # Label by first element
  conjtable = compose[compose,invmap[:,np.newaxis]]
  ccd = {}
  ccel = []
  ccsize = []
  for g in range(N):
    cc = set(conjtable[:,g])
    minel = min(cc)
    if minel != g:
      assert minel in ccd and ccd[minel] == cc
    else:
      ccd[g] = cc
      ccel.append(g)
      ccsize.append(len(cc))
  ccinv = invmap[ccel]
  # Initialize character table
  nreps = len(ccel)
  chars = np.ones((1,nreps),dtype=CPLXTYPE)
  ridx = 1
  # Get order of elements
  orders = np.zeros(nreps,dtype=int)
  for i,g in enumerate(ccel):
    n = 1
    gn = g
    while gn != 0:
      gn = compose[gn,g]
      n += 1
    orders[i] = n
  # First step: get (approximate) irreps from regular representation
  T = np.random.rand(N,N)
  T = (T+T.T) + 1j*(T-T.T)
  T /= linalg.norm(T)
  # Remove trace to take care of trivial representation
  T -= np.identity(N)*np.trace(T)/N
  Tsym = T[compose[:,:,np.newaxis],compose[:,np.newaxis,:]]
  Tsym = np.sum(Tsym,axis=0)/N
  # Irrep decomposition corresponds to degeneracies
  eig,U = linalg.eigh(Tsym)
  eidx = 0
  emax = max(np.abs(eig))
  dims = [1]
  while eidx < N:
    #if abs(eig[eidx]) < tol:
    #  # Represents trivial irrep
    #  eidx += 1
    #  assert abs(eig[eidx]) > tol
    lam = eig[eidx]
    degen = np.abs(lam - eig)/emax < tol
    dim = sum(degen)
    assert all(degen[eidx:eidx+dim]) # Should be in order
    eidx += dim
    # Projector from eigenvectors
    proj = U[:,degen]
    projd = proj.T.conj()
    # Obtain characters
    char = [dim]
    for i,g in enumerate(ccel):
      if i == 0:
        continue
      R = projd.dot(proj[compose[g,:],:])
      if dim == 1:
        lams = R.flatten()
      else:
        lams = linalg.eigvals(R)
      # Exact value is sum of order(g)-th roots of unity
      order = orders[i]
      roots = np.exp((pi*2j*np.arange(order,dtype=int))/order)
      tr = 0 # Precise trace
      for lam in lams:
        nroot = np.argmin(np.abs(lam-roots))
        tr = roots[nroot]+tr
      char.append(tr)
    # Check: already belongs to character table?
    if any(np.abs(chars.conj().dot(np.multiply(ccsize,char))) > tol):
      # Repeat of multidimensional, or trivial
      assert dim > 1 or abs(sum(np.multiply(ccsize,char))-N)<tol
      continue
    # Add to character table
    chars = np.append(chars,[char],axis=0)
    dims.append(dim)
  # Character properties
  assert chars.shape[0] == nreps
  assert sum(np.power(dims,2)) == N
  ccip = np.multiply(chars.conj(),np.array(ccsize)[np.newaxis,:]).dot(chars.T)/N
  assert np.allclose(ccip, np.identity(nreps),atol=tol,rtol=0)
  # Frobenius-Schur indicator
  sqclass = []
  for i,g in enumerate(ccel):
    # Find representative element for g^2
    for h in ccd[g]:
      if compose[h,h] in ccd:
        sqclass.append(ccel.index(compose[h,h]))
        break
  charsq = chars[:,sqclass]
  FSI = np.sum(np.multiply(charsq,[ccsize]),axis=1)/N
  assert linalg.norm(FSI-np.rint(FSI.real)) < tol
  FSI = np.rint(FSI.real).astype(int)
  multid = np.array(dims) != 1
  # Use characters to project regular representation onto irreps
  projectors = np.multiply(np.identity(N,dtype=CPLXTYPE)[np.newaxis,:,:],
                          np.array(dims)[multid,np.newaxis,np.newaxis])
  #projectors = np.multiply(broadcast_to(projectors,(sum(multid),N,N)).copy()
  for i,g in enumerate(ccel):
    if g == 0:
      continue
    charconj = chars[multid,i].conj()
    for h in ccd[g]:
      projectors[:,compose[h,:],range(N)] += charconj[:,np.newaxis]
  projectors /= N
  mdl = list(np.arange(nreps)[multid])
  reps = []
  for k,d in enumerate(dims):
    if d == 1:
      rep = np.zeros(N,dtype=CPLXTYPE)
      for i,g in enumerate(ccel):
        rep[list(ccd[g])] = chars[k,i]
      reps.append(rep)
      continue
    kd = np.argmax(np.abs(chars.dot(np.multiply(ccsize,chars[k,:]))))
    if kd < k:
      assert FSI[k] == 0 and FSI[kd] == 0
      reps.append(int(kd))
      continue
    #print(d,FSI[k])
    ik = mdl.index(kd)
    projector = projectors[ik]*d
    if FSI[k] == 1:
      dtype = REALTYPE
      projector = projector.real
    else:
      dtype = CPLXTYPE
    vecs = np.zeros((0,N),dtype=dtype)
    # Gram-Schmidt on random vectors projected to subspace
    # (effectively, diagonalize projector)
    while len(vecs) < d**2:
      vec = np.random.randn(N).astype(REALTYPE)
      vec /= linalg.norm(vec)
      vec = projector.dot(vec)
      if len(vecs):
        vec -= vecs.conj().dot(vec).dot(vecs)
      if linalg.norm(vec) > .1:
        vecs = np.append(vecs,vec[np.newaxis,:]/linalg.norm(vec),axis=0)
    assert linalg.norm(vecs.T.dot(vecs.conj())-projector) < tol
    # matrices for irrep^d
    rep2 = np.tensordot(vecs[:,compose],vecs.conj(),(2,1)).transpose((1,0,2))
    #print(linalg.norm(np.tensordot(rep2,rep2,(2,1)).transpose((0,2,1,3))-rep2[compose,:,:]))
    passing = False
    while not passing:
      # Random symmetric matrix at regular precision
      M = np.random.randn(d**2,d**2)
      if FSI[k] == 1:
        M = (M+M.T)
      else:
        M = (M+M.T) + 1j*(M-M.T)
      M = np.einsum('jk,imj,ink',M,rep2,rep2.conj())/N
      w = linalg.eigvalsh(M)
      # Isolate largest-magnitude eigenspace
      wdistinct = np.sort(w)[d//2::d]
      assert np.allclose(np.repeat(wdistinct,d),sorted(w),atol=tol,rtol=0)
      maxplus = (wdistinct[-1] > -wdistinct[0])
      for i in range(d):
        if (i+maxplus)%d:
          M = M.dot(M-wdistinct[i]*np.identity(d**2,dtype=int))
          wdistinct *= (wdistinct-wdistinct[i])
      lmax = wdistinct[-int(maxplus)]
      M /= lmax
      P = M.astype(dtype)
      P = P.dot(P.T.conj())
      # Refine by squaring until M is a projector to within float128 precision
      err0,err1 = 2,1
      while err1 < err0 and err1 != 0:
        # Stabilize
        P = np.einsum('jk,imj,ink',P,rep2,rep2.conj())/N
        # Square
        P = P.dot(P.T.conj())
        # Normalize
        P *= d/np.trace(P)
        # Test condition 
        err0,err1 = err1,abs(np.trace(P.dot(P))-d)
        print('errs',err0,err1)
        #assert err1<tol
      if err1 > tollow:
        continue
      #print(sorted(linalg.eigvalsh(P)))
      v0 = np.random.randn(d**2).astype(REALTYPE)
      vecs1 = P.dot(v0)
      vecs1 = vecs1[np.newaxis,:]/linalg.norm(vecs1)
      while len(vecs1) < d:
        v1 = np.random.randn(d**2).astype(REALTYPE)
        v1 /= linalg.norm(v1)
        v1 = P.dot(v1)
        v1 -= vecs1.conj().dot(v1).dot(vecs1)
        if linalg.norm(v1) > .1:
          vecs1 = np.append(vecs1,v1[np.newaxis,:]/linalg.norm(v1),axis=0)
      # Single copy of irrep, finally
      rep = np.einsum('ijk,mj,nk->imn',rep2,vecs1.conj(),vecs1)
      if FSI[k] == 1:
        # Real -- must perform basis transformation here
        # Get "charge-conjugation matrix" S: S^H R_g S = R_g^*
        # and then perform Takagi decomposition
        assert np.isrealobj(rep)
        #S0 = np.random.randn(d,d).astype(REALTYPE)
        #S0 += S0.T
        #S0 = np.einsum('jk,imj,ink->mn',S0,rep,rep)/N
        #S0 *= d/linalg.norm(S0)
        #assert linalg.norm(S0-S0.T) < tol
        #assert linalg.norm(S0.dot(S0.conj()) - np.identity(d)) < tol
        #V = takagi_stable(S0)
        #rep = np.einsum('ijk,jm,kn->imn',rep,V.conj(),V)
      passing = np.allclose(np.tensordot(rep,rep,(2,1)).transpose((0,2,1,3)),rep[compose,:,:],atol=tollow,rtol=0)
      print(d,FSI[k],linalg.norm(np.tensordot(rep,rep,(2,1)).transpose((0,2,1,3))-rep[compose,:,:]))
    reps.append(rep)
  return reps

def finiteSL(n,p):
  # Get SL_n(F) for finite field of prime order p
  # Collect determinant-1 matrices
  I = np.identity(n,dtype=int)
  mats = [I]
  for M in itertools.product(itertools.product(range(p),repeat=n),repeat=n):
    if round(linalg.det(M)%p) == 1 and np.any(I != M):
      mats.append(M)
  N = len(mats)
  mats = np.array(mats)
  #dots = np.tensordot(mats,mats,(2,1))%p
  #dots = np.swapaxes(dots,1,2)
  # Finally produce composition table
  compose = np.zeros((N,N),dtype=np.intp)
  compose[0,:] = range(N)
  compose[:,0] = range(N)
  for g in range(1,N):
    dot = np.tensordot(mats[g,:,:],mats[1:,:,:],(1,1))%p
    dot = dot.transpose((1,0,2))
    # Find permutation taking original order to this
    gh = g
    cmp_mats = np.delete(mats,g,axis=0)
    cmp_idx = np.delete(np.arange(N),g)
    for h in range(1,N):
      cmp = np.all(cmp_mats == dot[h-1,np.newaxis,:,:],axis=(1,2))
      gh = np.nonzero(cmp)[0]
      assert len(gh) == 1
      ghi = gh[0]
      compose[g,h] = cmp_idx[ghi]
      cmp_mats = np.delete(cmp_mats,ghi,axis=0)
      cmp_idx = np.delete(cmp_idx,ghi)
  print(N)
  reps = reps_from_compose(compose)
  print(len(reps))
  pickle.dump((compose,reps),open(f'sparseinv/reps_SL{n}(F{p}).p','wb'))
  return FiniteGroup(f'SL_{n}F_{p}',reps), compose

def reloadSL(n,p):
  compose,reps = pickle.load(open(f'sparseinv/reps_SL{n}(F{p}).p','rb'))
  return FiniteGroup(f'SL_{n}F_{p}',reps), compose
