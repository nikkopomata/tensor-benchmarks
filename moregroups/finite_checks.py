# Sanity-test properties of (finite) groups
import numpy as np
from scipy import linalg
import gc

# Set verbose to an threshold to use asserts instead of printing
threshold = False

def cprint(s, *errs):
  # Print or check error
  if threshold is False:
    print(f'{s}:', *errs)
  else:
    print(s)
    for e in errs:
      assert abs(e) < threshold

def charprops(G):
  # Check properties of character tables
  dims = [G._dims[r] for r in range(G.nreps)]
  cprint('Identity element',linalg.norm(G._chars[:,0]-dims))
  dot = G._chars.conj().dot(G._chars.T)/G.order
  #print('"Orthonormality" of characters:',linalg.norm(dot-np.diag(np.power(dims,2))))
  cprint('"Orthonormality" of characters',linalg.norm(dot-np.identity(G.nreps)))
  #print('Frobenius-Schur indicator...')
  # TODO fix or just paste FSI code from __init__

tables = {}
def getcomposetable(G):
  # Table of values g x h
  if G.label in tables:
    return tables[G.label]
  print('Generating table of composition operation...')
  compose = np.zeros((G.order,G.order),dtype=np.intp)
  for g in range(G.order):
    for h in range(G.order):
      compose[g,h] = G.compose(g,h)
  tables[G.label] = compose
  return compose

inverses = {}
def getinverses(G):
  # Map from element to inverse
  if G.label in inverses:
    return inverses[G.label]
  compose = getcomposetable(G)
  indarr, invarr = np.nonzero(compose == 0)
  assert len(indarr) == G.order
  indsort = np.argsort(indarr)
  assert all(indarr[indsort] == range(G.order))
  invmap = invarr[indsort]
  inverses[G.label] = invmap
  return invmap

def groupbasic(G):
  # Check axiomatic properties of group (i.e. verify compose table)
  compose = getcomposetable(G)
  print('Identity property...')
  assert np.all(compose[0,:] == range(G.order))
  assert np.all(compose[:,0] == compose[0,:])
  print('Inverses...')
  invmap = getinverses(G)
  assert all(invmap[invmap] == range(G.order))
  invcomp = compose[invmap,invmap[:,np.newaxis]]
  assert np.all(invmap[compose] == invcomp)
  print('Associativity...')
  assert np.all(compose[compose,:] == compose[:,compose])
  # Squares are diagonals of compose table
  chi2 = G._chars[:,compose[np.diag_indices(G.order)]]
  FSI = np.sum(chi2,axis=1)/G.order
  cprint('Frobenius-Schur indicators',linalg.norm(FSI-G._FSI))

def subgroup(G,gen):
  # Generate subgroup of G from gen
  # Make sure identity is in gen
  compose = getcomposetable(G)
  gen = set(gen) | {0}
  subdim = 1
  while subdim < len(gen):
    subdim = len(gen)
    genl = np.array(list(gen))
    gen = set(compose[genl[:,np.newaxis],genl].flatten())
  return list(gen)

def conjugacyclasses(G):
  # Compare conjugacy classes to irrep properties
  print('Conjugacy classes...')
  compose = getcomposetable(G)
  invmap = getinverses(G)
  conjtable = compose[compose,invmap[:,np.newaxis]]
  # Label conjugacy classes by first element
  ccs = {}
  counts = np.zeros(G.order,dtype=int)
  for g in range(G.order):
    cc = set(conjtable[:,g])
    minel = min(cc)
    if minel != g:
      assert minel in ccs and ccs[minel] == cc
    else:
      ccs[g] = cc
      counts[sorted(cc)] += 1
  assert all(counts == 1)
  assert ccs[0] == {0}
  assert len(ccs) == G.nreps
  #ccsize = sorted(len(cc) for cc in ccs.values())
  #assert ccsize == sorted(G._dims[k] for k in range(G.nreps))
  print('Characters match over conjugacy classes...')
  cclist = [[0]]
  cclist += [sorted(cc) for cc in ccs.values() if 0 not in cc]
  ccchars = np.zeros((G.nreps,G.nreps),dtype=np.complex256)
  ccdim = np.zeros(G.nreps,dtype=int)
  for icc,cc in enumerate(cclist):
    ccchars[icc,:] = G._chars[:,cc[0]]
    if len(cc) > 1:
      cccheck = linalg.norm(G._chars[:,cc[1:]] - G._chars[:,cc[:1]])
      if threshold:
        assert cccheck < threshold
      else:
        print(cccheck)
    ccdim[icc] = len(cc)
  cprint('Characters are orthogonal across conjugacy classes:',
    linalg.norm(ccchars.conj().T.dot(np.diag(ccdim)).dot(ccchars)/G.order-np.identity(G.nreps)))
  print('Commutator subgroup...')
  commutators = compose[conjtable,invmap[np.newaxis,:]]
  commsub = subgroup(G,set(commutators.flatten()))
  assert G.order == len(commsub) * sum(G._dims[k] == 1 for k in range(G.nreps))

def fusionprops(G, testgen=None):
  if testgen is None or len(testgen) == 0:
    testgen = slice(None)
    ntest = G.order
  else:
    ntest = len(testgen)
  print('Checking fusion properties...')
  for k1,d1 in G._dims.items():
    for k2,d2 in G._dims.items():
      print(f'[{k1:>2d} x {k2:>2d}]',end=' ')
      f,CG = G.fusionCG(k1,k2)
      assert G.sumdims(f) == d1*d2
      T = np.zeros((d1,d2,d1*d2),dtype=np.complex256)
      charprod = G._chars[abs(k1),:] * G._chars[abs(k2),:]
      charsum = np.zeros_like(charprod)
      sumrep = np.zeros((ntest,d1*d2,d1*d2),dtype=np.complex256)
      idx = 0
      for k,n in f:
        charsum += n*G._chars[abs(k),:]
        d = G.dim(k)
        if CG[k] is None:
          if d1 == 1:
            T = np.expand_dims(np.identity(d2),axis=0)
          elif d2 == 1:
            T = np.expand_dims(np.identity(d1),axis=1)
        else:
          T[:,:,idx:idx+n*d] = CG[k].transpose((0,1,3,2)).reshape(d1,d2,n*d)
        if d == 1:
          sumrep[:,idx,idx] = G._irreps[k][testgen].conj()
        else:
          rep = G._irreps[abs(k)][testgen]
          if n != 1:
            rep = np.kron(np.identity(n),rep)
          if k < 0:
            sumrep[:,idx:idx+n*d,idx:idx+n*d] = rep
          else:
            sumrep[:,idx:idx+n*d,idx:idx+n*d] = rep.conj()
        idx += n*d
      if threshold:
        assert linalg.norm(charprod-charsum) < threshold
      else:
        print(linalg.norm(charprod-charsum),end=' ')
      if k1 < 0:
        S = G.S(-k1)
        rep1 = np.einsum('ijk,lj,mk->ilm',G._irreps[-k1][testgen,:,:],S.conj(),S)
      elif d1 == 1:
        rep1 = np.expand_dims(G._irreps[k1][testgen],axis=(1,2))
      else:
        rep1 = G._irreps[k1][testgen,:,:]
      if k2 < 0:
        S = G.S(-k2)
        rep2 = np.einsum('ijk,lj,mk->ilm',G._irreps[-k2][testgen,:,:],S.conj(),S)
      elif d2 == 1:
        rep2 = np.expand_dims(G._irreps[k2][testgen],axis=(1,2))
      else:
        rep2 = G._irreps[k2][testgen,:,:]
      Tsym = np.einsum('ijk,imn,ipq,knq->ijmp',rep1,rep2,sumrep,T,optimize=True)
      if threshold:
        assert linalg.norm(Tsym-T[np.newaxis,...]) < threshold
        print()
      else:
        print(linalg.norm(Tsym-T[np.newaxis,...]))
      gc.collect()

def unitarity(G):
  # Unitarity & properties of dual representations
  print('Checking unitarity...')
  for k,rep in enumerate(G._irreps):
    d = G._dims[k]
    if d == 1:
      assert rep.shape == (G.order,)
      cprint(f'[{k}]',linalg.norm(np.abs(rep)-1))
    else:
      assert rep.shape == (G.order,d,d)
      cprint(f'[{k}]',linalg.norm(np.einsum('ijk,ilk->ijl',rep,rep.conj())-np.identity(d,dtype=np.float128)[np.newaxis,:,:]))
  print('Checking dual representations...')
  for k,rep in enumerate(G._irreps):
    if G._FSI[k] == 1:
      print(f'[{k}]')
      assert np.isrealobj(rep)
    elif G._FSI[k] == 0:
      cprint(f'[{k}]',linalg.norm(G._irreps[G._duals[k]] - rep.conj()))
    else:
      S = G.S(k)
      Rstar = np.einsum('kj,ikl,lm',S.conj(),rep,S)
      cprint(f'[{k}]',linalg.norm(Rstar - rep.conj()))

def intertwiners(G,testgen=None):
  if testgen is None or len(testgen) == 0:
    testgen = slice(None)
    ntest = G.order
  else:
    ntest = len(testgen)
  print('Checking elementary intertwiners...')
  for (k1,k2,k3),I in G._reference_intertwiners.items():
    d1,d2,d3,n = I.shape
    rep1 = G._irreps[k1][testgen]
    if d1 == 1:
      rep1 = rep1[:,np.newaxis,np.newaxis]
    rep2 = G._irreps[k2][testgen]
    if d2 == 1:
      rep2 = rep2[:,np.newaxis,np.newaxis]
    rep3 = G._irreps[k3][testgen]
    if d3 == 1:
      rep3 = rep3[:,np.newaxis,np.newaxis]
    RI = np.einsum('ijk,imn,ipq,knql->ijmpl',rep1,rep2,rep3,I,optimize=True)
    diff = linalg.norm((I[np.newaxis,...]-RI).reshape(ntest,-1),axis=1)
    cprint(f'[{k1}x{k2}x{k3}]',linalg.norm(diff))
    if isinstance(testgen,np.ndarray):
      yield (k1,k2,k3),testgen[np.nonzero(diff>1e-15)]
    else:
      yield (k1,k2,k3),np.nonzero(diff>1e-15)[0]

def all3j(G,testgen=None):
  if testgen is None or len(testgen) == 0:
    testgen = slice(None)
    ntest = G.order
  else:
    ntest = len(testgen)
  print('Checking all basic intertwiners...')
  for k1,d1 in G._dims.items():
    for k2,d2 in G._dims.items():
      f = G.fusion(k1,k2)
      kl = []
      for k,n in f:
        if n == 0:
          continue
        d = G._dims[k]
        kl.append((G._duals[k],d,n))
        if G._FSI[k] == -1:
          kl.append((k,d,n))
      for k3,d3,n in kl:
        I = G.get3j(k1,k2,k3,n)
        if I is None:
          assert d1 == 1 and d2 == 1 and d3 == 1
          continue
        ortho = linalg.norm(np.tensordot(I.conj(),I,((0,1,2),(0,1,2)))-np.identity(n))
        unit1 = linalg.norm(np.tensordot(I.conj(),I,((1,2),(1,2))).reshape(d1*n,d1*n)*d1-np.identity(d1*n))
        unit2 = linalg.norm(np.tensordot(I.conj(),I,((0,2),(0,2))).reshape(d2*n,d2*n)*d2-np.identity(d2*n))
        unit3 = linalg.norm(np.tensordot(I.conj(),I,((0,1),(0,1))).reshape(d3*n,d3*n)*d3-np.identity(d3*n))
        nstr = '' if n == 1 else f'({n})'
        if threshold:
          print(f'[{k1:^3}x{k2:^3}x{k3:^3} {nstr}]')
          assert ortho < threshold
          assert unit1 < threshold
          assert unit2 < threshold
          assert unit3 < threshold
        else:
          print(f'[{k1:^3}x{k2:^3}x{k3:^3} {nstr}]: {ortho:^7.2g} {unit1:^7.2g} {unit2:^7.2g} {unit3:^7.2g}',end=' ')
        rep1 = G._irreps[abs(k1)][testgen]
        if d1 == 1:
          rep1 = rep1[:,np.newaxis,np.newaxis]
        elif k1 < 0:
          rep1 = rep1.conj()
        rep2 = G._irreps[abs(k2)][testgen]
        if d2 == 1:
          rep2 = rep2[:,np.newaxis,np.newaxis]
        elif k2 < 0:
          rep2 = rep2.conj()
        rep3 = G._irreps[abs(k3)][testgen]
        if d3 == 1:
          rep3 = rep3[:,np.newaxis,np.newaxis]
        elif k3 < 0:
          rep3 = rep3.conj()
        RI = np.einsum('ijk,imn,ipq,knql->ijmpl',rep1,rep2,rep3,I,optimize=True)
        diff = linalg.norm((I[np.newaxis,...]-RI).reshape(ntest,-1),axis=1)
        if threshold:
          assert linalg.norm(diff) < threshold
        else:
          print(linalg.norm(diff))
        if isinstance(testgen,np.ndarray):
          yield (k1,k2,k3),testgen[np.nonzero(diff>1e-15)]
        else:
          yield (k1,k2,k3),np.nonzero(diff>1e-15)[0]
        gc.collect()

