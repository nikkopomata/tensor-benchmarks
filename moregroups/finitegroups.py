from quantrada.groups import *
import itertools

def asmatrixvalued(arr, ndim):
  # If arr has rank ndim, return "matrix-valued array" with two additional
  # indices
  if arr.ndim == ndim:
    return arr[...,np.newaxis,np.newaxis]
  else:
    assert arr.ndim == ndim+2
    return arr

class FiniteGroup(Group):
  def __init__(self, label, irreps, S={}):
    self.order = len(irreps[0])
    self._irreps = list(irreps)
    self.nreps = len(self._irreps)
    # Character table
    self._chars = np.zeros((len(irreps),self.order),dtype=CPLXTYPE)
    FSI = np.zeros(len(irreps),dtype=CPLXTYPE)
    dims = {}
    duals = {}
    for k in range(len(irreps)):
      rep = irreps[k]
      if isinstance(rep,list):
        rep = np.array(rep)
      if isinstance(rep,np.ndarray):
        if np.all(np.isreal(rep)):
          rep = rep.real.astype(REALTYPE)
        else:
          rep = rep.astype(CPLXTYPE)
      if isinstance(rep,int):
        kd = rep
        assert kd<k
        rep = irreps[kd].conj()
        assert abs(FSI[kd]) < 1e-10 # Frobenius-Schur indicator is 0
        FSI[k] = 0
        self._chars[k,:] = self._chars[kd,:].conj()
        duals[k] = kd
        duals[kd] = k
        d = dims[kd]
      elif rep.ndim == 1 or rep.shape[1] == 1:
        d = 1
        rep = rep.squeeze()
        FSI[k] = np.sum(np.power(rep,2))/self.order
        self._chars[k,:] = rep
      else:
        d = rep.shape[1]
        self._chars[k,:] = np.trace(rep,axis1=1,axis2=2)
        FSI[k] = np.tensordot(rep,rep,((0,1,2),(0,2,1)))/self.order
      if np.real(FSI[k])>.9 and rep.dtype == CPLXTYPE:
        # Cast to real
        assert linalg.norm(rep.imag) < 1e-10
        rep = rep.real
      self._irreps[k] = rep
      dims[k] = d
    self._FSI = np.rint(FSI.real).astype(int)
    assert linalg.norm(FSI-self._FSI) < 1e-10
    super().__init__()
    self._dims.update(dims)
    self._duals.update(duals)
    if self.quaternionic:
      self._Ss.update(S)
    # Duals
    for k,ind in enumerate(self._FSI):
      if ind == 1:
        self._duals[k] = k
      elif ind == -1:
        self._duals[k] = -k
        self._duals[-k] = k
        self._dims[-k] = dims[k]
      else:
        assert ind == 0
        if k not in self._duals:
          # Find dual
          chardot = self._chars.dot(self._chars[k,:])
          kd = np.argmax(np.abs(chardot))
          chardot[kd] -= self.order
          assert linalg.norm(chardot) < 1e-10
          assert linalg.norm(irreps[k] - irreps[kd].conj()) < 1e-10
          assert self._FSI[kd] == 0
          self._duals[kd] = k
          self._duals[k] = kd
    self._reference_intertwiners = {}

  @classmethod
  def get_identifier(cls, label, irreps, S={}):
    return label

  @property
  def quaternionic(self):
    return any(self._FSI == -1)

  def _fdim(self, r):
    # No need to implement since self._dims is already complete
    pass
  
  def _fdual(self, r):
    # Same as with _fdim
    pass

  def isrep(self, r):
    return (isinstance(r,int) or isinstance(r,np.int64)) and abs(r)<len(self._irreps) and (r>=0 or self._FSI[-r] == -1)

  def indicate(self, r):
    if r < 0:
      return -2
    else:
      return self._FSI[r]

  def _fS(self, r):
    if r < 0:
      return self.S(-r).T.conj()
    # Compute by averaging R(g)@R@R(g)^T
    # Get first group element for which this is "large enough"
    d = self._dims[r]
    rep = self._irreps[r]
    for i in range(1,self.order):
      R = rep[i,:,:]
      Savg = np.tensordot(rep,R,1)
      Savg = np.tensordot(Savg,rep,((0,2),(0,2)))
      Snorm = linalg.norm(Savg)
      if Snorm > 1/self.order:
        break
    if Snorm < np.sqrt(self.order):
      # "Amplify" for greater precision
      Savg = np.tensordot(rep,Savg,1)
      Savg = np.tensordot(Savg,rep,((0,2),(0,2)))
      Snorm = linalg.norm(Savg)
    return Savg * np.sqrt(REALTYPE(d))/Snorm

  def _firrep(self, r, g):
    return self._irreps[r][g]

  def _firrep1d(self, r, g):
    return self._irreps[r][g]

  def get3j(self, r1, r2, r3, degen):
    # Obtain the reference intertwiners used for Clebsch-Gordan coefficients,
    # a.k.a. 3j-symbols
    if (r1,r2,r3) in self._reference_intertwiners:
      return self._reference_intertwiners[r1,r2,r3]
    # Check "secondary" quaternionic
    if r1 < 0:
      return np.tensordot(self.S(-r1).conj(), self.get3j(-r1,r2,r3, degen),(0,0))
    if r2 < 0:
      return np.tensordot(self.S(-r2).conj(), self.get3j(r1,-r2,r3, degen), (0,1)).transpose((1,0,2,3))
    if r3 < 0:
      return np.tensordot(self.get3j(r1,r2,-r3, degen), self.S(-r3).conj(), (2,0)).transpose((0,1,3,2))
    # Check sorted
    if r2<r1 or r3<r2:
      isort = np.argsort((r1,r2,r3))
      iinv = np.zeros_like(isort)
      iinv[isort] = range(3) # Invert permutation
      T = self.get3j(*sorted((r1,r2,r3)), degen)
      if T is None:
        return None
      return T.transpose(tuple(iinv)+(3,))
    # Lexical comparison with dual
    rd = sorted((abs(self._duals[r1]),abs(self._duals[r2]),abs(self._duals[r3])))
    if rd[0]<r1 or (rd[0]==r1 and (rd[1]<r2 or (rd[1]==r2 and rd[2]<r3))):
      I = self.get3j(self._duals[r1],self._duals[r2],self._duals[r3], degen)
      if isinstance(I,np.ndarray):
        return I.conj()
      else:
        return I
    if r1 == 0: # Trivial -- implies r2 ~ r3*
      d = self._dims[r2]
      if self._FSI[r2] == -1:
        # Quaternionic (proportional to S)
        M = self.S(r2)
      else:
        M = np.identity(d,dtype=REALTYPE)
      T = np.expand_dims(M/np.sqrt(REALTYPE(d)),axis=(0,3))
      self._reference_intertwiners[r1,r2,r3] = T
      return T
    # Compute
    d1 = self._dims[r1]
    d2 = self._dims[r2]
    d3 = self._dims[r3]
    if (d1 == 1 and d2 == 1 and d3 == 1) or degen == 0:
      return None
    # Build invariant vector
    if self._FSI[r1] == 1 and self._FSI[r2] == 1 and self._FSI[r3] == 1:
      dtype = REALTYPE
    else:
      dtype = CPLXTYPE
    #v0 = np.identity(d1*d2*d3,dtype=dtype).reshape(d1,d2,d3,-1)
    rep1 = asmatrixvalued(self._irreps[r1],1)
    rep2 = asmatrixvalued(self._irreps[r2],1)
    rep3 = asmatrixvalued(self._irreps[r3],1)
    # Symmetrize all basis elements in r1 x r2 x r3
    vsym = np.einsum('ijk,imn,ipq->jmpknq',rep1,rep2,rep3,optimize=True)/self.order
    vsym = vsym.reshape(d1,d2,d3,-1)
    #print(f'[{r1} x {r2} x {r3}] collected:',linalg.norm(vsym))
    # Collect best-approximation intertwiners
    vest = []
    while vsym.shape[3]:
      norm = linalg.norm(vsym.reshape(d1*d2*d3,-1),axis=0)
      # Stabilized argmax to ensure quasi-deterministic ordering of basis
      nmax = np.max(norm)
      idx = np.argwhere(norm > nmax*(1-1e-5))[0,0]
      v = vsym[...,idx]/norm[idx]
      vest.append(v)
      if len(vest) == degen:
        break
      vsym = np.delete(vsym, idx, axis=3)
      dot = np.tensordot(v.conj(),vsym,3)
      vsym -= np.tensordot(v,dot,0)
    # Refine
    vest = np.array(vest).transpose((1,2,3,0))
    #print(f'[{r1} x {r2} x {r3}] first pass:',linalg.norm(vest))
    vsym = np.einsum('ijk,imn,ipq,knqr->jmpr',rep1,rep2,rep3,vest,optimize=True)/self.order
    #print(f'[{r1} x {r2} x {r3}] refined:',linalg.norm(vsym))
    assert linalg.norm(vsym)>linalg.norm(vest)*.9 # Close to an actual intertwiner
    for idx in range(degen):
      v = vsym[...,idx]
      v /= linalg.norm(v)
      if idx < degen-1:
        dot = np.tensordot(v.conj(),vsym[...,idx+1:],3)
        vsym[...,idx+1:] -= np.tensordot(v,dot,0)
      vsym[...,idx] = v
    #print(f'[{r1} x {r2} x {r3}] refined w/G.S.:',linalg.norm(vsym))
    self._reference_intertwiners[r1,r2,r3] = vsym
    return vsym

  def _ffusion(self, r1, r2):
    if r1 == 0:
      if r2 < 0:
        yield (-r2,np.expand_dims(self.S(-r2).conj(),(0,3)))
      else:
        yield (r2,None)
      return
    if r2 == 0:
      if r1 < 0:
        yield (-r1,np.expand_dims(self.S(-r1).conj(),(1,3)))
      else:
        yield (r1,None)
      return
    chardot = np.einsum('i,i,ji->j',self._chars[abs(r1),:],self._chars[abs(r2),:],self._chars.conj())/self.order
    counts = np.rint(chardot.real).astype(int)
    reps = np.nonzero(counts)[0]
    assert linalg.norm(chardot-counts) < 1e-10
    if self._dims[r1] == 1 and self._dims[r2] == 1:
      yield (reps[0],None)
    else:
      for k in reps:
        n = counts[k]
        yield k,self.get3j(r1,r2,self._duals[k],n)*np.sqrt(REALTYPE(self._dims[k]))
    return

  def Haargen(self):
    return RNG.integers(self.order)

  def compose(self, g, h):
    if g == 0:
      return h
    elif h == 0:
      return g
    candidates = np.delete(np.arange(self.order,dtype=np.intp),[g,h])
    for k in sorted(range(self.nreps),key=self.dim):
      rep = self._irreps[k]
      if k == 0:
        continue
      #print(f'before {k}:',candidates)
      if len(candidates) == 1:
        break
      if self._dims[k] == 1:
        r = rep[g]*rep[h]
        indicator = np.abs(rep[candidates]-r)
      else:
        R = rep[g,:,:].dot(rep[h,:,:])
        #indicator = linalg.norm(self._irreps[k][candidates,:,:]-np.repeat(R,[len(candidates),1,1]),axis=(1,2))
        indicator = linalg.norm(rep[candidates,:,:]-R[np.newaxis,:,:],axis=(1,2))
      candidates = candidates[indicator<1e-10]
    assert len(candidates) == 1
    return candidates[0]

  def complex_pairorder(self, r):
    return r > self._duals[r]

  def __reduce__(self):
    if self.quaternionic:
      return self.__class__, (self._irreps, self._Ss)
    else:
      return self.__class__, (self._irreps,)

def dihedral(N):
  """Dihedral group D_N (symmetry of polygon with N/2 vertices)"""
  assert N%2 == 0
  M = N//2
  # Elements 1,a,...,a^l,...,x,ax,...,a^lx
  # 1D irreps:
  if M%2 == 0:
    # Abelianization is G/a**2 ~= V
    reps = [np.ones(N,dtype=int) for k in range(4)]
    reps[1][M:] = -1
    reps[2][1::2] = -1
    reps[3][1:M:2] = -1
    reps[3][M::2] = -1
  else:
    # Abelianization is G/a ~= Z2
    reps = np.ones((2,N),dtype=int)
    reps[1][M:] = -1
    reps = list(reps)
  # 2D irreps
  max2d = (M+1)//2
  angle = 2*pi*np.tensordot(range(1,max2d),range(M),0)/M
  c = np.cos(angle)
  s = np.sin(angle)
  rep2 = np.concatenate(([[c,-s],[s,c]],[[c,s],[s,-c]]),axis=3)
  reps.extend(rep2.transpose((2,3,0,1)))
  return FiniteGroup(f'dihedral{N}',reps)

def cyclic(N):
  """Cyclic group Z_N (or C_N)"""
  reps = [np.ones(N,dtype=int)]
  angle = 2*pi*np.tensordot(range(1,(N+1)//2),range(N),0)/N
  reps.extend(np.exp(CPLXTYPE(1j)*angle))
  if N%2==0:
    reps.append(reps[0].copy())
    reps[-1][1::2] = -1
  reps.extend(range((N-1)//2,0,-1))
  return FiniteGroup(f'Z{N}',reps)

def SL23():
  """SL_2(F_3) (or, universal cover of A4)"""
  N = 24
  reps = [np.ones(N,dtype=int)]
  # Organize according to: element (n,r,i) is
  # nth (0 or 1) cover of composition of
  #   rotation (0 1 2)^r with
  #   ith reflection element (in (), (01)(23), (02)(13), (03)(12) )
  # 1D irreps (abelianization is rotations)
  angle = 2*pi*np.tensordot([1,2],range(3),0)/3
  rabel = np.exp(CPLXTYPE(1j)*angle)
  reps.extend(np.tile(rabel[:,np.newaxis,:,np.newaxis],(1,2,1,4)).reshape(2,24))
  # 3D irrep: can take
  #   "reflections" as pi rotations around x,y,&z axes
  #   generating rotation as by 2pi/3 around (1,1,1)
  # Diagonal part of reflections
  Rrefs = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=int)
  # Rotation generation
  # SO(3) generators Lx+Ly+Lz
  Lxyz = np.array([[0,-1,1],[1,0,-1],[-1,1,0]],dtype=int)
  Rrot = expm_128(2*pi/3/np.sqrt(REALTYPE(3))*Lxyz)
  Rrots = np.array([np.identity(3,dtype=REALTYPE),Rrot,Rrot.dot(Rrot)])
  R3d = np.einsum('i,jlm,km->ijklm',[1,1],Rrots,Rrefs)
  reps.append(R3d.reshape(24,3,3))
  # 2D quaternionic irrep:
  # "reflections" are i*Paulis
  # rotations are (I +- i s_x +- i s_y +- i s_z)/2
  ipauli = np.array([[[1,0],[0,1]],
                    [[0,1j],[1j,0]],
                    [[0,1],[-1,0]],
                    [[1j,0],[0,-1j]]],dtype=CPLXTYPE)
  # Flip i is ipauli[i]
  # Rotation (0 1 2)^r is R_r[a,b] = drots[r,i]/2*ipauli[i,a,b] for:
  drots = [[2,0,0,0],[1,1,1,-1],[1,-1,-1,1]]
  repq = np.einsum('i,jl,lmp,kpn->ijkmn',[1,-1],drots,ipauli,ipauli)
  repq = repq.reshape((24,2,2))/2
  reps.append(repq)
  reps.append(np.einsum('i,ijk->ijk',reps[1],repq))
  reps.append(5)
  return FiniteGroup('SL2F3',reps)

def partitions(N,nmax=None):
  """Returns partitions of integer N. Helper for Sn"""
  if not nmax:
    nmax = N
  if nmax >= N:
    yield (N,)
  for n in range(min(N-1,nmax),0,-1):
    for l in partitions(N-n,n):
      yield (n,) + l

def symmetric(nletters):
  N = np.math.factorial(nletters)
  reps = []
  # Generators corresponding to flips (i i+1)
  genflip = []
  dims = []
  # Iterate through Young tableaux
  for lam in partitions(nletters):
    # Get rows & columns of Young tableau
    print(lam)
    k = len(lam) # Height
    l = lam[0] # Length
    # Array form
    youngarr = np.zeros((k,l),dtype=np.intp)
    idx = 0
    lamsl = []
    for i,p in enumerate(lam):
      youngarr[i,:p] = range(idx,idx+p)
      lamsl.append(slice(idx,idx+p))
      idx += p
    # Row form
    youngr = [youngarr[i,:p] for i,p in enumerate(lam)]
    # Column form
    youngc = [youngarr[:,j] for j in range(l)]
    lamT = [k]
    for j in range(1,l):
      youngc[j] = youngc[j][youngc[j]>0]
      lamT.append(len(youngc[j]))
    # Obtain repreesentation by (following Diaconis)
    # taking action on "tabloids", equivalence classes under row permutations
    tabloids = [] # Representative elements of tabloids
    tabindex = {}
    standardtab = [] # Indices of standard tableaux
    rowlabels = [] # Generate tabloid representatives
    nlabels = nletters
    for p in lam:
      rowlabels.append(itertools.combinations(range(nlabels),p))
      nlabels -= p
    idx = 0
    # Indices where an intra-column comparison is valid
    colcomp_idx = [(i,j) for j in range(l) for i in range(lamT[j]-1)]
    # Collect tabloids
    for rows in itertools.product(*rowlabels):
      letters = np.arange(nletters,dtype=int)
      tab = []
      for row in rows:
        tab.append(tuple(letters[row,]))
        letters = np.delete(letters,row)
      tab = tuple(tab)
      tabloids.append(tab)
      tabindex[tab] = idx
      # check if standard
      if all(tab[i][j] < tab[i+1][j] for i,j in colcomp_idx):
        standardtab.append(idx)
      idx += 1
    ntab = idx
    dim = len(standardtab)
    dims.append(dim)
    # Obtain "polytabloids" corresponding to standard tableaux
    polytabloids = np.zeros((dim,ntab),dtype=int)
    # Array representation of standard tableaux
    standardarr = np.zeros((dim,nletters),dtype=np.intp)
    for i,idx in enumerate(standardtab):
      m = 0
      for j,row in enumerate(tabloids[idx]):
        mm = m+lam[j]
        standardarr[i,m:mm] = row
        m = mm
    colperms = (itertools.permutations(yc) for yc in youngc)
    # Column-order to row-order
    #print(youngr,youngc)
    rtoc = np.concatenate(youngc,axis=0)
    ctor = np.zeros_like(rtoc)
    ctor[rtoc] = range(nletters)
    for permc in itertools.product(*colperms):
      # Resulting tableaux
      permr = np.concatenate(permc,axis=0)[ctor]
      #permuted = np.concatenate([standardarr[:,p] for p in permc],axis=1)[:,ctor]
      permuted = standardarr[:,permr]
      # Sign of permutation
      asgn = 0
      for p in permc:
        pa = np.array(p)
        for ii,j in enumerate(p):
          asgn += sum(pa[:ii] > j)
      sgn = 1-2*(asgn%2)
      # For each tableau find tabloid representative
      rows = [permuted[:,sl] for sl in lamsl]
      for idx,tab in enumerate(zip(*rows)):
        #tab = []
        #for i in lam:
        #  row,tabl = tabl[:i],tabl[i:]
        #  tab.append(tuple(sorted(row)))
        #polytabloids[idx,tabindex[tuple(tab)]] += sgn
        #print(standardarr[idx,:],tuple(tuple(sorted(row)) for row in tab),permc,sgn)
        polytabloids[idx,tabindex[tuple(tuple(sorted(row)) for row in tab)]] += sgn
    # Perform Gram-Schmidt on polytabloid basis
    polytabbasis = polytabloids.astype(REALTYPE)
    for idx in range(dim):
      v = polytabbasis[idx,:]
      v /= linalg.norm(v)
      vdot = polytabbasis[idx+1:,:].dot(v)
      polytabbasis[idx+1:,:] -= np.tensordot(vdot,v,0)
    # Can generate the group from (0 1) and (0 1 ... nletters-1)
    # Determine action in tabloid basis
    permflip = []
    permrot = []
    for idx,tab in enumerate(tabloids):
      if any(row[:2] == (0,1) for row in tab):
        # Tabloid is unchanged by (0 1)
        permflip.append(idx)
      else:
        tabflip = list(tab)
        for i in range(k):
          if tab[i][0] == 0:
            tabflip[i] = (1,)+tab[i][1:]
          elif tab[i][0] == 1:
            tabflip[i] = (0,)+tab[i][1:]
        # Already ordered
        idxflip = tabindex[tuple(tabflip)]
        permflip.append(idxflip)
      tabrot = tuple(tuple(sorted((i+1)%nletters for i in row)) for row in tab)
      permrot.append(tabindex[tabrot])
    # Project onto polytabloid subspace
    genrot = polytabbasis[:,permrot].dot(polytabbasis.T)
    flips = np.zeros((nletters-1,dim,dim),dtype=REALTYPE)
    flips[0,:,:] = polytabbasis.dot(polytabbasis[:,permflip].T)
    for i in range(1,nletters-1):
      flips[i,:,:] = genrot.dot(flips[i-1,:,:]).dot(genrot.T)
    genflip.append(flips)
  # Generate rest of group
  elsort = sorted(enumerate(itertools.permutations(range(nletters))),
    key=lambda l: sum(l[1][i] > l[1][j] for j in range(nletters) for i in range(j)))
  #print(elsort)
  for k,d in enumerate(dims):
    rep = np.zeros((N,d,d),dtype=REALTYPE)
    rep[0,:,:] = np.identity(d,dtype=int)
    reps.append(rep)
  perminv = {tuple(range(nletters)):0} # Invert permutation list on accounted-for subset
  for idx,perm in elsort:
    if idx == 0:
      continue
    # First flip
    for ii in range(nletters-1,0,-1):
      if perm[ii-1]>perm[ii]:
        break
    pperm = perm[:ii-1] + (perm[ii],perm[ii-1]) + perm[ii+1:]
    #print(len(perminv),perm,ii,pperm)
    if len(perminv) < nletters:
      # Simple flip -- already determined
      assert pperm == tuple(range(nletters))
      for k,rep in enumerate(reps):
        rep[idx,:,:] = genflip[k][ii-1,:,:]
    else:
      # Compose with calculated permutation
      pidx = perminv[pperm]
      for k,rep in enumerate(reps):
        rep[idx,:,:] = genflip[k][ii-1,:,:].dot(rep[pidx,:,:])
    perminv[perm] = idx
  return FiniteGroup(f'Symmetric{nletters}',reps)

def gamma(p,q=0, testgens=False):
  # Gamma group (aka Dirac group, or generators of Clifford algebra)
  # on p+q generators with signature (p,q)
  # if testgens just return generators for testing instead of instantiating
  #   group object
  assert isinstance(p,int) and p >= 0
  assert isinstance(q,int) and q >= 0
  ngen = p+q
  assert ngen > 1
  N = 2**(ngen+1)
  # 2^(n+1) elements: as a (2,2,...) array, element
  # [i,j1,j2,...,jn] is (-1)^i gamma1^j1 gamma2^j2 ... gamman^jn
  # 1D irreps are in correspondence with power set of generators
  if not testgens:
    rep1d = np.ones(2,dtype=int)
    simple = np.array([[1,1],[1,-1]],dtype=int)
    for i in range(ngen):
      rep1d = np.tensordot(simple,rep1d,0)
      rep1d = np.moveaxis(rep1d,(0,1),(i,-1))
    reps = list(rep1d.reshape(2**ngen,N))
  # Construct multidimensional irreps
  # Obtain Frobenius-Schur indicator to see what properties are to be considered
  FSI = [1,1,1,0,-1,-1,-1,0][(p-q)%8]
  # Number of generators to obtain before worrying about real basis
  sz = np.diag([1,-1])
  syi = np.array([[0,-1],[1,0]])
  sx = np.array([[0,1],[1,0]])
  generators = np.stack([sz,sx,syi],axis=0)
  dims = 1
  whereim = np.zeros(ngen,dtype=bool)
  if ngen < 3:
    # Cut out extraneous generators 
    generators = generators[:ngen,...]
  else:
    whereim[2] = True
    I2 = np.identity(2,dtype=int)
    I = 1
    for igen in range(3,ngen):
      # Dimensions for generators at this point
      if igen%2==1:
        dims += 1
        I = np.tensordot(I,I2,0)
        g = np.tensordot(I,sx,0)
        generators = np.tensordot(generators,sz,0)
      else:
        g = np.tensordot(I,syi,0)
        whereim[igen] = True
      generators = np.concatenate([generators,g[np.newaxis,...]],axis=0)
    generators = np.moveaxis(generators,range(3,2*dims+1,2),range(2,dims+1))
  if FSI == 1:
    # Real
    if p >= q:
      nV = (p-q)//8
      ndrest = ngen//2 - 4*nV
      # Separate whereim into parts affected & not affected by basis change
      whereneg = whereim.copy()
      whereim[8*nV+1:] = False
      whereneg[:8*nV+1] = False
      if (p-q)%8==0 and q!=0:
        # Need to flip last generator (change sx to -i*sy)
        assert whereneg[-1] == False
        generators[-1,...,1] *= -1
        whereneg[-1] = True
      assert sum(whereneg) == q
      if nV:
        # Apply nV copies of "Takagi matrix", V st VV^T=YXYX
        # 2*X = (VxR + i*VxI) * (VxR^T + i*VxI^T), etc.
        VxR = np.array([[0,1],[0,1]],dtype=int)
        VxI = np.array([[1,0],[-1,0]],dtype=int)
        VxxR = np.kron(VxR,VxR) - np.kron(VxI,VxI)
        VxxI = np.kron(VxI,VxR) + np.kron(VxR,VxI)
        VyyR = np.zeros((4,4),dtype=int)
        VyyR[[0,1,2,3],[0,2,2,0]] = [1,1,1,-1]
        VyyI = np.zeros((4,4),dtype=int)
        VyyI[[0,1,2,3],[3,1,1,3]] = [1,1,-1,1]
        VR = np.kron(VyyR,VxxR) - np.kron(VyyI,VxxI)
        VI = np.kron(VyyR,VxxI) + np.kron(VyyI,VxxR)
        VR = np.moveaxis(VR.reshape(8*(2,)),(1,5),(2,6)).reshape(16,16)
        VI = np.moveaxis(VI.reshape(8*(2,)),(1,5),(2,6)).reshape(16,16)
        # Separate generators into real & imaginary
        gshape = [ngen] + nV*[16] + [2**ndrest] + nV*[16] + [2**ndrest]
        genR = generators.reshape(*gshape)
        genI = genR.copy()
        genR[whereim] = 0
        genI[~whereim] = 0
        # Apply V^T on left
        for i in range(nV):
          genR,genI = np.tensordot(genR,VR,(1,0))-np.tensordot(genI,VI,(1,0)),\
                      np.tensordot(genI,VR,(1,0))+np.tensordot(genR,VI,(1,0))
        # Apply V^* on right
        for i in range(nV):
          genR,genI = np.tensordot(genR,VR,(2,0))+np.tensordot(genI,VI,(2,0)),\
                      np.tensordot(genI,VR,(2,0))-np.tensordot(genR,VI,(2,0))
        assert not np.any(genI)
        genR = np.moveaxis(genR,(1,2),(1+nV,2+2*nV))
        generators = genR.astype(REALTYPE)/8**nV
    elif p < q:
      nV = (q-p+2)//8
      ndrest = ngen//2 - 4*nV + 1
      # Apply nV-1 copies of 4-qubit "Takagi matrix", V4 st VV^T=YXYX
      # & one copy of V3 st VV^T=YXY
      # 2*X = (VxR + i*VxI) * (VxR^T + i*VxI^T), etc.
      VxR = np.array([[0,1],[0,1]],dtype=int)
      VxI = np.array([[1,0],[-1,0]],dtype=int)
      VxxR = np.kron(VxR,VxR) - np.kron(VxI,VxI)
      VxxI = np.kron(VxI,VxR) + np.kron(VxR,VxI)
      VyyR = np.zeros((4,4),dtype=int)
      VyyR[[0,1,2,3],[0,2,2,0]] = [1,1,1,-1]
      VyyI = np.zeros((4,4),dtype=int)
      VyyI[[0,1,2,3],[3,1,1,3]] = [1,1,-1,1]
      VR = np.kron(VyyR,VxxR) - np.kron(VyyI,VxxI)
      VI = np.kron(VyyR,VxxI) + np.kron(VyyI,VxxR)
      V4R = np.moveaxis(VR.reshape(8*(2,)),(1,5),(2,6)).reshape(16,16)
      V4I = np.moveaxis(VI.reshape(8*(2,)),(1,5),(2,6)).reshape(16,16)
      VR = np.kron(VxR,VyyR) - np.kron(VxI,VyyI)
      VI = np.kron(VxR,VyyI) + np.kron(VxI,VyyR)
      V3R = np.moveaxis(VR.reshape(6*(2,)),(0,3),(1,4)).reshape(8,8)
      V3I = np.moveaxis(VI.reshape(6*(2,)),(0,3),(1,4)).reshape(8,8)
      # Separate whereim into parts affected & not affected by basis change
      whereneg = whereim.copy()
      whereim[8*nV-1:] = False
      whereim[:8*nV-1] ^= True
      whereneg[:8*nV-1] = True
      if (p-q)%8==0 and q!=0:
        # Need to flip last generator (change sx to -i*sy)
        assert whereneg[-1] == False
        generators[-1,...,1] *= -1
        whereneg[-1] = True
      assert sum(whereneg) == q
      # Separate generators into real & imaginary
      gshape = [ngen] + 2*((nV-1)*[16] + [8,2**ndrest])
      genR = generators.reshape(*gshape)
      genI = genR.copy()
      genR[whereim] = 0
      genI[~whereim] = 0
      # Apply V^T on left
      for i in range(nV-1):
        genR,genI = np.tensordot(genR,V4R,(1,0))-np.tensordot(genI,V4I,(1,0)),\
                    np.tensordot(genI,V4R,(1,0))+np.tensordot(genR,V4I,(1,0))
      genR,genI = np.tensordot(genR,V3R,(1,0))-np.tensordot(genI,V3I,(1,0)),\
                  np.tensordot(genI,V3R,(1,0))+np.tensordot(genR,V3I,(1,0))
      # Apply V^* on right
      for i in range(nV-1):
        genR,genI = np.tensordot(genR,V4R,(2,0))+np.tensordot(genI,V4I,(2,0)),\
                    np.tensordot(genI,V4R,(2,0))-np.tensordot(genR,V4I,(2,0))
      genR,genI = np.tensordot(genR,V3R,(2,0))+np.tensordot(genI,V3I,(2,0)),\
                  np.tensordot(genI,V3R,(2,0))-np.tensordot(genR,V3I,(2,0))
      assert not np.any(genI)
      genR = np.moveaxis(genR,(1,2),(1+nV,2+2*nV))
      generators = genR.astype(REALTYPE)/2**(3*nV-1)
    # Sort generators based on signature
    generators = np.concatenate([generators[~whereneg],generators[whereneg]],axis=0)
  else:
    # Complex or quaternionic
    generators = generators.astype(CPLXTYPE)
    # Match signature
    whereim ^= p*[False]+q*[True]
    generators[whereim,...] *= 1j
  # Repeat as necessary
  n2d = 1 + (ngen%2)
  generators = np.broadcast_to(generators,(n2d,)+generators.shape).copy()
  if FSI == 0:
    # Irreps should be dual
    generators[1,whereim,...] *= -1
  elif n2d == 2:
    # Irreps are not dual: just invert last generator
    generators[1,-1,...] *= -1
  # Flatten qubits
  d = 2**dims
  generators = generators.reshape(n2d,ngen,d,d)
  if testgens:
    return generators
  # Obtain representation on all qubits
  I = np.identity(d,dtype=generators.dtype)
  I = np.broadcast_to(I,(n2d,d,d))
  rep = np.stack([I,generators[:,-1,...]],axis=1)
  for igen in range(ngen-2,-1,-1):
    # Add product with generator
    # Preserve first (irrep) index
    gtimes = np.einsum('i...kl,ilm->i...km',rep,generators[:,igen,...])
    rep = np.stack([rep,gtimes],axis=1)
  # Add center
  rep = np.swapaxes(np.tensordot([1,-1],rep,0),0,1)
  reps.extend(rep.reshape(n2d,N,d,d))
  return FiniteGroup(f'Gamma{p}_{q}',reps)
