# Defining group & representation structure
# NOMENCLATURE CONVENTION - in decomposing a product representation,
#   "fusion coefficients" give degeneracy of irreps in decomposition
#   "Clebsch-Gordan coefficients" give change-of-basis from product to sum
from .npstack import np,RNG,linalg,safesvd,sparse
from abc import ABCMeta, abstractmethod
import itertools
import functools
from . import tensors
from . import links
from . import config

REALTYPE = np.float128
CPLXTYPE = np.complex256
pi = REALTYPE('3.1415926535897932384626433832795029')
sqrthalf = np.sqrt(REALTYPE(.5))
# Not all functions used are compatible with higher-precision - use mpmath?

class GroupType(ABCMeta):
  _registry = {}

  def __call__(cls, *args, **kwargs):
    label = cls.get_identifier(*args)
    if label not in cls._registry:
      obj = super(GroupType,cls).__call__(*args,**kwargs)
      cls._registry[label] = obj
      obj.__setattr__('label',label)
      obj.label = label
    return cls._registry[label]


class Group(metaclass=GroupType):
  """Implementation of group, subject to the conditions that finite-dimensional
  irreps
  * are unitary
  * follow an equivalent of Schur's Lemma
  * are discretely indexed (any hashable index)
  Groups are singleton in the sense that they are given a unique identifier
    and retrieved on calls to __new__
  The following functionals should be implemented to yield necessary data
  for the group:
  - _firrep returns representation matrix of multidimensional irreps
  - _firrep1d returns phase for 1-dimensional irrep
  - _fdual returns dual representation
    * for quaternionic representations maps together a pair of isomorphic
      reps
    * stored in _duals
  - _fdim determines dimensions of irrep
    * stored in _dims
  - _ffusion maps (irrep, irrep) to iterable of
      (irrep, sequence of Clebsch-Gordan arrays where applicable)
    * Irrep list stored in _fusion
    * Clebsch-Gordan coefficients stored in _CG
    * (CG coefficients here used to refer to elementary intertwiners as in
      standard SU2 terminology)
  - indicate yields a modified Frobenius-Schur indicator
  - _fS, if set, maps quaternionic irreps to the "charge-conjugation" matrix
      S such that S^H R_g S = R_g*
    * stored in _Ss
  - isrep identifies whether irrep labels are good
  Additionally:
  - triv identifies the trivial irrep
  - quaternionic is a boolean variable indicating whether or not the group
    contains quaternionic/pseudoreal representations
  """

  def __init__(self):
    self._duals = {}
    self._dims = {}
    self._fusion = {}
    self._CG = {}
    if self.quaternionic:
      self._Ss = {}

  @classmethod
  @abstractmethod
  def get_identifier(cls, *args):
    """Return identifier which should uniquely specify the group""" 
    pass

  def __repr__(self):
    #By default, use identifier of singleton class
    return '<%s Group>'%self.label

  @abstractmethod
  def isrep(self):
    pass

  @property
  def triv(self):
    """By default expect the trivial irrep to be 0"""
    return 0

  @property
  def quaternionic(self):
    """By default expect no quaternionic irreps"""
    return False

  def verify_rep(self, r):
    if not self.isrep(r):
      raise ValueError('Group %s expected representation, got %s' \
        %(self.__repr__(), r))
  
  def dim(self, r):
    """Retrieve dimension of irrep r"""
    if r not in self._dims:
      self.verify_rep(r)
      self._dims[r] = self._fdim(r)
    return self._dims[r]

  @abstractmethod
  def _fdim(self, r):
    pass

  def S(self, r):
    """Retrieve charge-conjugation matrix for irrep r"""
    if r not in self._Ss:
      self.verify_rep(r)
      assert self.indicate(r) < 0
      self._Ss[r] = self._fS(r)
    return self._Ss[r]

  def _fS(self, r):
    raise NotImplementedError

  def fusionCG(self, r1, r2):
    """Retrieve fusion decomposition, Clebsch-Gordan coefficients
    (elementary intertwiners) for r1 x r2"""
    if (r1,r2) not in self._fusion:
      self.verify_rep(r1)
      self.verify_rep(r2)
      CG0 = {}
      CG1 = {}
      fusion = []
      for k,R in self._ffusion(r1,r2):
        if R is None:
          # Identity transformation; one or both should be 1d
          CG0[k] = None
          CG1[k] = None
          fusion.append((k,1))
        else:
          fusion.append((k,R.shape[3]))
          CG0[k] = R
          if r1 != r2:
            CG1[k] = R.transpose((1,0,2,3))
      self._CG[r1,r2] = CG0
      self._fusion[r1,r2] = fusion
      if r1 != r2:
        self._CG[r2,r1] = CG1
        self._fusion[r2,r1] = fusion
      return fusion, CG0
    else:
      return self._fusion[r1,r2], self._CG[r1,r2]

  @abstractmethod
  def _ffusion(self, r1, r2):
    pass
  
  def fusion(self, r1, r2):
    return self.fusionCG(r1,r2)[0]

  def CG(self, r1, r2):
    return self.fusionCG(r1,r2)[1]

  def dual(self, r):
    """Retrieve label of dual representation of r"""
    if r not in self._duals:
      self._duals[r] = self._fdual(r)
    return self._duals[r]

  @abstractmethod
  def _fdual(self, r):
    pass

  def irrepmatrix(self, r, g):
    """Retrieve matrix representation of irrep r at group element g"""
    self.verify_rep(r)
    if self.indicate(r) == -2:
      return self._firrep(self.dual(r),g).conj()
    elif self.dim(r) == 1:
      if self.indicate(r) == 1:
        return np.array([[np.real(self._firrep1d(r,g))]],dtype=REALTYPE)
      else:
        return np.array([[self._firrep1d(r,g)]],dtype=CPLXTYPE)
    else:
      return self._firrep(r,g)

  @abstractmethod
  def _firrep(self, r, g):
    pass

  @abstractmethod
  def _firrep1d(self, r, g):
    pass

  @abstractmethod
  def indicate(self, r):
    """Modified Frobenius-Schur indicator:
    +1 for real irrep
     0 for complex irrep
    -1 for quaternionic irrep (primary representation)
    -2 for quaternionic irrep (conjugate representation)"""
    pass

  def sumdims(self, rep):
    """Returns the sum of dimensions for a list-based representation
    (tuples (k,n) irrep label x multiplicity)"""
    return sum(self.dim(k)*n for k,n in rep)

  def Haargen(self):
    """Randomly sample from valid group elements using Haar measure. Optional"""
    raise NotImplementedError

  def compose(self, g, h):
    """Return element produced by the composition of g and h. Optional"""
    raise NotImplementedError

  def product(self, H):
    """Direct product with H"""
    return ProductGroup(self, H)
  
  def complex_pairorder(self, r):
    """For a complex irrep, give (fixed but arbitrary)
    ordering relative to dual irrep
      (i.e. return r > r* for some ordering ">" on complex irreps)
    Of use in product groups, where for s quaternionic need to
      select two out of rxs, r*xs, rxs*, r*xs*
      (so if r > r*, we use rxs and r*xs*)
    By default compare hash values
    These are not fully reliable because they may collide; for full
      reliability it is recommended to override"""
    h = hash(r)
    hd = hash(self.dual(r))
    assert h != hd
    return h > hd

  def __getnewargs__(self):
    """By default assume singleton"""
    return ()


class U1(Group):
  """Returns U(1). Representations are labeled by integers,
  elements by floats"""

  @classmethod
  def get_identifier(cls):
    return 'U1'

  def isrep(self, r):
    return isinstance(r, int)

  def _fdim(self, r):
    return 1

  def _ffusion(self, r1, r2):
    yield (r1 + r2,None)

  def _fdual(self, r):
    return -r

  def _firrep(self, r, theta):
    return

  def _firrep1d(self, r, theta):
    return np.exp(1j*r*theta)

  def indicate(self, r):
    if r == 0:
      return 1
    else:
      return 0

  def Haargen(self):
    return 2*pi*RNG.random()

  def compose(self, g, h):
    return (g+h)%(2*pi)

  def complex_pairorder(self, r):
    return r > 0


class O2(Group):
  """Returns O(2). Representations are labeled by integers with parity irrep
  labeled by -1; elements are (theta, 1) for rotations and (theta,-1) for
  reflections."""
  
  @classmethod
  def get_identifier(cls):
    return 'O2'

  def isrep(self, r):
    return isinstance(r, int) and r >= -1

  def _fdim(self, r):
    if r < 1:
      return 1
    else:
      return 2

  def _ffusion(self, r1, r2):
    if r1 < 1:
      if r2 < 1:
        # 1 x 1
        yield (r1^r2, None)
      elif r1 == 0:
        # Trivial transformation
        yield (r2, None)
      else:
        # Alternating transformation
        yield (r2, np.expand_dims([[0,1],[-1,0]],(0,3)))
      return
    elif r2 == 0:
      return [(r1, None)]
    elif r2 == -1:
      return [(r1, np.expand_dims([[0,1],[-1,0]],(1,3)))]
    # 2Dx2D: first r = r1+r2
    yield (r1+r2, sqrthalf*np.expand_dims([[[1,0],[0,1]],[[0,1],[-1,0]]],3))
    # r1-r2
    if r1 > r2:
      yield (r1 - r2, sqrthalf*np.expand_dims([[[1,0],[0,-1]],[[0,1],[1,0]]],3))
    elif r1 < r2:
      yield (r2 - r1, sqrthalf*np.expand_dims([[[1,0],[0,1]],[[0,-1],[1,0]]],3))
    else:
      # Decompose into trivial+alternating
      yield (0, sqrthalf*np.expand_dims([[1,0],[0,1]],(2,3)))
      yield (-1,sqrthalf*np.expand_dims([[0,1],[-1,0]],(2,3)))

  
  def _fdual(self, r):
    return r

  def _firrep1d(self, r, g):
    if r == -1: # Alternating
      return g[1]
    elif r == 0:
      return 1

  def _firrep(self, r, g):
    th,f = g
    s,c = np.sin(r*th),np.cos(r*th)
    if f == -1:
      return np.array([[c,s], [s,-c]],dtype=REALTYPE)
    else:
      return np.array([[c,s], [-s,c]],dtype=REALTYPE)

  def indicate(self, r):
    return 1

  def Haargen(self):
    return (2*pi*RNG.random(), 2*RNG.integers(2)-1)

  def compose(self, g, h):
    return ((h[1]*g[0] + h[0])%(2*pi), h[1]*g[1])


class SU2(Group):
  def __init__(self):
    super().__init__()
    self._Amats = {}

  @classmethod
  def get_identifier(cls):
    return 'SU2'

  @property
  def quaternionic(self):
    return True

  def isrep(self, r):
    return isinstance(r, int) and (r >= 0 or r % 2 == 1)

  # Specialized matrices used in SU2 calculations:
  def realbasis(self, r):
    """Matrix to transform from the spin-z basis to the real basis
    in the integer spin case"""
    if r == 0:
      return np.array([[1]])
    elif r in self._Amats:
      return self._Amats[r]
    j = r//2
    A = np.zeros((r+1, r+1), dtype=CPLXTYPE)
    if j % 2:
      A[j,j] = 1j
      sgn = -1
    else:
      A[j,j] = 1
      sgn = 1
    for n in range(1,j+1):
      sgn *= -1
      A[j-n,j-n] = CPLXTYPE(sqrthalf)
      A[j-n,j+n] = CPLXTYPE(sgn*sqrthalf)
      A[j+n,j-n] = 1j*sqrthalf
      A[j+n,j+n] = -1j*sgn*sqrthalf
    self._Amats[r] = A
    return A

  def toreal(self, M, r):
    """If r is a real irrep, transform M from spin-z to real basis; otherwise
    return without change"""
    if r % 2 or r == 0:
      return M
    A = self.realbasis(r)
    return A.dot(M).dot(A.T.conj())

  def spinmats(self, r):
    # Spin matrices (generators); (0,1,2)=(x,y,z)
    # Given as anti-Hermitian
    Ss = np.zeros((r+1,r+1,3), dtype=CPLXTYPE)
    Ss[:,:,2] = np.diag(range(r,-r-1,-2))
    Ss[:,:,2] /= 2

    S0 = np.diag(np.sqrt(np.array([m*(r-m+1) for m in range(1,r+1)],
      dtype=REALTYPE))/2)
    Ss[:r,1:,0] = S0
    Ss[1:,:r,0] += S0
    Ss[:r,1:,1] = -1j*S0
    Ss[1:,:r,1] += 1j*S0
    if r % 2 == 0:
      A = self.realbasis(r)
      return np.imag(np.tensordot(np.tensordot(A,Ss,1),
                A.conj(),(1,1)).transpose((0,2,1)))
    else:
      return -1j*Ss

  def _fS(self, r):
    return (-1)**(r<0) * np.flipud(np.diag([(-1)**m for m in range(abs(r)+1)]))

  def _fdim(self, r):
    return abs(r)+1

  def _fdual(self, r):
    if r%2:
      return -r
    else:
      return r

  def indicate(self, r):
    if r < 0:
      return -2
    elif r % 2 == 1:
      return -1
    else:
      return 1

  def _firrep(self, r, g):
    return linalg.expm(self.spinmats(r).dot(g))

  def _firrep1d(self, r, g):
    return 1

  def _ffusion(self, r1, r2):
    if r1 == 0:
      if r2 == 0:
        yield (0, None)
      elif r2 < 0:
        yield (-r2, np.expand_dims(-self.S(r2),(0,3)))
      else:
        yield (r2, None)
      return
    elif r2 == 0:
      if r1 < 0:
        yield (-r1, np.expand_dims(-self.S(r1),(1,3)))
      else:
        yield (r1, None)
      return
    s1 = abs(r1)
    s2 = abs(r2)
    # Revert to the case of s1<s2
    smin = min(s1,s2)
    smax = max(s1,s2)
    for s in range(smax-smin,smax+smin+1,2):
      Wk = np.zeros((smin+1,smax+1,s//2+1),dtype=REALTYPE)
      # 3 parts contribute to W:
      # sum part unique to each element of W,
      # product of m1,m2,m dependencies,
      # constant factor
      jdiff = (smax - smin - s)//2
      jsum = (smax + smin - s)//2
      for n1 in range(smin+1): #j1-m1
        for n2 in range(smax+1): #j2-m2
          # m in product dof determined by selection
          n = n1 + n2 - jsum
          if n < 0 or n > s//2:
            continue
          k0 = max(0, jdiff+n1, jsum-n2)
          k1 = min(jsum, n1, smax-n2)+1
          for k in range(k0, k1):
            Wk[n1,n2,n] += (-1)**k/REALTYPE(np.math.factorial(jsum-k) \
                                  * np.math.factorial(n1-k) \
                                  * np.math.factorial(smax-n2-k) \
                                  * np.math.factorial(k-n1-jdiff) \
                                  * np.math.factorial(k+n2-jsum) \
                                  * np.math.factorial(k))
      # Factor varying independently by m
      arr2 = np.sqrt(np.array([np.math.factorial(n2) for n2 in range(smax+1)],
        dtype=REALTYPE))
      arr1 = arr2[:smin+1].copy()
      arr3 = arr2[:s//2+1].copy()
      arr1 *= arr1[::-1]
      arr2 *= arr2[::-1]
      arr3 *= np.sqrt(np.array([np.math.factorial(s-n) for n in range(s//2+1)],
        dtype=REALTYPE))
      Wk *= np.tensordot(np.tensordot(arr1,arr2,0),arr3,0)

      Wk *= np.sqrt(REALTYPE((s+1)*np.math.factorial(jsum) \
             * np.math.factorial((s-smin+smax)//2) \
             * np.math.factorial((s+smin-smax)//2)) \
             / np.math.factorial((s+smin+smax)//2+1))
      # Negative Sz
      if s != 0:
        Wk = np.concatenate((Wk, (-1)**jsum * Wk[::-1,::-1,(s-1)//2::-1]), 2)
      # Corrections for real & conjugate-quaternionic representations
      if s1 > s2:
        Wk = Wk.transpose((1,0,2))
      if r1 < 0:
        Wk = np.tensordot(self.S(s1), Wk, (0,0))
      if r2 < 0:
        Wk = np.tensordot(self.S(s2), Wk, (0,1)).transpose((1,0,2))
      if r1 % 2 == 0:
        Wk = np.tensordot(self.realbasis(r1).conj(), Wk, (1,0))
      if r2 % 2 == 0:
        Wk = np.tensordot(self.realbasis(r2).conj(),Wk,(1,1)).transpose((1,0,2))
      if s % 2 == 0 and s != 0:
        Wk = np.tensordot(Wk, self.realbasis(s), (2,1))
        if r1%2==0 and r2%2==0:
          Wk = np.real(Wk)
      yield (s, np.expand_dims(Wk,3))

  def Haargen(self):
    p0 = 0
    # Rejection sampling on Haar measure in S^3 parametrization
    while RNG.random() >= p0:
      # Uniform distribution
      theta = RNG.random()*pi
      psi = RNG.random()*pi
      phi = 2*RNG.random()*pi
      # Calculate Haar measure
      p0 = np.sin(theta)**2 * np.sin(psi)
    return (2*theta*np.sin(psi)*np.cos(phi),
           2*theta*np.cos(psi),
           2*theta*np.sin(psi)*np.sin(phi))
  
  def compose(self, g, h):
    g = np.array(g,dtype=REALTYPE)
    h = np.array(h,dtype=REALTYPE)
    thetag = linalg.norm(g)
    thetah = linalg.norm(h)
    gnorm = g/thetag
    hnorm = h/thetah
    sg = np.sin(thetag/2)
    cg = np.cos(thetag/2)
    sh = np.sin(thetah/2)
    ch = np.cos(thetah/2)
    phi = np.arccos(cg*ch - gnorm.dot(hnorm)*sg*sh)
    gh = gnorm*sg*ch + hnorm*cg*sh + np.cross(gnorm,hnorm)*sg*sh
    return tuple(2*phi/np.sin(phi)*gh)


def takagi_stable(S):
  """Takagi decomposition of symmetric unitary S, at expected precision
  Produces unitary matrix U s.t. S=UU^T
  Required to derive real form of an irrep from its charge-conjugation matrix
  Algorithm based on stackexchange comment
    https://math.stackexchange.com/a/2026299
  with changes made to ensure high precision, determinism, & stability"""
  #A = S.copy()
  A = CPLXTYPE(S)
  d = S.shape[0]
  U = np.identity(d,dtype=CPLXTYPE)

  for n in range(d):
    # Start with vector y s.t. Ay*=y
    # For x = e0, either Ax+x or i(Ax-x)
    #   Triangle inequality: at least one must have norm >= 1
    #   Move slightly to avoid ambiguity in the case norm == 1
    assert A.shape[0] == d-n
    y0 = A[:,0].copy()
    y0[0] += 1
    if linalg.norm(y0) < REALTYPE('.95'):
      y0 = 1j*A[:,0]
      y0[0] -= 1j
    y = y0/linalg.norm(y0)
    Vi = np.identity(d-n, dtype=CPLXTYPE)
    Vi[:,0] = y
    # Produce unitary matrix with Gram-Schmidt
    for i in range(1,d-n):
      v = Vi[:,i]
      for k in range(i):
        v -= Vi[:,k].dot(Vi[:,k].conj().dot(v))
      Vi[:,i] = v/linalg.norm(v)
    A = Vi.T.conj().dot(A).dot(Vi.conj())
    A = A[1:,1:]
    U[:,n:] = U[:,n:].dot(Vi)
  return U

class ProductGroup(Group):
  """Direct product of two groups, G and H"""

  def __init__(self, G, H, realforms={}):
    # realforms are unitary matrices giving real forms of
    # quaternionicxquaternionic irreps, which is computed from
    # charge-conjugation matrices by Takagi decomposition;
    # this determines basis of irrep so memoization must be saved
    self.G = G
    self.H = H
    super().__init__()
    if isinstance(H, ProductGroup):
      self.__factors = 1+H.__factors
    else:
      self.__factors = 2
    self._realforms = realforms

  @classmethod
  def get_identifier(cls, G, H):
    return G.label + ' x ' + H.label

  def _split_rep(self, r):
    # Divide representation into representations corresponding to G and H,
    # depending on whether or not H is product itself
    # Also applies to group elements
    if self.__factors > 2:
      return r[0], r[1:]
    else:
      return r

  def _join_rep(self, rg, rh):
    # Inverse of _split_rep
    if self.__factors == 2:
      return (rg, rh)
    else:
      return (rg,) + rh

  def isrep(self, r):
    if not isinstance(r, tuple) or len(r) != self.__factors:
      return False
    rg,rh = self._split_rep(r)
    if not (self.G.isrep(rg) and self.H.isrep(rh)):
      return False
    ig = self.G.indicate(rg)
    ih = self.H.indicate(rh)
    if ig < 0:
      if ih == 0:
        # Quaternionic x complex: use ordering on complex irreps to determine
        # validity - if rg > rg* and rh is the "primary" rep,
        # then rg x rh and rg* x rh* are valid
        return (ig == -2) ^ self.H.complex_pairorder(rh)
      elif ih < 0:
        # Quaternionic x quaternionic = real: only -1x-1 is valid
        return ig == -1 and ih == -1
      else:
        return True
    elif ih < 0 and ig == 0:
      return (ih == -2) ^ self.G.complex_pairorder(rg)
    return True

  @property
  def triv(self):
    return self._join_rep(self.G.triv, self.H.triv)

  @property
  def quaternionic(self):
    return self.G.quaternionic or self.H.quaternionic

  def _fdim(self, r):
    rg,rh = self._split_rep(r)
    return self.G.dim(rg)*self.H.dim(rh)

  def indicate(self, r):
    # In all (permitted) cases, modified indicator is still the product
    rg,rh = self._split_rep(r)
    return self.G.indicate(rg)*self.H.indicate(rh)

  def _fS(self, r):
    rg,rh = self._split_rep(r)
    if self.G.indicate(rg) < 0:
      return linalg.kron(self.G.S(rg), np.identity(self.H.dim(rh)))
    elif self.H.indicate(rh) < 0:
      return linalg.kron(np.identity(self.G.dim(rg)), self.H.S(rh))

  def _fdual(self, r):
    rg,rh = self._split_rep(r)
    if self.G.indicate(rg) == -1 and self.H.indicate(rh) == -1:
      # Real: unchanged
      return r
    else:
      return self._join_rep(self.G.dual(rg), self.H.dual(rh))

  def quaternionic_to_real(self, rg, rh):
    """Change-of-basis from a pair of quaternionic irreps to its real form
    (U R_qq U^H = R_r)""" 
    if (rg,rh) not in self._realforms:
      S = np.kron(self.G.S(rg), self.H.S(rh))
      self._realforms[rg,rh] = takagi_stable(S).T.conj()
    return self._realforms[rg,rh]

  def _firrep(self, r, g):
    rg,rh = self._split_rep(r)
    eg,eh = self._split_rep(g)
    gd1 = (self.G.dim(rg) == 1)
    hd1 = (self.H.dim(rh) == 1)
    if not gd1:
      if self.G.indicate(rg) == -2:
        MG = self.G._firrep(self.G.dual(rg), eg).conj()
      else:
        MG = self.G._firrep(rg, eg)
    if not hd1:
      if self.H.indicate(rh) == -2:
        MH = self.H._firrep(self.H.dual(rh), eh).conj()
      else:
        MH = self.H._firrep(rh, eh)
    if gd1:
      return self.G._firrep1d(rg, eg) * MH
    elif hd1:
      return self.H._firrep1d(rh, eh) * MG
    mat = np.kron(MG, MH)
    if self.G.indicate(rg) == -1 and self.H.indicate(rh) == -1:
      U = self.quaternionic_to_real(rg, rh)
      mat = np.real(U.dot(mat).dot(U.T.conj()))
    return mat

  def _firrep1d(self, r, g):
    # Product of 2 1d irreps
    rg,rh = self._split_rep(r)
    eg,eh = self._split_rep(g)
    return self.G._firrep1d(rg,eg) * self.H._firrep1d(rh,eh)

  def _ffusion(self, r1, r2):
    # Product-group fusion rules:
    # All resulting pairs of irreps are unique
    r1g,r1h = self._split_rep(r1)
    r2g,r2h = self._split_rep(r2)
    i1g = self.G.indicate(r1g)
    i2g = self.G.indicate(r2g)
    i1h = self.H.indicate(r1h)
    i2h = self.H.indicate(r2h)
    d1g = self.G.dim(r1g)
    d2g = self.G.dim(r2g)
    d1h = self.H.dim(r1h)
    d2h = self.H.dim(r2h)
    d1 = d1g*d1h
    d2 = d2g*d2h
    q2r1 = (i1g == -1 and i1h == -1)
    if q2r1:
      U1 = self.quaternionic_to_real(r1g,r1h).conj()
    q2r2 = (i2g == -1 and i2h == -1)
    if q2r2:
      U2 = self.quaternionic_to_real(r2g,r2h).conj()
    fg,Wg = self.G.fusionCG(r1g, r2g)
    fh,Wh = self.H.fusionCG(r1h, r2h)
    for kg,ng in fg:
      Wkg = Wg[kg]
      ikg = self.G.indicate(kg)
      for kh,nh in fh:
        Wkh = Wh[kh]
        ikh = self.H.indicate(kh)
        q2r = (ikg == -1 and ikh == -1)
        if q2r:
          U = self.quaternionic_to_real(kg,kh).T
        if ikg == -1 and ikh == 0 and not self.H.complex_pairorder(kh):
          qcflip = 1
          k = self._join__rep(self.G.dual(kg),kh)
        elif ikg == 0 and ikh == -1 and not self.G.complex_pairorder(kg):
          qcflip = 2
          k = self._join_rep(kg, self.H.dual(kh))
        else:
          qcflip = 0
          k = self._join_rep(kg,kh)
        d = self.dim(k)
        if Wkg is None:
          if Wkh is None:
            assert d1*d2 == d
            if (d1h == 1 and d2g == 1) and not (q2r1 or q2r2 or q2r or qcflip):
              yield k, None
            else:
              if q2r:
                W = U.reshape((d1g,d2g,d1h,d2h,d))
              elif qcflip == 1:
                W = np.tensordot(self.G.S(kg).conj(), np.identity(d1h*d2h), 0)
                W = W.transpose((1,2,0,3)).reshape((d1g,d2g,d1h,d2h,d))
              elif qcflip == 2:
                W = np.tensordot(np.identity(d1g*d2g), self.G.S(kh).conj(), 0)
                W = W.transpose((0,3,1,2)).reshape((d1g,d2g,d1h,d2h,d))
              else:
                W = np.identity(d,dtype=REALTYPE).reshape((d1g,d2g,d1h,d2h,d))
              W = W.transpose((0,2,1,3,4)).reshape((d1,d2,d,1))
              if q2r1:
                W = np.tensordot(U1, W, (1,0))
              if q2r2:
                W = np.tensordot(U2, W, (1,1)).transpose((1,0,2,3))
              yield k, W
            return
          else:
            Wkg = np.identity(d1g*d2g,dtype=REALTYPE)
            Wkg = Wkg.reshape((d1g,d2g,d1g*d2g,1))
        elif Wkh is None:
          Wkh = np.identity(d1h*d2h,dtype=REALTYPE).reshape((d1h,d2h,d1h*d2h,1))
        if qcflip == 1:
          Wkg = np.tensordot(Wkg,self.G.S(kg).conj(),(2,0)).transpose((0,1,3,2))
        elif qcflip == 2:
          Wkh = np.tensordot(Wkh,self.H.S(kh).conj(),(2,0)).transpose((0,1,3,2))
        W = np.tensordot(Wkg,Wkh,0).transpose((0,4,1,5,2,6,3,7))
        W = W.reshape((d1,d2,-1,ng*nh))
        if q2r1:
          W = np.tensordot(U1, W, (1,0))
        if q2r2:
          W = np.tensordot(U2, W, (1,1)).transpose((1,0,2,3))
        if q2r:
          W = np.tensordot(W, U, (2,0)).transpose((0,1,3,2))
        yield k,W
  
  def Haargen(self):
    return self._join_rep(self.G.Haargen(), self.H.Haargen())

  def compose(self, g, h):
    g1,h1 = self._split_rep(g)
    g2,h2 = self._split_rep(h)
    return self._join_rep(self.G.compose(g1,g2), self.H.compose(h1,h2))

  def complex_pairorder(self, r):
    # Return ordering for first complex irrep encountered
    rg,rh = self._split_rep(r)
    if self.indicate(rg) == 0:
      return self.G.complex_pairorder(rg)
    else:
      return self.H.complex_pairorder(rh)

  def product(self, H2):
    # Associativity of product: H may be compound, G should be singular
    return self.G.product(self.H.product(H2))

  def __getnewargs_ex__(self):
    return (self.G, self.H), {'realforms':self._realforms}


class GroupDerivedType(ABCMeta):
  """Metaclass for types which each correspond to a single Group instance
  Must have a field regname which names the attribute of the Group instance that
    will point to the derived class instance"""
  def __new0__(mcl, name, bases, nmspc):
    if 'group' in nmspc:
      # Should be 1 registered type corresponding to group
      if '_regname' in nmspc:
        regname = nmspc['_regname']
      else:
        regname = getattr(bases[0], '_regname')
      assert not hasattr(nmspc['group'], regname)
      if hasattr(nmspc['group'], regname):
        # Already exists
        return getattr(nmspc['group'], regname)
      else:
        cls = super().__new__(mcl, name, bases, nmspc)
        # Register (should this be done in __init__ instead?)
        setattr(nmspc['group'], regname)
    else:
      cls = super().__new__(mcl, name, bases, nmspc)
    return cls
  
  def __init0__(cls, name, bases, nmspc):
    if hasattr(cls,'group') and hasattr(cls,'_required_types'):
      # Construct other GroupDerivedTypes required for operation
      # Do not need to save - will automatically register themselves with
      #   the group
      for cls2 in cls._required_types:
        cls2.derive(cls.group)

  def derive(cls, group):
    """Produce subclass corresponding to group"""
    if hasattr(group, cls._regname):
      return getattr(group, cls._regname)
    subcl = GroupDerivedType(str(group.label)+'_'+cls.__name__, (cls,),
          {'group':group, '__reduce__': lambda self: (cls.derive, (group,)) })
    setattr(group, cls._regname, subcl)
    if hasattr(cls, '_required_types'):
      for cls2 in cls._required_types:
        cls2.derive(group)
    return subcl


class SumRepresentation(links.VectorSpace,metaclass=GroupDerivedType):
  """Decomposed representation of a group, as indentifier for indices of
  gauge-invariant tensors"""

  _regname='SumRep'

  def __init__(self, replist):
    self._decomp = replist
    self._dimension = self.group.sumdims(replist)

  def dual(self):
    return self.__class__([(self.group.dual(k), n) for k,n in self._decomp])

  def __eq__(self, W):
    return self.__class__ is W.__class__ and self._decomp == W._decomp
  
  def __xor__(self, W):
    if self.__class__ is not W.__class__ or len(self._decomp) != len(W._decomp):
      return False
    for i in range(len(self._decomp)):
      if self._decomp[i][1] != W._decomp[i][1] or \
          self.group.dual(self._decomp[i][0]) != W._decomp[i][0]:
        return False
    return True

  @property
  def real(self):
    return all(self.group.indicate(k) == 1 for k,n in self._decomp)

  @property
  def qprimary(self):
    """Contains any 'primary' quaternionic irreps"""
    return any(self.group.indicate(k) == -1 for k,n in self._decomp)

  @property
  def qsecond(self):
    """Contains any 'secondary' quaternionic irreps"""
    return any(self.group.indicate(k) == -2 for k,n in self._decomp)

  def matrix(self, g):
    """Retrieve matrix for symmetry transformation of group element g on
    representation"""
    mat = np.zeros((self.dim,self.dim),
        dtype=(REALTYPE if self.real else CPLXTYPE))
    idx = 0
    for k,n in self._decomp:
      d = self.group.dim(k)
      R = self.group.irrepmatrix(k,g)
      for i in range(n):
        mat[idx:idx+d,idx:idx+d] = R
        idx += d
    return mat

  def __iter__(self):
    """Iterate over (irrep, degeneracy) pairs"""
    return iter(self._decomp)

  def __contains__(self,k):
    """Irrep is contained in decomposition (w/ nonzero degeneracy)"""
    for k1,n in self._decomp:
      if k1 == k:
        return bool(n)
    return False

  def idx_of(self, k):
    """Index at which the subspace that transforms as irrep k starts"""
    i0 = 0
    for k1,n in self._decomp:
      if k1 == k:
        return i0
      i0 += n*self.group.dim(k1)

  def degen_of(self, k):
    """Degeneracy of irrep in representation"""
    for k1,n in self._decomp:
      if k1 == k:
        return n
    return 0
