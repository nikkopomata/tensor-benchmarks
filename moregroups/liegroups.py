from quantrada.groups import *
import sympy
import scipy.sparse
import itertools,functools
from collections import defaultdict
from fractions import Fraction
from math import gcd,lcm,factorial
from .rationallinalg import invert as inverse_rational
from .rationallinalg import rones,diag,rzeros,rarray,fnorm
from scipy import linalg

no_strict_checks = False # Bypass potentially time-consuming assertions
_fix_3j_symmetries = True

from sympy.simplify.radsimp import radsimp, rad_rationalize, collect_sqrt

def mat_collect(A):
  return A.applyfunc(collect_sqrt)

def denestplus(expr):
  return sympy.expand_mul(sympy.sqrtdenest(expr))

def dynkin_to_cartan(series, rank):
  # Obtain Cartan matrix given label of Dynkin diagram
  # TODO is there a canonical ordering?
  A = 2*np.eye(rank,dtype=int)
  if series == 'A':
    # Off-diagonal elements all -1
    A[range(rank-1),range(1,rank)] = -1
    A[range(1,rank),range(rank-1)] = -1
  elif series == 'B':
    A[range(rank-2),range(1,rank-1)] = -1
    A[range(1,rank-1),range(rank-2)] = -1
    # Final root is shorter
    A[-2,-1] = -1
    A[-1,-2] = -2
  elif series == 'C':
    A[range(rank-2),range(1,rank-1)] = -1
    A[range(1,rank-1),range(rank-2)] = -1
    # Final root is longer
    A[-2,-1] = -2
    A[-1,-2] = -1
  elif series == 'D':
    A[range(rank-2),range(1,rank-1)] = -1
    A[range(1,rank-1),range(rank-2)] = -1
    # Edge between final root & 3rd-to-last root
    A[-3,-1] = -1
    A[-1,-3] = -1
  elif series == 'E':
    A[range(rank-2),range(1,rank-1)] = -1
    A[range(1,rank-1),range(rank-2)] = -1
    # -4 -- -3 -- -2 (covered) & -4 -- -1
    A[-4,-1] = -1
    A[-1,-4] = -1
    # Conforms to DiFrancesco's ordering except E7 (WTF is going on there)
  elif series == 'F':
    assert rank == 4
    A[0,1] = -1
    A[1,0] = -1
    A[1,2] = -1
    A[2,1] = -2 ###
    A[2,3] = -1
    A[3,2] = -1
  elif series == 'G':
    assert rank == 2
    A[0,1] = -3
    A[1,0] = -1
  return A

def gramschmidt_coefficients(gram):
  # Perform Gram-Schmidt procedure given Gram matrix
  # Returns change-of-basis matrix A, s.t. A[j,:] are the coefficients of
  # an orthonormal basis in the target space with respect to the original set
  gram = sympy.Matrix(gram)
  assert no_strict_checks or gram.is_hermitian
  assert no_strict_checks or gram.is_positive_semidefinite
  N = gram.rows
  cob = sympy.zeros(N,1)
  i0 = 0
  # Find first nontrivial element
  while gram[i0,i0] == 0:
    i0 += 1
    if i0 == N:
      raise ValueError('Expected nontrivial Gram matrix')
  cob[i0,0] = 1/sympy.sqrt(gram[i0,i0])
  ngram = 1
  for i in range(i0+1,N):
    if gram[i,i] == 0:
      continue
    coeff = sympy.zeros(N,1)
    coeff[i] = 1
    coeff[:i,0] = symdot(-cob[:i,:],symdot(cob[:i,:].T,gram[:i,i]))
    #gc = symdot(gram,coeff)
    #normsq = coeff.dot(gc,True)
    normsq = sympy.expand(gram[i,i] + gram[i,:i].dot(coeff[:i,0]))
    norm = sympy.expand(sympy.sqrtdenest(sympy.sqrt(normsq)))
    if norm == 0:
      continue
    normnum,normdenom = rad_rationalize(1,norm)
    cob = cob.col_insert(ngram,(normnum*coeff/normdenom).expand())
    ngram += 1
  assert no_strict_checks or (cob.adjoint()*cob).det().simplify() != 0
  assert no_strict_checks or (cob.adjoint()*gram*cob - sympy.eye(cob.cols)).is_zero_matrix
  return cob

def symdot(A,B):
  if not isinstance(A,sympy.MatrixBase):
    A = sympy.Matrix(A)
  if not isinstance(B,sympy.MatrixBase):
    B = sympy.Matrix(B)
  return A.multiply(B,True)

def commutator(A,B):
  return A.dot(B) - B.dot(A)

def ituple(arr):
  # This one's for numpy's integer arrays
  return tuple(int(i) for i in arr)

def negate(mu):
  return tuple(-mui for mui in mu)

def _BCH_dynkin_coefficient_XY(xystring):
  """Coefficient of sequence of Xs and Ys (ending in implicit [X,Y]) in
  Dynkin expansion of BCH formula
  Argument is tuple of 0s (representing X) and 1s (representing Y), with final
  (0,1) omitted (also counts -(1,0))"""
  # "Minimal" grouping of xs & ys
  rs = [0]
  ss = [0]
  try:
    xyiter = iter(xystring)
    while True:
      while not next(xyiter):
        rs[-1] += 1
      ss[-1] += 1
      while next(xyiter):
        ss[-1] += 1
      rs.append(1)
      ss.append(0)
  except StopIteration:
    if xystring[-1]:
      # Ended on Y
      xyval = _BCH_dynkin_coefficient(tuple(zip(rs,ss))+((1,1),))
      ss[-1] += 1
      yxval = _BCH_dynkin_coefficient(tuple(zip(rs,ss))+((1,0),))
    else:
      # Ended on X: for YX increment last s & add (1,0), for XY increment both
      ss[-1] += 1
      yxval = _BCH_dynkin_coefficient(tuple(zip(rs,ss))+((1,0),))
      rs[-1] += 1
      xyval = _BCH_dynkin_coefficient(tuple(zip(rs,ss)))
    return Fraction(1,len(xystring)+2) * (xyval - yxval)

@functools.cache
def _BCH_dynkin_coefficient(rss):
  n0 = len(rss)
  nmaxs = [r+s for r,s in rss]
  inner_coeff = 0
  for nis in itertools.product(*[range(1,nmax+1) for nmax in nmaxs]):
    n = sum(nis)
    term = Fraction((-1)**(n-1),n)
    for ni,(ri,si) in zip(nis,rss):
      term *= _dynkin_combinatorial(ri,si,ni)
    #print(xystring,rss,nis,term)
    inner_coeff += term
  return inner_coeff



@functools.cache
def _dynkin_combinatorial(r,s,n):
  """Combinatorial function occuring in calculation of Dynkin expansion of BCH
  Can be understood as follows: for G the free semigroup
  on letters a & b, let g in G be a^r b^s.
  Consider all ways to decompose g (note that all components will belong
  to the subset X of the form x = a^r' b^s'),
  and let f:X->R be defined by f(a^r' b^s') = 1/(r'! s'!). Then this function
  returns the sum, over all sequences {x_i} in X^n whose product evaluates to g,
  of the product of f(x_i)."""
  if r+s<n:
    # No options
    return 0
  if r+s==n:
    # Only one option -- all x are either a or b
    return 1
  if n == 1:
    # x = g
    return Fraction(1,factorial(r)*factorial(s))
  # Otherwise recurse over greater of r and s
  if r >= s:
    # Case where first x is a^r b^s' is equivalent to 1/r! times
    # result of r=0, s, n
    rv = Fraction(1,factorial(r)) * _dynkin_combinatorial(0,s,n)
    for rl in range(1,r+1):
      # Pull out x = a^rl
      rv += Fraction(1,factorial(rl)) * _dynkin_combinatorial(r-rl,s,n-1)
    return rv
  else:
    # Likewise for r,s reversed
    rv = Fraction(1,factorial(s)) * _dynkin_combinatorial(r,0,n)
    for sr in range(1,s+1):
      rv += Fraction(1,factorial(sr)) * _dynkin_combinatorial(r,s-sr,n-1)
    return rv

class SimpleLieAlgebra(Group):
  def __init__(self, series, rank):
    """Pass A-G series label & rank of Lie algebra"""
    self._CK_series = series
    self._rank = rank
    self._cartan_matrix = dynkin_to_cartan(series, rank)
    # Next: obtain simple roots
    lengthsq = 2 # It's *a* convention
    lsqs = [lengthsq]
    roots = sympy.zeros(1,rank)
    roots[0,0] = sympy.sqrt(lengthsq)
    coroots = sympy.Rational(2,lengthsq)*roots
    for i in range(1,rank):
      # Should be a unique j<i linked
      jlist, = self._cartan_matrix[:i,i].nonzero()
      assert len(jlist) == 1
      j = jlist[0]
      aij = self._cartan_matrix[i,j]
      aji = self._cartan_matrix[j,i]
      assert aij < 0 and aji < 0
      # Compute alpha on span of existing roots using existing coroots
      alpha = sympy.zeros(1,rank)
      alpha[0,:i] = symdot(coroots[:,:i].inv(),self._cartan_matrix[:i,i]).T
      # Now update lengthsq to get remaining nonzero coordinate
      assert lengthsq % (-aij) == 0
      lengthsq = (lengthsq * aji) // aij
      tmplengthsq = alpha.dot(alpha).expand()
      alpha[0,i] = denestplus(sympy.sqrt(lengthsq - tmplengthsq))
      assert no_strict_checks or lengthsq == alpha.norm()**2
      lsqs.append(lengthsq)
      # And insert
      roots = roots.row_insert(i,alpha)
      coroots = coroots.row_insert(i,sympy.Rational(2,lengthsq)*alpha)
    # Normalize according to convention long roots have lengths sqrt2,
    # (13.79) of DiFrancesco
    maxsq = max(lsqs)
    self._simple_roots = sympy.sqrt(sympy.Rational(2,maxsq)) * roots
    self._simple_coroots = sympy.sqrt(sympy.Rational(maxsq,2)) * coroots
    self._rls_sympy = [sympy.Rational(2*l,maxsq) for l in lsqs]
    self._rootlengthsquared = [Fraction(2*l,maxsq) for l in lsqs]
    # Hard-coded marks
    if series == 'A':
      self._marks = rank*[1]
    elif series == 'B':
      self._marks = [1] + (rank-1)*[2]
    elif series == 'C':
      self._marks = (rank-1)*[2] + [1]
    elif series == 'D':
      self._marks = [1] + (rank-3)*[2] + [1,1]
    elif series == 'E':
      if rank == 6:
        self._marks = [1,2,3,2,1,2]
      elif rank == 7:
        self._marks = [1,2,3,4,3,2,2]
      elif rank == 8:
        self._marks = [2,3,4,5,6,4,2,3]
    elif series == 'F':
      self._marks = [2,3,4,2]
    elif series == 'G':
      self._marks = [3,2]
    self._marks = tuple(self._marks)

    self._fund_weights = self._simple_coroots.T.inv()
    # Quadratic form without relying on explicit form of fundamental weights:
    # (From DiFrancesco but noting that they use transposed Cartan matrix)
    self._quad_form = inverse_rational(self._cartan_matrix)
    self._quad_form *= np.array(self._rootlengthsquared,dtype=object)[:,None]*Fraction(1,2)
    assert np.all(self._quad_form == self._quad_form.T)
    # Quadratic form expressed as integer matrix divided by scalar integer
    div = 1
    for entry in self._quad_form.flatten():
      div = lcm(div,entry.denominator)
    self._quad_form_int = (np.array(div*self._quad_form,dtype=int),div)
    # Obtain all roots (written in terms of Dynkin labels)
    by_level = [[self._marks]]
    height = sum(self._marks)
    for level in range(height-1,1,-1):
      newroots = []
      tested = set()
      for alpha in by_level[0]:
        for i in range(rank):
          if alpha[i] == 0:
            continue
          beta = list(alpha)
          beta[i] -= 1
          beta = ituple(beta)
          if beta in tested:
            continue
          # Test using "Cartan bracket" <alpha,beta> = 2(alpha,beta)/(alpha,alpha)
          bracket = self._cartan_matrix[i].dot(beta)
          tested.add(beta)
          if bracket > -2:
            # Definitely valid
            newroots.append(beta)
          elif bracket == -2:
            # Check if alpha + alpha_i is a root
            if len(by_level) < 2:
              continue
            gamma = list(alpha)
            gamma[i] += 1
            if ituple(gamma) in by_level[1]:
              newroots.append(beta)
          # Should only have <alpha,beta> < 0 and beta-alpha is a positive root
          # in a single case in G2 where beta-alpha is a simple root anyway
      by_level.insert(0, newroots)
    if height != 1:
      # Add simple roots unless highest root is one (only for A1)
      by_level.insert(0, np.eye(rank, dtype=int))
    self._positive_roots = np.concat(by_level,dtype=int)
    self._positive_by_level = by_level[::-1]
    self._cartan_ext = [ituple(ac) for ac in self._positive_roots.dot(self._cartan_matrix.T)]
    for betaind in self._positive_roots[rank:]:
      beta = symdot(self._simple_roots.T, betaind)
      self._rls_sympy.append(sympy.expand(beta.dot(beta)))
      # beta in coroot basis (x2)
      beta_coroots2 = betaind * self._rootlengthsquared[:rank]
      # Now Cartan matrix gives 2x length
      rls2 = beta_coroots2.dot(self._cartan_matrix).dot(betaind)
      self._rootlengthsquared.append(Fraction(rls2,2))
      assert self._rootlengthsquared[-1] == self._rls_sympy[-1]
    if self._CK_series == 'G':
      self._rls_int = np.array([int(rls*3) for rls in self._rootlengthsquared])
    else:
      assert all(rls.denominator == 1 for rls in self._rootlengthsquared)
      self._rls_int = np.array([int(rls) for rls in self._rootlengthsquared])
    self.chevalley_structure = None
    self._highest_weight_irreps = {}
    self._highest_weight_rational = {}
    self._dummy_irreps = {}
    self._3js = {}
    self._CWCnorms = None
    super().__init__()

  @classmethod
  def get_identifier(cls, series, rank):
    # TODO account for isomorphisms
    return '%s~%d'%(series,rank)

  @property
  def rank(self):
    return self._rank

  @property
  def quaternionic(self):
    # Admits quaternionic irreps (in compact form)
    if self._CK_series == 'A':
      return self._rank % 4 == 1
    elif self._CK_series == 'B':
      return self._rank % 4 in (1,2)
    elif self._CK_series == 'C':
      return True
    elif self._CK_series == 'D':
      return self._rank % 4 == 2
    elif self._CK_series == 'E':
      return self._rank == 7
    else:
      return False

  def raiseweight(self, weight, i):
    # Quick method to get mu -> mu + alpha_i (in tuple of Dynkin labels)
    w = np.array(weight) + self._cartan_matrix[:,i]
    return ituple(w)

  def lowerweight(self, weight, i):
    # Quick method to get mu -> mu - alpha_i (in Dynkin labels)
    w = np.array(weight) - self._cartan_matrix[:,i]
    return ituple(w)

  def raisebyroot1(self, mu, n, idx):
    mu2 = np.array(mu,dtype=int)
    mu2 += n*self._cartan_matrix.dot(self._positive_roots[idx])
    return ituple(mu2)

  def raisebyroot(self, mu, n, idx):
    mu2 = tuple(mui + n*aci for mui,aci in zip(mu,self._cartan_ext[idx]))
    return mu2

  def _fdual(self, lam):
    # Differs from getdualweight on quaternionic irreps
    # (distinguishes "primary"/"secondary"
    if any(li<0 for li in lam) or self.indicate(lam) == -1:
      return negate(lam)
    else:
      return self.getdualweight(lam)

  # TODO depricate one of these
  def getdualweight(self, lam):
    # What I believe is the negative of the action of the maximal element of
    # the Weyl group (assuming this corresponds to duality of Dynkin diagrams
    # iff exists and is unique - exception: Dn for n even)
    if self._CK_series == 'A':
      lbar = lam[::-1]
    elif self._CK_series == 'D' and self._rank % 2:
      lbar = lam[:-2] + lam[-1:-3:-1]
    elif self._CK_series == 'E' and self._rank == 6:
      lbar = lam[-2::-1] + (lam[-1],)
    else:
      lbar = lam
    if not no_strict_checks:
      rep = self.get_dummy(lam)
      assert lbar == rep.duallabel()
    return lbar

  def getduallabel(self, lam):
    return self.getdualweight(lam)

  def produce_chevalley_basis(self):
    if self.chevalley_structure is not None:
      return
    # First determine simple-root strings through roots
    # Recursively determine length in negative direction
    npos = self._positive_roots.shape[0]
    negative_strings = np.zeros((npos,self.rank),dtype=int)
    pos_root_indices = {ituple(beta):idx for idx,beta in enumerate(self._positive_roots)}
    # All "simple" strings terminate at simple roots
    for idx in range(self.rank,npos):
      beta = ituple(self._positive_roots[idx])
      for i in range(self.rank):
        bminus = beta[:i]+(beta[i]-1,)+beta[i+1:]
        if bminus in pos_root_indices:
          idxm = pos_root_indices[bminus]
          negative_strings[idx,i] = negative_strings[idxm,i]+1
    # Define non-simple basis elements by (p, i, idx) such that
    # E_beta = [E_alpha_i, E_root[idx]]/p
    chev_def = self.rank*[None]
    for idx in range(self.rank,npos):
      strings_at = negative_strings[idx]
      i = np.flatnonzero(strings_at)[0]
      bminus = self._positive_roots[idx].copy()
      bminus[i] -= 1
      idxm = pos_root_indices[ituple(bminus)]
      chev_def.append((strings_at[i],i,idxm))
    self._chevalley_definitions = chev_def
    # Easiest way to proceed may be to get a representation-
    # use adjoint because we know it's faithful
    # (hang on shouldn't they all be if it's simple)
    rep = self.get_adjoint_HW()
    assert rep.dim == 2*npos + self.rank
    repnmsq = rep.normsquare_array
    def rdagger(E):
      # Non-normalized basis so Hermitian adjoint involves raising & lowering
      # indices
      return E.T.raise_index(repnmsq,0).lower_index(repnmsq,1)
    matrices = list(rep.generators)
    for idx in range(self.rank,npos):
      p, i, idxm = chev_def[idx]
      matrices.append(commutator(matrices[i],matrices[idxm])/p)
    normsqs = [np.trace(rdagger(E).dot(E)) for E in matrices]
    hchev = [rep.diagonal_chevalley(i) for i in range(self.rank)]
    # In (un-normalized) Killing form, Gramian of diagonal
    # Chevalley generators is 2g * Gramian of coroots
    # for g dual Coxeter number, so change to dual basis is obtained
    # by inverting that with "quadratic form" matrix / 2g
    dualcoxeter = sum(self._marks[i]*self._rootlengthsquared[i]/2 \
                        for i in range(self.rank)) + 1
    assert dualcoxeter.denominator == 1
    hinvert = self._quad_form / (2*dualcoxeter)
    
    #Hweyl = [rep.cartanweyl_diagonal(i) for i in range(self.rank)]
    ## Form for extracting coefficients
    #coH = []
    #for Hi in Hweyl:
    #  killingnsq = np.sum(np.square(Hi))
    #  nsqnum,nsqdenom = rad_rationalize(1,killingnsq)
    #  coH.append(sympy.expand(nsqnum*Hi/nsqdenom))
    def rhs_coeff(E1,E2,newroot):
      idx3 = pos_root_indices[newroot]
      E3 = matrices[idx3]
      lhs = E1.dot(E2) - E2.dot(E1)
      #coeff = sympy.expand(symdot(E3.adjoint(),lhs).trace()*normnum/normdenom)
      coeff = rdagger(E3).dot(lhs).trace() / normsqs[idx3]
      return idx3,coeff
    chev_structure_pos = []
    chev_structure_neg = []
    nstart = npos+self.rank
    for idx in range(npos):
      beta = self._positive_roots[idx]
      E1 = matrices[idx]
      #beta_structure = sympy.SparseMatrix.zeros(2*npos+self.rank,2*npos+self.rank)
      beta_structure = rzeros((2*npos+self.rank,2*npos+self.rank))
      nbeta_structure = beta_structure.copy()
      # First, action on Cartan-subalgebra elements
      eigs = self._cartan_matrix.dot(beta)
      beta_structure[idx,npos:npos+self.rank] = -eigs
      nbeta_structure[idx+nstart,npos:npos+self.rank] = eigs
      for idx2 in range(npos):
        gamma = self._positive_roots[idx2]
        E2 = matrices[idx2]
        plusgamma = ituple(beta+gamma)
        if plusgamma in pos_root_indices:
          idx3,coeff = rhs_coeff(E1,E2,plusgamma)
          beta_structure[idx3,idx2] = coeff
          nbeta_structure[idx3+nstart,idx2+nstart] = -coeff
        minusgamma = ituple(beta-gamma)
        if minusgamma in pos_root_indices:
          idx3,coeff = rhs_coeff(E1,rdagger(E2),minusgamma)
          beta_structure[idx3,idx2+nstart] = coeff
          nbeta_structure[idx3+nstart,idx2] = -coeff
        gammaminus = ituple(gamma-beta)
        if gammaminus in pos_root_indices:
          idx3,coeff = rhs_coeff(rdagger(E1),E2,gammaminus)
          beta_structure[idx3+nstart,idx2+nstart] = -coeff
          nbeta_structure[idx3,idx2] = coeff
        elif idx == idx2:
          E2d = rdagger(E2)
          Hbeta = E1.dot(E2d) - E2d.dot(E1)
          Hbeta.rereduce()
          chevkilling = [hi.dot(Hbeta).trace() for hi in hchev]
          chevcoeffs = hinvert.dot(chevkilling)
          beta_structure[npos:nstart,idx2+nstart] = chevcoeffs
          nbeta_structure[npos:nstart,idx2] = -chevcoeffs
      chev_structure_pos.append(beta_structure)
      chev_structure_neg.append(nbeta_structure)
    self.chevalley_structure = chev_structure_pos
    # Now action of Cartan-subalgebra (diagonal) elements
    Haction = self._cartan_matrix.dot(self._positive_roots.T)
    Haction = np.hstack([Haction,
                np.zeros((self.rank,self.rank),dtype=int),-Haction])
    for i in range(self.rank):
      # sympy.diag appears to break if numpy types provided
      #Hi = sympy.ImmutableSparseMatrix.diag(ituple(Haction[i]))
      Hi = np.diag(Haction[i])
      self.chevalley_structure.append(Hi)
    self.chevalley_structure.extend(chev_structure_neg)

  def check_chevalley(self,logger=None,matrix_checking='exact',tolerance=0):
    # Check the Chevalley structure coefficients determined in
    #   produce_chevalley_basis
    # First checks individual coefficients against theory, then
    #   confirms that structure coefficients form a representation
    # If a logger is provided, writes inconsistencies to logger & then returns
    #   False if one or more are encountered after a full check;
    #   otherwise error is thrown immediately
    # To alter second part (structure as representation), can provide
    # matrix_checking as:
    #   -'symbolic' (depricated): use sparse SymPy matrices, check relations
    #     exactly
    #   - 'exact' (default): use RationalArrays
    #   -'none': skip this part
    #   -'numpy' or 'longdouble': convert to numpy float64 or longdouble,
    #     respectively, and then check with numpy.isclose using
    #     atol = rtol = tolerance
    #     (set to 1e-14 for float64 and 1e-18 for float128 if not provided)
    #   -'int'/'short': use numpy integer arrays
    #     (throws error if noninteger values are encountered)

    # Get all relevant string lengths, i.e. for any (possibly negative)
    # beta,gamma, the greatest p>=0 such that beta - p gamma is a root
    npos = self._positive_roots.shape[0]
    pos_root_indices = {ituple(beta):idx for idx,beta in enumerate(self._positive_roots)}
    sum_matrix = np.zeros((2,npos,2,npos,2),dtype=int)
    # Indices 0 & 2 are sign bit
    # (sgn1,idx1,sgn2,idx2) -> [sgn3,idx3],
    # where sgn3 is -1 if there is no root
    sum_matrix[:,:,:,:,0] = -1
    for idx1,beta in enumerate(self._positive_roots):
      for idx2,gamma in enumerate(self._positive_roots):
        plus = ituple(beta+gamma)
        if plus in pos_root_indices:
          idx3 = pos_root_indices[plus]
          sum_matrix[0,idx1,0,idx2] = [0,idx3]
          sum_matrix[1,idx1,1,idx2] = [1,idx3]
        minus = beta-gamma
        if ituple(minus) in pos_root_indices:
          idx3 = pos_root_indices[ituple(minus)]
          sum_matrix[0,idx1,1,idx2] = [0,idx3]
          sum_matrix[1,idx1,0,idx2] = [1,idx3]
        elif ituple(-minus) in pos_root_indices:
          idx3 = pos_root_indices[ituple(-minus)]
          sum_matrix[0,idx1,1,idx2] = [1,idx3]
          sum_matrix[1,idx1,0,idx2] = [0,idx3]
    stringlengths = np.zeros((2,npos,2,npos),dtype=int)
    for idx1 in range(npos):
      for idx2 in range(npos):
        for sgn in range(2):
          sgndiff,idxdiff = sum_matrix[sgn,idx1,1,idx2]
          while sgndiff != -1:
            stringlengths[sgn,idx1,0,idx2] += 1
            sgndiff,idxdiff = sum_matrix[sgndiff,idxdiff,1,idx2]
          stringlengths[1-sgn,idx1,1,idx2] = stringlengths[sgn,idx1,0,idx2]
    nstart = npos + self.rank
    def loginfo(msg,*args):
      if logger is not None:
        logger.info(msg,*args)
    def badlog(msg,*args):
      if logger is not None:
        logger.warn('~~'+msg+'~~',*args)
        badlog.correct = False
      else:
        raise ValueError(msg%args)
    badlog.correct = True
    def idx_to_msg(idx):
      if idx < npos:
        return 'E_%s',self._positive_roots[idx]
      elif idx < nstart:
        return 'h_%d',idx-npos
      else:
        return 'E_%s',-self._positive_roots[idx-nstart]
    def write_nonzeros(array, msg0, args0, exclude=None, xcartan=False):
      msg = str(msg0)
      args = list(args0)
      for idx in np.argwhere(array):
        # One of idxl & idxr should be 0 (and if both are, it's 0 we want)
        v = array[idx]
        if isinstance(idx,tuple):
          idx = max(*idx)
        if idx == exclude:
          continue
        elif idx < npos:
          idstr = 'E_%s'
          idarg = self._positive_roots[idx]
        elif idx >= nstart:
          idstr = 'E_%s'
          idarg = -self._positive_roots[idx-nstart]
        elif xcartan:
          continue
        else:
          idstr = 'h_%d'
          idarg = idx-npos
        msg += '%+0.4g '+idstr
        args.extend([v,idarg])
      if len(args) > len(args0):
        badlog(msg, *args)

    # First check (computed) structure constants against expected relations
    struc = self.chevalley_structure
    rls = rarray(self._rootlengthsquared)
    for idx1,beta in enumerate(self._positive_roots):
      loginfo('Checking E_+-%s relations',beta)
      pmatrix = struc[idx1]
      nmatrix = struc[idx1+nstart]
      # Expect [E_+-beta, E_+-beta] = 0; [E_+-beta,E_-+beta] = H(+-beta^v)
      write_nonzeros(pmatrix[:,idx1],'[E_%s,E_%s] has nonzero form ',
        [beta,beta])
      write_nonzeros(nmatrix[:,idx1+nstart],'[E_%s,E_%s] has nonzero form ',
        [-beta,-beta])
      coeffs = beta * rls[:self.rank]
      coeffs /= rls[idx1]
      for i in range(self.rank):
        #assert coeffs[i].is_integer # You'll find out soon enough
        if pmatrix[npos+i,idx1+nstart] != coeffs[i]:
          badlog('Expected coefficient %d of h_%d in [E_%s,E_%s], got %0.4g',
            coeffs[i],i,beta,-beta,pmatrix[npos+i,idx1+nstart])
        if nmatrix[npos+i,idx1] != -coeffs[i]:
          badlog('Expected coefficient %d of h_%d in [E_%s,E_%s], got %0.4g',
            -coeffs[i],i,-beta,beta,nmatrix[npos+i,idx1])
      write_nonzeros(pmatrix[:,idx1+nstart],
        'Expected [E_%s,E_%s] to belong to Cartan subalgebra, '
        'contains additional component ', [beta,-beta], xcartan=True)
      write_nonzeros(nmatrix[:,idx1],
        'Expected [E_%s,E_%s] to belong to Cartan subalgebra, '
        'contains additional component ', [-beta,beta], xcartan=True)
      loginfo('E_%s with Cartan subalgebra...',beta)
      dots = self._cartan_matrix.dot(beta)
      for i in range(self.rank):
        if pmatrix[idx1,i+npos] != -dots[i]:
          badlog('Expected [E_%s,h_%d] = %d E_%s, instead '
          'have coefficient %0.4g', beta,i,-dots[i],beta,pmatrix[idx1,i+npos])
        write_nonzeros(pmatrix[:,i+npos],'Expected [E_%s,h_%d]=%d E_%s, '
          'contains additional component', [beta,i,-dots[i],beta],exclude=idx1)
        if nmatrix[idx1+nstart,i+npos] != dots[i]:
          badlog('Expected [E_%s,h_%d] = %d E_%s, instead have '
          'coefficient %0.4g',-beta,i,dots[i],-beta,nmatrix[idx1+nstart,i+npos])
        write_nonzeros(nmatrix[:,i+npos],'Expected [E_%s,h_%d]=%d E_%s, '
          'contains additional component', [-beta,i,dots[i],-beta],
          exclude=idx1+nstart)
      for idx2,gamma in enumerate(self._positive_roots):
        if idx2 == idx1:
          continue
        sumdiffs = sum_matrix[:,idx1,:,idx2]
        loginfo('Checking [E_+-%s,E_+-%s]',beta,gamma)
        if sumdiffs[0,0,0] == -1:
          write_nonzeros(pmatrix[:,idx2],'Expected [E_%s,E_%s]=0, instead =',
            [beta,gamma])
          write_nonzeros(nmatrix[:,idx2+nstart],
            'Expected [E_%s,E_%s]=0, instead =', [-beta,-gamma])
        else:
          idx3 = sumdiffs[0,0,1]
          p = stringlengths[0,idx1,0,idx2]
          if abs(pmatrix[idx3,idx2]) != p+1:
            badlog('Expected [E_%s,E_%s] = +-%d E_%s, but coefficient is %0.4g',
              beta,gamma,p+1,self._positive_roots[idx3],pmatrix[idx3,idx2])
          write_nonzeros(pmatrix[:,idx2],
            'Expected [E_%s,E_%s] = +-%d E_%s, but has additional ',
            [beta,gamma,p+1,self._positive_roots[idx3]],exclude=idx3)
          if abs(nmatrix[idx3+nstart,idx2+nstart]) != p+1:
            badlog('Expected [E_%s,E_%s] = +-%d E_%s, but coefficient is %0.4g',
              -beta,-gamma,p+1,-self._positive_roots[idx3],
              nmatrix[idx3+nstart,idx2+nstart])
          write_nonzeros(nmatrix[:,idx2+nstart],
            'Expected [E_%s,E_%s] = +-%d E_%s, but has additional ',
            [-beta,-gamma,p+1,-self._positive_roots[idx3]],exclude=idx3+nstart)
        if sumdiffs[0,1,0] == -1:
          write_nonzeros(pmatrix[:,idx2+nstart],
            'Expected [E_%s,E_%s]=0, instead =',[beta,-gamma])
          write_nonzeros(nmatrix[:,idx2],
            'Expected [E_%s,E_%s]=0, instead =',[-beta,gamma])
        else:
          sbit = sumdiffs[0,1,0]
          idx3 = sumdiffs[0,1,1]
          p = stringlengths[0, idx1, 1, idx2]
          bg = (1-2*sbit)*self._positive_roots[idx3]
          #print(beta,gamma,sbit,bg)
          if abs(pmatrix[idx3+sbit*nstart,idx2+nstart]) != p+1:
            badlog('Expected [E_%s,E_%s] = +-%d E_%s, but coefficient is %0.4g',
              beta,-gamma,p+1,bg, pmatrix[idx3+sbit*nstart,idx2+nstart])
          write_nonzeros(pmatrix[:,idx2+nstart],
            'Expected [E_%s,E_%s] = +-%d E_%s, but has additional ',
            [beta,-gamma,p+1,bg], exclude=idx3+sbit*nstart)
          if abs(nmatrix[idx3+(1-sbit)*nstart,idx2]) != p+1:
            badlog('Expected [E_%s,E_%s] = +-%d E_%s, but coefficient is %0.4g',
              -beta,gamma,p+1,-bg, pmatrix[idx3+(1-sbit)*nstart,idx2])
          write_nonzeros(nmatrix[:,idx2],
            'Expected [E_%s,E_%s] = +-%d E_%s, but has additional ',
            [-beta,gamma,p+1,-bg], exclude=idx3+(1-sbit)*nstart)
    # Diagonal elements
    for i in range(self.rank):
      hi = struc[i+npos]
      for j in range(self.rank):
        write_nonzeros(hi[:,j+npos],'In Cartan subalgebra [h_%d,h_%d]=', [i,j])
      for idx,beta in enumerate(self._positive_roots):
        dot = self._cartan_matrix[i].dot(beta)
        if hi[idx,idx] != dot:
          badlog('Expected [h_%d,E_%s] = %d E_%s, '
            'instead have coefficient %0.4g', i,beta,dot,beta,hi[idx,idx])
        write_nonzeros(hi[:,idx],'Expected [h_%d,E_%s]=%dE_%s, contains '
          'additional component ',[i,beta,dot,beta],exclude=idx)
        if hi[idx+nstart,idx+nstart] != -dot:
          badlog('Expected [h_%d,E_%s] = %d E_%s, '
            'instead have coefficient %0.4g', i,-beta,-dot,-beta,
            hi[idx+nstart,idx+nstart])
        write_nonzeros(hi[:,idx+nstart],'Expected [h_%d,E_%s]=%dE_%s, contains '
          'additional component ',[i,-beta,-dot,-beta],exclude=idx+nstart)
    
    if matrix_checking == 'none':
      pass
    elif matrix_checking == 'exact':
      loginfo('Checking structure coefficients as representation (exact)')
      #killing = sympy.zeros(npos+nstart,npos+nstart)
      #for idx1 in range(npos+nstart):
      #  for idx2 in range(npos+nstart):
      #    killing[idx2,idx1] = symdot(struc[idx1], struc[idx2]).trace().simplify()
      for idx1 in range(npos+nstart):
        for idx2 in range(npos+nstart):
          #res = sympy.SparseMatrix.zeros(npos+nstart,npos+nstart)
          res = rzeros((npos+nstart,npos+nstart))
          for idx3, in np.argwhere(struc[idx1][:,idx2]):
            res += struc[idx1][idx3,idx2]*struc[idx3]
          compare = struc[idx1].dot(struc[idx2])-struc[idx2].dot(struc[idx1])
          if np.any(compare-res):
            s1,v1 = idx_to_msg(idx1)
            s2,v2 = idx_to_msg(idx2)
            badlog('Error in [%s,%s]'%(s1,s2)+': off by %0.4g (%0.4g/%0.4g)',
              v1,v2,(compare-res).norm(),fnorm(res),
              fnorm(struc[idx1])*fnorm(struc[idx2]))
    elif matrix_checking == 'symbolic':
      loginfo('Checking structure coefficients as representation (exact)')
      #killing = sympy.zeros(npos+nstart,npos+nstart)
      #for idx1 in range(npos+nstart):
      #  for idx2 in range(npos+nstart):
      #    killing[idx2,idx1] = symdot(struc[idx1], struc[idx2]).trace().simplify()
      for idx1 in range(npos+nstart):
        for idx2 in range(npos+nstart):
          #res = sympy.SparseMatrix.zeros(npos+nstart,npos+nstart)
          res = sympy.zeros(npos+nstart,npos+nstart)
          for (idx3,x),v in struc[idx1][:,idx2].iter_items():
            res += v*struc[idx3]
          compare = struc[idx1]*struc[idx2]-struc[idx2]*struc[idx1]
          if not (compare-res).is_zero_matrix:
            s1,v1 = idx_to_msg(idx1)
            s2,v2 = idx_to_msg(idx2)
            badlog('Error in [%s,%s]'%(s1,s2)+': off by %0.4g (%0.4g/%0.4g)',
              v1,v2,(compare-res).norm(),res.norm(),
              struc[idx1].norm()*struc[idx2].norm())
    elif matrix_checking in ['int','short','numpy','longdouble']:
      if matrix_checking == 'int':
        dtype = int
      elif matrix_checking == 'short':
        dtype = np.short
      elif matrix_checking == 'longdouble':
        dtype = np.longdouble
      else:
        dtype = float
      inttype = (matrix_checking in ['int','short'])
      struc_numeric = np.zeros((npos+nstart,npos+nstart,npos+nstart),dtype=dtype)
      dim = npos+nstart
      for idx1 in range(dim):
        for idx3,idx2 in np.argwhere(struc[idx1]):
          val = struc[idx1][idx3,idx2]
          if matrix_checking in ['int','short'] and not val.denominator == 1:
            s1,v1 = idx_to_msg(idx1)
            s2,v2 = idx_to_msg(idx2)
            s3,v3 = idx_to_msg(idx3)
            msg = 'Non-integer coefficient [%s,%s] = %%s %s (+...) encountered'%(s1,s2,s3)
            raise ValueError(msg%(v1,v2,val,v3))
          elif matrix_checking == 'longdouble':
            # There MUST be a better way
            val = str(val.evalf(22))
          struc_numeric[idx1,idx2,idx3] = val
      if matrix_checking == 'numpy' and not tolerance:
        tolerance = 1e-14
      elif matrix_checking == 'longdouble' and not tolerance:
        tolerance = 1e-18
      transposed  = struc_numeric.transpose((1,0,2))
      if inttype:
        antisym = (struc_numeric == -transposed)
      else:
        antisym = np.isclose(struc_numeric,-transposed,rtol=tolerance,atol=tolerance)
      for idx1,idx2,idx3 in np.argwhere(np.logical_not(antisym)):
        s1,v1 = idx_to_msg(idx1)
        s2,v2 = idx_to_msg(idx2)
        s3,v3 = idx_to_msg(idx3)
        msg = 'Failure of antisymmetry for (%%d,%%d),%%d (%s,%s,%s)'%(s1,s2,s3)
        badlog(msg,idx1,idx2,idx3,v1,v2,v3)
      #jacobi_term = np.zeros((dim,dim,dim,dim),dtype=dtype)
      #jacobi_term = np.tensordot(struc_numeric,struc_numeric,(1,2))
      # Raised index is 1; Jacobi identity is cyclic permutation of 0,2,3
      for idxd in range(dim):
        jacobit1 = np.tensordot(struc_numeric[:,:,idxd],struc_numeric,(1,2))
        jacobit2 = jacobit1.transpose((1,2,0))
        jacobit3 = jacobit2.transpose((1,2,0))
        if inttype:
          jacobi = (jacobit2+jacobit3 == -jacobit1)
        else:
          jacobi = np.isclose(jacobit2+jacobit3,-jacobit1,rtol=tolerance,atol=tolerance)
        # TODO was that supposed to be idxd instead of idxb
        for idxa,idxb,idxc in np.argwhere(np.logical_not(jacobi)):
          sa,va = idx_to_msg(idxa)
          sb,vb = idx_to_msg(idxb)
          sc,vc = idx_to_msg(idxc)
          sd,vd = idx_to_msg(idxd)
          msg = 'Failure of Jacobi identity, for indices a,b,c,d -> %s,%s,%s,%s'%(sa,sb,sc,sd)
          badlog(msg,va,vb,vc,vd)
    else:
      raise ValueError('Unrecognized value "%s" provided for matrix_checking'%\
        matrix_checking)
    return badlog.correct


  def get_symbrep(self, lambda_labels):
    if lambda_labels not in self._highest_weight_irreps:
      if not any(lambda_labels):
        self._highest_weight_irreps[lambda_labels] = TrivialRepresentation(self)
      else:
        self._highest_weight_irreps[lambda_labels] = SymbolicRepresentation(lambda_labels,self)
    return self._highest_weight_irreps[lambda_labels]

  def get_ratrep(self, lambda_labels, dimcheck=False):
    if lambda_labels not in self._highest_weight_rational:
      if not any(lambda_labels):
        self._highest_weight_rational[lambda_labels] = TrivialRepresentation(self)
      else:
        dualweight = self.getdualweight(lambda_labels)
        if dualweight < lambda_labels:
          dual = self.get_ratrep(dualweight, dimcheck=dimcheck)
          if dual is None:
            return None
          self._highest_weight_rational[lambda_labels] = RationalRepresentation(lambda_labels,self,dual=dual)
        else:
          try:
            self._highest_weight_rational[lambda_labels] = RationalRepresentation(lambda_labels,self,dimcheck=dimcheck)
            if dualweight == lambda_labels:
              self._highest_weight_rational[lambda_labels]._fix_dual()
          except DimensionCheckException:
            return None
    return self._highest_weight_rational[lambda_labels]

  def get_dummy(self, lambda_labels):
    if lambda_labels not in self._dummy_irreps:
      self._dummy_irreps[lambda_labels] = DummyIrrep(lambda_labels,self)
    return self._dummy_irreps[lambda_labels]

  def order(self, lam):
    # TODO depricate in favor of irrep_order
    return self.irrep_order(lam) 

  def irrep_order(self, lam):
    # Tuple (dimension, Dynkin indices) to yield complete ordering on irreps
    return (self.dim(lam),lam)

  def get_adjoint_HW(self):
    labels = ituple(self._cartan_matrix.dot(self._marks))
    return self.get_ratrep(labels)

  def tensor_decompose_symbolic(self, lambda1, lambda2):
    rep1 = self.get_irrep(lambda1)
    rep2 = self.get_irrep(lambda2)
    rank = self.rank
    # First identify weights & generator representations of product
    weights_by_depth = []
    weight_products = {}
    degeneracies = {}
    A = self._cartan_matrix
    def wraise(weight,i):
      w = np.array(weight) + A[:,i]
      return ituple(w)
    height1 = len(rep1._weights_by_depth)
    height2 = len(rep2._weights_by_depth)
    height = height1+height2-1
    # First map component weights onto product weights
    for lvl in range(height):
      weights_by_depth.append([])
      for lvl1 in range(max(0,lvl-height2+1),min(height1,lvl+1)):
        lvl2 = lvl - lvl1
        for mu1,d1 in zip(rep1._weights_by_depth[lvl1],rep1._degeneracies_by_depth[lvl1]):
          for mu2,d2 in zip(rep2._weights_by_depth[lvl2],rep2._degeneracies_by_depth[lvl2]):
            mu = tuple(mu1[i]+mu2[i] for i in range(rank))
            if mu in weight_products:
              idxm,d1m,d2m,mu1m,mu2m = weight_products[mu][-1]
              idx0 = idxm + d1m*d2m
              degeneracies[mu] += d1*d2
            else:
              idx0 = 0
              weight_products[mu] = []
              weights_by_depth[-1].append(mu)
              degeneracies[mu] = d1*d2
            weight_products[mu].append((idx0,d1,d2,mu1,mu2))
    degen1 = dict(zip(rep1.weights,rep1.degeneracies))
    degen2 = dict(zip(rep2.weights,rep2.degeneracies))
    # Then start constructing lowering operators
    lowering_blocks = [{} for i in range(rank)]
    for lvl,weights in enumerate(weights_by_depth):
      if lvl == 0:
        continue
      for mu in weights:
        for i in range(rank):
          muplus_i = wraise(mu,i)
          if muplus_i in degeneracies:
            block = sympy.zeros(degeneracies[mu],degeneracies[muplus_i])
            for idx,d1,d2,mu1,mu2 in weight_products[mu]:
              mu1p = wraise(mu1,i)
              mu2p = wraise(mu2,i)
              for idxb,d1b,d2b,mu1b,mu2b in weight_products[muplus_i]:
                if mu1p == mu1b and mu1p in rep1.lowering_blocks[i]:
                  assert mu2b == mu2
                  mu1a,Ei = rep1.lowering_blocks[i][mu1p]
                  for k2 in range(d2):
                    # TODO can't assign to slices with steps?
                    for k1a in range(d1):
                      for k1b in range(degen1[mu1p]):
                        block[idx+k2+d2*k1a,idxb+k2+d2*k1b] += Ei[k1a,k1b]
                elif mu2p == mu2b and mu2p in rep2.lowering_blocks[i]:
                  assert mu1b == mu1
                  mu2a,Ei = rep2.lowering_blocks[i][mu2p]
                  for k1 in range(d1):
                    block[idx+k1*d2:idx+(k1+1)*d2,
                          idxb+k1*d2b:idxb+(k1+1)*d2b] += Ei
            lowering_blocks[i][muplus_i] = block
    # Now have enough information to perform decomposition
    components = [] # Highest weights & multiplicities
    # Projections onto components for each participating weight
    decomp_by_weight = {mu:{} for mu in degeneracies}
    degen_remaining = dict(degeneracies) # Degeneracy of each weight unaccounted for
    for lambda_new in itertools.chain.from_iterable(weights_by_depth):
      # Candidate highest weight must have nonnegative coefficients
      if min(lambda_new) < 0:
        continue
      elif max(lambda_new) == 0:
        # When we've reached 0 no (other) candidate highest weights left
        break
      if degen_remaining[lambda_new] == 0:
        continue
      N = degen_remaining[lambda_new]
      degen_remaining[lambda_new] = 0
      components.append((lambda_new,N))
      if degeneracies[lambda_new] != N:
        #print(lambda_new,degeneracies[lambda_new],N)
        # Need to account for vectors already projected out
        proj_out = sympy.Matrix.vstack(*itertools.chain.from_iterable(decomp_by_weight[lambda_new].values()))
        # TODO nullspace can hang indefinitely - better alternative?
        #spanning = proj_out.nullspace(True)
        proj_on = sympy.eye(degeneracies[lambda_new]) - symdot(proj_out.H,proj_out)
        spanning = proj_on.columnspace()
        basis = sympy.Matrix.orthogonalize(*spanning,normalize=True,rankcheck=True)
        # Projection onto highest weight
        onto_hw = [v.T.expand().applyfunc(denestplus) for v in basis]
      else:
        onto_hw = sympy.eye(N)
        onto_hw = [onto_hw.row(a) for a in range(N)]
      decomp_by_weight[lambda_new][lambda_new] = onto_hw
      # Get representation info
      irrep = self.get_irrep(lambda_new)
      irrep_iter = zip(irrep.weights,irrep.degeneracies)
      next(irrep_iter)
      # Produce projection for remaining weights one-by-one
      for mu,d in irrep_iter:
        projectors = [sympy.zeros(d,degeneracies[mu]) for a in range(N)]
        # Iterate through "simple lowering operators"
        for i,coeffs in enumerate(irrep._construction[mu]):
          if coeffs is None:
            continue
          muplus = wraise(mu,i)
          Ei = lowering_blocks[i][muplus].T
          proj_plus = decomp_by_weight[muplus][lambda_new]
          for a in range(N):
            projectors[a] += symdot(coeffs,symdot(proj_plus[a],Ei))
        decomp_by_weight[mu][lambda_new] = [mat_collect(P) for P in projectors]
        degen_remaining[mu] -= d*N
    # Then deal with 0
    zerow = rank*(0,)
    if degen_remaining.get(zerow,0):
      assert rep1.duallabel() == lambda2
      assert degen_remaining[zerow] == 1
      components.append((zerow,1))
      project_out = sympy.Matrix.vstack(*itertools.chain.from_iterable(decomp_by_weight[zerow].values()))
      #projector, = project_out.nullspace(True)
      project_on = sympy.eye(degeneracies[zerow]) - symdot(project_out.H,project_out)
      projector, = project_on.columnspace()
      decomp_by_weight[zerow][zerow] = [projector.normalized().T]
      degen_remaining[zerow] = 0
    else:
      assert rep1.duallabel() != lambda2
    # Sanity checks
    if not no_strict_checks:
      for mu,decomp in decomp_by_weight.items():
        Ptot = sympy.Matrix.vstack(*itertools.chain.from_iterable(decomp.values()))
        assert Ptot.is_square
        assert symdot(Ptot.H,Ptot) == sympy.eye(degeneracies[mu])
    assert sum(N*self.get_irrep(lam).dim for lam,N in components) == self.get_irrep(lambda1).dim*self.get_irrep(lambda2).dim
    # Finally build rank-3(+1) tensors from V1xV2->V_
    intertwiners = {}
    # For now won't try sympy nd arrays, just numpy-array-of-object approach
    for lam,N in components:
      D = self._highest_weight_irreps[lam].dim
      mat = sympy.zeros(N*D,rep1.dim*rep2.dim)
      intertwiners[lam] = sympy.matrix2numpy(mat).reshape((N,D,rep1.dim,rep2.dim))
    # Weight-by-weight
    indices = {}
    for lam,_w in [(lambda1,None),(lambda2,None)]+components:
      irrep = self._highest_weight_irreps[lam]
      blockidxs = [0] + list(irrep.block_indices)
      indices[lam] = {mu:blockidxs[idx] for mu,idx in irrep.weight_indices.items()}
    for mu in itertools.chain.from_iterable(weights_by_depth):
      # Prepare relevant data for target irreps
      irreps_iter = []
      for lam,W in decomp_by_weight[mu].items():
        irrep = self._highest_weight_irreps[lam]
        d = irrep.degeneracy(mu)
        idx = indices[lam][mu]
        irreps_iter.append((lam,d,idx,W))
      # Iterate over decomposition as sum of product weights
      for idx0,d1,d2,mu1,mu2 in weight_products[mu]:
        idx1 = indices[lambda1][mu1]
        idx2 = indices[lambda2][mu2]
        for lam,d,idx,W in irreps_iter:
          for a,Wa in enumerate(W):
            Wblock = sympy.matrix2numpy(Wa[:,idx0:idx0+d1*d2])
            intertwiners[lam][a,idx:idx+d,idx1:idx1+d1,idx2:idx2+d2] \
              = Wblock.reshape((d,d1,d2))
    return components, intertwiners

  def tensor_decompose_rational(self, lambda1, lambda2):
    decomp = RationalTensorDecomposition(self, lambda1, lambda2, True)
    #for lambda3 in decomp._3js:
    return decomp

  def get3j_rational(self, lambda1, lambda2, lambda3):
    # Get rational-element 3j symbol(s) corresponding to lambda1xlambda2xlambda3
    return self._get3jrat(lambda1,lambda2,lambda3,False)

  def _get3jrat(self, lambda1, lambda2, lambda3, optional):
    # Get 3j symbol(s) corresponding to lambda1xlambda2xlambda3.
    # In order to construct 3j symbol uses RationalTensorDecomposition,
    # which in turn calls back with optional flag set to True, for which case
    # only returns anything when either
    # 3j(lambda1,lambda2,lambda3) has already been calculated, or when
    # this is not the preferred order for lambda1,lambda2,lambda3
    # (for purposes of consistency)
    if (lambda1,lambda2,lambda3) not in self._3js:
      if [lambda1,lambda2,lambda3] != sorted([lambda1,lambda2,lambda3],key=self.order):
        # Change order first
        self._get3jrat(*sorted([lambda1,lambda2,lambda3],key=self.order),False)
        return self._3js[lambda1,lambda2,lambda3]
      elif not any(lambda1):
        # Trivial representation: use `2j symbol'
        self._set3js(Rational3j(self,lambda2))
      else:
        # First check if dual reps take precedence
        ld1 = self.getdualweight(lambda1)
        ld2 = self.getdualweight(lambda2)
        ld3 = self.getdualweight(lambda3)
        if min([lambda1,ld1,ld2,ld3],key=self.order) != lambda1:
          self._get3jrat(ld1,ld2,ld3,False)
        elif optional:
          return
        else:
          # Now calculate
          decomp = self.tensor_decompose_rational(lambda1,lambda2)
          # Will have set other 3j symbols from there
    return self._3js[lambda1,lambda2,lambda3]
  
  def _set3js(self, symbol, setduals=True):
    # Set 3j symbols corresponding to symbol
    l1 = symbol.rep1._highest
    l2 = symbol.rep2._highest
    l3 = symbol.rep3._highest
    if _fix_3j_symmetries:
      if l1 == l2:
        if setduals:
          # Otherwise should already be fixed
          symbol = symbol.fixflip()
          Rdiag = symbol.Rs[0,1]
        else:
          Rdiag = symbol.C.Rs[0,1]
          Rdiag = RationalRacahOperator(Rdiag,(symbol,True,True),
            (symbol,False,False))
      elif l2 == l3:
        if setduals:
          symbol = symbol.fixflip2()
          Rdiag = symbol.Rs[1,2]
        else:
          Rdiag = symbol.C.Rs[1,2]
          Rdiag = RationalRacahOperator(Rdiag,(symbol,True,True),
            (symbol,False,False))
    self._3js[l1,l2,l3] = symbol
    if l1 != l2:
      s01 = symbol.transpose((1,0,2))
      self._3js[l2,l1,l3] = s01
    if l2 != l3:
      s12 = symbol.transpose((0,2,1))
      s021 = symbol.transpose((2,0,1))
      self._3js[l1,l3,l2] = s12
      self._3js[l3,l1,l2] = s021
      if l1 != l2:
        s012 = s12.transpose((2,1,0))
        self._3js[l2,l3,l1] = s012
    if l1 != l2:
      # If l2 = l3 still need (l2,l2,l1)
      s02 = self._3js[l3,l1,l2].transpose((0,2,1))
      self._3js[l3,l2,l1] = s02
    if _fix_3j_symmetries:
      if l1 == l2 == l3:
        # Only (0,1) has been fixed
        # Need one more to compute the full symmetry group
        R12 = symbol.rotator((1,2),symbol)
        # Note that these will act on positions not values of letters,
        # thus "R01" @ "R12" sends (0,1,2)->(0,2,1)->(2,0,1)
        R01 = Rdiag
        R021 = R01.dot(R12)
        symbol.Rs[1,2] = R12
        symbol.Rs[0,2,1] = R021
        symbol.Rs[0,1,2] = R12.dot(R01)
        symbol.Rs[0,2] = R021.dot(R01)
      else:
        if l1 == l2:
          mismatches = [(1,0,2),(1,2,0),(2,1,0)]
        elif l2 == l3:
          mismatches = [(0,2,1),(2,0,1),(2,1,0)]
        else:
          mismatches = []
        labels = (l1,l2,l3)
        for pb in itertools.permutations(range(3)):
          lb = tuple(labels[i] for i in pb)
          symbb = self._3js[lb]
          for plabel in {(0,1),(0,2),(1,2),(0,1,2),(0,2,1)}:# - set(symbb.Rs):
            pc = list(pb)
            for i in range(len(plabel)):
              pc[plabel[i-1]] = pb[plabel[i]]
            lc = tuple(labels[i] for i in pc)
            if (tuple(pc) in mismatches) ^ (pb in mismatches):
              #self._3js[lc].Rs[plabel] = (symbb,Rdiag)
              symbb.Rs[plabel] = RationalRacahOperator(Rdiag,
                (self._3js[lc],True,True),(symbb,False,False))
            else:
              #self._3js[lc].Rs[plabel] = (symbb,1)
              #symbb.Rs[plabel] = (self._3js[lc],1)
              symbb.Rs[plabel] = RationalRacahOperator(1,
                (self._3js[lc],True,True),(symbb,False,False))

    if setduals:
      # If non-isomorphic to dual symbol (under rotations), use to set duals
      clabels = tuple(rep.duallabel() for rep in symbol.reps)
      if sorted(clabels) != sorted((l1,l2,l3)):
        # Note that set() will not catch (l,l,l*) !~= (l*,l*,l)*
        # Will need to make sure that conjugated symbol knows about original
        symbc = symbol.conj()
        # Temporarily set symbc.C to symbol for ease of finding
        symbc.C = symbol
        self._set3js(symbc,setduals=False)
        if _fix_3j_symmetries:
          # If exactly one of the representations is quaternionic,
          # means Cs must be related by negative sign
          sgn = [1,-1][sum(rep.FSI == -1 for rep in symbol.reps) % 2]
          for bothlabels in itertools.permutations(zip((l1,l2,l3),clabels)):
            x,xc = zip(*bothlabels)
            self._3js[x].C = RationalRacahOperator(1,
              (self._3js[xc],True,True),(self._3js[x],True,False))
            #(self._3js[xc],1)
            #self._3js[xc].C = (self._3js[x],1)
            self._3js[xc].C = RationalRacahOperator(sgn,
              (self._3js[x],True,True),(self._3js[xc],True,False))
      elif _fix_3j_symmetries:
        # clabels should already be in self._3js
        C = symbol.conjugator(self._3js[clabels])
        for bothlabels in itertools.permutations(zip((l1,l2,l3),clabels)):
          x,xc = zip(*bothlabels)
          self._3js[x].C = RationalRacahOperator(C,
            (self._3js[xc],True,True),(self._3js[x],True,False))
          self._3js[xc].C = self._3js[x].C.H.conj()

  def dummy_decompose(self, lambda1, lambda2):
    rep1 = self.get_dummy(lambda1)
    rep2 = self.get_dummy(lambda2)
    rank = self.rank
    # First identify weights & generator representations of product
    weights_by_depth = []
    multiplicities = {}
    A = self._cartan_matrix
    wraise = self.raiseweight
    height1 = len(rep1._weights_by_depth)
    height2 = len(rep2._weights_by_depth)
    m1 = rep1.multiplicities
    m2 = rep2.multiplicities
    height = height1+height2-1
    # First map component weights onto product weights
    for lvl in range(height):
      weights_by_depth.append([])
      for lvl1 in range(max(0,lvl-height2+1),min(height1,lvl+1)):
        lvl2 = lvl - lvl1
        for mu1 in rep1._weights_by_depth[lvl1]:
          for mu2 in rep2._weights_by_depth[lvl2]:
            mu = tuple(mu1[i]+mu2[i] for i in range(rank))
            if mu not in multiplicities:
              multiplicities[mu] = m1[mu1] * m2[mu2]
              weights_by_depth[-1].append(mu)
            else:
              multiplicities[mu] += m1[mu1] * m2[mu2]
    components = [] # Highest-weight reps & multiplicities
    dimleft = rep1.dim * rep2.dim
    for lambda_new in itertools.chain.from_iterable(weights_by_depth):
      # Candidate highest weight must have nonnegative coefficients
      if min(lambda_new) < 0:
        continue
      elif max(lambda_new) < 0:
        # When we've reached 0 no (other) candidate highest weights left
        break
      D = multiplicities[lambda_new]
      for rep,N in components:
        if lambda_new in rep.multiplicities:
          D -= rep.multiplicities[lambda_new]*N
      if D == 0:
        continue
      assert D > 0
      rep = self.get_dummy(lambda_new)
      components.append((rep,D))
      dimleft -= rep.dim*D
      if max(lambda_new) == 0:
        break
    assert dimleft == 0
    return components

  def check_fusion(self, logger, lambda1, lambda2, components, intertwiners,
      tolerance=1e-12):
    rep1 = self.get_irrep(lambda1)
    rep2 = self.get_irrep(lambda2)
    rank = self.rank
    dtot = sum(N*self.get_irrep(lam).dim for lam,N in components)
    assert rep1.dim*rep2.dim == dtot
    def badlog(msg,*args):
      if logger is not None:
        logger.warn('~~'+msg+'~~',*args)
        badlog.correct = False
      else:
        raise ValueError(msg%args)
    badlog.correct = True
    # Get Chevalley generators of product
    hs = []
    es = []
    I1 = scipy.sparse.eye_array(rep1.dim)
    I2 = scipy.sparse.eye_array(rep2.dim)
    for i in range(rank):
      h1i = rep1.fp_chevalley('h',i)
      h2i = rep2.fp_chevalley('h',i)
      h12 = scipy.sparse.kron(h1i,I2) + scipy.sparse.kron(I1,h2i)
      hs.append(h12)
      e1i = rep1.fp_chevalley('e',i)
      e2i = rep2.fp_chevalley('e',i)
      e12 = scipy.sparse.kron(e1i,I2) + scipy.sparse.kron(I1,e2i)
      es.append(e12)
    for lam,N in components:
      rep = self.get_irrep(lam)
      Ws = intertwiners[lam]
      hprod = [rep.fp_chevalley('h',i) for i in range(rank)]
      eprod = [rep.fp_chevalley('e',i) for i in range(rank)]
      for a in range(N):
        # Convert to sparse scipy array
        rows = []
        cols = []
        data = []
        for i in range(rep.dim):
          for j in range(rep1.dim):
            for k in range(rep2.dim):
              if Ws[a,i,j,k] != 0:
                rows.append(i)
                cols.append(rep2.dim*j+k)
                data.append(float(Ws[a,i,j,k]))
        Wcoo = scipy.sparse.coo_array((data,(rows,cols)),
          shape=(rep.dim,rep1.dim*rep2.dim))
        W = Wcoo.tocsr()
        WT = Wcoo.tocsc().T
        WWT = W.dot(WT).toarray()
        if not np.allclose(WWT,np.eye(rep.dim),atol=tolerance,rtol=tolerance):
          badlog('Failure of unitarity in %sx%s->%s_%d (%0.4g)',
            lambda1,lambda2,lam,a,linalg.norm(WWT-np.eye(rep.dim)))
        for i in range(rank):
          hi = hprod[i].toarray()
          hcomp = W.dot(hs[i]).dot(WT).toarray()
          if not np.allclose(hi,hcomp,atol=tolerance,rtol=tolerance):
            badlog('Failure to reproduce h_%d in %sx%s->%s_%d (%0.4g)',
              i,lambda1,lambda2,lam,a,linalg.norm(hi-hcomp))
          ei = eprod[i].toarray()
          ecomp = W.dot(es[i]).dot(WT).toarray()
          if not np.allclose(ei,ecomp,atol=tolerance,rtol=tolerance):
            badlog('Failure to reproduce e_%d in %sx%s->%s_%d (%0.4g)',
              i,lambda1,lambda2,lam,a,linalg.norm(ei-ecomp))
    return badlog.correct

  @property
  def dimension(self):
    return 2*self._positive_roots.shape[0] + self.rank

  @property
  def triv(self):
    return self.rank*(0,)

  def isrep(self, l):
    if not isinstance(l,tuple) or not all(isinstance(li,int) for li in l):
      return False
    if len(l) != self.rank:
      return False
    if any(li < 0 for li in l):
      # Only valid for "secondary" quaternionic representation 
      # (all indices negative of "primary"
      if any(li > 0 for li in l):
        return False
      return self.get_dummy(negate(l)).FSI == -1
    else:
      return True

  def indicate(self, l):
    if any(li<0 for li in l):
      assert self.get_dummy(negate(l)).FSI == -1
      return -2
    return self.get_dummy(l).FSI

  def _firrep(self, lam, coeffs):
    """Group elements given as exp(g) for g given from coefficients by
    compact_irrep"""
    if any(li<0 for li in lam):
      assert self.indicate(lam) == -2
      return linalg.expm(self.compact_irrep(negate(lam),coeffs).conj())
    return linalg.expm(self.compact_irrep(lam,coeffs))

  def compact_irrep(self, lam, coeffs):
    """Elements of compact real form of irrep lam determined by
    coefficients of Cartan-Weyl-based generators:
    First r are iH_j (Cartan subalgebra)
    Remaining are in pairs corresponding to positive roots alpha:
    with normalization [E_alpha,E_-alpha] = H_alpha (H_alpha defined wrt
    root NOT coroot),
    iX_alpha = iE_alpha + iE_-alpha
    iY_alpha = E_alpha - E_-alpha"""
    return self.chevalley_irrep(lam, self._CWcompact_to_chevalley(coeffs))

  def chevalley_irrep(self, lam, coeffs):
    irrep = self.get_ratrep(lam)
    es = []
    mat = np.zeros((irrep.dim,irrep.dim),dtype=complex)
    npos = self._positive_roots.shape[0]
    for i in range(self.rank):
      hi = irrep.fp_chevalley('h',i,realize=True)
      mat += coeffs[npos+i] * hi
      ei = irrep.fp_chevalley('e',i,realize=True)
      es.append(ei)
    for idx in range(self.rank,npos):
      p,i,idx0 = self._chevalley_definitions[idx]
      es.append((es[i].dot(es[idx0]) - es[idx0].dot(es[i]))/p)
    for idx in range(npos):
      mat += coeffs[idx] * es[idx]
      mat += coeffs[npos+self.rank+idx] * es[idx].T.conj()
    return mat
    
  def _CW_to_chevalley_norms(self):
    npos = len(self._positive_roots)
    nstart = npos + self.rank
    if self._CWCnorms is not None:
      return self._CWCnorms
    self._CWCnorms = []
    for idx,alpha in enumerate(self._positive_roots):
      chev_h = self.chevalley_structure[idx][npos:npos+self.rank,idx+nstart]
      chev_h = [2*chev_h[i]/self._rootlengthsquared[i] for i in range(self.rank)]
      normalizer = None
      for i,ai in enumerate(alpha):
        if ai:
          if normalizer is None:
            normalizer = chev_h[i]/ai
          else:
            assert ai * normalizer == chev_h[i]
        else:
          assert chev_h[i] == 0
      if idx < self.rank:
        assert 2/self._rootlengthsquared[idx] == normalizer
      self._CWCnorms.append(np.sqrt(normalizer.numerator)/np.sqrt(normalizer.denominator))
    return self._CWCnorms

  def _CWcompact_to_chevalley(self, coeffs):
    # See compact_irrep() for basis
    npos = len(self._positive_roots)
    nstart = npos + self.rank
    # Cartan subalgebra: convert from Cartan-Weyl by applying transpose of
    # fundamental weight definition matrix
    cartan_coeffs = self._fund_weights.T * sympy.Matrix(coeffs[:self.rank])
    cartan_coeffs = [1j*complex(a) for a in cartan_coeffs]
    pos_coeffs = []
    neg_coeffs = []
    for idx,norm in enumerate(self._CW_to_chevalley_norms()):
      cX,cY = coeffs[self.rank+2*idx:self.rank+2*idx+2]
      pos_coeffs.append((1j*cX + cY)/norm)
      neg_coeffs.append((1j*cX - cY)/norm)
    return pos_coeffs + cartan_coeffs + neg_coeffs

  def _chevalley_to_CWcompact(self, coeffs):
    npos = len(self._positive_roots)
    nstart = npos + self.rank
    cartan_coeffs = self._simple_coroots * sympy.Matrix(coeffs[npos:nstart])
    CWcoeffs = [-1j*complex(a) for a in cartan_coeffs]
    for idx,norm in enumerate(self._CW_to_chevalley_norms()):
      cpos = coeffs[idx]
      cneg = coeffs[idx+nstart]
      cX = (cpos + cneg)/2j * norm
      cY = (cpos - cneg)/2 * norm
      CWcoeffs.extend([cX,cY])
    return CWcoeffs

  def _firrep1d(self, l, g):
    assert l == self.triv
    return 1

  def _fdim(self, lam):
    if not any(lam):
      return 1
    if any(li < 0 for li in lam):
      lam = negate(lam)
    weylvec = rones((self.rank,))
    # Weyl vector & weyl+lambda in "coweight" basis dual to simple roots
    rls = rarray(self._rootlengthsquared[:self.rank])
    coweyl = 2*weylvec*rls
    coweyllam = 2*(weylvec + rarray(lam))*rls
    d = 1
    for alpha in self._positive_roots:
      d *= coweyllam.dot(alpha)
      d /= coweyl.dot(alpha)
    assert d.denominator == 1
    return d.numerator

  def _ffusion(self, lambda1, lambda2):
    if any(li < 0 for li in lambda1+lambda2):
      return NotImplemented
    if self.irrep_order(lambda1) > self.irrep_order(lambda2):
      return NotImplemented
    if not any(lambda1):
      return [(lambda2,None)]
    return self._fusion_from_decomp(lambda1,lambda2)

  def _fS(self, lam):
    return self.get_ratrep(lam).charge_conjugator_fp().todense()

  def _fusion_from_decomp(self, lambda1, lambda2):
    # TODO is this inefficient enough to make it preferable to directly use 3js?
    decomp = self.tensor_decompose_rational(lambda1, lambda2)
    for lam3,symbol in decomp._3js.items():
      yield self.getduallabel(lam3),symbol.asdenseunitary()
    

  # TODO override Haargen, compose
  # TODO override product() with SemisimpleLieAlgebra
  def compose(self, g, h, maxiter=30, realcheck=1e-14):
    """Compose sufficiently small elements of (compact) Lie group 
    (see compact_irrep for basis)
    Uses Dynkin expansion of BCH formula (not heavily simplified, so nth
    iteration has 2^(n-1) terms - may be very slow even for given value of
    maxiter)
    If g & h have all-real components, will return real elements
      If realcheck is nonzero, will compare imaginary parts of computed
      components to realcheck * naive norm of gh"""
    isreal = all(isinstance(xi,(float,np.floating)) for x in (g,h) for xi in x)
    ghchev = self.compose_chevalley(self._CWcompact_to_chevalley(g),
      self._CWcompact_to_chevalley(h), maxiter=maxiter)
    ghCW = self._chevalley_to_CWcompact(ghchev)
    if isreal and realcheck:
      normcheck = realcheck*linalg.norm(ghCW)
      ghreal = []
      for i,x in enumerate(ghCW):
        if abs(x.imag) > normcheck:
          raise ValueError('%dth component of gh, %0.3e + %0.3e i, has '
            'excessive imaginary component (compared to norm %0.3e of gh,'
            'or %0.3e, %0.3e of g and h' % (i,x.real,x.imag,
            linalg.norm(ghCW),linalg.norm(g),linalg.norm(h)))
        ghreal.append(float(x.real))
      return tuple(ghreal)
    return ghCW
            

  def compose_chevalley(self, X, Y, maxiter=30):
    eps = 1e-16
    # Exit when 2 sequential terms have (naive) relative norm < epsilon
    # Construct adjoint actions of X and Y operating in the Chevalley basis
    adX = np.zeros((self.dimension,self.dimension),dtype=CPLXTYPE)
    adY = np.zeros((self.dimension,self.dimension),dtype=CPLXTYPE)
    for i in range(self.dimension):
      adX += CPLXTYPE(X[i]) * self.chevalley_structure[i]
      adY += CPLXTYPE(Y[i]) * self.chevalley_structure[i]
    Z = np.array(X,dtype=CPLXTYPE) + np.array(Y,dtype=CPLXTYPE)
    # All commutators encountered so far, indexed by sequence of 0s and 1s
    # representing Xs and Ys respectively, final pair omitted (always [X,Y])
    commutators = {():adX.dot(Y)}
    ads = [adX,adY]
    last_eps = linalg.norm(commutators[()]/2) < linalg.norm(Z)*eps
    Z += commutators[()]/2
    for N in range(3,maxiter):
      term = np.zeros_like(Z)
      for xystring in itertools.product((0,1),repeat=N-2):
        commutators[xystring] = ads[xystring[0]].dot(commutators[xystring[1:]])
        coeffXY = _BCH_dynkin_coefficient_XY(xystring)
        term += float(coeffXY) * commutators[xystring]
      curr_eps = linalg.norm(term) < linalg.norm(Z)*eps
      Z += term
      if last_eps and curr_eps:
        print(N)
        return Z
      last_eps = curr_eps
    raise OverflowError('Baker-Campbell-Hausdorff series failed to converge after %d iterations'%maxiter)
    

  def complex_pairorder(self, l):
    # Use lexical ordering on Dynkin labels
    return l < self.dual(l)

  def __reduce__(self):
    # TODO save irreps, 3js?
    return self.__class__, (self._CK_series, self.rank)


class HighestWeightRepresentation:
  """Base class for HighestWeightRepresentation
  Computation of coefficients or matrix elements is left to base classes
  (to be done either with sympy or RationalArrays)"""
  # Not making GroupDerivedType for now bc all use should be within 
  # SimpleLieAlgebra class definition
  def __init__(self, highest_weight, group, dual=None, **kw_args):
    # Highest weight should be provided as tuple of Dynkin indices
    assert all(li>=0 for li in highest_weight)
    self._group = group
    self._highest = highest_weight

    A = group._cartan_matrix
    rank = group._rank
    self._lambda = symdot(group._fund_weights.T, highest_weight)
    self._construction = {} # Specific basis used wrt lowering operators
                            # - useful for tensor product decomposition

    if dual is None:
      has_zero = self._construct_weights(**kw_args)

      self._lowest = self._weights_by_depth[-1][0]
      self._dual_weight = tuple(-l for l in self._lowest)
      if self._dual_weight != highest_weight:
        self.FSI = 0
      elif has_zero:
        self.FSI = 1
      else:
        # Check if (compact form of) algebra admits quaternionic irreps 
        # TODO confirm if 'contains weight zero' test works when this is the
        # case - seems like should be true based on idea that HW with quaternionic
        # irreps form sublattice of order 2 in self-dual irreps?
        # Underpredicts for D6?
        if group.quaternionic:
          if group._CK_series == 'A':
            if highest_weight[rank//2]%2:
              self.FSI = -1
            else:
              self.FSI = 1
          elif group._CK_series == 'B':
            self.FSI = 1-2*(highest_weight[-1]%2)
          elif group._CK_series == 'C':
            if any(li%2 for li in highest_weight[::2]):
              self.FSI = -1
            else:
              self.FSI = 1
          elif group._CK_series == 'D':
            if (highest_weight[-2]+highest_weight[-1])%2:
              self.FSI = -1
            else:
              self.FSI = 1
          elif group._CK_series == 'E':
            # ???
            self.FSI = -1
        else:
          self.FSI = 1
      self._wbd_depth_order = self._weights_by_depth
      self._reversed = False
      self._assign_weight_indices()

      self.raising_blocks = [{} for i in range(rank)]
      self.lowering_blocks = [{} for i in range(rank)]
      *_, self.dim = self.block_indices
      self._construct_generators()
      
      self._CC = None
    else:
      self._construct_from_dual(dual)
      self._wbd_depth_order = self._weights_by_depth[::-1]
      self._reversed = True
      self._assign_weight_indices()
      self._lowest = self._weights_by_depth[-1]
      self.dim = dual.dim
      self._dual_weight = dual._highest
      self.FSI = 0

  def _assign_weight_indices(self):
    self.weight_indices = {}
    ifull = 0
    for depth,weights in enumerate(self._weights_by_depth):
      for idx,mu in enumerate(weights):
        self.weight_indices[mu] = ifull
        ifull += 1

  @property
  def label(self):
    return self._highest

  @property
  def weights(self):
    return itertools.chain.from_iterable(self._weights_by_depth)

  @property
  def degeneracies(self):
    return itertools.chain.from_iterable(self._degeneracies_by_depth)

  @property
  def block_indices(self):
    return itertools.accumulate(self.degeneracies)

  def degeneracy(self,mu,strict=True):
    for weights,degens in zip(self._weights_by_depth,self._degeneracies_by_depth):
      for mu0,d in zip(weights,degens):
        if mu == mu0:
          return d
    if strict:
      raise KeyError('Weight %s not found'%(mu,))
    else:
      return 0

  @property
  def depth(self):
    return len(self._weights_by_depth)

  def __contains__(self, mu):
    # Has weight mu
    return mu in self.weight_indices

  def duallabel(self):
    # Return Dynkin label for dual rep
    return self._dual_weight

  @property
  def dual(self):
    return self._group.get_irrep(self._dual_weight)

  @property
  def block_ind_dict(self):
    blockidx = [0] + list(self.block_indices)
    return {mu:blockidx[idx] for mu,idx in self.weight_indices.items()}

  @property
  def block_slice_dict(self):
    blockidx = [0] + list(self.block_indices)
    return {mu:slice(*blockidx[idx:idx+2]) for mu,idx in self.weight_indices.items()}

class TrivialRepresentation(HighestWeightRepresentation):
  def __init__(self, group):
    self._group = group
    self._highest = group.rank*(0,)
    self._lambda = self._highest
    self._CC = (self, {self._highest:rones((1,1))})
    self._norm_squares = {self._highest:rones((1,))}
    self.raising_blocks = group.rank*[{}]
    self.lowering_blocks = group.rank*[{}]

  def __contains__(self,mu):
    return mu == self._highest

  @property
  def weights(self):
    return [self._highest]

  @property
  def degeneracies(self):
    return [1]

  @property
  def block_indices(self):
    return [1]

  @property
  def weight_indices(self):
    return {self._highest:0}

  @property
  def depth(self):
    return 1

  @property
  def dim(self):
    return 1

  def duallabel(self):
    return self._highest

  @property
  def FSI(self):
    return 1

  @property
  def dual(self):
    return self

  def degeneracy(self,mu,strict=True):
    if mu == self._highest:
      return 1
    elif strict:
      raise KeyError('Weight %s does not belong to trivial representation'%mu)
    else:
      return 0

  def diagonal_chevalley(self,i):
    return sympy.zeros(1,1)
  
  def fp_chevalley(self, which, i, precision='double'):
    if precision == 'longdouble':
      dtype = np.longdouble
    else:
      dtype = float
    return sparse.csr_array((1,1),dtype=dtype)

  def _construct_charge_conjugator(self, dualrep=None):
    return self._CC[1]

  def raise_idx_block(self, mu, array, idx, inplace=False):
    if inplace:
      return array
    else:
      return array.copy()

  def lower_idx_block(self, mu, array, idx, inplace=False):
    if inplace:
      return array
    else:
      return array.copy()

  @property
  def _weights_by_depth(self):
    return [[self._highest]]

class DimensionCheckException(Exception):
  pass

class DummyIrrep:
  # Determine weights & multiplicities with Freudenthal formula, but
  # don't determine basis
  # TODO subclass HighestWeightRepresentation?
  # TODO current form of recursion is very inefficient
  # (but, in use case for supporting methods that are actually calculating
  # matrix elements, is it 
  def __init__(self, highest_weight, group):
    assert all(li>=0 for li in highest_weight)
    self._group = group
    rank = group.rank
    self._highest = highest_weight
    if not any(highest_weight): # Trivial rep
      self._weights_by_depth = [[highest_weight]]
      self.multiplicities = {highest_weight:1}
    else:
      self._construct_weights()
    assert len(self._weights_by_depth[-1]) == 1
    self._lowest = self._weights_by_depth[-1][0]
    self._dual_weight = tuple(-l for l in self._lowest)
    if self._dual_weight != highest_weight:
      self.FSI = 0
    elif rank*(0,) in self.multiplicities:
      self.FSI = 1
    else:
      if group.quaternionic:
        if group._CK_series == 'A':
          if highest_weight[rank//2]%2:
            self.FSI = -1
          else:
            self.FSI = 1
        elif group._CK_series == 'B':
          self.FSI = 1-2*(highest_weight[-1]%2)
        elif group._CK_series == 'C':
          if any(li%2 for li in highest_weight[::2]):
            self.FSI = -1
          else:
            self.FSI = 1
        elif group._CK_series == 'D':
          if (highest_weight[-2]+highest_weight[-1])%2:
            self.FSI = -1
          else:
            self.FSI = 1
        elif group._CK_series == 'E':
          # ???
          self.FSI = -1
      else:
        self.FSI = 1

  def _construct_weights(self):
    group = self._group
    A = group._cartan_matrix
    rank = group._rank
    lower = group.lowerweight
    weights_by_depth = []
    multiplicities = {self._highest:1}
    weights_curr = [self._highest]
    if group._CK_series == 'G':
      rlsdiv = 3
    else:
      rlsdiv = 1
    F,Fdiv = group._quad_form_int
    idx_by_level = []
    iroot = 0
    for roots in group._positive_by_level[::-1]:
      ir2 = iroot+len(roots)
      idx_by_level.insert(0,np.arange(iroot,ir2))
      iroot = ir2
    copos = group._positive_roots * group._rls_int[None,:rank]
    depth = 0
    heighttot = len(group._positive_by_level)
    lamlam = np.outer(self._highest,self._highest)
    while len(weights_curr):
      depth += 1
      assert depth < 1000 # For now
      weights_by_depth.append(weights_curr)
      weights_last,weights_curr = weights_curr,[]
      tested = set()
      for mu0 in weights_last:
        for i in range(rank):
          mu = lower(mu0,i)
          if mu in tested:
            continue
          tested.add(mu)
          # Denominator (with global factor of 1/rlsdiv)
          num = 0
          for lvl,roots in enumerate(group._positive_by_level):
            height = heighttot-lvl
            for idx,alpha in zip(idx_by_level[lvl],roots):
              for n in range(1,depth//height+1):
                muplus = self._group.raisebyroot(mu,n,idx)
                if muplus not in multiplicities:
                  continue
                # (mu + n alpha, alpha) * rlsdiv
                mupdot = copos[idx].dot(muplus)
                num += mupdot*multiplicities[muplus]
          if num:
            # Otherwise multiplicity = 0
            denommat = lamlam - np.outer(mu,mu)
            for i in range(rank):
              denommat[i] += 2*(self._highest[i] - mu[i])
            denom = np.sum(F * denommat) # / Fdiv
            assert (num*Fdiv)%(denom*rlsdiv) == 0
            m = (num*Fdiv)//(denom*rlsdiv)
            assert m > 0
            multiplicities[mu] = m
            weights_curr.append(mu)
    self._weights_by_depth = weights_by_depth
    self.multiplicities = multiplicities

  @property
  def weights(self):
    return itertools.chain.from_iterable(self._weights_by_depth)
  
  @property
  def dim(self):
    return sum(self.multiplicities.values())

  def duallabel(self):
    # Return Dynkin label for dual rep
    return self._dual_weight

  @property
  def label(self):
    return self._highest



from .rationallie import RationalRepresentation,RationalTensorDecomposition,Rational3j,RationalRacahOperator
