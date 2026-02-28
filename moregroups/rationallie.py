from .liegroups import *
from fractions import *
from .rationallinalg import *
from numbers import Number
from copy import copy,deepcopy
# This module uses a lot of dtype=Fraction. It doesn't mean anything different
# from dtype=object, but you can always hope


def solve_overcomplete(M, B):
  # Solve an overcomplete system of equations MX = B
  # (i.e. overdetermined but should have an exact solution)
  # TODO faster to start with "QR" decomposition?
  reducer = rarray(M)
  solver = rarray(B)
  N,m = reducer.shape
  N2,n = solver.shape
  assert N == N2
  for ileft in range(m):
    # Identify row which (when reduced) has nonzero entry in ileft position
    for i2 in range(ileft,N2):
      for j in range(ileft):
        if reducer[i2,j]:
          shift = reducer[i2,j]
          reducer[i2] -= shift*reducer[j]
          solver[i2] -= shift*solver[j]
      if reducer[i2,ileft]:
        if i2 != ileft:
          # Swap rows
          reducer[[i2,ileft],:] = reducer[[ileft,i2],:]
          solver[[i2,ileft],:] = solver[[ileft,i2],:]
        # Normalize
        factor = reducer[ileft,ileft]
        reducer[ileft] /= factor
        solver[ileft] /= factor
  reducer = reducer[:m].rereduce()
  solver = solver[:m].rereduce()
  # Now zero remaining elements
  for i in range(m-2,-1,-1):
    for j in range(i+1,m):
      shift = reducer[i,j]
      if shift:
        reducer[i] -= shift*reducer[j]
        solver[i] -= shift*solver[j]
  return solver.rereduce()

def takagi_rational(S, normsq):
  # TODO untested, not in terms of RationalArray
  # Takagi decomposition of real rational-entry transformation S
  # st sqrt(normsq) * S * sqrt(normsq)^-1 is symmetric & unitary
  # Decomposition S = normsq^-1 * U * Delta * U^T
  # (Delta is diagonal, rational-entry, U is rational-but-possibly-complex
  # entry, sqrt(normsq)^-1 * U * Delta^1/2 is unitary)
  N = S.shape[0]
  if not no_strict_checks:
    Ssym = S.raise_index(normsq,1)
    assert np.all(Ssym.T == Ssym)
    assert np.all(S.dot(S) == reye(N))
  URe = rzeros((N,0))
  UIm = rzeros((N,0))
  normnew = rzeros((0,))
  hasim = False
  for k in range(N):
    for sgn in (1,-1):
      y = S[:,k].copy()
      y[k] += sgn
      ylow = normsq*y
      normy = y.dot(ylow)
      # Orthogonalize
      if k:
        norminv = normnew.reciprocal()
        coeffR = URe.T.dot(ylow)
        normy -= coeffR.dot(norminv*coeffR)
        if hasim:
          coeffI = UIm.T.dot(ylow)
          normy -= coeffI.dot(norminv*coeffI)
      if normy == 0:
        assert sgn == 1
        # Try -1
        continue
      if k:
        yR = y - URe.dot(norminv*coeffR)
        if hasim:
          yR -= UIm.dot(norminv*coeffI)
          yI = URe.dot(norminv*coeffI) - UIm.dot(norminv*coeffR)
          yI.rereduce()
      else:
        yR = y
      yR.rereduce()
      if not hasim:
        yI = rzeros((N,))
      nout,nrad = sqrt_reduce(normy)
      normnew = np.append(normnew,nrad)
      if sgn == -1:
        # Multiply by i
        yR,yI = -yI,yR
        hasim = True
      URe = np.hstack((URe,yR[:,None]/nout))
      UIm = np.hstack((UIm,yI[:,None]/nout))
      break # Skip other sign if applicable
  if not no_strict_checks:
    # Unitarity
    #norminv = Fraction(1)/normnew
    UReL = URe.T * normsq[None,:]
    UImL = UIm.T * normsq[None,:]
    lhs = UReL.dot(URe)
    if hasim:
      lhs += UImL.dot(UIm)
      assert np.all(UImL.dot(URe) - UReL.dot(UIm) == 0)
    rhs = np.diag(normnew)
    assert np.all(lhs == rhs)
    # Decomposition
    UReR = URe.T.raise_index(normnew,0)
    UImR = UIm.T.raise_index(normnew,0)
    lhs = URe.dot(UReR)
    if hasim:
      lhs -= UIm.dot(UImR)
      assert np.all(URe.dot(UImR) == -UIm.dot(UReR))
    assert np.all(lhs == Ssym)
  return (URe.rereduce(),UIm.rereduce()),normnew

def plus_assign(d,k,arr):
  if k in d:
    d[k] += arr
  else:
    d[k] = arr

class RationalRepresentation(HighestWeightRepresentation):
  # Highest weight representation with rational (Fraction) data
  def _construct_weights(self, dimcheck=None):
    group = self._group
    rank = group.rank
    weights_by_depth = [[self._highest]]
    degen_by_depth = [[1]]
    gen_action_by_depth = [[rank*[None]]]
    self._norm_squares = {self._highest: rones((1,))}

    weights_curr = []
    degen_curr = []
    gen_curr = []
    normsq_curr = []
    lower = group.lowerweight
    wraise = group.raiseweight
    dim = 1
    # Next-highest level
    for i in range(rank):
      norm = self._highest[i]
      if norm:
        mu = lower(self._highest,i)
        weights_curr.append(mu)
        degen_curr.append(1)
        gen_curr.append(rank*[None])
        # Improper normalization: while the generator is 1 in this basis,
        # the stored object is (norms of target space)*generator
        gen_curr[-1][i] = rarray([[norm]])
        #print(i,norm)
        self._norm_squares[mu] = rarray([norm])
        constr = rank*[None]
        constr[i] = rones((1,1))
        #self._construction[mu] = self._norm_squares[mu], constr
        self._construction[mu] = constr
        dim += 1

    # General levels
    reached_lowest = False
    zero = rank*(0,)
    has_zero = (zero in weights_curr)
    while len(weights_curr):
      weights_by_depth.append(weights_curr)
      degen_by_depth.append(degen_curr)
      gen_action_by_depth.append(gen_curr)

      weights_last,weights_curr = weights_curr,[]
      degen_last,degen_curr = degen_curr,[]
      gen_last,gen_curr = gen_curr,[]

      enumeration = {w:i for i,w in enumerate(weights_last)}

      for i0 in range(rank):
        for idx0,muplus in enumerate(weights_last):
          mu = lower(muplus,i0)
          if mu in weights_curr:
            # Already covered
            continue
          elif muplus[i0] == 0 and gen_last[idx0][i0] is None:
            # Does not contribute
            continue
          gram_rows = []
          contributing = []
          for j in range(i0,rank):
            muplus_j = wraise(mu,j)
            if muplus_j not in weights_last:
              continue
            jdx = enumeration[muplus_j]
            Ej_j = gen_last[jdx][j]
            if muplus_j[j] == 0 and Ej_j is None:
              continue
            dj = degen_last[jdx]
            # Diagonal block
            jdotj = muplus_j[j] * np.diag(self._norm_squares[muplus_j])
            if Ej_j is not None:
              Ejjr = self.raise_idx_block(wraise(muplus_j,j),Ej_j,0)
              jdotj += Ej_j.T.dot(Ejjr)
            if not np.any(jdotj):
              # Does not contribute
              continue
            # Other blocks with known contributors
            row = []
            for iblock,(i,idx) in enumerate(contributing):
              Ej_i = gen_last[idx][j]
              Ei_j = gen_last[jdx][i]
              if Ej_i is None or Ei_j is None:
                block = rzeros((degen_last[idx],dj))
              else:
                Eijr = self.raise_idx_block(wraise(muplus_j,i),Ei_j,0)
                block = Ej_i.T.dot(Eijr)
              gram_rows[iblock] = np.hstack((gram_rows[iblock],block))
              row.append(block.T)
            row.append(jdotj)
            gram_rows.append(np.hstack(row))
            contributing.append((j,jdx))
          if not contributing:
            # No valid contributors found with full test
            continue
          weights_curr.append(mu)
          if mu == zero:
            has_zero = True
          gram = np.vstack(gram_rows)
          basis,normsq = rational_gramschmidt_implicit(gram)
          degen_curr.append(len(normsq))
          dim += len(normsq)
          if dimcheck and dim > dimcheck:
            raise DimensionCheckException()
          self._norm_squares[mu] = normsq
          #print(mu,normsq.astype(float))
          # Action of Chevalley generators from orthonormal basis:
          genaction = rank*[None]
          self._construction[mu] = rank*[None]
          k0 = 0
          for iblock,(i,idx) in enumerate(contributing):
            k1 = k0 + degen_last[idx]
            genaction[i] = gram[k0:k1,:].dot(basis).rereduce()
            self._construction[mu][i] = basis[k0:k1,:].T.copy().rereduce()
            k0 = k1
            #assert no_strict_checks or np.all(self._construction[mu][i] == genaction[i].T * Fraction(1)/normsq[:,None])
          gen_curr.append(genaction)

    assert len(weights_by_depth[-1]) == 1
    self._weights_by_depth = weights_by_depth
    # NOTE _generators_by_depth are not true matrix elements
    # (require normalization on one side or the other for each case)
    self._generators_by_depth = gen_action_by_depth
    self._degeneracies_by_depth = degen_by_depth
    return has_zero

  def _construct_generators(self):
    for depth,weights in enumerate(self._weights_by_depth):
      for idx,mu in enumerate(weights):
        for i,Eimu in enumerate(self._generators_by_depth[depth][idx]):
          if Eimu is None:
            continue
          muplus = self._group.raiseweight(mu,i)
          assert muplus in self.weight_indices # Just to be sure
          # Raising operator: normalize in target space, V_muplus
          Eraise = self.raise_idx_block(muplus,Eimu,0)
          Eraise.rereduce()
          self.raising_blocks[i][mu] = (muplus,Eraise)
          assert muplus not in self.lowering_blocks[i] # Again
          # Lowering operator: normalize in target space, V_mu
          Elower = self.raise_idx_block(mu,Eimu.T,0)
          Elower.rereduce()
          self.lowering_blocks[i][muplus] = (mu,Elower)
          #assert no_strict_checks or np.all(self._construction[mu][i] == self.lowering_blocks[i][muplus][1])
        if not no_strict_checks and mu != self._highest:
          cmp = None
          for i in range(self._group.rank):
            if mu in self.raising_blocks[i]:
              muplus = self._group.raiseweight(mu,i)
              f = self.lowering_blocks[i][muplus][1]
              A = self._construction[mu][i]
              assert A is not None
              if cmp is None:
                cmp = f.dot(A.T)
              else:
                cmp += f.dot(A.T)
          assert np.all(cmp == reye(cmp.shape[0]))
    # After this "_generators_by_depth" should not be needed,
    # kill them to avoid confusion
    self._generators_by_depth = None

  def _construct_from_dual(self, dual):
    assert dual.FSI == 0
    assert dual._dual_weight == self._highest
    self.lowering_blocks = []
    self.raising_blocks = []
    self._norm_squares = {}
    self._CC = (dual,{})
    dual._CC = (self,{})
    self._weights_by_depth = []
    self._generators_by_depth = None
    self._degeneracies_by_depth = []
    rank = self._group.rank
    for dd in range(dual.depth):
      weights = []
      for idx,mud in enumerate(dual._weights_by_depth[dd]):
        d = dual._degeneracies_by_depth[dd][idx]
        mu = negate(mud)
        weights.append(mu)
        self._norm_squares[mu] = dual._norm_squares[mud]
        I = reye(d)
        self._CC[1][mu] = I
        dual._CC[1][mud] = I
        self._construction[mu] = rank*[None]
      self._degeneracies_by_depth.append(dual._degeneracies_by_depth[dd])
      self._weights_by_depth.append(weights)

    for i in range(self._group.rank):
      lowerer = {}
      for mud,(mupd,ei) in dual.raising_blocks[i].items():
        mu = negate(mud)
        muminus = negate(mupd)
        lowerer[mu] = (muminus,-ei)
      self.lowering_blocks.append(lowerer)
      raiser = {}
      for mud,(mumd,fi) in dual.lowering_blocks[i].items():
        raiser[negate(mud)] = (negate(mumd),-fi)
      self.raising_blocks.append(raiser)
    # Now build "construction" from pseudo-inverse of stacked fs
    for mu in self.weights:
      if mu == self._highest:
        continue
      fs = []
      for i in range(rank):
        if mu in self.raising_blocks[i]:
          muplus = self._group.raiseweight(mu,i)
          fs.append(self.lowering_blocks[i][muplus][1])
          
      fstack = np.hstack(fs)
      A = inverse_rational(fstack.dot(fstack.T)).dot(fstack)
      idx0 = 0
      for i in range(rank):
        if mu in self.raising_blocks[i]:
          muplus,block = self.raising_blocks[i][mu]
          idx1 = idx0 + block.shape[0]
          self._construction[mu][i] = A[:,idx0:idx1]
          idx0 = idx1
      if not no_strict_checks:
        cmp = None
        for i in range(self._group.rank):
          if mu in self.raising_blocks[i]:
            muplus = self._group.raiseweight(mu,i)
            f = self.lowering_blocks[i][muplus][1]
            A = self._construction[mu][i]
            assert A is not None
            if cmp is None:
              cmp = f.dot(A.T)
            else:
              cmp += f.dot(A.T)
        assert np.all(cmp == reye(cmp.shape[0]))


  @property
  def generators(self):
    # Dense matrices, so should be instantiated sparingly
    gens = []
    indices = [0] + list(self.block_indices)
    for i in range(self._group.rank):
      Ei = rzeros((self.dim,self.dim))
      for mu1,(mu2,Eblock) in self.raising_blocks[i].items():
        idx1 = self.weight_indices[mu1]
        idx2 = self.weight_indices[mu2]
        Ei[indices[idx2]:indices[idx2+1],indices[idx1]:indices[idx1+1]] \
          = Eblock
      gens.append(Ei)
    return gens

  @property
  def normsquare_array(self):
    return np.concatenate([self._norm_squares[mu] for mu in self.weights])

  def raise_idx_block(self, mu, array, idx, inplace=False):
    """Raise index of array that corresponds to weight mu"""
    return array.raise_index(self._norm_squares[mu],idx,inplace)

  def lower_idx_block(self, mu, array, idx, inplace=False):
    """Lower index of array that corresponds to weight mu"""
    return array.lower_index(self._norm_squares[mu],idx,inplace)

  @property
  def dual(self):
    return self._group.get_ratrep(self._dual_weight)

  def fp_chevalley(self, which, i, precision='double', unitary=True,
      realize=False):
    """Chevalley generators (which='h','e','f'), evaluated as floats in a 
    scipy.sparse array
    If unitary, converts to unitary form; if realize and irrep is real,
    also applies change-of-basis to real matrix elements
      (in compact real form; note that we are working in split form which
      has real matrix elements that become complex)"""
    shape = (self.dim,self.dim)
    if which == 'h':
      diag = []
      for mu,d in zip(self.weights,self.degeneracies):
        diag.extend(d*[mu[i]])
      hi = scipy.sparse.diags_array(diag,shape=shape, dtype=int,format='csc')
      if realize and self.FSI == 1:
        cob = self.real_cob()
        hi = cob.dot(hi).dot(cob.T.conj())
      return hi
    else:
      indices = [0] + list(self.block_indices)
      if which == 'e':
        blocks = self.raising_blocks[i]
      else:
        blocks = self.lowering_blocks[i]
      # TODO longdouble precision
      dtype = float
      indptr = [0]
      rowind = []
      data = []
      dataidx = 0
      norms = {mu:np.sqrt(self._norm_squares[mu].asfarray()) for mu in self.weights}
      for mu1,idx1,d in zip(self.weights,indices,self.degeneracies):
        if mu1 not in blocks:
          indptr.extend(d*[dataidx])
        else:
          mu2,Eblock = blocks[mu1]
          idx2 = indices[self.weight_indices[mu2]]
          Efloat = Eblock.asfarray()
          if unitary:
            Efloat = norms[mu2][:,None]*Efloat/norms[mu1][None,:]
          for i1 in range(d):
            rows, = np.nonzero(Eblock[:,i1])
            dataidx += len(rows)
            indptr.append(dataidx)
            rowind.extend(rows+idx2)
            data.extend(Efloat[rows,i1])
      mat = scipy.sparse.csc_array((data,rowind,indptr),
        shape=(self.dim,self.dim))
      if realize and self.FSI == 1:
        assert unitary
        cob = self.real_cob()
        return cob.tocsc().dot(mat).dot(cob.T.conj())
      return mat

  def diagonal_chevalley(self,i):
    diag = np.zeros((self.dim,),dtype=int)
    for mu,d,idx in zip(self.weights,self.degeneracies,self.block_indices):
      diag[idx-d:idx] = mu[i]
    return np.diag(diag)

  def check_representation(self, logger, matrix_checking='rational',
      unitary=True,tolerance=0):
    # Compare representation definitons with Chevalley structure coefficients
    if matrix_checking == 'numpy':
      dtype = complex
      num = True
      if not tolerance:
        tolerance = 1e-14
    elif matrix_checking == 'longdouble':
      # TODO implement or discard
      assert False
      dtype = np.complex256
      num = True
      if not tolerance:
        tolerance = 1e-18
    else:
      assert matrix_checking == 'rational'
      num = False
    g = self._group
    rank = g.rank
    npos = len(g._positive_roots)
    shape = (self.dim,self.dim)
    if num:
      hs = [self.fp_chevalley('h',i,unitary=unitary) for i in range(rank)]
      Es = [self.fp_chevalley('e',i,unitary=unitary) for i in range(rank)]
      Fs = [self.fp_chevalley('f',i,unitary=unitary) for i in range(rank)]
      zeros = scipy.sparse.csc_array((self.dim,self.dim),dtype=dtype)
    else:
      hs = [self.diagonal_chevalley(i) for i in range(rank)]
      Es = list(self.generators)
      nmsq = self.normsquare_array
      Fs = []
      for i in range(rank):
        fi = Es[i].T.raise_index(nmsq,0).lower_index(nmsq,1)
        fi.rereduce()
        Fs.append(fi)
      zeros = rzeros((self.dim,self.dim))
    for idx in range(rank,npos):
      p,i,idxm = g._chevalley_definitions[idx]
      if num:
        div = 1/p
      else:
        div = Fraction(1,p)
      Es.append((Es[i].dot(Es[idxm]) - Es[idxm].dot(Es[i]))*div)
      Fs.append((Fs[idxm].dot(Fs[i]) - Fs[i].dot(Fs[idxm]))*div)
    basis = Es + hs + Fs
    labels = ['E_%s'%g._positive_roots[idx] for idx in range(npos)] \
      + ['h_%d'%i for i in range(rank)] \
      + ['F_%s'%g._positive_roots[idx] for idx in range(npos)]
    # Finally compare commutation with Chevalley structure coefficients
    def badlog(msg,*args):
      if logger is not None:
        logger.warn('~~'+msg+'~~',*args)
        badlog.correct = False
      else:
        raise ValueError(msg%args)
    badlog.correct = True
    def numnorm(mat,sub=None):
      if sub is not None:
        mat = mat-sub
      #if not num:
        #mat = mat.astype(dtype)
      return np.linalg.norm(mat)
    for idx1,(M1,s1) in enumerate(zip(basis,labels)):
      for idx2,(M2,s2) in enumerate(zip(basis,labels)):
        compare = zeros.copy()
        c_all = g.chevalley_structure[idx1][:,idx2]
        coeffs = [(i,c_all[i]) for i, in np.argwhere(c_all)]
        t1 = M1.dot(M2)
        t2 = M2.dot(M1)
        if num:
          t1 = t1.toarray()
          t2 = t2.toarray()
        if len(coeffs) == 0:
          if not (num and np.allclose(t1,t2,atol=tolerance,rtol=tolerance)) \
              and not (not num and np.all(t1 == t2)):
            badlog('%s and %s do not commute (failure %0.4g/%0.4g)',
              s1,s2,numnorm(t1,t2), max(numnorm(t1),numnorm(t2)))
        else:
          for idx3,coeff in coeffs:
            compare += int(coeff)*basis[idx3]
          if not (num and np.allclose(t1-t2,compare.toarray(),atol=tolerance,rtol=tolerance)) and not (not num and np.all(t1-t2==compare)):
            badlog('Failure in [%s,%s] (%0.4g/%0.4g)',s1,s2,
              numnorm(t1-t2,compare.toarray()),numnorm(compare.toarray()))
    return badlog.correct

  def charge_conjugator(self, dualrep=None):
    # Isometry mapping dual space of self to a dualrep
    # (self if not provided, i.e. if real or quaternionic)
    # Returns exact (dense, Fraction-type, non-unitary) matrix
    self._construct_charge_conjugator(dualrep)
    dualrep,blocks = self._CC
    indices = [0] + list(self.block_indices)
    dualindices = [0] + list(dualrep.block_indices)
    matrix = rzeros((self.dim,self.dim))
    #matrix[0,-1] = 1
    #weights = self.weights
    #next(weights)
    for mu in self.weights:
      idx = self.weight_indices[mu]
      didx = dualrep.weight_indices[tuple(-mui for mui in mu)]
      matrix[slice(*dualindices[didx:didx+2]),slice(*indices[idx:idx+2])] \
        = blocks[mu]
    return matrix

  def charge_conjugator_fp(self, dualrep=None, unitary=True):
    # Floating-point, sparse, unitary matrix for charge-conjugator
    # TODO longdouble precision
    self._construct_charge_conjugator(dualrep)
    dualrep,blocks = self._CC
    dtype = float
    indptr = [0]
    rowind = []
    data = []
    dataidx = 0
    indices = [0] + list(self.block_indices)
    dualindices = [0] + list(dualrep.block_indices)
    for mu,idx,d in zip(self.weights,indices,self.degeneracies):
      mud = negate(mu) 
      idxd = dualindices[dualrep.weight_indices[mud]]
      Cfloat = blocks[mu].asfarray()
      if unitary:
        normright = np.sqrt(self._norm_squares[mu].asfarray())
        normleft = np.sqrt(dualrep._norm_squares[mud].asfarray())
        Cfloat = normleft[:,None]*Cfloat/normright[None,:]
      for i in range(d):
        rows, = np.nonzero(blocks[mu][:,i])
        dataidx += len(rows)
        indptr.append(dataidx)
        rowind.extend(rows+idxd)
        data.extend(Cfloat[rows,i])
    mat = scipy.sparse.csc_array((data,rowind,indptr),
      shape=(self.dim,self.dim))
    return mat

  def _construct_charge_conjugator(self, dualrep=None):
    if dualrep is None:
      assert self.FSI
      dualrep = self
    if self._CC is not None:
      assert self._CC[0] is dualrep
      return self._CC[1]
    assert self.depth == dualrep.depth
    assert self.dim == dualrep.dim
    rank = self._group.rank
    blocks = {}
    # Fix CC(|lambda>*) = |-lambda>
    blocks[self._highest] = dualrep._norm_squares[dualrep._lowest].reciprocal().reshape((1,1))
    indices = [0] + list(self.block_indices)
    dualindices = [0] + list(dualrep.block_indices)
    weights = self.weights
    next(weights)
    for mu in weights:
      mud = negate(mu)
      EL = []
      ER = []
      for i in range(rank):
        if mu in self.raising_blocks[i]:
          muplus,Ei = self.raising_blocks[i][mu]
          mudminus,Edi = dualrep.lowering_blocks[i][mud]
          assert mudminus == negate(muplus)
          # Hermitian adjoint wrt non-normalized basis requires
          # normalization factors
          CH = self.raise_idx_block(muplus,blocks[muplus].T,0)
          dualrep.lower_idx_block(mudminus,CH,1,True)
          EL.append(CH.dot(Edi).rereduce())
          ER.append(Ei)
      EL = np.vstack(EL)
      ER = np.vstack(ER)
      blocks[mu] = -solve_overcomplete(EL,ER)
      # Solution should still be exact
      if not no_strict_checks:
        assert np.all(EL.dot(blocks[mu])==-ER)
        normleft = Fraction(1)/self._norm_squares[mu]
        normright = dualrep._norm_squares[mud]
        adj = self.raise_idx_block(mu,blocks[mu].T,0)
        dualrep.lower_idx_block(mud,adj,1,True)
        assert np.all(blocks[mu].dot(adj)==reye(blocks[mu].shape[0]))
    self._CC = (dualrep,blocks)

  def check_cc(self,logger,dualrep,matrix_checking='rational',tolerance=0,
      unitary=None):
    if matrix_checking == 'numpy':
      dtype = float
      num = True
      if not tolerance:
        tolerance = 1e-14
      if unitary is None:
        unitary = True
    elif matrix_checking == 'longdouble':
      # TODO implement or discard
      assert False
      dtype = np.float128
      num = True
      if not tolerance:
        tolerance = 1e-18
      if unitary is None:
        unitary = True
    else:
      assert matrix_checking == 'rational'
      num = False
      unitary = False
    # Should send e_i -> e_i^T, h_i -> -h_i
    if num:
      S = self.charge_conjugator_fp(dualrep,unitary=unitary)
      if unitary:
        Sd = S.T
      else:
        norms = self.normsquare_array.asfarray(dtype)
        normsd = dualrep.normsquare_array.asfarray(dtype)
        Sd = 1/norms[:,None] * S.T * normsd[None,:]
      def matcheck(A,B):
        return np.allclose(A.toarray(),B.toarray(),atol=tolerance,rtol=tolerance)
      def numnorm(mat,sub=None):
        if sub is not None:
          mat = mat-sub
        return np.sqrt((mat**2).sum())
    else:
      S = self.charge_conjugator(dualrep)
      norms = self.normsquare_array
      normsd = dualrep.normsquare_array
      Sd = Fraction(1)/norms[:,None] * S.T * normsd[None,:]
      def matcheck(A,B):
        return np.all(A==B)
      def numnorm(mat,sub=None):
        if sub is not None:
          mat = mat-sub
        mat = np.array(mat,dtype=float)
        return np.linalg.norm(mat)
    def badlog(msg,*args):
      if logger is not None:
        logger.warn('~~'+msg+'~~',*args)
        badlog.correct = False
      else:
        raise ValueError(msg%args)
    badlog.correct = True
    if self is dualrep:
      strs = ['','symmetric','antisymmetric']
      if matcheck(S,Sd):
        fsi_obs = 1
      elif matcheck(S,-Sd):
        fsi_obs = -1
      else:
        badlog('S should be %s but is neither symmetric nor antisymmetric: failure %0.4g',strs[self.FSI],numnorm((Sd-self.FSI*S)))
        fsi_obs = 0
      if fsi_obs and fsi_obs != self.FSI:
        badlog('S should be %s but is instead %s',strs[self.FSI],strs[fsi_obs])
    rank = self._group.rank
    if num:
      hs = [self.fp_chevalley('h',i,unitary=unitary) for i in range(rank)]
      hds = [dualrep.fp_chevalley('h',i,unitary=unitary) for i in range(rank)]
    else:
      hs = [self.diagonal_chevalley(i) for i in range(rank)]
      hds = [dualrep.diagonal_chevalley(i) for i in range(rank)]
    for i in range(rank):
      hcheck = S.dot(hs[i].dot(Sd))
      if not matcheck(hcheck,-hds[i]):
        badlog('Charge conjugation of h_%d gives incorrect result (%0.4g/%0.4g)',i,numnorm(hcheck,-hds[i]),numnorm(hds[i]))
    if not num:
      gens = self.generators
      gend = dualrep.generators
    for i in range(rank):
      if num:
        ei = self.fp_chevalley('e',i,unitary=unitary)
        fd = dualrep.fp_chevalley('f',i,unitary=unitary)
      else:
        ei = gens[i]
        fd = gend[i].T.raise_index(normsd,0).lower_index(normsd,1)
      Echeck = S.dot(ei).dot(Sd)
      if not matcheck(Echeck,-fd):
        badlog('Charge conjugation of e_%d gives incorrect result (%0.4g/%0.4g)',i,numnorm(Echeck,-fd),numnorm(fd))
    return badlog.correct

  def _fix_dual(self):
    # For self-dual (real or quaternionic) representation, set nonzero weights
    # to have S_mu = +- identity
    # If zero is a weight (implies real), obtain Takagi decomposition
    assert self.FSI
    self._construct_charge_conjugator()
    for mu in self.weights:
      nmu = negate(mu)
      # Use the one with higher lexical order as "reference" weight
      if mu > nmu:
        nmsqp = self._norm_squares[mu]
        nmsqm = self._norm_squares[nmu]
        self._norm_squares[nmu] = nmsqp
        S = self._CC[1][mu]
        Sd = S.T.raise_index(nmsqp,0).lower_index(nmsqm,1)
        for i in range(self._group.rank):
          if nmu in self.raising_blocks[i]:
            nmuplus,Eraise = self.raising_blocks[i][nmu]
            Eraise = Eraise.dot(S)
            self.raising_blocks[i][nmu] = (nmuplus,Eraise.rereduce())
            _,Elower = self.lowering_blocks[i][nmuplus]
            Elower = Sd.dot(Elower)
            self.lowering_blocks[i][nmuplus] = (nmu,Elower.rereduce())
            constr = S.T.dot(self._construction[nmu][i])
            self._construction[nmu][i] = constr.rereduce()
          if nmu in self.lowering_blocks[i]:
            nmuminus,Elower = self.lowering_blocks[i][nmu]
            Elower = Elower.dot(S)
            self.lowering_blocks[i][nmu] = (nmuminus,Elower.rereduce())
            _,Eraise = self.raising_blocks[i][nmuminus]
            Eraise = Sd.dot(Eraise)
            self.raising_blocks[i][nmuminus] = (nmu,Eraise.rereduce())
            constr = self._construction[nmuminus][i].dot(Sd.T)
            self._construction[nmuminus][i] = constr.rereduce()
        self._CC[1][mu] = reye(self.degeneracy(mu))
        self._CC[1][nmu] = self.FSI * self._CC[1][mu]
    zerow = self._group.triv
    if zerow in self:
      self._takagi0 = takagi_rational(self._CC[1][zerow],
        self._norm_squares[zerow])
    else:
      self._takagi0 = None

  def real_cob(self):
    # Change into basis where compact real form has real matrix elements
    # i.e. use Takagi decomposition
    # Basis is zero weight space (if present), then paired weight spaces
    # in order
    assert self.FSI == 1
    if hasattr(self, '_real_cob'):
      return self._real_cob
    idx = 0
    row = []
    col = []
    data = []
    blockind = self.block_ind_dict
    if self._takagi0 is not None:
      j0 = blockind[self._group.triv]
      (Ure,Uim),nmsqT = self._takagi0
      norms0 = np.sqrt(self._norm_squares[self._group.triv])
      normsT = np.sqrt(nmsqT)
      for j,i in np.argwhere(np.logical_or(Ure,Uim)):
        row.append(i)
        col.append(j+j0)
        data.append(1/normsT[i] * (Ure[j,i] + 1j*Uim[j,i]) * norms0[j])
      idx += len(nmsqT)
    for mu,d in zip(self.weights,self.degeneracies):
      if mu > negate(mu):
        iplus = blockind[mu]
        iminus = blockind[negate(mu)]
        for imu in range(d):
          row.extend([idx,idx,idx+1,idx+1])
          col.extend(2*[iplus+imu,iminus+imu])
          data.extend([x*np.sqrt(.5) for x in [1,1,1j,-1j]])
          idx += 2
    self._real_cob = scipy.sparse.csr_array((data,(row,col)))
    return self._real_cob


class RationalTensorDecomposition:
  def __init__(self, group, lambda1, lambda2, use_3j=False):
    self.group = group
    self.rep1 = group.get_ratrep(lambda1)
    self.rep2 = group.get_ratrep(lambda2)
    self.weights_by_depth = []
    self.weight_products = {}
    self.degeneracies = {}
    self.normsquares = {}
    self._find_weights()
    self._find_accessible()
    #self._construct_lowering()
    self._3js = {}
    self._decompose(use_3j)
    self.fusion_degen = {lam:N for lam,N,nmsq in self.components}
    self._construct_3js()

  def _find_weights(self):
    if isinstance(self.rep1,TrivialRepresentation):
      self.weights_by_depth = deepcopy(self.rep2._wbd_depth_order)
      self.degeneracies = dict(zip(self.rep2.weights,self.rep2.degeneracies))
      self.normsquares = dict(self.rep2._norm_squares)
      self.weight_products = {mu:[(0,1,d,self.rep1.label,mu)] for mu,d in self.degeneracies.items()}
      return
    elif isinstance(self.rep2,TrivialRepresentation):
      self.weights_by_depth = deepcopy(self.rep1._wbd_depth_order)
      self.degeneracies = dict(zip(self.rep1.weights,self.rep1.degeneracies))
      self.normsquares = dict(self.rep1._norm_squares)
      self.weight_products = {mu:[(0,d,1,mu,self.rep2.label)] for mu,d in self.degeneracies.items()}
      return
    # Map component weights onto product weights
    height1 = len(self.rep1._weights_by_depth)
    height2 = len(self.rep2._weights_by_depth)
    self.height = height1+height2-1
    rank = self.group.rank
    # First map component weights onto product weights
    for lvl in range(self.height):
      self.weights_by_depth.append([])
      for lvl1 in range(max(0,lvl-height2+1),min(height1,lvl+1)):
        lvl2 = lvl - lvl1
        if self.rep1._reversed:
          lvl1 = height1 - lvl1 - 1
        if self.rep2._reversed:
          lvl2 = height2 - lvl2 - 1
        for mu1,d1 in zip(self.rep1._weights_by_depth[lvl1],self.rep1._degeneracies_by_depth[lvl1]):
          for mu2,d2 in zip(self.rep2._weights_by_depth[lvl2],self.rep2._degeneracies_by_depth[lvl2]):
            mu = tuple(mu1[i]+mu2[i] for i in range(rank))
            normsqs = np.ravel(np.outer(self.rep1._norm_squares[mu1],
              self.rep2._norm_squares[mu2]))
            if mu in self.weight_products:
              idx0 = self.degeneracies[mu]
              self.degeneracies[mu] += d1*d2
              self.normsquares[mu] = np.concatenate((self.normsquares[mu],normsqs))
            else:
              idx0 = 0
              self.weight_products[mu] = []
              self.weights_by_depth[-1].append(mu)
              self.degeneracies[mu] = d1*d2
              self.normsquares[mu] = normsqs
            self.weight_products[mu].append((idx0,d1,d2,mu1,mu2))

  def _find_accessible(self):
    # Weights accessible by raising operator from principal Weyl chamber
    by_depth = []
    # First get weights in principal Weyl chamber
    nlvls = (len(self.weights_by_depth)+1)//2
    for weights in self.weights_by_depth[:nlvls]:
      at_depth = set()
      for mu in weights:
        if all(mui >= 0 for mui in mu):
          at_depth.add(mu)
      by_depth.append(at_depth)
    while not by_depth[-1]:
      by_depth.pop()
      nlvls -= 1
    # Now expand
    for lvl in range(nlvls-1,1,-1):
      # Iterate over level below; 0 is accounted for--highest weight--so stop at 2
      candidates = set()
      for muminus in by_depth[lvl]:
        for i in range(self.group.rank):
          candidates.add(self.group.raiseweight(muminus,i))
      by_depth[lvl-1] |= candidates & set(self.weights_by_depth[lvl-1])
    self.accessible_by_depth = by_depth

  def _construct_lowering(self):
    # Construct lowering operators
    # TODO depricated?
    rank = self.group.rank
    self.lowering_blocks = [{} for i in range(rank)]
    rep1,rep2 = self.rep1,self.rep2
    wraise = self.group.raiseweight
    for lvl,weights in enumerate(self.weights_by_depth):
      if lvl == 0:
        continue
      for mu in weights:
        for i in range(rank):
          muplus_i = wraise(mu,i)
          if muplus_i in self.degeneracies:
            block = np.zeros((self.degeneracies[mu],self.degeneracies[muplus_i]),dtype=Fraction)
            for idx,d1,d2,mu1,mu2 in self.weight_products[mu]:
              mu1p = wraise(mu1,i)
              mu2p = wraise(mu2,i)
              for idxb,d1b,d2b,mu1b,mu2b in self.weight_products[muplus_i]:
                if mu1p == mu1b and mu1p in rep1.lowering_blocks[i]:
                  assert mu2b == mu2
                  mu1a,Ei = rep1.lowering_blocks[i][mu1p]
                  for k2 in range(d2):
                    block[idx+k2:idx+d1*d2:d2,idxb+k2:idxb+d1b*d2:d2] += Ei
                elif mu2p == mu2b and mu2p in rep2.lowering_blocks[i]:
                  assert mu1b == mu1
                  mu2a,Ei = rep2.lowering_blocks[i][mu2p]
                  for k1 in range(d1):
                    block[idx+k1*d2:idx+(k1+1)*d2,
                          idxb+k1*d2b:idxb+(k1+1)*d2b] += Ei
            self.lowering_blocks[i][muplus_i] = block

  def lowering_dot(self, mu, i, arr, axis):
    # Apply lowering operator f_i|_mu to relevant axis of arr
    muminus = self.group.lowerweight(mu,i)
    # Transpose to 1st index
    arrT = arr.transpose((axis,)+tuple(range(axis))+tuple(range(axis+1,arr.ndim)))
    oshape = arrT.shape[1:]
    arrM = arrT.reshape((arrT.shape[0],-1))
    wpminus = self.weight_products[muminus]
    wpplus = self.weight_products[mu]
    # Shape array into block form, & get component-to-index dicts for blocks
    arrblocks = []
    bi1 = {}
    bi2 = {}
    for bi,(idx,d1,d2,mu1,mu2) in enumerate(wpplus):
      arrblocks.append(arrM[idx:idx+d1*d2].reshape((d1,d2,-1)))
      bi1[mu1] = bi
      bi2[mu2] = bi
    outblocks = []
    for idx,d1,d2,mu1,mu2 in wpminus:
      oblock = None
      mu1p = self.group.raiseweight(mu1,i)
      if mu1p in bi1:
        Ei = self.rep1.lowering_blocks[i][mu1p][1]
        bi = bi1[mu1p]
        oblock = np.tensordot(Ei,arrblocks[bi],1)
      mu2p = self.group.raiseweight(mu2,i)
      if mu2p in bi2:
        Ei = self.rep2.lowering_blocks[i][mu2p][1]
        bi = bi2[mu2p]
        ob2 = np.tensordot(Ei,arrblocks[bi],(1,1)).transpose((1,0,2))
        if oblock is None:
          oblock = ob2
        else:
          oblock += ob2
      if oblock is None:
        outblocks.append(rzeros((d1*d2,arrM.shape[1])))
      else:
        outblocks.append(oblock.reshape((d1*d2,-1)))
    return np.vstack(outblocks).reshape((-1,)+oshape)
      

  def _decompose(self, use_3j=False):
    if isinstance(self.rep1,TrivialRepresentation) or isinstance(self.rep2,TrivialRepresentation):
      if isinstance(self.rep1,TrivialRepresentation):
        rep = self.rep2
      else:
        rep = self.rep1
      self.components = [(rep.label,1,rones((1,)))]
      self.decomposition = {}
      for mu in rep.weights:
        self.decomposition[mu] = {rep.label:
          (reye(self.degeneracies[mu])[None,:,:],
          rep._norm_squares[mu][None,:])}
      return
    wraise = self.group.raiseweight
    components = [] # Highest weights & multiplicities
    # Projections onto components for each participating weight
    decomp_by_weight = {mu:{} for mu in self.degeneracies}
    degen_remaining = dict(self.degeneracies) # Degeneracy of each weight unaccounted for
    nlvls = len(self.accessible_by_depth)
    for lvl in range(nlvls):
      for lambda_new in self.accessible_by_depth[lvl]:
        # Candidate highest weight must have nonnegative Dynkin indices
        if min(lambda_new) < 0:
          continue
        elif max(lambda_new) == 0:
          # When we've reached 0 no (other) candidate highest weights left
          break
        if degen_remaining[lambda_new] == 0:
          continue
        N = degen_remaining[lambda_new]
        N0 = self.degeneracies[lambda_new]
        #print(lambda_new,':',N,N0)
        degen_remaining[lambda_new] = 0
        #print(lambda_new,N,N0)
        # Get representation info
        irrep = self.group.get_ratrep(lambda_new)
        if use_3j:
          symbol = self.group._get3jrat(self.rep1._highest,self.rep2._highest,irrep.duallabel(),True)
        else:
          symbol = None
        if symbol is not None:
          # Already calculated
          assert symbol.degeneracy == N
          nmsqs = []
          divs = []
          for nm in symbol.normsquares:
            nmout,nmrad = sqrt_reduce(Fraction(nm,irrep.dim))
            nmsqs.append(nmrad)
            divs.append(nmout)
          divs = rarray(divs)
          nmsqs = rarray(nmsqs)
          components.append((lambda_new,N,nmsqs))
          self._3js[irrep.duallabel()] = symbol
          for mu in irrep.weights:
            d3 = irrep.degeneracy(mu)
            injectors = rzeros((N,d3,self.degeneracies[mu]))
            mu3 = negate(mu)
            for idx0,d1,d2,mu1,mu2 in self.weight_products[mu]:
              if (mu1,mu2,mu3) in symbol.blocks_out:
                W = symbol.block_OOI(mu1,mu2,mu3)
                injectors[:,:,idx0:idx0+d1*d2] = W.reshape((N,d1*d2,d3)).transpose((0,2,1))
            injectors.raise_index(divs,0,True)
            decomp_by_weight[mu][lambda_new] = (injectors.rereduce(),
                np.outer(nmsqs,irrep._norm_squares[mu]))
            degen_remaining[mu] -= d3*N
        else:
          # Account for possibility that (if defined by dual) highest-weight vector
          # might not be normalized
          lamnorm = irrep._norm_squares[lambda_new][0]
          if N0 != N:
            #print(lambda_new,degeneracies[lambda_new],N)
            # Need to account for vectors already projected out
            As,nsqs = zip(*decomp_by_weight[lambda_new].values())
            basis0 = np.vstack([A.reshape((-1,N0)) for A in As])
            normsq0 = np.concatenate([nsq.flatten() for nsq in nsqs])
            #print(basis0.shape,normsq0.shape,self.normsquares[lambda_new].shape)
            basis,normsq_lam = complete_basis(basis0,normsq0,self.normsquares[lambda_new])
          else:
            basis = reye(N0)
            normsq_lam = self.normsquares[lambda_new]
          decomp_by_weight[lambda_new][lambda_new] = (basis[:,None,:],normsq_lam[:,None])
          normsq_lam = normsq_lam / lamnorm
          components.append((lambda_new,N,normsq_lam))
          irrep_iter = itertools.chain.from_iterable(irrep._wbd_depth_order)
          next(irrep_iter)
          # Produce projection for remaining weights one-by-one
          # Change in convention: "injection" (acts on irrep) instead of
          # "projection" (acts on tensor-product space)
          # But will still put irrep index first unless it becomes too confusing
          for mu in irrep_iter:
            d = irrep.degeneracy(mu)
            # degen x mu(lam) x mu(x)
            injectors = rzeros((N,d,self.degeneracies[mu]))
            # Iterate through "simple lowering operators"
            for i,coeffs in enumerate(irrep._construction[mu]):
              # mu(lam)* x muplus(lam) (?)
              if coeffs is None:
                continue
              #nm,coeffs = constr
              muplus = wraise(mu,i)
              # mu(x) x muplus(x)*
              # degen x muplus(lam)* x muplus(x)
              Wplus = decomp_by_weight[muplus][lambda_new][0]
              # degen x muplus(lam)* x mu(x)
              fW = self.lowering_dot(muplus,i,Wplus,2)
              injectors += np.tensordot(coeffs,fW,(1,2)).transpose((2,0,1))
            normsqs = np.outer(normsq_lam,irrep._norm_squares[mu])
            if not no_strict_checks:
              lhs = np.tensordot(injectors,injectors*self.normsquares[mu][None,None,:],(2,2))
              rhs = normsqs[:,:,None,None]*reye(N*d).reshape((N,d,N,d))
              assert np.all(lhs == rhs)
            decomp_by_weight[mu][lambda_new] = (injectors.rereduce(),normsqs)
            degen_remaining[mu] -= d*N
    # Then deal with 0
    zerow = self.group.rank*(0,)
    #print(self.rep1._highest,self.rep2._highest,degen_remaining[zerow],self.degeneracies[zerow])
    if degen_remaining.get(zerow,0):
      assert self.rep1.duallabel() == self.rep2._highest
      assert degen_remaining[zerow] == 1
      if use_3j:
        # This is (proportional to) "2j symbol" aka charge conjugator
        symbol = self.group._get3jrat(self.rep1._highest,self.rep2._highest,zerow,False)
        self._3js[zerow] = symbol
        assert symbol.degeneracy == 1
        nmout,nmrad = sqrt_reduce(symbol.normsquares[0])
        normsqs = rarray([nmrad])
        injector = rzeros((1,1,self.degeneracies[zerow]))
        for idx0,d1,d2,mu1,mu2 in self.weight_products[zerow]:
          assert (mu1,mu2,zerow) in symbol.blocks_out
          # "OOI block" is the same as out block
          W = symbol.blocks_out[mu1,mu2,zerow]
          injector[:,:,idx0:idx0+d1*d2] = W.reshape((1,1,d1*d2))
      else:
        N0 = self.degeneracies[zerow]
        As,nsqs = zip(*decomp_by_weight[lambda_new].values())
        #print(N0,':')
        #print([A.shape for A in As])
        #print([ns.shape for ns in nsqs])
        basis0 = np.vstack([A.reshape((-1,N0)) for A in As])
        normsq0 = np.concatenate([nsq.flatten() for nsq in nsqs])
        basis,normsqs = complete_basis(basis0,normsq0,self.normsquares[zerow])
        injector = basis[:,None,:]
      decomp_by_weight[zerow][zerow] = (injector,normsqs[:,None])
      components.append((zerow,1,normsqs))
      degen_remaining[zerow] = 0
    else:
      assert self.rep1.duallabel() != self.rep2._highest
    # Sanity checks
    if not no_strict_checks:
      for mu,decomp in decomp_by_weight.items():
        N0 = self.degeneracies[mu]
        Lflat = []
        Rflat = []
        for lam,(W,nmsq) in decomp.items():
          Lflat.append(W.reshape(-1,N0).T)
          Rflat.append((W/nmsq[:,:,None]).reshape(-1,N0))
        L = np.hstack(Lflat)
        R = np.vstack(Rflat)
        assert L.shape == (N0,N0)
        assert np.all(L.dot(R) == np.diag(self.normsquares[mu].reciprocal()))
    assert sum(N*self.group.get_ratrep(lam).dim for lam,N,_ in components) == self.rep1.dim*self.rep2.dim
    self.components = components
    self.decomposition = decomp_by_weight

  def sparse_intertwiner(self, lam, degen_label=0, unitary=True):
    # Construct explicit, sparse intertwiner to mapping (Kronecker product)
    # rep1xrep2 from irrep with highest weight lam; degen_label identifies
    # index of degeneracy if there is more than 1
    N = self.fusion_degen[lam]
    irrep = self.group.get_ratrep(lam)
    indices = irrep.block_ind_dict
    ind1 = self.rep1.block_ind_dict
    ind2 = self.rep2.block_ind_dict
    D2 = self.rep2.dim
    indptr = [0]
    rowind = []
    data = []
    dataidx = 0
    #print(lam,N,':')
    for mu in irrep.weights:
      d = irrep.degeneracy(mu)
      idx = indices[mu]
      Wmu,nmsq = self.decomposition[mu][lam]
      #print(mu,d,self.degeneracies[mu],Wmu.shape,nmsq.shape)
      Wmu = Wmu[degen_label].T
      Wfloat = Wmu.asfarray()
      nmsq = nmsq[degen_label].asfarray()
      #nmsq_irrep = irrep._norm_squares[mu].astype(float)
      nmleft = self.normsquares[mu].asfarray()
      if unitary:
        Wfloat = np.sqrt(nmleft[:,None])*Wfloat/np.sqrt(nmsq[None,:])
      idx_out = []
      for idx0,d1,d2,mu1,mu2 in self.weight_products[mu]:
        idx1 = ind1[mu1]
        idx2 = ind2[mu2]
        idxprod = np.arange(idx2,idx2+d2)[None,:] + D2*np.arange(idx1,idx1+d1)[:,None]
        idx_out.append(idxprod.flatten())
      idx_out = np.concatenate(idx_out)
      for ilam in range(d):
        rows, = np.nonzero(Wmu[:,ilam])
        dataidx += len(rows)
        indptr.append(dataidx)
        rowind.extend(idx_out[rows])
        data.extend(Wfloat[rows,ilam])
    dim = self.rep1.dim*self.rep2.dim
    return scipy.sparse.csc_array((data,rowind,indptr),
      shape=(self.rep1.dim*self.rep2.dim,irrep.dim))

  def check_fusion(self, logger,tolerance=1e-12):
    rep1 = self.rep1
    lambda1 = rep1._highest
    rep2 = self.rep2
    lambda2 = rep2._highest
    rank = self.group.rank
    dtot = sum(N*self.group.get_ratrep(lam).dim for lam,N,nmsq in self.components)
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
      h12 = scipy.sparse.kron(h1i,I2,format='csc') + scipy.sparse.kron(I1,h2i,format='csc')
      hs.append(h12)
      e1i = rep1.fp_chevalley('e',i)
      e2i = rep2.fp_chevalley('e',i)
      e12 = scipy.sparse.kron(e1i,I2,format='csc') + scipy.sparse.kron(I1,e2i,format='csc')
      es.append(e12)
    for lam,N,nmsq in self.components:
      logger.debug('Checking %s x %s -> %d%s',rep1.label,rep2.label,N,lam)
      rep = self.group.get_ratrep(lam)
      hprod = [rep.fp_chevalley('h',i) for i in range(rank)]
      eprod = [rep.fp_chevalley('e',i) for i in range(rank)]
      for a in range(N):
        WT = self.sparse_intertwiner(lam,a)
        W = WT.T.tocsc()
        WWT = W.dot(WT).toarray()
        if not np.allclose(WWT,np.eye(rep.dim),atol=tolerance,rtol=tolerance):
          badlog('Failure of unitarity in %sx%s->%s_%d (%0.4g)',
            lambda1,lambda2,lam,a,np.linalg.norm(WWT-np.eye(rep.dim)))
        for i in range(rank):
          hi = hprod[i].toarray()
          hcomp = W.dot(hs[i]).dot(WT).toarray()
          if not np.allclose(hi,hcomp,atol=tolerance,rtol=tolerance):
            badlog('Failure to reproduce h_%d in %sx%s->%s_%d (%0.4g)',
              i,lambda1,lambda2,lam,a,np.linalg.norm(hi-hcomp))
          ei = eprod[i].toarray()
          ecomp = W.dot(es[i]).dot(WT).toarray()
          if not np.allclose(ei,ecomp,atol=tolerance,rtol=tolerance):
            badlog('Failure to reproduce e_%d in %sx%s->%s_%d (%0.4g)',
              i,lambda1,lambda2,lam,a,np.linalg.norm(ei-ecomp))
    return badlog.correct

  def _construct_3js(self):
    for lam,N,nmsq in self.components:
      lstar = self.group.get_ratrep(lam).duallabel()
      if lstar not in self._3js:
        threej = Rational3j(self, lam, N, nmsq)
        self._3js[lstar] = threej
        self.group._set3js(threej)

  def get_3j(self, lam):
    return self._3js[lam]


from .racah import Rational3j,RationalRacahSymbol,RationalRacahOperator
