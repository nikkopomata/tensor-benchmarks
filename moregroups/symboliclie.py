from liegroups import *

class SymbolicRepresentation(HighestWeightRepresentation):
  """Highest-weight representation with coefficients/matrix elements
  computed symbolically"""

  def _construct_weights(self):
    # Recursion data for highest & next-highest weights
    group = self._group
    rank = group.rank
    weights_by_depth = [[self._highest]]
    degen_by_depth = [[1]]
    gen_action_by_depth = [[rank*[None]]]
    weights_curr = []
    degen_curr = []
    gen_curr = []
    lower = group.lowerweight
    wraise = group.raiseweight
    # Next-highest level
    for i in range(rank):
      norm = self._highest[i]
      if norm:
        weights_curr.append(lower(self._highest,i))
        degen_curr.append(1)
        gen_curr.append(rank*[None])
        gen_curr[-1][i] = sympy.Matrix([[sympy.sqrt(norm)]])
        constr = rank*[None]
        constr[i] = sympy.Matrix([[radsimp(1/sympy.sqrt(norm))]])
        self._construction[weights_curr[-1]] = constr
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
            # Does not contribute (not the only way but I expected it would be the most common)
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
            jdotj = muplus_j[j] * sympy.eye(dj)
            if Ej_j is not None:
              jdotj += symdot(Ej_j.T, Ej_j)
            if jdotj.is_zero_matrix:
              # Does not contribute
              continue
            # Other blocks with known contributors
            row = []
            for iblock,(i,idx) in enumerate(contributing):
              Ej_i = gen_last[idx][j]
              Ei_j = gen_last[jdx][i]
              if Ej_i is None or Ei_j is None:
                block = sympy.zeros(degen_last[idx],dj)
              else:
                block = symdot(Ej_i.T, Ei_j)
              gram_rows[iblock] = gram_rows[iblock].row_join(block)
              row.append(block.T)
            row.append(jdotj)
            gram_rows.append(sympy.Matrix.hstack(*row))
            contributing.append((j,jdx))
          if not contributing:
            # No valid contributors found with full test
            continue
          weights_curr.append(mu)
          if mu == zero:
            has_zero = True
          gram = sympy.Matrix.vstack(*gram_rows)
          basis = gramschmidt_coefficients(gram)
          degen_curr.append(basis.cols)
          # Action of Chevalley generators from orthonormal basis:
          genaction = rank*[None]
          self._construction[mu] = rank*[None]
          k0 = 0
          for iblock,(i,idx) in enumerate(contributing):
            k1 = k0 + degen_last[idx]
            genaction[i] = symdot(gram[k0:k1,:], basis)
            self._construction[mu][i] = basis[k0:k1,:].T
            k0 = k1
          gen_curr.append(genaction)

    assert len(weights_by_depth[-1]) == 1
    #assert len(weights_by_depth) == sum(highest_weight)+1
    self._weights_by_depth = weights_by_depth
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
          self.raising_blocks[i][mu] = (muplus,Eimu)
          assert muplus not in self.lowering_blocks[i] # Again
          self.lowering_blocks[i][muplus] = (mu,Eimu.T)
    indices = [0] + list(self.block_indices)
    self.generators = []
    for i in range(self._group.rank):
      Ei = sympy.zeros(self.dim,self.dim)
      for mu1,(mu2,Eblock) in self.raising_blocks[i].items():
        idx1 = self.weight_indices[mu1]
        idx2 = self.weight_indices[mu2]
        Ei[indices[idx2]:indices[idx2+1],indices[idx1]:indices[idx1+1]] \
          = Eblock
      #self.generators.append(sympy.ImmutableSparseMatrix(Ei))
      self.generators.append(Ei)

  def diagonal_chevalley(self,i):
    # Diagonal Chevalley generators (i.e. belonging to Cartan subalgebra)
    h = sympy.zeros(self.dim,self.dim)
    for mu,d,idx in zip(self.weights,self.degeneracies,self.block_indices):
      mui = int(mu[i])
      if d == 1:
        h[idx-1,idx-1] = mui
      else:
        h[idx-d:idx,idx-d:idx] = mui*sympy.eye(d)
    return h

  def fp_chevalley(self, which, i, precision='double'):
    # Chevalley generators (which='h','e','f'), evaluated as floats in a 
    # scipy.sparse array
    shape = (self.dim,self.dim)
    if which == 'h':
      diag = []
      for mu,d in zip(self.weights,self.degeneracies):
        diag.extend(d*[mu[i]])
      return scipy.sparse.diags_array(diag,shape=shape, dtype=int,format='csr')
    if which in 'ef':
      idxs,data = zip(*self.generators[i].iter_items())
      rows,cols = zip(*idxs)
      if precision == 'longdouble':
        # HAS to be a better way
        dtype = np.longdouble
        data = [np.longdouble(str(val.evalf(22))) for val in data]
      else:
        dtype = float
        data = np.array(data,dtype=dtype)
      ei = scipy.sparse.coo_array((data,(rows,cols)),shape=shape,dtype=dtype)
      if which == 'e':
        return ei.tocsr()
      else:
        return ei.tocsc().T

  def diagonal_cartanweyl(self,i):
    # Diagonal Cartan-Weyl generators
    # Need ith element of each fundamental weight
    omega_ji = self._group._fund_weights[:,i]
    diag = []
    for mu,d in zip(self.weights,self.degeneracies):
      diag.extend(d*[omega_ji.dot(mu).expand()])
    #return sympy.ImmutableSparseMatrix.diag(diag)
    return sympy.diag(diag,unpack=True)

  def charge_conjugator(self, dualrep=None):
    # Isometry mapping dual space of self to a dualrep
    # (self if not provided, i.e. if real or quaternionic)
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
    blocks[self._highest] = sympy.eye(1)
    indices = [0] + list(self.block_indices)
    dualindices = [0] + list(dualrep.block_indices)
    #matrix = sympy.SparseMatrix.zeros(self.dim,self.dim)
    matrix = sympy.zeros(self.dim,self.dim)
    matrix[0,-1] = 1
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
          EL.append(symdot(blocks[muplus].adjoint(), -Edi))
          ER.append(Ei)
      EL = sympy.Matrix.vstack(*EL)
      ER = sympy.Matrix.vstack(*ER)
      #print('%dx%d LLSQ'%EL.shape)
      blocks[mu] = EL.solve_least_squares(ER)
      #print('LLSQ complete')
      # Solution should still be exact
      assert no_strict_checks or (symdot(EL, blocks[mu])-ER).is_zero_matrix
      assert no_strict_checks or (symdot(blocks[mu],blocks[mu].adjoint())-sympy.eye(blocks[mu].rows)).is_zero_matrix
      idx = self.weight_indices[mu]
      didx = dualrep.weight_indices[mud]
      #print(mu,mud)
      #print(indices[idx:idx+2],dualindices[didx:didx+2])
      #print(matrix.shape,blocks[mu].shape)
      matrix[slice(*indices[idx:idx+2]),slice(*dualindices[didx:didx+2])] \
        = blocks[mu].T
    #self._CC = (dualrep,sympy.ImmutableSparseMatrix(matrix))
    self._CC = (dualrep,matrix)
    return matrix

  def check_representation(self, logger, matrix_checking='symbolic',
      tolerance=0):
    # Compare representation definitons with Chevalley structure coefficients
    if matrix_checking == 'numpy':
      dtype = complex
      num = True
      if not tolerance:
        tolerance = 1e-14
    elif matrix_checking == 'longdouble':
      dtype = np.complex256
      num = True
      if not tolerance:
        tolerance = 1e-18
    else:
      assert matrix_checking == 'symbolic'
      num = False
    g = self._group
    rank = g.rank
    npos = len(g._positive_roots)
    shape = (self.dim,self.dim)
    hs = [self.diagonal_chevalley(i) for i in range(self._group.rank)]
    Es = list(self.generators)
    Fs = []
    if num:
      for i in range(rank):
        l = hs[i].diagonal()
        if matrix_checking == 'numpy':
          l = np.array(l).flatten()
        else:
          l = [dtype(str(v.evalf(22))) for v in l] # Has to be a better way
        hs[i] = scipy.sparse.diags_array(l,shape=shape,
          dtype=dtype,format='csr')
      for i in range(rank):
        idxs,data = zip(*Es[i].iter_items())
        rows,cols = zip(*idxs)
        if matrix_checking == 'longdouble':
          # HAS to be a better way
          data = [dtype(str(val.evalf(22))) for val in data]
        else:
          data = np.array(data,dtype=dtype)
        #data = []
        #rows = []
        #cols = []
        #for (idx1,idx2),val in Es[i].iter_items():
        #  if matrix_checking == 'numpy':
        #    data[i].append(float(val))
        #  else:
        #    data[i].append(dtype(str(val.evalf(22))))
        #  rows.append(idx1)
        #  cols.append(idx2)
        ei = scipy.sparse.coo_array((data,(rows,cols)),shape=shape,dtype=dtype)
        Es[i] = ei.tocsr()
        Fs.append(ei.T.tocsr())
      def mdot(A,B):
        return A.dot(B)
      zeros = scipy.sparse.csr_array(shape,dtype=dtype)
    else:
      def mdot(A,B):
        return A.multiply(B,True)
      #zeros = sympy.SparseMatrix.zeros(self.dim)
      zeros = sympy.zeros(self.dim)
    for idx in range(rank,npos):
      p,i,idxm = g._chevalley_definitions[idx]
      Es.append((mdot(Es[i], Es[idxm]) - mdot(Es[idxm], Es[i]))/p)
      Fs.append((mdot(Fs[idxm], Fs[i]) - mdot(Fs[i], Fs[idxm]))/p)
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
      if not num:
        mat = np.array(mat,dtype=complex)
      return np.linalg.norm(mat)
    for idx1,(M1,s1) in enumerate(zip(basis,labels)):
      for idx2,(M2,s2) in enumerate(zip(basis,labels)):
        compare = zeros.copy()
        coeffs = list(g.chevalley_structure[idx1][:,idx2].iter_items())
        t1 = mdot(M1,M2)
        t2 = mdot(M2,M1)
        if num:
          t1 = t1.toarray()
          t2 = t2.toarray()
        if len(coeffs) == 0:
          if not (num and np.allclose(t1,t2,atol=tolerance,rtol=tolerance)) \
              and not (not num and (t1-t2).is_zero_matrix):
            badlog('%s and %s do not commute (failure %0.4g/%0.4g)',
              s1,s2,numnorm(t1,t2), max(numnorm(t1),numnorm(t2)))
        else:
          for (idx3,_),coeff in coeffs:
            compare += int(coeff)*basis[idx3]
          if not (num and np.allclose(t1-t2,compare.toarray(),atol=tolerance,rtol=tolerance)) and not (not num and (t1-t2-compare).is_zero_matrix):
            badlog('Failure in [%s,%s] (%0.4g/%0.4g)',s1,s2,
              numnorm(t1-t2,compare.toarray()),numnorm(compare.toarray()))
    return badlog.correct

  def check_cc(self,logger,dualrep):
    # Should send e_i -> e_i^T, h_i -> -h_i
    S = self.charge_conjugator(dualrep)
    Sd = S.adjoint()
    def badlog(msg,*args):
      if logger is not None:
        logger.warn('~~'+msg+'~~',*args)
        badlog.correct = False
      else:
        raise ValueError(msg%args)
    badlog.correct = True
    if self is dualrep:
      strs = ['','symmetric','antisymmetric']
      if S.is_symmetric():
        fsi_obs = 1
      elif S.is_anti_symmetric():
        fsi_obs = -1
      else:
        badlog('S should be %s but is neither symmetric nor antisymmetric: failure %0.4g',strs[self.FSI],(S.T-self.FSI*S).norm())
        fsi_obs = 0
      if fsi_obs and fsi_obs != self.FSI:
        badlog('S should be %s but is instead %s',strs[self.FSI],strs[fsi_obs])
    rank = self._group.rank
    hs = [self.diagonal_chevalley(i) for i in range(rank)]
    hds = [dualrep.diagonal_chevalley(i) for i in range(rank)]
    def numnorm(mat,sub=None):
      if sub is not None:
        mat = mat-sub
      mat = np.array(mat,dtype=float)
      return np.linalg.norm(mat)
    for i in range(rank):
      #hcheck = symdot(symdot(Sd, hs[i]), S)
      hcheck = Sd * hs[i] * S
      if not (hcheck + hds[i]).is_zero_matrix:
        badlog('Charge conjugation of h_%d gives incorrect result (%0.4g/%0.4g)',i,numnorm(hcheck,-hds[i]),numnorm(hds[i]))
    for i in range(rank):
      #Echeck = symdot(symdot(Sd, self.generators[i]), S)
      Echeck = Sd * self.generators[i] * S
      Echeck = Echeck.expand()
      fd = dualrep.generators[i].T
      if not (Echeck + fd).is_zero_matrix:
        badlog('Charge conjugation of e_%d gives incorrect result (%0.4g/%0.4g)',i,numnorm(Echeck,-fd),numnorm(fd))
    return badlog.correct

