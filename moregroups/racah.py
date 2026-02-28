from .rationallie import *


class Rational3j:
  # Represent generalized 3j-symbol of highest-weight representations 
  # in terms of rational-valued blocks, as 4D arrays of
  # degeneracy x mu1 x mu2 x mu3
  def __init__(self, *args):
    if len(args) == 2:
      # 0 x lambda x lambda* "2j symbol"
      self._init_from_identity(*args)
    elif args: #isinstance(args[0],RationalTensorDecomposition):
      self._init_from_decomposition(*args)
    # Otherwise set empty, assume setting from within conjugate or transposed
    # symbol
    self.Rs = {}
    self.C = None
  
  def _init_from_decomposition(self, fusion, lamstar, N, normsq):
    self.group = fusion.group
    self.rep1 = fusion.rep1
    #self.rep1._construct_charge_conjugator(self.rep1.dual)
    #Sb1 = self.rep1._CC[1]
    self.rep2 = fusion.rep2
    #self.rep2._construct_charge_conjugator(self.rep2.dual)
    #Sb2 = self.rep2._CC[1]
    repout = self.group.get_ratrep(lamstar)
    self.rep3 = repout.dual
    repout._construct_charge_conjugator(self.rep3)
    Sb3d = repout._CC[1]
    self.degeneracy = N
    nmsqs = []
    divs = []
    for nm in normsq:
      nmout,nmrad = sqrt_reduce(repout.dim * nm)
      nmsqs.append(nmrad)
      divs.append(nmout)
    self.normsquares = rarray(nmsqs)
    divs = rarray(divs)
    self.blocks_in = {}
    self.blocks_out = {}
    for mu in repout.weights:
      Wmu,nmsq = fusion.decomposition[mu][lamstar]
      Wmu = Wmu.raise_index(divs,0)
      d3 = repout.degeneracy(mu)
      mu3 = negate(mu)
      for idx0,d1,d2,mu1,mu2 in fusion.weight_products[mu]:
        Wblock = Wmu[:,:,idx0:idx0+d1*d2].reshape(N,d3,d1,d2)
        # Need to convert index 3
        outin = repout.raise_idx_block(mu,Sb3d[mu],1)
        Wblock = np.tensordot(Wblock,outin,(1,1)).rereduce()
        # "In" blocks: need to lower indices# 1 & 2
        Win = self.rep1.lower_idx_block(mu1,Wblock,1)
        self.rep2.lower_idx_block(mu2,Win,2,True)
        self.rep3.lower_idx_block(mu3,Win,3,True)
        self.blocks_in[mu1,mu2,mu3] = Win.rereduce()
        ## "Out" blocks: need to raise index 3
        #Wout = Wblock*Fraction(1)/self.rep3._norm_squares[mu3][None,None,None,:]
        self.blocks_out[mu1,mu2,mu3] = Wblock

  def _init_from_identity(self, group, lambda2):
    # Special case of initializing from decomposition:
    # mapping lamda -> 0 x lambda is (effectively) identity
    self.group = group
    lambda1 = group.rank*(0,)
    self.rep1 = group.get_ratrep(lambda1)
    self.rep2 = group.get_ratrep(lambda2)
    self.rep3 = self.rep2.dual
    self.rep2._construct_charge_conjugator(self.rep3)
    Sb3d = self.rep2._CC[1]
    self.degeneracy = 1
    nmout,nmrad = sqrt_reduce(self.rep2.dim)
    self.normsquares = rarray([nmrad])
    self.blocks_in = {}
    self.blocks_out = {}
    for mu2,d in zip(self.rep2.weights,self.rep2.degeneracies):
      mu3 = negate(mu2)
      outin = self.rep2.raise_idx_block(mu2,Sb3d[mu2],1)
      # Tensordot with Wmu <-> contraction with identity
      Wblock = np.expand_dims(outin.T,(0,1)) / nmout
      self.blocks_out[lambda1,mu2,mu3] = Wblock.rereduce()
      Win = self.rep2.lower_idx_block(mu2,Wblock,2)
      self.rep3.lower_idx_block(mu3,Win,3,True)
      self.blocks_in[lambda1,mu2,mu3] = Win.rereduce()

  def transpose(self, perm):
    # Create transposed symbol
    T = Rational3j()
    T.group = self.group
    T.rep1 = self.reps[perm[0]]
    T.rep2 = self.reps[perm[1]]
    T.rep3 = self.reps[perm[2]]
    T.normsquares = self.normsquares
    T.degeneracy = self.degeneracy
    T.blocks_in = {}
    T.blocks_out = {}
    permplus = (0,)+tuple(i+1 for i in perm)
    for mus in self.blocks_in:
      muperm = tuple(mus[ip] for ip in perm)
      T.blocks_in[muperm] = self.blocks_in[mus].transpose(permplus)
      T.blocks_out[muperm] = self.blocks_out[mus].transpose(permplus)
    return T

  def conj(self,nocreate=False):
    # Create "conjugated" symbol (in->out on dual labels & vice versa)
    dual = Rational3j()
    dual.group = self.group
    dual.rep1 = self.rep1.dual
    dual.rep2 = self.rep2.dual
    dual.rep3 = self.rep3.dual
    dual.normsquares = self.normsquares
    dual.degeneracy = self.degeneracy
    dual.blocks_in = {}
    dual.blocks_out = {}
    for rep in self.reps:
      rep._construct_charge_conjugator(rep.dual)
    for mus in self.blocks_in:
      musn = tuple(negate(mu) for mu in mus)
      blockout = self.blocks_out[mus]
      for mu,rep in zip(mus,self.reps):
        blockout = np.tensordot(blockout,rep._CC[1][mu],(1,1))
      blockout.rereduce()
      blockin = blockout.copy()
      for i,rep in enumerate(self.reps):
        rep.dual.lower_idx_block(musn[i],blockin,i+1,True)
      blockin.rereduce()
      dual.blocks_in[musn] = blockin
      dual.blocks_out[musn] = blockout
    return dual

  def transform(self, M, nmsqs):
    # Apply change-of-basis matrix M; resulting symbol has norm-squares as
    # indicated
    T = Rational3j()
    T.group = self.group
    T.rep1,T.rep2,T.rep3 = self.reps
    T.normsquares = nmsqs
    T.degeneracy = self.degeneracy
    T.blocks_in = {}
    T.blocks_out = {}
    for mus in self.blocks_in:
      T.blocks_in[mus] = np.tensordot(M,self.blocks_in[mus],1).rereduce()
      T.blocks_out[mus] = np.tensordot(M,self.blocks_out[mus],1).rereduce()
    return T

  @property
  def reps(self):
    return (self.rep1, self.rep2, self.rep3)

  @property
  def labels(self):
    return (self.rep1._highest,self.rep2._highest,self.rep3._highest)

  def block_OOI(self, mu1,mu2,mu3):
    # "Out-out-in" block in the sense provided by decomposition
    repout = self.rep3.dual
    repout._construct_charge_conjugator(self.rep3)
    S = repout._CC[1][negate(mu3)]
    inout = self.rep3.lower_idx_block(mu3,S.T,1)
    return np.tensordot(self.blocks_out[mu1,mu2,mu3],inout,(3,1)).rereduce()

  def rotator(self, label,symb2):
    if label in self.Rs and self.Rs[label].target is symb2:
      return self.Rs[label]
    permuter = np.arange(3)
    permuter[label,] = permuter[label[1:]+(label[0],),]
    permute1 = (0,) + tuple(int(i+1) for i in permuter)
    permute0 = tuple(int(i) for i in permuter)
    # Determine from highest weight of c
    R = None
    lamc = self.rep3._highest
    #print(permute0,permute1)
    for mus in self.blocks_in:
      if mus[-1] == lamc:
        block1 = self.blocks_in[mus]
        muperm = [mus[i] for i in permute0]
        #print(mus,muperm)
        block2 = symb2.blocks_out[tuple(muperm)]
        #print(block1.shape,block2.shape)
        term = np.tensordot(block2,block1,((1,2,3),permute1[1:]))
        if R is None:
          R = term
        else:
          R += term
    R *= self.rep3.dim
    RT = R.T.copy()
    # Need to raise first index?
    R.raise_index(symb2.normsquares,0,True)
    R.rereduce()
    self.Rs[label] = RationalRacahOperator(R,(symb2,True,True),
      (self,False,False))
    if symb2 is not self:
      #RT.raise_index(self.normsquares,0,True)
      #RT.rereduce()
      labinv = label[:1] + label[-1:0:-1]
      #symb2.Rs[labinv] = (self,RT)
      symb2.Rs[labinv] = self.Rs[label].T
    return self.Rs[label]

  def check_rot(self, logger, label):
    permuter = np.arange(3)
    permuter[label,] = permuter[label[1:]+(label[0],),]
    permute1 = (0,) + ituple(permuter+1)
    permute0 = ituple(permuter)
    R = self.Rs[label]._array_def
    symb2 = self.Rs[label].target
    name = 'R(%s)'%(' '.join(str(i) for i in label))
    for mus,block1 in self.blocks_in.items():
      muperm = tuple(mus[i] for i in permute0)
      block2 = symb2.blocks_in[muperm]
      lhs = block1.transpose(permute1)
      if self.Rs[label]._k in '1s':
        rhs = R*block2
      elif self.Rs[label]._k in 'd-':
        rhs = block2.copy()
        for i,sgn in enumerate(R):
          rhs[i] *= sgn
      else:
        rhs = np.tensordot(R,block2,(0,0))
      if not np.all(lhs == rhs):
        logger.warning('Failure of proportionality in weights %sx%sx%s '
          '(%0.4g vs %0.4g)', *mus, fnorm(lhs-rhs),fnorm(lhs))

  def conjugator(self, symbc):
    # Determine from highest weight of c
    if self.C is not None and self.C.target is symbc:
      return self.C
    C = None
    lamc = self.rep3._highest
    for rep in self.reps:
      rep._construct_charge_conjugator(rep.dual)
    for mus in self.blocks_out:
      if mus[-1] == lamc:
        block1 = self.blocks_out[mus]
        nmu = tuple(negate(mu) for mu in mus)
        block2 = symbc.blocks_out[nmu]
        for mu,rep in zip(mus,self.reps):
          nmsq = rep.dual._norm_squares[negate(mu)]
          #"X" because it's two arrows facing each other
          X = nmsq[:,None] * rep._CC[1][mu]
          block1 = np.tensordot(block1,X,(1,1)) #Rotates indices all the way
        term = np.tensordot(block2,block1,((1,2,3),(1,2,3)))
        if C is None:
          C = term
        else:
          C += term
    C *= self.rep3.dim
    CT = C.T
    # Need to raise first index?
    C = C.raise_index(symbc.normsquares,0).rereduce()
    self.C = RationalRacahOperator(C,(symbc,True,True),(self,True,False))
    if symbc is not self:
      #symbc.C = (self,CT.raise_index(self.normsquares,0).rereduce())
      symbc.C = self.C.H.conj()
    return self.C

  def check_conj(self,logger):
    C = self.C._array_def
    symbc = self.C.target
    for mus,block1 in self.blocks_out.items():
      nmus = tuple(negate(mu) for mu in mus)
      for mu,nmu,rep in zip(mus,nmus,self.reps):
        nmsq = rep.dual._norm_squares[nmu]
        X = rep._CC[1][mu].lower_index(nmsq,0)
        block1 = np.tensordot(block1,X,(1,1))
      block2 = symbc.blocks_in[nmus]
      lhs = block1
      if self.C._k in '1s':
        rhs = C * block2
      elif self.C._k in 'd-':
        rhs = np.expand_dims(rarray(C),(1,2,3)) * block2
      if not np.all(lhs == rhs):
        logger.warning('Failure of proportionality in weights %sx%sx%s '
          '(%0.4g vs %0.4g)', *mus, fnorm(lhs-rhs),fnorm(lhs))

  def fixflip(self):
    # For a symbol with rep 1 == rep 2, change basis so that R(01) is diagonal
    assert self.rep1 is self.rep2
    if (0,1) in self.Rs and isinstance(self.Rs[0,1][1],list):
      return self
    R = self.rotator((0,1),self)
    newsymb,l = self._fixflip(R)
    newsymb.Rs[0,1] = RationalRacahOperator(l,(newsymb,True,True),
      (newsymb,False,False))
    return newsymb

  def fixflip2(self):
    # Same as fixflip but for case rep 2 == rep3 (now fixing R(12) )
    assert self.rep2 is self.rep3
    if (1,2) in self.Rs and isinstance(self.Rs[1,2][1],list):
      return self
    R = self.rotator((1,2),self)
    newsymb,l = self._fixflip(R)
    newsymb.Rs[1,2] = RationalRacahOperator(l,(newsymb,True,True),
      (newsymb,False,False))
    return newsymb

  def _fixflip(self, R):
    # Fix to eigenvectors of R, which may be (0,1) or (1,2) rotation
    # Gram-Schmidt on image of R+1 (1-eigenspace)
    N = self.degeneracy
    if N == 1:
      # Already diagonal
      assert R[0,0] == 1 or R[0,0] == -1
      return self,[int(R[0,0])]
    cob = rzeros((0,N))
    vplus = cob
    normsq = []
    for i in range(N):
      # (R+1)@e_i
      vec = R[:,i].copy()
      vec[i] += 1
      vec -= vplus.T.dot(cob.dot(vec))
      norm = vec.dot(vec*self.normsquares)
      if norm:
        nout,nrad = sqrt_reduce(norm)
        vec /= Fraction(nout)
        normsq.append(nrad)
        cob = np.vstack((cob,vec.T.lower_index(self.normsquares,1)/nrad))
        vplus = np.vstack((vplus,vec.T))
    vplus.rereduce()
    Nplus = cob.shape[0]
    eigs = Nplus*[1] + (N-Nplus)*[-1]
    if not Nplus or Nplus == N:
      # R is plus or minus identity - no need to transform
      vecs = np.identity(N,dtype=object)
      norms = self.normsquares
    else:
      normsq = rarray(normsq)
      vminus,nminus = complete_basis(vplus,normsq,self.normsquares)
      vecs = np.vstack((vplus,vminus))
      norms = np.concatenate((normsq,nminus))
    return self.transform(vecs,norms), eigs

  def check_unitary(self,logger):
    g = self.group
    ainout = {}
    binout = {}
    cinout = {}
    logger.info('Checking unitarity...')
    for (mua,mub,muc),block in self.blocks_in.items():
      bout = self.blocks_out[mua,mub,muc]
      bb = np.tensordot(block,bout,((2,3),(2,3)))
      plus_assign(ainout,mua,bb)
      bb = np.tensordot(block,bout,((1,3),(1,3)))
      plus_assign(binout,mub,bb)
      bb = np.tensordot(block,bout,((1,2),(1,2)))
      plus_assign(cinout,muc,bb)
    for blocks,rep,label in zip((ainout,binout,cinout),self.reps,'abc'):
      assert set(blocks) == set(rep.weights)
      factors = Fraction(1,rep.dim) * self.normsquares
      for mu,arr in blocks.items():
        name = '%s_%s'%(label,mu)
        N,d = arr.shape[:2]
        I2 = np.eye(N*d,dtype=object).reshape((N,d,N,d))
        if np.any(arr != factors[:,None,None,None]*I2):
          for i in range(N):
            for j in range(N):
              mat = arr[i,:,j]
              if i == j:
                fact2 = mat.trace()/d
                if np.all(mat == fact2*np.eye(d,dtype=object)):
                  if fact2 == factors[i]:
                    continue
                  else:
                    logger.warning('**Unitarity test in %s [%d] produces incorrect factor: %sxI instead of %sxI',name,i,fact2,factors[i])
                else:
                  logger.warning('**Failure of unitarity in %s [%d] (%0.4g of %0.4g)',
                    name,i,fnorm(mat - factors[i]*np.eye(d)),
                    float(factors[i])*np.sqrt(d))
              elif np.any(mat):
                logger.warning('**Failure of unitarity in %s [%d,%d] (%0.4g,'
                  ' compare %0.4g)', name, i, j, fnorm(mat),
                  np.sqrt(d*float(factors[i]*factors[j])))

  def check_invariance(self,logger):
    g = self.group
    repa,repb,repc = self.reps
    for i in range(g.rank):
      logger.info('Checking action of e_%d/f_%d',i,i)
      eaction_in = {}
      faction_in = {}
      eaction_out = {}
      faction_out = {}
      rw = lambda mu: g.raiseweight(mu,i)
      lw = lambda mu: g.lowerweight(mu,i)
      for (mua,mub,muc),block in self.blocks_out.items():
        # Out block is ordinary, in block is transposed
        blockin = self.blocks_in[mua,mub,muc]
        if mua in repa.raising_blocks[i]:
          eia = repa.raising_blocks[i][mua][1]
          block_ea = np.tensordot(block,eia,(1,1)).transpose((0,3,1,2))
          plus_assign(eaction_out,(rw(mua),mub,muc),block_ea)
          ifia = repa.lowering_blocks[i][rw(mua)][1]
          blockifa = np.tensordot(blockin,ifia,(1,0)).transpose((0,3,1,2))
          plus_assign(faction_in,(rw(mua),mub,muc),blockifa)
        if mua in repa.lowering_blocks[i]:
          fia = repa.lowering_blocks[i][mua][1]
          block_fa = np.tensordot(block,fia,(1,1)).transpose((0,3,1,2))
          plus_assign(faction_out,(lw(mua),mub,muc),block_fa)
          ieia = repa.raising_blocks[i][lw(mua)][1]
          blockiea = np.tensordot(blockin,ieia,(1,0)).transpose((0,3,1,2))
          plus_assign(eaction_in,(lw(mua),mub,muc),blockiea)
        if mub in repb.raising_blocks[i]:
          eib = repb.raising_blocks[i][mub][1]
          block_eb = np.tensordot(block,eib,(2,1)).transpose((0,1,3,2))
          plus_assign(eaction_out,(mua,rw(mub),muc),block_eb)
          ifib = repb.lowering_blocks[i][rw(mub)][1]
          blockifb = np.tensordot(blockin,ifib,(2,0)).transpose((0,1,3,2))
          plus_assign(faction_in,(mua,rw(mub),muc),blockifb)
        if mub in repb.lowering_blocks[i]:
          fib = repb.lowering_blocks[i][mub][1]
          block_fb = np.tensordot(block,fib,(2,1)).transpose((0,1,3,2))
          plus_assign(faction_out,(mua,lw(mub),muc),block_fb)
          ieib = repb.raising_blocks[i][lw(mub)][1]
          blockieb = np.tensordot(blockin,ieib,(2,0)).transpose((0,1,3,2))
          plus_assign(eaction_in,(mua,lw(mub),muc),blockieb)
        if muc in repc.raising_blocks[i]:
          eic = repc.raising_blocks[i][muc][1]
          block_ec = np.tensordot(block,eic,(3,1))
          plus_assign(eaction_out,(mua,mub,rw(muc)),block_ec)
          ific = repc.lowering_blocks[i][rw(muc)][1]
          blockifc = np.tensordot(blockin,ific,(3,0))
          plus_assign(faction_in,(mua,mub,rw(muc)),blockifc)
        if muc in repc.lowering_blocks[i]:
          fic = repc.lowering_blocks[i][muc][1]
          block_fc = np.tensordot(block,fic,(3,1))
          plus_assign(faction_out,(mua,mub,lw(muc)),block_fc)
          ieic = repc.raising_blocks[i][lw(muc)][1]
          blockiec = np.tensordot(blockin,ieic,(3,0))
          plus_assign(eaction_in,(mua,mub,lw(muc)),blockiec)
      outputs = {('e','in'):eaction_in,('e','out'):eaction_out,
                 ('f','in'):faction_in,('f','out'):faction_out}
      for ef in 'ef':
        for io in ('in','out'):
          #op = '%s_%d'%(ef,i)
          #symb = '"%s" symbol'%io
          for (mua,mub,muc),block in outputs[ef,io].items():
            assert block.shape == (self.degeneracy,repa.degeneracy(mua),repb.degeneracy(mub),repc.degeneracy(muc))
            if np.any(block):
              logger.warning('**%s_%d does not vanish in %sx%sx%s block'
                ' of "%s" symbol (%0.4g)',
                ef,i,mua,mub,muc,io,fnorm(block))

  def __contains__(self,mus):
    return mus in self.blocks_in

  def asdenseunitary(self, realize=True):
    # Convert to dense, IIO form
    if isinstance(self.rep3,TrivialRepresentation):
      # Identity or "charge conjugator"
      if isinstance(self.rep1,TrivialRepresentation):
        return None
      if self.rep1.FSI == 0 or (self.rep1.FSI == 1 and realize):
        return np.expand_dims(np.identity(self.rep1.dim),(2,3))/np.sqrt(self.rep1.dim)
      else:
        S = self.rep1.charge_conjugator_fp()/np.sqrt(self.rep1.dim)
        return np.expand_dims(S.todense(),(2,3))
        
    sl1 = self.rep1.block_slice_dict
    sl2 = self.rep2.block_slice_dict
    repout = self.rep3.dual
    sl3 = repout.block_slice_dict
    d1,d2,d3,D = (self.rep1.dim,self.rep2.dim,self.rep3.dim,self.degeneracy)
    W = np.zeros((d1,d2,d3,D))
    renorm = np.sqrt(self.normsquares/repout.dim)
    from pprint import pprint
    #print(self.rep1.label,self.rep2.label,repout.label)
    for (mu1,mu2,mu3),block in self.blocks_in.items():
      #print(mu1,mu2,negate(mu3))
      S = self.rep3._CC[1][mu3]
      # Convert "in" to "out" index
      outin = self.rep3.raise_idx_block(mu3,S,1)
      block = np.tensordot(block,outin,(3,1))
      #pprint(block)
      # Change all indices to unitary basis
      block = block/np.sqrt(self.rep1._norm_squares[mu1])[None,:,None,None]
      block /= np.sqrt(self.rep2._norm_squares[mu2])[None,None,:,None]
      block *= np.sqrt(repout._norm_squares[negate(mu3)][None,None,None,:])
      block /= renorm[:,None,None,None]
      #pprint(block)
      W[sl1[mu1],sl2[mu2],sl3[negate(mu3)],:] = block.transpose((1,2,3,0))
    #pprint(np.squeeze(W))
    if realize:
      if self.rep1.FSI == 1:
        Wmat = W.reshape(self.rep1.dim,-1)
        Wmat = self.rep1.real_cob().conj().dot(Wmat)
        W = Wmat.reshape(d1,d2,d3,D)
      if self.rep2.FSI == 1:
        Wmat = W.transpose(1,0,2,3).reshape(self.rep2.dim,-1)
        Wmat = self.rep2.real_cob().conj().dot(Wmat)
        W = Wmat.reshape(d2,d1,d3,D).transpose(1,0,2,3)
      if repout.FSI == 1:
        Wmat = W.transpose(2,0,1,3).reshape(repout.dim,-1)
        Wmat = repout.real_cob().dot(Wmat)
        W = Wmat.reshape((d3,d1,d2,D)).transpose(1,2,0,3)
    return W


class RationalRacahSymbol:
  def __init__(self, array, *indices):
    """For each index provide (3j symbol, "out" (True) or "in" (False),
    raised (True) or lowered (False)"""
    self.array = array
    assert isinstance(array,RationalArray)
    array.rereduce()
    threejs = []
    self._isout = []
    self._israised = []
    assert array.ndim == len(indices)
    for i,(symb,io,rl) in enumerate(indices):
      assert isinstance(symb,Rational3j)
      assert symb.degeneracy == array.shape[i]
      threejs.append(symb)
      assert isinstance(io,bool)
      self._isout.append(io)
      assert isinstance(rl,bool)
      self._israised.append(rl)
    self._3js = tuple(threejs)
    self._readonly = False
  
  def freeze(self):
    """Make read-only"""
    self._readonly = True
    self._isout = tuple(self._isout)
    self._israised = tuple(self._israised)
    
  def raise_index(self, idx, inplace=False):
    assert not self._israised[idx]
    if inplace:
      assert not self._readonly
      self.array.raise_index(self._3js[idx].normsquares,idx,True)
      self._israised[idx] = True
    else:
      array = self.array.raise_index(self._3js[idx].normsquares,idx)
      israised = list(self._israised)
      israised[idx] = True
      return self.init_like(array)
    
  def lower_index(self, idx, inplace=False):
    assert self._israised[idx]
    if inplace:
      assert not self._readonly
      self.array.lower_index(self._3js[idx].normsquares,idx,True)
      self._israised[idx] = False
    else:
      array = self.array.lower_index(self._3js[idx].normsquares,idx)
      israised = list(self._israised)
      israised[idx] = False
      return self.init_like(array)

  def __getitem__(self, indices):
    value = self.array[indices]
    if self._readonly:
      return copy(value)
    else:
      return value

  def __setitem__(self, indices, value):
    assert not self._readonly
    self.array[indices] = value

  def init_like(self, array):
    return RationalRacahSymbol(array, *self.indices)

  def __add__(self, rhs):
    if isinstance(rhs, RationalRacahSymbol):
      assert self.ndim == rhs.ndim
      for i in range(self.ndim):
        assert self._3js[i] is rhs._3js[i]
        assert self._isout[i] == rhs._isout[i]
        assert self._israised[i] == rhs._israised[i]
      rhs = rhs.array
    return self.init_like(self.array+rhs)

  def __radd__(self, lhs):
    return self.__add__(lhs)

  def __iadd__(self, rhs):
    assert not self._readonly
    if isinstance(rhs, RationalRacahSymbol):
      assert self.ndim == rhs.ndim
      for i in range(self.ndim):
        assert self._3js[i] is rhs._3js[i]
        assert self._isout[i] == rhs._isout[i]
        assert self._israised[i] == rhs._israised[i]
      rhs = rhs.array
    self.array += rhs
    return self

  def __neg__(self):
    return self.init_like(-self.array)

  def __sub__(self, rhs):
    return self.__add__(-rhs)

  def __rsub__(self, lhs):
    return self.__neg__().__add__(lhs)

  def __isub__(self, rhs):
    return self.__iadd__(-rhs)

  def __mul__(self, rhs):
    assert isinstance(rhs,Number)
    return self.init_like(self.array*rhs)

  def __rmul__(self, lhs):
    return self.__mul__(lhs)

  def __imul__(self, rhs):
    assert isinstance(rhs,Number)
    assert not self._readonly
    self.array *= rhs
    return self

  def __truediv__(self, rhs):
    assert isinstance(rhs,Number)
    return self.init_like(self.array/rhs)

  def __itruediv__(self, rhs):
    assert isinstance(rhs,Number)
    assert not self._readonly
    self.array /= rhs
    return self

  @property
  def indices(self):
    """Indices as provided to init"""
    return tuple(zip(self._3js,self._isout,self._israised))

  @property
  def ndim(self):
    return self.array.ndim

  @property
  def shape(self):
    return self.array.shape

  def get3j(self, index):
    """3j symbol associated with a given index"""
    return self._3js[index]
  
  def contract(self, symb2, indices):
    """Contract two symbols. Indices supplied as in np.tensordot"""
    assert isinstance(symb2,RationalRacahSymbol)
    if isinstance(indices,int):
      indices = (tuple(range(self.ndim-indices,self.ndim)),
        tuple(range(symb2.ndim-indices,symb2.ndim)))
    idx1,idx2 = indices
    if isinstance(idx1,int):
      assert isinstance(idx2,int)
      idx1 = (idx1,)
      idx2 = (idx2,)
    for i1,i2 in zip(idx1,idx2):
      if self._3js[i1] is not symb2._3js[i2]:
        raise ValueError('Contracted symbols do not match at indices %d-%d '
          '(<%s,%s,%s> vs <%s,%s,%s>)' % ((i1,i2)+self._3js[i1].labels+symb2._3js[i2].labels))
      if self._isout[i1] == symb2._isout[i2]:
        raise ValueError('Contracted symbols are incompatible at indices %d-%d '
          '(both "%s")' % (i1,i2,['in','out'][self._isout[i1]]))
      if self._israised[i1] == symb2._israised:
        raise ValueError('Contracted symbols are incompatible at indices %d-%d '
          '(both %s)' % (i1,i2,['lowered','raised'][self._israised[i1]]))
    result = np.tensordot(self.array,symb2.array,indices)
    iout = []
    for i,idx in enumerate(self.indices):
      if i not in idx1:
        iout.append(idx)
    for i,idx in enumerate(symb2.indices):
      if i not in idx2:
        iout.append(idx)

    return RationalRacahSymbol(result, *iout)

  def transpose(self, *axes):
    array = self.array.transpose(*axes).copy()
    if isinstance(axes[0],(tuple,list)):
      axes, = axes
    iold = self.indices
    indices = [iold[i] for i in axes]
    return RationalRacahSymbol(array, *indices)

  def conj(self):
    """Since there are no complex values, only flips in/out"""
    return RationalRacahSymbol(self.array,
      *[(threej,not io, rl) for (threej,io,rl) in self.indices])

  def norm(self):
    return np.linalg.norm(self.array)

  def rereduce(self):
    self.array.rereduce()

  def copy(self):
    return RationalRacahSymbol(self.array.copy(),*self.indices)

  def __copy__(self):
    return self.copy()

  def __eq__(self, symb2):
    if self.indices != symb2.indices:
      raise ValueError('"Symbol" arrays can only be compared when indices match')
    return np.all(self.array == symb2.array)

  def zeros_like(self):
    return RationalRacahSymbol(zeros_like(self.array),*self.indices)


class RationalRacahOperator(RationalRacahSymbol):
  def __init__(self, array, *indices):
    if not indices:
      assert isinstance(array,RationalRacahSymbol)
      # Convert to "operator"
      assert array.ndims == 2
      self._3js = tuple(array._3js)
      self._isout = tuple(array._isout)
      self._israised = tuple(array._israised)
      array = rarray(array.array)
    else:
      if isinstance(array,RationalRacahOperator):
        # Same matrix, different (presumably matching) symbols
        array = array.__array
      assert len(indices) == 2
      symbs, ios, rls = zip(*indices)
      self._3js = tuple(symbs)
      self._isout = tuple(ios)
      self._israised = tuple(rls)

    # "kinds" identity, scalar, diagonal, or array,
    # stored as 1,s,d,a
    if isinstance(array,RationalArray):
      array.rereduce()
      if np.all(self._3js[0].normsquares == self._3js[1].normsquares):
        # If diagonal - with matching metrics - store as such
        if not np.any(array - np.diag(np.diag(array))):
          array = np.diag(array)
      if array.ndim == 1:
        array = tuple(array)
      else:
        self._k = 'a'
    if isinstance(array,list):
      array = tuple(array)
    if isinstance(array,tuple):
      if all(x == array[0] for x in array[1:]):
        # Scalar-valued
        array = array[0]
      elif all(abs(x) == 1 for x in array):
        self._k = '-'
        array = tuple(int(x) for x in array)
      else:
        self._k = 'd'
    if isinstance(array,(int,Fraction)):
      if array == 1:
        self._k = '1'
      else:
        self._k = 's'
    self.__array = array

  @property
  def array(self):
    if self._k in '1s':
      return self.__array * reye(self._3js[0].degeneracy)
    elif self._k in 'd-':
      return diag(self.__array)
    else:
      return self.__array

  @property
  def _array_def(self):
    return self.__array

  @property
  def _readonly(self):
    return False

  @property
  def ndim(self):
    return 2

  @property
  def shape(self):
    return (self._3js[0].degeneracy,self._3js[1].degeneracy)

  @property
  def kind(self):
    return {'1':'identity',
            's':'scalar',
            'd':'rational diagonal',
            '-':'diagonal signs',
            'a':'array'}[self._k]

  @property
  def T(self):
    if self._k == 'a':
      array = self.__array.T
    else:
      array = self.__array
    return RationalRacahOperator(array, *(self.indices[::-1]))

  def conj(self):
    return RationalRacahOperator(self.__array,
      *[(threej,not io, rl) for (threej,io,rl) in self.indices])

  @property
  def H(self):
    if not bool.__xor__(*self._israised):
      raise ValueError('Hermitian transpose only permitted on raised-lowered '
        'or lowered-raised (symbol is %s-%s)' % self._israised)
    if self._k != 'a':
      # TODO may be valid to have list with different norm-squares on each side?
      array = self.__array
    else:
      array = self.__array.T
      if self._israised[0]:
        array = array.lower_index(self._3js[0].normsquares,0)
        array.raise_index(self._3js[1].normsquares,1,True)
      else:
        array = array.raise_index(self._3js[0],normsquares,0)
        array.lower_index(self._3js[1].normsquares,1,True)
    return RationalRacahOperator(array,
      *[(threej,not io, not rl) for (threej,io,rl) in self.indices[::-1]])

  def apply(self, symbol, idx):
    if self._k == 'a':
      symbout = self.contract(symbol,(1,idx))
      return symbout.transpose((*range(1,idx+1),0,*range(idx+1,symbol.ndim)))
    else:
      newindices = [*symbol.indices[:idx],self.indices[0],
        *symbol.indices[idx+1:]]
      assert self.domain is symbol.get3j(idx)
      assert self._isout[1] ^ symbol._isout[idx]
      assert self._israised[1] ^ symbol._israised[idx]
      if self._k == '1':
        return RationalRacahSymbol(symbol.array,*newindices)
      elif self._k == 's':
        return RationalRacahSymbol(self.__array*symbol.array,*newindices)
      else:
        array = symbol.array.copy()
        arrT = array.transpose((idx,*range(idx),*range(idx+1,symbol.ndim)))
        for i,li in enumerate(self.__array):
          arrT[i] *= li
        return RationalRacahSymbol(array, *newindices)

  def dot(self, O2):
    assert isinstance(O2,RationalRacahOperator)
    assert self._3js[1] is O2._3js[0]
    assert self._isout[1] ^ O2._isout[0]
    assert self._israised[1] ^ O2._israised[0]
    if self._k in '1s':
      if O2._k in 'd-':
        array = tuple(self.__array * li for li in O2.__array)
      else:
        array = self.__array * O2.__array
    elif self._k in 'd-':
      if O2._k in '1s':
        array = tuple(li * O2.__array for li in self.__array)
      elif O2._k in 'd-':
        array = tuple(l1*l2 for l1,l2 in zip(self.__array,O2.__array))
      else:
        array = O2.__array.copy()
        for i,li in enumerate(self.__array):
          array[i] *= li
    else:
      if O2._k in '1s':
        array = self.__array * O2.__array
      elif O2._k in 'd-':
        array = self.__array.copy()
        for i,li in enumerate(O2.__array):
          array[:,i] *= li
      else:
        array = self.__array.dot(O2.__array)
    return RationalRacahOperator(array, self.indices[0], O2.indices[1])

  @property
  def domain(self):
    return self._3js[1]

  @property
  def target(self):
    return self._3js[0]



def getFsymbol(g, a, b, c, d, e, f):
  r'''Compute rational "F-symbol" representing recoupling
      a   b     c    a     b   c
       \ /     /      \     \ /
        \     /  ==>   \     /
         e   /          \   f
          \ /            \ /
           |              |
           d              d
  (alternatively, proportional to generalized 6j symbol)
  Note that, in addition to being defined with respect to a non-normalized
  basis of 3j symbols, symbol has global factor of sqrt(d_e/d_f) with respect
  to 'true' F symbol
  Indices in the symbol returned are in the order
  (b,c,fbar),(a,f,dbar),(a,b,ebar),(e,c,dbar)
  (first two are raised, last two lowered'''
  dbar = g.get_dummy(d).duallabel()
  repe = g.get_ratrep(e)
  repf = g.get_ratrep(f)
  bcf = g.get3j_rational(b, c, repf.duallabel())
  afd = g.get3j_rational(a, f, dbar)
  abe = g.get3j_rational(a, b, repe.duallabel())
  ecd = g.get3j_rational(e, c, dbar)
  F = ra.rzeros((bcf.degeneracy, afd.degeneracy, abe.degeneracy, ecd.degeneracy))
  mud = dbar
  repe._construct_charge_conjugator(repe.dual)
  repf._construct_charge_conjugator(repf.dual)
  Ses = repe._CC[1]
  Sfs = repf._CC[1]
  for mua,mub,nmue in abe.blocks_in:
    mue = negate(nmue)
    muc = tuple(-ai-bi-di for ai,bi,di in zip(mua,mub,mud))
    muf = tuple(bi+ci for bi,ci in zip(mub,muc))
    nmuf = negate(muf)
    if (mub,muc,nmuf) not in bcf:
      continue
    if (mue,muc,mud) not in ecd:
      continue
    if (mua,muf,mud) not in afd:
      continue
    abeblock = abe.blocks_in[mua,mub,nmue]
    bcfblock = bcf.blocks_out[mub,muc,nmuf]
    ecdblock = ecd.blocks_in[mue,muc,mud]
    afdblock = afd.blocks_out[mua,muf,mud]
    Xf = repf.dual.lower_idx_block(nmuf,Sfs[muf],0)
    Oe = repe.raise_idx_block(mue,Ses[mue],1)
    abeblock = np.tensordot(abeblock,Oe,1)
    bcfblock = np.tensordot(bcfblock,Xf,1)
    abcd_e = np.tensordot(abeblock,ecdblock,(3,1)) # I3 a b I4 c d
    afd_env = np.tensordot(bcfblock,abcd_e,((1,2),(2,4))) # I1 f I3 a I4 d
    Fblock = np.tensordot(afdblock,afd_env,((1,2,3),(3,1,5)))
    Fblock.rereduce()
    F += Fblock.transpose(1,0,2,3)
  F = rationallie.RationalRacahSymbol(F,(bcf,True,False),(afd,True,False),
    (abe,False,False),(ecd,False,False))
  F.raise_index(0,True)
  F.raise_index(1,True)
  F *= repf.dim * g.get_dummy(d).dim
  F.array.rereduce()
  return F 
