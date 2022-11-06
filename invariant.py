# Tensors invariant under group symmetries
from .npstack import np,RNG,linalg,safesvd,sparse
import re
import itertools
import functools
import collections
from abc import abstractmethod
from . import links,config,tensors,groups
from .groups import Group,GroupDerivedType,SumRepresentation
from .tensors import Tensor,FusionPrimitive,_endomorphic,_eig_vec_process

class InvariantFusion(FusionPrimitive,metaclass=GroupDerivedType):
  """Determine fusion for indices invariant under group action
  Calculates 'Clebsch-Gordan' matrices"""
  _regname='Fusion'
  _required_types=(SumRepresentation,)

  def __init__(self, vs_in, v_out=None, idx_in=None, CG=None):
    FusionPrimitive.__init__(self, vs_in, v_out=v_out, idx_in=idx_in)
    if CG is not None:
      if isinstance(CG,int):
        CG = None
      self._CGmat = CG
      return
    if self._rank == 1:
      if self._out.qsecond:
        # Must apply some charge-conjugation
        blocks = []
        rep = []
        for k,n in self._out._decomp:
          if self.group.indicate(k) == -2:
            kd = self.group.dual(k)
            blocks.extend(n*[self.group.S(kd).T.conj()])
            rep.append((kd, n))
          else:
            nn = n*self.group.dim(k)
            blocks.append(sparse.eye(nn, dtype=int, format='csr'))
            rep.append((k,n))
        self._CGmat = sparse.block_diag(blocks, format='csr')
        self._out = self.group.SumRep(rep)
      else:
        self._CGmat = None
    else:
      self._out, self._CGmat = self.CG_tree(self._spaces_in)
    self._rep_idx = None
    self._dimeff = None

  @classmethod
  def CG_tree(cls, reps):
    """Method for recursively building Clebsch-Gordan coefficients
    for a product representation
    Returns fused decomposition, CG matrix"""
    if len(reps) == 1:
      # Yield identity
      # NOTE: Should not be root case
      return reps[0], sparse.eye(reps[0].dim, dtype=int, format='csr')
    if len(reps) == 2:
      # No further assembly required
      return cls.CG_matrix(*reps)
    # Find optimal dividing line
    # NOTE: may be further optimized by permitting reordering, I am choosing
    # not to do this for the time being
    dims = tuple(v.dim for v in reps)
    rank = len(reps)
    # Determine possible pairs of dimensions
    dls = [dims[0]]
    drs = [dims[-1]]
    for idiv in range(1,rank-1):
      dls.append(dls[-1]*dims[idiv])
      drs.insert(0, drs[0]*dims[-idiv-1])
    # Find minimum mismatch
    for id0 in range(rank-1):
      if dls[id0] > drs[id0]:
        # Crossover
        if id0 == 0 or dls[id0]/drs[id0] < drs[id0-1]/dls[id0-1]:
          idiv = id0
        else:
          idiv = id0-1
        break
    if dls[-1] < drs[-1]:
      idiv = rank-2
    # Generate left decomposition, CG
    vl,CGl = cls.CG_tree(reps[:idiv+1])
    # right, same
    vr,CGr = cls.CG_tree(reps[idiv+1:])
    # Produce CG coefficients between right & left decompositions
    vo,CGo = cls.CG_matrix(vl, vr)
    return vo, CGo.dot(sparse.kron(CGl, CGr))

  @classmethod
  def CG_matrix(cls, repleft, repright):
    """Method for determining a sparse Clebsch-Gordan matrix for a pair of
    sum representations"""
    # Unrolling block-diagonal basis into sum-decomposition basis
    idxO = collections.defaultdict(list)
    ffin = collections.defaultdict(int)
    # Clebsch-Gordan coefficients, as blocks for diagonal matrix
    CGblocks = []
    # Re-rolling Kronecker-product basis into block-diagonal basis
    idxI = []
    ML = repleft.dim
    MR = repright.dim
    iblock = 0
    iL = 0
    for kl,nl in repleft._decomp:
      dl = cls.group.dim(kl)
      iR = 0
      for kr,nr in repright._decomp:
        dr = cls.group.dim(kr)
        # Collect unrolled (dl*nl)x(dr*nr) block
        # into nl*nr blocks of size dl*dr
        for bil in range(nl): # left blocks
          for bir in range(nr): # right blocks
            for idxl in range(dl): # left irrep index
              idx0 = MR*(iL+bil*dl+idxl) + iR+bir*dr
              idxI.extend(range(idx0,idx0+dr))
        # CG blocks
        fo,Wo = cls.group.fusionCG(kl,kr)
        if next(iter(Wo.values())) is None:
          k = fo[0][0]
          # May, in this case, require charge-conjugation
          if cls.group.indicate(k) == -2:
            k = cls.group.dual(k)
            W = cls.group.S(k).T.conj()
          else:
            # Otherwise: identity mapping
            W = sparse.eye(dl*dr, dtype=int, format='csr')
          idxO[k].extend(range(iblock,iblock+nl*nr*dl*dr))
          ffin[k] += nl*nr
        else:
          W = []
          ib = iblock
          for k,n in fo:
            do = cls.group.dim(k)
            W.append(Wo[k].reshape((dl*dr,do*n)).T)
            # Put each copy with the appropriate irrep
            for io in range(nl*nr):
              idxO[k].extend(range(ib + io*dl*dr,ib+io*dl*dr+n*do))
            ib += n*do
            ffin[k] += nl*nr*n
          W = np.concatenate(W,axis=0)
        iblock += nl*nr*dl*dr
        CGblocks.extend(nl*nr*[W])
        iR += dr*nr
      iL += dl*nl
    MO = ML*MR
    # Input permutation
    permI = sparse.csr_matrix((MO*[1],idxI,range(MO+1)),shape=(MO,MO),dtype=int)
    # Block-diagonal Clebsch-Gordan matrix
    W = sparse.block_diag(CGblocks, format='csr')
    idxOl = []
    fo = []
    for k in idxO:
      fo.append((k,ffin[k]))
      idxOl.extend(idxO[k])
    # Output permutation
    permO=sparse.csr_matrix((MO*[1],idxOl,range(MO+1)),shape=(MO,MO),dtype=int)
    return cls.group.SumRep(fo), permO.dot(W).dot(permI)
  
  def conj(self):
    if self._CGmat is None:
      CGc = 0
    else:
      CGc = self._CGmat.conj()
    return self.__class__([~v for v in self._spaces_in],
      ~self._out,self._idxs,CG=CGc)

  @property
  def irrep_idx(self):
    if self._rep_idx is None:
      self._rep_idx = self._out.idx_of(self.__class__.group.triv)
    return self._rep_idx

  @property
  def effective_dim(self):
    if self._dimeff is None:
      self._dimeff = self._out.degen_of(self.__class__.group.triv)
    return self._dimeff

  def vector_convert(self, T, idx=None):
    # Select invariant subspace
    v = super().vector_convert(T,idx)
    return v[self.irrep_idx:self.irrep_idx+self.effective_dim]

  def tensor_convert(self, v, idx=None):
    if idx is None:
      idx = self._idxs
    v1 = np.zeros((self._out.dim,),dtype=config.FIELD)
    v1[self.irrep_idx:self.irrep_idx+self.effective_dim] = np.squeeze(v)
    V = self.group.Tensor(v1,(0,),(self._out,))
    return V._do_unfuse({0:(idx,self)})

  def tensor_convert_multiaxis(self, T, idx_newax, ax_old=0, **kw_args):
    shape = list(T.shape)
    ax_old = ax_old%len(shape)
    shape[ax_old] = self.dim
    T1 = np.zeros(tuple(shape),config.FIELD)
    sl = slice(self.irrep_idx,self.irrep_idx+self.effective_dim)
    sls = (len(shape)-1)*[slice(None)]
    sls.insert(ax_old,sl)
    T1[tuple(sls)] = T
    return super().tensor_convert_multiaxis(T1, idx_newax, ax_old, **kw_args)

  @classmethod
  @property
  def _tensor_class(cls):
    return cls.group.Tensor

  def empty_like(self, idxs=None):
    if idxs is None:
      idxs = self._idxs
    T = np.empty(self.shape,dtype=config.FIELD)
    return self.group.Tensor(T, idxs, self._spaces_in)


class SubstantiatingFusion(InvariantFusion):
  """Given a ChargedTensor (i.e. a tensor with an implicit index transforming
  under irrep*) determine relations between that & an invariant tensor
  (i.e. one where the implicit index has been substantiated)"""
  _regname='Substantiation'
  _required_types=(InvariantFusion,)

  def __init__(self, vs_in, v_out=None, idx_in=None, CG=None,
      irrep=None, subst_info=None):
    InvariantFusion.__init__(self, vs_in, v_out, idx_in, CG)
    self._irrep = irrep
    if subst_info is None:
      self._out_subst, self._CG_subst = self.CG_tree((self._out,
        self.group.SumRep([(self.group.dual(irrep), 1)])))
    else:
      self._out_subst, self._CG_subst = subst_info

  def substantiating_fusion_matrix(self, *reps):
    """Given remaining details of the structure of a tensor
    (other representations in order),
    produce transformation from
    (flattened array w/implicit index) -> (flattened invariant array)"""
    # Fusion of explicit indices
    f,CG = self.CG_tree((self._out,)+reps)
    d = self.group.dim(self._irrep)
    M = self._out.dim
    N = CG.shape[0]
    N0 = N//M
    # Implicit index made explicit:
    CGk = sparse.kron(CG.T.conj(),sparse.eye(d,dtype=int,format='csr'))
    # Fill out flattened sector to identity blocks:
    i0 = self._out.idx_of(self._irrep)
    n = self._out.degen_of(self._irrep)
    idx = []
    idxptr = i0*[0]
    for a in range(n):
      # input (row):  single index i0 + a*d
      # output (column): identity matrix i0+a*d+j + N*j
      idxptr.extend(d*[idxptr[-1]+d])
      idx.extend(i0 + a*d+j + N*j for j in range(d))
    idblocks = sparse.csr_matrix((d*n*[1],idx,idxptr),dtype=int,shape=(d*N,N))
    mat = self._CG_subst.dot(CGk.dot(idblocks.dot(CG)))
    return mat

  def conj(self):
    if self._CGmat is None:
      CG = None
    else:
      CG = self._CGmat.conj()
    if self._CG_subst is None:
      CGsubst = self._CG_subst.conj()
    return self.__class__([~v for v in self._spaces_in],~self._out,self._idxs,
        CG=CG,irrep=self.group.dual(self._irrep),
        subst_info=(~self._out_subst,CGsubst))

  @classmethod
  @property
  def _tensor_class(cls):
    return cls.group.ChargedTensor

  def empty_like(self, idxs=None):
    if idxs is None:
      idxs = self._idxs
    T = np.empty(self.shape,dtype=config.FIELD)
    return self.group.ChargedTensor(T, idxs, self._spaces_in, self._irrep)


class GroupTensor(Tensor,metaclass=GroupDerivedType):
  """Abstract class encompassing tensors which transform (trivially or
  nontrivially) under group action"""
  _regname='TensorAbstract'

  def tensor(cls, t, idx):
    raise NotImplementedError

  @classmethod
  def _vspace_from_arg(cls, A):
    try:
      A = list(A)
    except TypeError:
      if not isinstance(A,int) or A <= 0:
        raise ValueError('Invariant index must be provided as positive integer')
      A = [(cls.group.triv,A)]
    repset = set()
    for x in A:
      if not isinstance(x,tuple) or len(x) != 2:
        raise ValueError('irrep list must be of ordered pairs')
      k,n = x
      if not cls.group.isrep(k):
        raise ValueError('invalid irrep \'%s\''%k)
      if not isinstance(n,int) or n < 0:
        raise ValueError('second element of irrep pair must be valid '
          'degeneracy')
      if k in repset:
        raise ValueError('irrep \'%s\' repeated'%k)
      elif cls.group.indicate(k) < 0 and cls.group.dual(k) in repset:
        raise ValueError('quaternionic \'%s\' double represented'%k)
      repset.add(k)
    return cls.group.SumRep(A)

  def getFusion(self, ls, *args, **kw_args):
    return self.group.Fusion(ls,self,*args,**kw_args)

  def singletonFusion(self, l):
    return self.group.Fusion((l,),self,CG=0)

  @classmethod
  def buildFusion(cls, ls, *args, **kw_args):
    return cls.group.Fusion(ls, *args, **kw_args)

  def _do_fuse(self, *args, prims=None):
    neworder = []
    newshape = []
    permutation = []
    newvs = []
    if prims is None: # Provided if computation is being repeated
      prims = self._get_fuse_prim(*args)
    for arg in args:
      if len(arg) == 1:
        l1, = arg
        permutation.append(self._idxs.index(l1))
      else:
        l1,l0s = arg[:2]
        for ll in l0s:
          permutation.append(self._idxs.index(ll))
      neworder.append(l1)
      newvs.append(prims[l1].V)
      newshape.append(prims[l1].dim)
    T = self._T.transpose(permutation).reshape(newshape)
    # Apply Clebsch-Gordan matrices 
    for i in range(len(newshape)):
      CG = prims[neworder[i]]._CGmat
      if CG is not None:
        T = T.reshape((newshape[i],-1))
        T = CG.dot(T).T
        T = T.reshape(newshape[i+1:]+newshape[:i+1])
      else:
        T = np.moveaxis(T, 0, -1)
    return self._tensorfactory(T.astype(config.FIELD),
      tuple(neworder), tuple(newvs)), prims

  def _do_unfuse(self, fuseinfo):
    newvs = []
    newshape = []
    newidxs = []
    T = self._T
    subindicate = [isinstance(fuseinfo[ll],tuple) and \
        isinstance(fuseinfo[ll][1],SubstantiatingFusion) and \
        fuseinfo[ll][1]._out_subst == self._dspace[ll] for ll in self._idxs]
    subst = any(subindicate)
    if subst:
      # Exists substantiated implicit index to be unfused/unsubstantiated
      if sum(subindicate) > 1 or isinstance(self, ChargedTensor):
        raise ValueError('Attempting to unfuse multiple implicit indices')
      idx = subindicate.index(True)
      ll = self._idxs[ll]
      sfuse = fuseinfo[ll][1]
      irrep = sfuse._irrep
      T = T.moveaxis(idx,0)
      shape = T.shape
      T = T.reshape((shape[idx],-1))
      T = sfuse._CG_subst.transpose().conj().dot(T)
      T = T.reshape((shape[idx],self.group.dim(irrep),-1))[:,0,:]
      T = T.moveaxis(0,idx).reshape(shape[:idx]+(shape[idx],)+shape[idx+1:])
    else:
      irrep = None
    for ll in self._idxs:
      if isinstance(fuseinfo[ll],tuple):
        names,fusion = fuseinfo[ll]
        vs = fusion._spaces_in
        newidxs.extend(names)
        newvs.extend(vs)
        newshape.extend((V.dim for V in vs))
        if fusion._CGmat is not None:
          d = T.shape[0]
          sh = T.shape[1:]
          # Reshape & apply Clebsch-Gordan matrix
          T = T.reshape((d,-1))
          T = fusion._CGmat.conj().transpose().dot(T)
          T = T.T.reshape(sh+(d,))
        else:
          T = np.moveaxis(T,0,-1)
      else:
        newidxs.append(fuseinfo[ll])
        V = self._dspace[ll]
        newvs.append(V)
        newshape.append(V.dim)
        T = np.moveaxis(T,0,-1)
    T = T.reshape(newshape)
    return self._tensorfactory(T, tuple(newidxs), tuple(newvs), irrep=irrep)
  
  @abstractmethod
  def _rand_init(self, **settings):
    """Initialize as random tensor
    (assumed pre-initialized as zero)"""
    pass

  def rand_like(self, **settings):
    A = self.zeros_like()
    A._rand_init(**settings)
    return A

  @classmethod
  def rand_from(cls, parsestr, *tens, **settings):
    A = cls.zeros_from(parsestr, *tens)
    A._rand_init(**settings)
    return A

  @abstractmethod
  def symmetrized(self):
    """Enforce group symmetry"""
    pass

  def group_action(self, g, *idxs):
    """Apply action of group element g on indices
    If not supplied, will be all indices (is this a bad idea?)"""
    if not idxs:
      idxs = self._idxs
    T = self._T
    for ll in self._idxs:
      if ll in idxs:
        Rg = self._dspace[ll].matrix(g)
        T = np.tensordot(T, Rg, (0,1)).astype(config.FIELD)
      else:
        T = np.moveaxis(T,0,-1)
    return self._init_like(T)

  def _do_contract(self, T2, contracted, out1, out2, conj):
    # Various cases depending on invariance
    switch = False
    if not isinstance(T2, GroupTensor):
      # Non-invariant tensor
      switch = True
    elif self.group is not T2.group:
      raise ValueError('Cannot contract tensors with non-matching groups')
    elif isinstance(T2, InvariantTensor):
      switch = isinstance(self, InvariantTensor)
    elif isinstance(self, ChargedTensor):
      if isinstance(T2, ChargedTensor):
        # Both are charged - not implemented yet
        raise NotImplementedError('Contraction of charged tensors not yet'
        ' implemented')
    elif isinstance(T2, ChargedTensor):
      switch = True
    if switch:
      cswitch = [(r,l) for l,r in contracted]
      if conj:
        T2 = T2.conj()
      return Tensor._do_contract(T2, self, cswitch, out2, out1, False)
    return Tensor._do_contract(self, T2, contracted, out1, out2, conj)

  def renamed(self, idxmap, view=False, strict=False):
    # For the time being return copy, not view
    return Tensor.renamed(self, idxmap, False, strict)


class InvariantTensor(GroupTensor,metaclass=GroupDerivedType):
  """Tensors which are invariant under transformations
  Densely-represented"""
  _regname='Tensor'
  _required_types=(SumRepresentation,InvariantFusion)

  def __init__(self, T, idxs, spaces):
    super().__init__(T, idxs, spaces)
    self._irrep = self.group.triv

  def matches(self, A, conj):
    if not isinstance(A, self.group.Tensor):
      raise ValueError('Failure to match symmetry group')
    Tensor.matches(self, A, conj)

  def _tensorfactory(self, T, idxs, spaces, irrep=None):
    if irrep is None or irrep == self._irrep:
      return self.group.Tensor(T, idxs, spaces)
    else:
      return self.group.ChargedTensor(T, idxs, spaces, irrep)

  def _rand_init(self, **settings):
    if self.rank == 1:
      # Set identity sector
      i0 = 0
      for k,n in self._spaces[0]._decomp:
        if k == self.group.triv:
          self._T[i0:i0+n] = tensors._np_rand((n,), **settings)
          break
        i0 += n*self.group.dim(k)
    else:
      # Divide in two
      left, right = self._split_indices()
      # Matrix that will be block-diagonal-ish w/ identity sub-blocks
      M,fs = self._svd_fuse(left, right)
      V2 = fs[1].V
      i1 = 0
      for k,n in fs[0].V:
        d = self.group.dim(k)
        kd = self.group.dual(k)
        # Find right index of block
        if kd in V2:
          i2 = V2.idx_of(kd)
          n2 = V2.degen_of(kd)
          t_rand = np.kron(tensors._np_rand((n,n2), **settings),np.identity(d))
          M._T[i1:i1+d*n,i2:i2+d*n2] = t_rand
        i1 += d*n
      self._T = M._do_unfuse({0:(left,fs[0]), 1:(right,fs[1])})._T
    
  def _split_indices(self):
    """Split indices into left/right for closest to 'square' matrix"""
    if self.rank == 2:
      return [self._idxs[0]],[self._idxs[1]]
    nsq = np.sqrt(self.numel)
    sdiff = np.inf
    d = 1
    for i in range(self.rank-2):
      d *= self._spaces[i].dim
      sd = d - nsq
      if abs(sd) < sdiff:
        i0 = i
        sdiff = abs(sd)
      if sd < 0:
        break
    return self._idxs[:i0+1], self._idxs[i0+1:]

  def _svd_fuse(self, left, right):
    """Fuse into matrix such that quaternionic irreps are appropriately
    divided ("primary" on left, "secondary" on right)"""
    fpl = self.group.Fusion(left, self)
    fpr = self.group.Fusion([~self._dspace[lr] for lr in right],
      idx_in=right).conj()
    return self._do_fuse((0,left), (1,right), prims={0:fpl,1:fpr})

  def _do_svd(self, chi, tolerance, chi_ratio, sv_tensor):
    """Helper function for svd;
    must provide rank-2 tensor indexed 0,1
      (if quaternionic irreps are present, 0 must have "primary" form &
      1 must have "secondary" - i.e. use _svd_fuse)"""
    T = self.permuted((0,1))
    VL = self._dspace[0]
    VR = self._dspace[1]
    Us = {} # Sector-wise left unitaries
    ss = {} # Sector-wise singular values
    Vs = {} # Sector-wise right unitaries
    s_presort = [] # Sortable list of singular values w/ irrep info
    il = 0
    snorm = np.linalg.norm(T)
    D0 = 0
    for k,n in VL:
      d = self.group.dim(k)
      kd = self.group.dual(k)
      if kd in VR:
        # Get block
        ir = VR.idx_of(kd)
        nr = VR.degen_of(kd)
        Tk = T[il:il+d*n:d,ir:ir+d*nr:d]
        U,s,Vd = safesvd(Tk)
        # First truncate based on singular value tolerance
        m = len(s)
        if tolerance:
          while m>0 and s[m-1]/snorm < tolerance:
            m -= 1
        Us[k] = U[:,:m]
        ss[k] = s[:m]
        Vs[k] = Vd[:m,:]
        D0 += m*d
        if chi:
          s_presort.extend((sv,k,d) for sv in ss[k])
      il += d*n
    if chi and D0 > chi:
      s_sort = sorted(s_presort, key=lambda st:-st[0])
      D = 0
      Dks = collections.defaultdict(int)
      while s_sort and D < chi:
        sv,k,d = s_sort.pop(0)
        D += d
        Dks[k] += 1
      if config.degen_tolerance:
        if chi_ratio: # Find maximum allowable value to follow degeneracies
          if isinstance(chi_ratio,int):
            chi_max = chi_ratio
          else:
            chi_max = int(np.floor(chi_ratio*chi))
          chi_max = min(chi_max,VL.dim,VR.dim)
        else:
          chi_max = min(VL.dim,VR.dim)
        while s_sort and D < chi_max:
          sv1,k,d = s_sort.pop(0)
          if 1-sv1/sv > config.degen_tolerance:
            break
          Dks[k] += 1
          D += d
          sv1 = sv
    else:
      Dks = {k:len(ss[k]) for k in ss}
    V_cr = self.group.SumRep(list(Dks.items()))
    D = V_cr.dim
    V_cl = V_cr.dual()
    U = np.zeros((VL.dim, D), dtype=config.FIELD)
    s = []
    Vd = np.zeros((D, VR.dim), dtype=config.FIELD)
    ic = 0
    for k,n in V_cr:
      if n == 0:
        continue
      d = self.group.dim(k)
      il = VL.idx_of(k)
      ir = VR.idx_of(self.group.dual(k))
      Uk = np.kron(Us[k][:,:n],np.identity(d))
      U[il:il+Uk.shape[0],ic:ic+d*n] = Uk
      s.extend(np.repeat(ss[k][:n],d))
      Vk = np.kron(Vs[k][:n,:],np.identity(d))
      Vd[ic:ic+d*n,ir:ir+Vk.shape[1]] = Vk
      ic += d*n
    U = self.__class__(U, (0,1), (VL,V_cl))
    Vd = self.__class__(Vd, (0,1), (V_cr,VR))
    if sv_tensor:
      s = self.__class__(np.diag(s), (0,1), (V_cl,V_cr))
    return U,s,Vd

  def symmetrized(self):
    if self.rank == 1:
      # Set identity sector
      V = self._spaces[0]
      Tsym = self.zeros_like()
      i0 = V.idx_of(self.group.triv)
      n = V.degen_of(self.group.triv)
      Tsym._T[i0:i0+n] = self._T[i0:i0+n]
      return Tsym
    # Turn into matrix, set "diagonal" blocks
    left,right = self._split_indices()
    M,fs = self._svd_fuse(left, right)
    Msym = M.zeros_like()
    V2 = fs[1].V
    i1 = 0
    for k,n in fs[0].V:
      if n == 0:
        continue
      d = self.group.dim(k)
      kd = self.group.dual(k)
      # Find right index of block
      if kd in V2:
        i2 = V2.idx_of(kd)
        n2 = V2.degen_of(kd)
        if n2 == 0:
          continue
        # Get identity component
        t0 = M._T[i1:i1+d*n,i2:i2+d*n2].reshape((n,d,n2,d))
        t0 = t0.trace(axis1=1,axis2=3)/d
        Msym._T[i1:i1+d*n,i2:i2+d*n2] = np.kron(t0,np.identity(d))
      i1 += d*n
    return Msym._do_unfuse({0:(left,fs[0]), 1:(right,fs[1])})

  @_endomorphic
  def eig(self, lidx, ridx, herm=True, selection=None, left=False, vecs=True,
      mat=False, discard=False, zero_tol=None, irrep=None, **kw_args):
    """Additional argument irrep can be 'neutral' (for invariant subspace),
    single irrep, list of irreps, or 'all' (equivalent to list of all irreps
    appearing in tensor-product decomposition)
      if mat=False, list or 'all' yields irrep:eigenset maps"""
    A, info = self._do_fuse((1,ridx),(0,lidx,1,True))
    if discard:
      del self._T
    if mat:
      if irrep is None:
        irrep = 'all'
      raise NotImplementedError()
    else:
      if irrep is None:
        irrep = 'neutral'
      if irrep == 'all':
        irrep = [self.group.dual(k) for k,n in A._dspace[0]]
      if irrep == 'neutral':
        irrep = self.group.triv
      if isinstance(irrep, list):
        rvs = {}
        for k in irrep:
          rv = A._eig_irrep(k, herm, selection, left, vecs, mat, zero_tol,
            **kw_args)
          if rv is None or (not vecs and len(rv)==0) or (vecs and len(rv[0])==0):
            continue
          if vecs:
            d_unfuse = {0:(lidx,info[0])}
            if left:
              d_unfuse_l = {1:(ridx,info[1])}
            for i in range(len(rv[0])):
              rv[1][i] = rv[1][i]._do_unfuse(d_unfuse)
              if left:
                rv[2][i] = rv[2][i]._do_unfuse(d_unfuse_l)
          rvs[k] = rv
        return rvs
      else:
        rv = A._eig_irrep(irrep, herm, selection, left, vecs, mat, zero_tol,
          **kw_args)
        if vecs:
          d_unfuse = {0:(lidx,info[0])}
          if left:
            d_unfuse_l = {1:(ridx,info[1])}
          for i in range(len(rv[0])):
            rv[1][i] = rv[1][i]._do_unfuse(d_unfuse)
            if left:
              rv[2][i] = rv[2][i]._do_unfuse(d_unfuse_l)
        return rv

  def _eig_irrep(self, irrep, herm, selection, left, vecs, mat, zero_tol,
      **kw_args):
    assert self.idxset == {0,1}
    kd = self.group.dual(irrep)
    N = self._dspace[0].degen_of(kd)
    if not N:
      return None
    idx = self._dspace[0].idx_of(kd)
    d = self.group.dim(irrep)
    A = self.permuted((0,1))[idx:idx+d*N:d,idx:idx+d*N:d]
    if selection is not None:
      if isinstance(selection, int):
        if selection < 0:
          selection = (selection%N,N-1)
        else:
          selection = (0, selection-1)
      elif not isinstance(selection, tuple):
        raise ValueError('selection argument must be integer or tuple')
      nn = selection[1] - selection[0] + 1
    else:
      nn = A.shape[0]
    # Check for reversal
    reverse = ('reverse' in kw_args and kw_args['reverse'])
    # Diagonalize
    if vecs:
      w,v,vl = _eig_vec_process(A, herm, left, selection, reverse, zero_tol)
      if mat:
        return w,v,vl
      else:
        vs = []
        init_args = ((0,), (self._spaces[0],))
        v1 = np.zeros((self.dshape[0],nn),config.FIELD)
        v1[idx:idx+d*N:d,:] = v
        for i in range(nn):
          vs.append(self._tensorfactory(v1[:,i], *init_args,irrep=irrep))
        if left: 
          vls = []
          init_args = ((1,), (self._spaces[1]))
          vl1 = np.zeros_like(v1)
          vl1[idx:idx+d*N:d,:] = v
          for i in range(nn):
            vl = vl.conj()
            # NOTE: may need to modify for quaternionic irreps
            vls.append(self._tensorfactory(vl1[:,i], *init_args,irrep=kd))
            return w,vs,vls
        return w,vs
    else:
      if herm:
        ws = linalg.eigvalsh(A._T, eigvals=selection)
      elif selection:
        ws = linalg.eigvals(A._T)
        ws = sorted(ws, key=lambda z: -abs(z))[selection[0]:selection[1]+1]
      else:
        ws = linalg.eigvals(A._T)
      if reverse:
        return ws[::-1]
      else:
        return ws


class ChargedTensor(GroupTensor,metaclass=GroupDerivedType):
  """Tensors which transform under a (nontrivial) representation k of a Group
  In case of dim(k) > 1, tensor is implicitly of the form
  T^j_i0,i1,...ir, where index j transforms under k;
  ndarray self._T is explicitly T^0_i0,i1,...ir
  NOTE: this means, if T is rank-1, the nonzero part will 
    belong to the ___k*___ sector"""
  _regname='ChargedTensor'
  _required_types=(InvariantTensor,SubstantiatingFusion)
  def __init__(self, T, idxs, spaces, irrep):
    super().__init__(T, idxs, spaces)
    self._irrep = irrep

  def _tensorfactory(self, T, idxs, spaces, irrep=None):
    if irrep is None:
      irrep = self._irrep
    elif irrep == self.group.triv:
      return self.group.Tensor(T, idxs, spaces)
    return self.group.ChargedTensor(T, idxs, spaces, irrep)

  def matches(self, A, conj):
    if not isinstance(A, self.group.ChargedTensor):
      raise ValueError('Failure to match symmetry group')
    Tensor.matches(self, A, conj)
    if (not conj and self._irrep != A._irrep) or \
        (conj and self._irrep != self.group.dual(A._irrep)):
      raise ValueError('Failure to match overall transformation')

  @classmethod
  def _identity(cls, left, right, spaces):
    # Use InvariantTensor method instead
    return cls.group.Tensor._identity(left, right, spaces)

  def _rand_init(self, **settings):
    M,f = self._do_fuse((0,self._idxs))
    V = f[0].V
    kd = self.group.dual(self._irrep)
    d = self.group.dim(self._irrep)
    if kd in V:
      i0 = V.idx_of(kd)
      n = V.degen_of(kd)
      M._T[i0:i0+n*d:d] = tensors._np_rand((n,),**settings)
    elif self.group.indicate(kd) < 0 and self._irrep in V:
      # Need to charge-conjugate
      # In this case self._irrep should be "primary"
      # S^H*v will transform correctly
      i0 = V.idx_of(self._irrep)
      n = V.idx_of(self._irrep)
      S = self.group.S(kd)
      v0 = np.kron(tensors._np_rand((n,),**settings),S[0,:].conj())
      M._T[i0:i0+n*d] = v0
    else:
      raise ValueError('Tensor specified does not have %s sector'%kd)
    self._T = M._do_unfuse({0:(self._idxs,f[0])})._T

  def _svd_fuse(self, left, right):
    """Fuse in implicit index"""
    fpl = self.group.Substantiation(left, self, irrep=self._irrep)
    fpr = self.group.Fusion([~self._dspace[lr] for lr in right], right).conj()
    M, info = self._do_fuse((0,left), (1,right), prims={0:fpl,1:fpr})
    T0 = M._T.flatten()
    subst = fpl.substantiating_fusion_matrix(fpr)
    T1 = subst.dot(T0)
    return self.group.InvariantTensor(T1.reshape((-1,M.shape[1])), (0,1),
      (fpr._out_subst, fpr.V))

  def symmetrized(self):
    kd = self.group.dual(self._irrep)
    d = self.group.dim(self._irrep)
    if self.group.indicate(kd) == -2:
      fp = self.group.Fusion([~v for v in self._spaces], self._idxs).conj()
      M,f = self._do_fuse((0,self._idxs), prims={0:fp})
    else:
      M,f = self._do_fuse((0,self._idxs))
    Msym = M.zeros_like()
    V = f[0].V
    if kd in V:
      i0 = V.idx_of(kd)
      n = V.degen_of(kd)
      Msym._T[i0:i0+n*d:d] = M._T[i0:i0+n*d:d]
    return Msym._do_unfuse({0:(self._idxs,f[0])})

  def conjugate(self):
    T = Tensor.conjugate(self)
    T._irrep = self.group.dual(self._irrep)
    return T


class InvariantTransposedView(InvariantTensor, tensors.TensorTransposedView,
    metaclass=GroupDerivedType):
  _regname='TransposedView'
  _required_types=(InvariantTensor,)
  pass
