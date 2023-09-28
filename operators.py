import numpy as np
import re
from scipy import sparse,linalg
import scipy.sparse.linalg
from .tensors import Tensor,FusionPrimitive,_eig_vec_process

from . import links, config, networks
from copy import copy,deepcopy
from abc import ABC, abstractmethod
from numbers import Number
from scipy.sparse.linalg import ArpackError

class TensorOperator(sparse.linalg.LinearOperator, ABC):
  """Linear operator that takes in & outputs a Tensor
  Functional subclass is NetworkOperator; otherwise processes sums, products,
  etc."""

  @property
  def is_endomorphic(self):
    return self._fuse_in is self._fuse_out

  @property
  def idx_in(self):
    return self._fuse_in._idxs

  @property
  def Vin(self):
    return self._fuse_in._out

  @property
  def Vout(self):
    return self._fuse_out._out

  @property
  def idx_out(self):
    return self._fuse_out._idxs

  def _matvec(self, v):
    config.linalg_log.log(15, 'Iteration #%d', self.iters)
    self.iters += 1
    T0 = self.tensor_in(v)
    T1 = self.tensor_action(T0)
    return self.vector_out(T1)

  def _adjoint(self, v):
    config.linalg_log.log(15, 'Iteration (adjoint) #%d', self.iters)
    self.iters += 1
    T0 = self.tensor_adjoint_in(v)
    T1 = self.adjoint_tensor_action(T0)
    return self.vector_adjoint_out(T1)

  @abstractmethod
  def tensor_action(self, T):
    pass

  @abstractmethod
  def adjoint_tensor_action(self, T):
    pass

  def tensor_in(self, v):
    """Convert input vector to Tensor"""
    return self._fuse_in.tensor_convert(v)

  def vector_out(self, T):
    """Convert output Tensor to vector"""
    return self._fuse_out.vector_convert(T)

  def tensor_adjoint_in(self, v):
    """Convert input vector to Tensor"""
    return self._fuse_out.tensor_convert(v)

  def vector_adjoint_out(self, T):
    """Convert output Tensor to vector"""
    return self._fuse_in.vector_convert(T)

  def connect_with(self, O2, which='both', switch=False, c=False):
    """Change fuse_in or fuse_out to match O2
    which (in, out, or both) identifies which to change
    if switch, identify with opposite
    if c, identify with conjugate
    O2 may optionally be a FusionPrimitive instead"""
    # TODO is there actually a use case for this?
    if which == 'both':
      matchin,matchout = True,True
    elif which == 'in':
      matchin,matchout = True,False
    else:
      assert which == 'out'
      matchin,matchout = False,True
    if not isinstance(O2,TensorOperator):
      assert isinstance(O2, FusionPrimitive)
      assert which != 'both'
      if matchin:
        fin = O2
      else:
        fout = O2
    else:
      fin,fout = self._fuse_in,self._fuse_out
      if switch:
        fin,fout = fout,fin
    if matchin:
      assert self._fuse_in.cmp(fin, c=c)
      if c:
        fin = fin.conj()
      self._fuse_in = fin
    if matchout:
      assert self._fuse_out.cmp(fout, c=c)
      if c:
        fout = fout.conj()
      self._fuse_out = fout

  def toendomorphism(self):
    """Change fuse_out to match fuse_out"""
    if not self._fuse_in.cmp(self._fuse_out):
      raise ValueError('Input & output indices not compatible')
    self._fuse_out = self._fuse_in

  @property
  def dtype(self):
    return np.dtype(config.FIELD)

  @property
  def shape(self):
    return (self._fuse_out.effective_dim,self._fuse_in.effective_dim)

  def scalar_times(self, factor):
    return ScaledOperator(self, factor)

  def __add__(self, O2):
    if isinstance(O2,Number):
      O2 = O2*IdentityOperator(self)
    elif not isinstance(O2, TensorOperator):
      return NotImplemented
    return SumOperator(self, O2)

  def __sub__(self, O2):
    if isinstance(O2,Number):
      return SumOperator(self, -O2*IdentityOperator(self))
    elif not isinstance(O2, TensorOperator):
      return NotImplemented
    return SumOperator(self, O2.scalar_times(-1))

  def __radd__(self, O2):
    if isinstance(O2,Number):
      return SumOperator(self, O2*IdentityOperator(self))
    else:
      return NotImplemented

  def __rsub__(self, O2):
    if isinstance(O2,Number):
      return SumOperator(O2*IdentityOperator(self),self.scalar_times(-1))
    else:
      return NotImplemented

  def __mul__(self, factor):
    if isinstance(factor, Number):
      return self.scalar_times(factor)
    else:
      return NotImplemented

  def __rmul__(self, factor):
    if isinstance(factor, Number):
      return self.scalar_times(factor)
    else:
      return NotImplemented

  def __div__(self, denom):
    if isinstance(denom, Number):
      return self.scalar_times(1/denom)
    else:
      return NotImplemented

  def __truediv__(self, denom):
    return self.__div__(denom)

  def __neg__(self):
    return self.scalar_times(-1)

  def dot(self, O2):
    return ProductOperator(self, O2)

  def __matmul__(self, O2):
    return self.dot(O2)

  def __pow__(self, n):
    if not isinstance(n, int):
      return NotImplemented
    if n < 0:
      raise ValueError('Cannot raise to negative power')
    if n == 0:
      return IdentityOperator(self)
    if n == 1:
      return self
    return ProductOperator(self, self.__pow__(n-1))
  
  def id_from(self):
    return IdentityOperator(self)

  @abstractmethod
  def transpose(self):
    pass

  @property
  def T(self):
    return self.transpose()

  @abstractmethod
  def conj(self):
    pass

  @abstractmethod
  def adjoint(self):
    pass

  @property
  def H(self):
    return self.adjoint()

  def project(self, projector, apply_in=True, apply_out=True, zero=False):
    """Apply projector to input and/or output indices
    Pass 'projector' as isometric matrix acting on effective index
    If zero, use ZeroingFusion instead of ProjectionFusion
      (assume provided as complement instead of original)"""
    #TODO alternative inputs
    #TODO processing through fusion ensures disagreement when directly called
    # with tensor_action 
    projcls = ZeroingFusion if zero else ProjectionFusion
    if apply_in and apply_out and self.is_endomorphic:
      self._fuse_in = projcls(self._fuse_in, projector)
      self._fuse_out = self._fuse_in
    else:
      if apply_in:
        self._fuse_in = projcls(self._fuse_in, projector)
      if apply_out:
        self._fuse_out = projcls(self._fuse_out, projector)

  def project_out(self, vectors, apply_in=True, apply_out=True, tol=1e-14, tol_absolute=False, zero=False):
    """Project out vectors (provided as a list of rank-n tensors or single
    rank-n+1 tensor, for n rank of input space--must use same index names)
    by calculating projector onto orthogonal completement & applying as in
    project()
    if tol is 0 or None project out linear combinations of any amplitude
    (effectively, do full-rank unless exact duplicate is provided); otherwise
    will be the minimum amplitude"""
    if apply_in and apply_out and not self.is_endomorphic:
      # Apply separately
      self.project_out(vectors, apply_in=False, apply_out=True, tol=tol, zero=zero)
      apply_out = False
    assert apply_in or apply_out
    fuse = self._fuse_in if apply_in else self._fuse_out
    if isinstance(vectors,Tensor):
      # TODO more efficient, extensible way of converting
      assert vectors.rank == fuse.rank+1
      assert set(fuse._idxs).issubset(vectors.idxset)
      iextra = (vectors.idxset - set(fuse._idxs)).pop()
      dextra = vectors.shape[iextra]
      T = vectors.permuted((iextra,)+tuple(fuse._idxs))
      vectors = [vectors.init_fromT(T[n], ','.join(f'{idx}|0.{idx}' for idx in fuse._idxs)) for n in range(dextra)]
    mat = np.array([fuse.vector_convert(v,check=True) for v in vectors])
    config.linalg_log.debug('Projecting out %d vectors in dimension-%d space',*mat.shape)
    if zero:
      L,s,R = linalg.svd(mat,full_matrices=False)
    else:
      L,s,R = linalg.svd(mat,full_matrices=True)
    if not tol:
      bi = s==0
    else:
      maxval = max(abs(v) for v in vectors)
      if tol_absolute:
        cutoff = tol
      else:
        cutoff = maxval*tol
      bi = s < cutoff
    if zero:
      # Interested in basis for vectors provided - opposite of alternative
      bi = ~bi
    else:
      bi = np.concatenate([bi, np.ones(R.shape[0]-len(s),dtype=bool)])
    w = R[bi,:].conj()
    self.project(w, apply_in=apply_in,apply_out=apply_out,zero=zero)

  def eigs(self, k, herm=True, vecs=True, nv=None, guess=None, mat=False,
      **eigs_kw):
    """Eigendecomposition of linear transformation obtained from network,
    using ARPACK methods
    k is the number of eigenvalues to collect
    vecs indicates whether or not to return eigenvectors
    nv indicates number of Lanczos/Arnoldi vectors (supplied to eigs/eigsh)
    which is the eigenvalue specification per eigs
    guess is an optional tensor
      (as an initial guess)"""
    if not self.is_endomorphic:
      raise ValueError('Can only find eigenvalues of endomorphism')
    if k >= self._fuse_out.effective_dim-1:
      # Vector space is small; construct explicitly
      try:
        M = self.compute_dense_matrix()
      except:
        if self._fuse_out.effective_dim == 0:
          raise ValueError('Linear operator is 0-dimensional')
        else:
          raise
      if vecs:
        rv = _eig_vec_process(M, herm, False, None, False, None)
        rv = rv[:2]
      elif herm:
        return linalg.eigvalsh(M)
      else:
        return linalg.eigvals(M)
    else:
      # Do the actual sparse computation
      config.linalg_log.log(18, 'Diagonalizing operator with effective dimension %d',self._fuse_out.effective_dim)
      if isinstance(guess, Tensor):
        self._fuse_in.matchtensor(guess)
        guess = self.vector_out(guess)
      elif guess:
        raise TypeError('guess, if provided, must be Tensor')
      self.iters = 0
      if herm:
        try:
          rv = sparse.linalg.eigsh(self, k, ncv=nv, return_eigenvectors=vecs,
            v0=guess,**eigs_kw)
        except sparse.linalg.ArpackError as e:
          if 'Starting vector is zero' in str(e):
            config.linalg_log.error('Guess vector determined as zero '
              '(norm %0.4g); retrying without',linalg.norm(guess))
            rv = sparse.linalg.eigsh(self, k, ncv=nv, return_eigenvectors=vecs,
              **eigs_kw)
          else:
            raise
        config.linalg_log.log(13,'Lanczos diagonalization complete')
      else:
        try:
          rv = sparse.linalg.eigs(self, k, ncv=nv, return_eigenvectors=vecs,
            v0=guess,**eigs_kw)
        except sparse.linalg.ArpackError as e:
          if 'Starting vector is zero' in str(e):
            config.linalg_log.error('Guess vector determined as zero '
              '(norm %0.4g); retrying without',linalg.norm(guess))
            rv = sparse.linalg.eigs(self, k, ncv=nv, return_eigenvectors=vecs,
              **eigs_kw)
          else:
            raise
        config.linalg_log.log(13,'Arnoldi diagonalization complete')
      if 'tol' in eigs_kw and eigs_kw['tol'] and k>1:
        # Check difference in eigenvalues is outside tolerance
        w = rv[0] if vecs else rv
        if abs(w[0] - w[1]) < abs(w[0])*eigs_kw['tol']:
          config.linalg_log.warn('Eigenvalue difference outside tolerance: %0.2g versus %0.2g', abs(1-w[1]/w[0]), eigs_kw['tol'])
          # Fall back to explicit
          M = self.compute_dense_matrix()
          if vecs:
            rv = _eig_vec_process(M, herm, False, None, False, None)
            rv = rv[:2]
          elif herm:
            return linalg.eigvalsh(M)
          else:
            return linalg.eigvals(M)
      if not vecs:
        return rv
    w, vs = rv
    nn = len(w)
    if mat:
      T = self._fuse_out.tensor_convert_multiaxis(vs,('c',))
      config.linalg_log.log(10, 'Eigenvectors converted to matrix form')
      return w, T
    else:
      Ts = []
      for i in range(nn):
        v = vs[:,i]
        Ts.append(self.tensor_in(v/linalg.norm(v)))
      config.linalg_log.log(10, 'Eigenvectors packaged in individual tensor form')
      return w, Ts

  def asdense(self, inprefix='', outprefix='', naive=False):
    # TODO check asdense cases
    assert naive
    shape_in = self._fuse_in.shape
    shape_out = self._fuse_out.shape
    din = self.Vin.dim
    T = None
    rn_in = {l:inprefix+l for l in self._fuse_in._idxs}
    rn_out = {l:outprefix+l for l in self._fuse_out._idxs}
    for n in range(din):
      v = np.zeros((din,))
      v[n] = 1
      v = v.reshape(shape_in)
      vin = self._fuse_in.empty_like()
      vin._T = v
      vout = self.tensor_action(vin)
      T1 = vout.renamed(rn_out).contract(vin.renamed(rn_in),'~*')
      if T is None:
        T = T1
      else:
        T += T1
    return T

  def charge(self, irrep, flip_quaternionic=False):
    """Convert input & output to charged tensors transforming under irrep
    If flip_quaternionic and irrep is a quaternionic rep whose dual is found in
      either fused indices, will use dual instead for that case"""
    G = self._fuse_in.group
    if irrep not in self._fuse_in._out:
      if flip_quaternionic and G.indicate(irrep)<0 and G.dual(irrep) in self._fuse_in._out:
        irrep = G.dual(irrep)
      else:
        raise ValueError(f'input cannot transform under {irrep}')
    fin = self._fuse_in.substantiating(irrep)
    if self.is_endomorphic:
      fout = fin
    else:
      if irrep not in self._fuse_out._out:
        if flip_quaternionic and G.indicate(irrep)<0 and G.dual(irrep) in self._fuse_out._out:
          irrep = G.dual(irrep)
        else:
          raise ValueError(f'output cannot transform under {irrep}')
      fout = self._fuse_out.substantiating(irrep)
    self._fuse_in,self._fuse_out = fin,fout

  def compute_dense_matrix(self):
    self.iters = 0
    return self.matmat(np.identity(self._fuse_in.effective_dim,dtype=config.FIELD))


class NetworkOperator(TensorOperator):
  """Abstract linear-operator wrapping for Network
  Provide original network, name of tensor representing the vector being acted
    on, optionally order of contraction for adjoint, boolean variable
    determining whether or not to treat the operator as an endomorphism
  Alternative:
    (1) valid network constructor string, network constructor arguments, then
      name of tensor representing vector acted upon, or
    (2) network constructor string as above, but with name of tensor
      representing vector in parentheses, no concrete vector provided
    in these cases may provide order"""
  
  def __init__(self, net, t, *args, adjoint_order=None, endomorphism=True, order=None):
    if isinstance(net,str):
      args = (t,)+args
      # Network constructor argument
      m = re.search(r'(?:^|;)\((\w+)\);', net)
      if m:
        # vector not explicitly provided
        t = m.group(1)
        net = re.sub(r'\((\w+)\)',r'\1',net)
        # Find vector spaces as appropriate
        didx = {}
        # Partially parse string "net"
        clauses = net.split(';')
        # args represent tensors provided to network(), *minus* t0
        if len(clauses) != len(args)+3:
          # Autocompleted output
          if len(clauses) != len(args)+2 or net[-1] != '~':
            raise ValueError('Network creation string \'%s\' has incorrect '
              'number of subclauses (expected %d, got %d)' \
              %(net,len(args)+3,len(clauses)))
          clauses[-1] = clauses[-1][:-1]
          clauses.append('~')
        tensors = {}
        argidx = 0
        for clause in clauses[:-2]:
          if clause == t:
            tidx = argidx
            continue
          ts = clause.split(',')
          for t1 in ts:
            if t1[-1] == '*':
              tensors[t1[:-1]] = (args[argidx],True)
            else:
              tensors[t1] = (args[argidx],False)
          argidx += 1
        for t0,l0,t1,l1 in re.findall(r'(\w+)\.(\w+)-(\w+)\.(\w+)',clauses[-2]):
          if t0 == t:
            if t1 not in tensors:
              raise KeyError(f'Cannot find contracted tensor {t1} in ({t0}).{l0}-{t1}.{l1}')
            T,c = tensors[t1]
            if l1 not in T:
              raise KeyError(f'Cannot find contracted index {t1}.{l1} in ({t0}).{l0}-{t1}.{l1}')
            V = T._dspace[l1]
            if c:
              didx[l0] = V
            else:
              didx[l0] = ~V
          elif t1 == t:
            if t0 not in tensors:
              raise KeyError(f'Cannot find contracted tensor {t0} in {t0}.{l0}-({t1}).{l1}')
            T,c = tensors[t0]
            if l0 not in T:
              raise KeyError(f'Cannot find contracted index {t0}.{l0} in {t0}.{l0}-({t1}).{l1}')
            V = T._dspace[l0]
            if c:
              didx[l1] = V
            else:
              didx[l1] = ~V
        self._fuse_in = args[0].buildFusion(didx.values(),idx_in=tuple(didx.keys()))
        T0 = self._fuse_in.empty_like()
        targs = list(args)
        targs.insert(tidx, T0)
        self.network = networks.Network.network(net, *targs)
      else:
        # TODO was it being used the other way around anywhere?
        t = args[0]
        self.network = networks.Network.network(net,*args[1:])
        T0 = self.network[t]
        self._fuse_in = T0.getFusion(T0._idxs)
      if order is not None:
        self.network.setorder(order)
    else:
      self.network = copy(net)
      T0 = self.network[t]
      self._fuse_in = T0.getFusion(T0._idxs)
    self._t0 = t
    try:
      idx = next(self.network.freeindices(unsetonly=True))
    except:
      pass
    else:
      raise ValueError('Output index %s.%s not named'%idx)
    outdict = self.network._out
    if endomorphism:
      for t1,ll in self.network.freeindices():
        l = outdict[t1,ll]
        if l not in T0._idxs or not self.network[t1]._dspace[ll]==T0._dspace[l]:
          raise ValueError('Index %s.%s>%s does not match'%(t1,ll,l))
      self._fuse_out = self._fuse_in
    else:
      vs = []
      idxs = []
      for t1,ll in self.network.freeindices():
        l = outdict[t1,ll]
        idxs.append(l)
        vs.append(self.network[t1]._dspace[ll])
      self._fuse_out = args[0].buildFusion(vs, idx_in=idxs)
      T1 = self._fuse_out.empty_like()
    # Create network representing transposed action
    if endomorphism:
      T1 = T0
    bondstrs = []
    outstrs = []
    for t1,ll in self.network.freeindices():
      l1 = outdict[t1,ll]
      if t1 == t:
        # Uncontracted index of tensor acted upon - becomes same in transpose
        outstrs.append('%s.%s>%s'%(t,l1,ll))
      else:
        # Contracted index of tensor acted on by transpose
        bondstrs.append('%s.%s-%s.%s'%(t1,ll,t,l1))
    bdict = self.network._tbonds[t]
    for l0 in bdict:
      # Output index of transposed network
      outstrs.append('%s.%s>%s'%(bdict[l0]+(l0,)))
    self.transposed_net = self.network.derive('-%s;+%s*;%s;%s'%(t,t,
        ','.join(bondstrs),','.join(outstrs)), T1)
    if adjoint_order is not None:
      self.transposed_net.setorder(adjoint_order)
    else:
      self.transposed_net.optimize()
    self.adjoint_net = self.transposed_net.conj()

  def tensor_action(self, T):
    self.network[self._t0] = T
    return self.network.contract()

  def adjoint_tensor_action(self, T):
    self.adjoint_net.replace(self._t0, T)
    return self.adjoint_net.contract()

  def transpose(self):
    return TransposedOperator(self)

  def conj(self):
    return ConjugatedOperator(self)

  def adjoint(self):
    return AdjointOperator(self)

  def asdense(self, inprefix='', outprefix='', naive=False, order=None):
    if naive:
      return TensorOperator.asdense(self, inprefix, outprefix, naive)
    out = {(t,ll):outprefix+lo for (t,ll),lo in self.network._out.items()}
    t0bonds = self.network._tbonds[self._t0]
    out.update({tl:inprefix+ll for ll,tl in t0bonds.items()})
    net = self.network.subnetwork([self._t0], out)
    if order is None:
      net.optimize()
    else:
      net.setorder(order)
    return net.contract()
    

class AdjointOperator(TensorOperator):
  def __init__(self, base):
    self._base_operator = base
    self._fuse_in = base._fuse_out
    self._fuse_out = base._fuse_in

  def tensor_action(self, T):
    return self._base_operator.adjoint_tensor_action(T)

  def adjoint_tensor_action(self, T):
    return self._base_operator.tensor_action(T)

  def transpose(self):
    return self._base_operator.conj()

  def conj(self):
    return self._base_operator.transpose()

  def adjoint(self):
    return self._base_operator

  def asdense(self, inprefix='', outprefix='', **kwargs):
    return self._base_operator.asdense(inprefix=outprefix, outprefix=inprefix, **kwargs).conj()


class TransposedOperator(TensorOperator):
  def __init__(self, base):
    self._base_operator = base
    self._fuse_in = base._fuse_out.conj()
    if base.is_endomorphic:
      self._fuse_out = self._fuse_in
    else:
      self._fuse_out = base._fuse_in.conj()
    self._net = base.transposed_net

  def tensor_action(self, T):
    self._net.replace(self._base_operator._t0, T, c=False)
    return self._net.contract()

  def adjoint_tensor_action(self, T):
    return self._base_operator.tensor_action(T.conj()).conj()

  def transpose(self):
    return self._base_operator

  def adjoint(self):
    return self._base_operator.conj()

  def conj(self):
    return self._base_operator.adjoint()

  def asdense(self, inprefix='', outprefix='', **kwargs):
    return self._base_operator.asdense(inprefix=outprefix, outprefix=inprefix, **kwargs)


class ConjugatedOperator(TensorOperator):
  def __init__(self, base):
    self._base_operator = base
    self._fuse_in = base._fuse_in.conj()
    if base.is_endomorphic:
      self._fuse_out = self._fuse_in
    else:
      self._fuse_out = base._fuse_out.conj()

  def tensor_action(self, T):
    return self._base_operator.tensor_action(T.conj()).conj()

  def adjoint_tensor_action(self, T):
    net = self._base_operator.transposed_net
    net.replace(self._base_operator._t0, T, c=False)
    return net.contract()

  def transpose(self):
    return self._base_operator.adjoint()

  def adjoint(self):
    return self._base_operator.transpose()

  def conj(self):
    return self._base_operator

  def asdense(self, **kwargs):
    return self._base_operator.asdense(**kwargs).conj()


class ScaledOperator(TensorOperator):
  def __init__(self, base, factor):
    self._base_operator = base
    self._coefficient = factor
    self._fuse_in = base._fuse_in
    self._fuse_out = base._fuse_out

  def tensor_action(self, T):
    return self._coefficient * self._base_operator.tensor_action(T)
  
  def adjoint_tensor_action(self, T):
    return np.conjugate(self._coefficient) * \
      self._base_operator.adjoint_tensor_action(T)

  def scalar_times(self, factor):
    return ScaledOperator(self._base_operator, factor*self._coefficient)

  def transpose(self):
    return ScaledOperator(self._base_operator.transpose(), self._coefficient)

  def adjoint(self):
    return ScaledOperator(self._base_operator.adjoint(),
      np.conjugate(self._coefficient))

  def conj(self):
    return ScaledOperator(self._base_operator.conj(),
      np.conjugate(self._coefficient))

  def asdense(self, **kwargs):
    return self._coefficient * self._base_operator.asdense(**kwargs)


class SumOperator(TensorOperator):
  def __init__(self, base1, base2, *additional_bases):
    if not base1._fuse_in.cmp(base2._fuse_in) or \
        not base1._fuse_out.cmp(base2._fuse_out):
      raise ValueError('Operators are not summable')
    self._fuse_in = base1._fuse_in
    self._fuse_out = base1._fuse_out
    # Needs to not recurse excessively
    if isinstance(base1,SumOperator):
      bo1 = base1._base_ops
    else:
      bo1 = (base1,)
    if isinstance(base2,SumOperator):
      bo2 = base2._base_ops
    else:
      bo2 = (base2,)
    for base in additional_bases:
      if not base1._fuse_in.cmp(base._fuse_in) or \
          not base1._fuse_out.cmp(base._fuse_out):
        raise ValueError('Operators are not summable')
      bo2 += (base,)
    self._base_ops = bo1+bo2
    # TODO check case of more than two summands

  def tensor_action(self, T):
    action = self._base_ops[0].tensor_action(T)
    for op in self._base_ops[1:]:
      action += op.tensor_action(T)
    return action
  
  def adjoint_tensor_action(self, T):
    action = self._base_ops[0].adjoint_tensor_action(T)
    for op in self._base_ops[1:]:
      action += op.adjoint_tensor_action(T)

  def scalar_times(self, factor):
    return SumOperator(*(op.scalar_times(factor) for op in self._base_ops))

  def transpose(self):
    return SumOperator(*(op.transpose() for op in self._base_ops))

  def adjoint(self):
    return SumOperator(*(op.adjoint() for op in self._base_ops))

  def conj(self):
    return SumOperator(*(op.conj() for op in self._base_ops))

  def asdense(self, **kwargs):
    T = self._base_ops[0].asdense(**kwargs)
    for op in self._base_ops[1:]:
      T += op.asdense(**kwargs)
    return T


class ProductOperator(TensorOperator):
  def __init__(self, left, right):
    if not left._fuse_in.cmp(right._fuse_out):
      raise ValueError('Operators cannot be composed')
    if isinstance(right, ScaledOperator):
      # Shift coefficient to the 1st factor
      left = left.scalar_times(right._coefficient)
      right = right._base_operator
    self._op_left = left
    self._op_right = right
    self._fuse_in = right._fuse_in
    if self._fuse_in.cmp(left._fuse_out):
      self._fuse_out = self._fuse_in
    else:
      self._fuse_out = left._fuse_out

  def tensor_action(self, T):
    return self._op_left.tensor_action(self._op_right.tensor_action(T))
  
  def adjoint_tensor_action(self, T):
    T1 = self._op_left.adjoint_tensor_action(T)
    return self._op_right.adjoint_tensor_action(T1)

  def scalar_times(self, factor):
    return ProductOperator(self._op_left.scalar_times(factor), self._op_right)

  def transpose(self):
    return ProductOperator(self._op_right.transpose(),
        self._op_left.transpose())

  def adjoint(self):
    return ProductOperator(self._op_right.adjoint(), self._op_left.adjoint())

  def conj(self):
    return ProductOperator(self._op_left.conj(), self._op_right.conj())

  def asdense(self, inprefix='', outprefix='', naive=False):
    if naive:
      return TensorOperator.asdense(self, inprefix=inprefix, outprefix=outprefix, naive=True)
    Tleft = self._op_left.asdense(inprefix=inprefix, outprefix=outprefix) 
    Tright = self._op_right.asdense(inprefix=inprefix, outprefix=outprefix) 
    contracted = [(inprefix+ll,outprefix+ll) for ll in self._op_left.idx_in]
    outl = (outprefix+ll for ll in self.idx_out)
    outl = {ol:ol for ol in outl}
    outr = (inprefix+ll for ll in self.idx_in)
    outr = {ir:ir for ir in outr}
    return Tleft._do_contract(Tright, contracted, outl, outr, False)


class IdentityOperator(TensorOperator):
  def __init__(self, base):
    if isinstance(base, TensorOperator):
      if not base.is_endomorphic:
        raise ValueError('Can only derive identity operator from endomorphic'
          ' operator')
      self._fuse_in = base._fuse_in
    elif isinstance(base, FusionPrimitive):
      self._fuse_in = base
    elif isinstance(base, Tensor):
      self._fuse_in = base.getFusion(base._idxs)
    else:
      raise NotImplementedError()
    self._fuse_out = self._fuse_in

  def tensor_action(self, T):
    return T

  def adjoint_tensor_action(self, T):
    return T

  def adjoint(self):
    return self

  def transpose(self):
    return IdentityOperator(self._fuse_in.conj())

  def conj(self):
    return self.transpose()

  def asdense(self, inprefix='', outprefix='', **kwargs):
    inidx = [inprefix+ll for ll in self.idx_in]
    outidx = [outprefix+ll for ll in self.idx_out]
    assert len(set(inidx) | set(outidx)) == 2*len(inidx)
    return self._fuse_in.tensorclass._identity(outidx, inidx,
      self._fuse_in._spaces_in)


class ProjectionFusion(FusionPrimitive):
  """Represent projector (in the sense of isometry) from "output"
  effective space of fusion to subspace"""
  def __init__(self, base, w):
    assert isinstance(base,FusionPrimitive)
    if isinstance(base, ProjectionFusion) and not isinstance(base,ZeroingFusion):
      # Compose projectors
      w = w.dot(base.isometry)
      base = base.__base
    assert isinstance(w,np.ndarray) or isinstance(w,sparse.linalg.LinearOperator)
    self.__base = base
    self.isometry = w
    assert w.shape[1] == base.effective_dim
    self._rank = self.__base._rank

  def conj(self):
    return self.__class__(self.__base.conj(), self.isometry.conj())

  @property
  def tensorclass(self):
    return self.__base.tensorclass

  @property
  def _spaces_in(self):
    return self.__base._spaces_in

  @property
  def _idxs(self):
    return self.__base._idxs

  @property
  def _out(self):
    return self.__base._out

  @property
  def dim(self):
    return self.__base.dim

  def vector_convert(self, T, idx=None, check=False):
    return self.isometry.dot(self.__base.vector_convert(T,idx=idx,check=check))

  def tensor_convert(self, v, idx=None):
    vx = self.isometry.T.conj().dot(v)
    return self.__base.tensor_convert(vx,idx=idx)

  def tensor_convert_multiaxis(self, T, idx_newax, ax_old=0, **kw_args):
    # TODO check operationality
    T1 = T.moveaxis(ax_old,0)
    T1 = self.isometry.T.conj().dot(T1)
    T1 = T1.moveaxis(0,ax_old)
    return self.__base.tensor_convert_multiaxis(T1,idx_newax,ax_old=ax_old,**kw_args)

  @property
  def effective_dim(self):
    return self.isometry.shape[0]


class ZeroingFusion(ProjectionFusion):
  """Instead of projecting out in the sense of O -> W O W^H,
  use O -> (1 - P) O (1 - P)
  where 1 - P = W^H W and P is given in the form V^H V
  """
  def __init__(self, base, isometry_complement):
    assert isinstance(base,FusionPrimitive)
    assert isinstance(isometry_complement,np.ndarray) or isinstance(isometry_complement,sparse.linalg.LinearOperator)
    self.__base = base
    self._ProjectionFusion__base = base
    # TODO compose -- requires SVD tolerance?
    self.iso_comp = isometry_complement
    self.iso_comp_H = isometry_complement.T.conj()
    assert isometry_complement.shape[1] == base.effective_dim
    self._rank = self.__base._rank

  def conj(self):
    return self.__class__(self.__base.conj(), self.iso_comp.conj())

  def vector_convert(self, T, idx=None, check=False):
    v0 = self.__base.vector_convert(T,idx=idx,check=check)
    return v0 - self.iso_comp_H.dot(self.iso_comp.dot(v0))

  def tensor_convert(self, v, idx=None):
    vx = v - self.iso_comp_H.dot(self.iso_comp.dot(v))
    return self.__base.tensor_convert(vx,idx=idx)

  def tensor_convert_multiaxis(self, T, idx_newax, ax_old=0, **kw_args):
    # TODO check operationality
    T1 = T.moveaxis(ax_old,0)
    T1 = T1 - self.iso_comp_H.dot(self.iso_comp.dot(T1))
    T1 = T1.moveaxis(0,ax_old)
    return self.__base.tensor_convert_multiaxis(T1,idx_newax,ax_old=ax_old,**kw_args)

  @property
  def effective_dim(self):
    return self.__base.effective_dim
