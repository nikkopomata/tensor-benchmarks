import numpy as np
import re
from scipy import sparse,linalg
import scipy.sparse.linalg
from .tensors import Tensor,FusionPrimitive

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
    if config.lin_iter_verbose:
      print('Iteration #%d'%self.iters,flush=config.flush)
    self.iters += 1
    T0 = self.tensor_in(v)
    T1 = self.tensor_action(T0)
    return self.vector_out(T1)

  def _adjoint(self, v):
    if config.lin_iter_verbose:
      print('Iteration (adjoint) #%d'%self.iters,flush=config.flush)
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
      k = self._fuse_out.effective_dim-2
      assert k > 0
    if isinstance(guess, Tensor):
      self._fuse_in.matchtensor(guess)
      guess = self.vector_out(guess)
    elif guess:
      raise TypeError('guess, if provided, must be Tensor')
    self.iters = 0
    if herm:
      rv = sparse.linalg.eigsh(self, k, ncv=nv, return_eigenvectors=vecs,
        v0=guess,**eigs_kw)
    else:
      rv = sparse.linalg.eigs(self, k, ncv=nv, return_eigenvectors=vecs,
        v0=guess,**eigs_kw)
    if not vecs:
      return rv
    w, vs = rv
    nn = len(w)
    if mat:
      T = self._fuse_out.tensor_convert_multiaxis(vs,('c',))
      return w, T
    else:
      Ts = []
      for i in range(nn):
        v = vs[:,i]
        Ts.append(self.tensor_in(v/linalg.norm(v)))
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

  def charge(self, irrep):
    """Convert input & output to charged tensors transforming under irrep"""
    fin = self._fuse_in.substantiating(irrep)
    if self.is_endomorphic:
      fout = fin
    else:
      fout = self._fuse_out.substantiating(irrep)
    self._fuse_in,self._fuse_out = fin,fout

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
        t = args[-1]
        self.network = networks.Network.network(net,*args[:-1])
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
