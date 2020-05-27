import numpy as np
from scipy import sparse,linalg
import scipy.sparse.linalg
from .tensors import Tensor

from . import links, config, networks
from copy import copy,deepcopy
from abc import ABC, abstractmethod
from numbers import Number

class TensorOperator(sparse.linalg.LinearOperator, ABC):
  """Linear operator that takes in & outputs a Tensor
  Functional subclass is NetworkOperator; otherwise processes sums, products,
  etc."""

  def _derive_in(self, Oin):
    """Identifying features of domain based on operator with same domain"""
    self._fuse_in = Oin._fuse_in
    self._idx_in = Oin._idx_in
    self._Vin = Oin._Vin
    self._tensor_constructor = Oin._tensor_constructor

  def _derive_out(self, Oout):
    """Identifying features of range based on operator with same range"""
    self._fuse_out = Oout._fuse_out
    self._idx_out = Oout._idx_out
    self._Vout = Oout._Vout

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
    if not self.has_adjoint:
      raise NotImplementedError()
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
    if len(v.shape) == 2:
      v = v.flatten()
    T0 = self._tensor_constructor(v, (0,), (self._Vin,))
    return T0._do_unfuse({0:(self._idx_in,False)+self._fuse_in[1:]})

  def vector_out(self, T):
    """Convert output Tensor to vector"""
    return T._do_fuse((0,self._idx_out,self._fuse_out,False))[0]._T

  def tensor_adjoint_in(self, v):
    """Convert input vector to Tensor"""
    if len(v.shape) == 2:
      v = v.flatten()
    T0 = self._tensor_constructor(v, (0,), (self._Vout))
    return T0._do_unfuse({0:(self._idx_out,False)+self._fuse_out[1:]})

  def vector_adjoint_out(self, T):
    """Convert output Tensor to vector"""
    return T._do_fuse((0,self._idx_in,self._fuse_in,False))._T

  @property
  def dtype(self):
    return np.dtype(config.FIELD)

  @property
  def shape(self):
    return (self._Vout.dim, self._Vin.dim)

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

  def eigs(self, k, herm=True, vecs=True, nv=None, which='LM', maxiter=None,
      guess=None):
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
    if isinstance(guess, Tensor):
      guess = self.vector_out(guess)
    elif guess:
      raise TypeError('guess, if provided, must be Tensor')
    self.iters = 0
    if herm:
      rv = sparse.linalg.eigsh(self, k, ncv=nv, return_eigenvectors=vecs,
        which=which,v0=guess,maxiter=maxiter)
    else:
      rv = sparse.linalg.eigs(self, k, ncv=nv, return_eigenvectors=vecs,
        which=which,v0=guess,maxiter=maxiter)
    if not vecs:
      return rv
    w, vs = rv
    nn = len(w)
    Ts = []
    for i in range(nn):
      v = vs[:,i]
      Ts.append(self.tensor_in(v/linalg.norm(v)))
    return w, Ts


class NetworkOperator(TensorOperator):
  """Abstract linear-operator wrapping for Network
  Provide original network, name of tensor representing the vector being acted
    on, optionally order of contraction for adjoint, boolean variable
    determining whether or not to treat the operator as an endomorphism"""
  
  def __init__(self, net, t, adjoint_order=None, endomorphism=True):
    self.network = copy(net)
    self._t0 = t
    T0 = self.network[t]
    self._tensor_constructor = T0.__class__
    self._idx_in = T0._idxs
    self._fuse_in = T0._get_fuse_info((0,T0._idxs))[0]
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
      self._idx_out = self._idx_in
    else:
      vs = []
      idxs = []
      for t1,ll in self.network.freeindices():
        l = outdict[t1,ll]
        idxs.append(l)
        vs.append(self.network[t1]._dspace[ll])
      T1 = Tensor(None, idxs, tuple(vs))
      self._fuse_out = T1._get_fuse_info((0,idxs))[0]
      self._idx_out = tuple(idxs)
    self.is_endomorphic = bool(endomorphism)
    self._Vin = self._fuse_in[0]
    self._Vout = self._fuse_out[0]
    #if adjoint_order is not None:
    # Create network representing transposed action
    self.has_adjoint = True
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
    #else:
      #self.has_adjoint = False
      #self.transposed_net = None

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


class AdjointOperator(TensorOperator):
  def __init__(self, base):
    if not base.has_adjoint:
      raise NotImplementedError()
    self._base_operator = base
    self._fuse_in = base._fuse_out
    self._idx_in = base._idx_out
    self._Vin = base._Vout
    self._tensor_constructor = base._tensor_constructor
    self._fuse_out = base._fuse_in
    self._idx_out = base._idx_in
    self._Vout = base._Vin
    self.has_adjoint = True
    self.is_endomorphic = base.is_endomorphic

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


class TransposedOperator(TensorOperator):
  def __init__(self, base):
    if not base.has_adjoint:
      raise NotImplementedError()
    self._base_operator = base
    self._idx_in = base._idx_out
    self._Vin = ~base._Vout
    self._idx_out = base._idx_in
    self._Vout = ~base._Vin
    self._fuse_in = (self._Vin,(~v for v in base._fuse_out))
    self._tensor_constructor = base._tensor_constructor
    self._fuse_out = (self._Vout,(~v for v in base._fuse_in))
    self._net = base.transposed_net
    self.has_adjoint = True
    self.is_endomorphic = base.is_endomorphic

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


class ConjugatedOperator(TensorOperator):
  def __init__(self, base):
    self._base_operator = base
    self._idx_in = base._idx_in
    self._Vin = ~base._Vin
    self._idx_out = base._idx_out
    self._Vout = ~base._Vout
    self._fuse_in = (self._Vin,(~v for v in base._fuse_in))
    self._tensor_constructor = base._tensor_constructor
    self._fuse_out = (self._Vout,(~v for v in base._fuse_out))
    self.has_adjoint = base.has_adjoint
    self.is_endomorphic = base.is_endomorphic

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


class ScaledOperator(TensorOperator):
  def __init__(self, base, factor):
    self._base_operator = base
    self._coefficient = factor
    self.has_adjoint = base.has_adjoint
    self._derive_in(base)
    self._derive_out(base)
    self.is_endomorphic = base.is_endomorphic

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


class SumOperator(TensorOperator):
  def __init__(self, base1, base2):
    self._base_op1 = base1
    self._base_op2 = base2
    self.has_adjoint = base1.has_adjoint and base2.has_adjoint
    self.is_endomorphic = base1.is_endomorphic or base2.is_endomorphic
    self._derive_in(base1)
    self._derive_out(base1)

  def tensor_action(self, T):
    return self._base_op1.tensor_action(T) + self._base_op2.tensor_action(T)
  
  def adjoint_tensor_action(self, T):
    return self._base_op1.adjoint_tensor_action(T) + \
      self._base_op2.adjoint_tensor_action(T)

  def scalar_times(self, factor):
    return SumOperator(self._base_op1.scalar_times(factor),
        self._base_op2.scalar_times(factor))

  def transpose(self):
    return SumOperator(self._base_op1.transpose(),
        self._base_op2.transpose())

  def adjoint(self):
    return SumOperator(self._base_op1.adjoint(),
        self._base_op2.adjoint())

  def conj(self):
    return SumOperator(self._base_op1.conj(), self._base_op2.conj())


class ProductOperator(TensorOperator):
  def __init__(self, base1, base2):
    if isinstance(base2, ScaledOperator):
      # Shift coefficient to the 1st factor
      base1 = base1.scalar_times(base2._coefficient)
      base2 = base2._base_operator
    self._base_op1 = base1
    self._base_op2 = base2
    self.has_adjoint = base1.has_adjoint and base2.has_adjoint
    self._derive_in(base2)
    self._derive_out(base1)
    self.is_endomorphic = (base1.is_endomorphic and base2.is_endomorphic) \
      or (base1._fuse_in == base2._fuse_out)

  def tensor_action(self, T):
    return self._base_op1.tensor_action(self._base_op2.tensor_action(T))
  
  def adjoint_tensor_action(self, T):
    T1 = self._base_op1.adjoint_tensor_action(T)
    return self._base_op2.adjoint_tensor_action(T1)

  def scalar_times(self, factor):
    return ProductOperator(self._base_op1.scalar_times(factor), self._base_op2)

  def transpose(self):
    return ProductOperator(self._base_op2.transpose(),
        self._base_op1.transpose())

  def adjoint(self):
    return ProductOperator(self._base_op2.adjoint(), self._base_op1.adjoint())

  def conj(self):
    return ProductOperator(self._base_op1.conj(), self._base_op2.conj())


class IdentityOperator(TensorOperator):
  def __init__(self, base):
    if isinstance(base, TensorOperator):
      if not base.is_endomorphic:
        raise ValueError('Can only derive identity operator from endomorphic'
          ' operator')
      self.has_adjoint = True
      self._derive_in(base)
      self._derive_out(base)
    elif isinstance(base, tuple):
      self._idx_in, self._fuse_in, self._tensor_constructor = base
      self._idx_out = self._idx_in
      self._fuse_out = self._fuse_in
      self._Vin = self._Vout = self._fuse_in[0]
    else:
      raise NotImplementedError()
    self.is_endomorphic = True

  def tensor_action(self, T):
    return T

  def adjoint_tensor_action(self, T):
    return T

  def adjoint(self):
    return self

  def transpose(self):
    return IdentityOperator(self._idx_in, (~self._Vin,(~v for v in self._fuse_in[1])),self._tensor_constructor)

  def conj(self):
    return self.transpose()
