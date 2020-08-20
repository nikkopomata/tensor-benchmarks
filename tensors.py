#import numpy as np
#from scipy import linalg
#import numpy.random as rand
import os
import os.path
import re
import time
import functools
from copy import copy,deepcopy
from .npstack import np,RNG,linalg,safesvd,sparse

from . import links
links._tensorimported = True
from . import config

def _np_rand(shape, dist='normal'):
  """Get np.ndarray with appropriate features"""
  T = getattr(RNG, dist)(size=shape).astype(config.FIELD)
  if issubclass(config.FIELD,complex) or \
      issubclass(config.FIELD,np.complexfloating):
    T += 1j*getattr(RNG, dist)(size=shape)
  return T

def _findrep1(ll):
  """Find first repetition in list of strings ll"""
  if isinstance(ll,dict):
    ll = ll.values()
  ls = sorted(ll)
  for ii in range(len(ls)-1):
    if ls[ii] == ls[ii+1]:
      return ls[ii]
  raise ValueError('Repetiton expected, not found')

def _endomorphic(func):
  """Decorator that processes functions that treat a tensor as an endomorphism
  parsestr may be of the form l0-r0,l1-r1,l2-r2 or l0,l1,l2-r0,r1,r2
  Function will accept lists [l0,l1,l2], [r0,r1,r2]"""
  def endo_wrap(self, parsestr, *args, **kw_args):
    if re.fullmatch(r'(?!,)(,?\w+-\w+)*', parsestr):
      ridx = []
      lidx = []
      for ll,rl in re.findall('(\w+)-(\w+)', parsestr):
        ridx.append(rl)
        lidx.append(ll)
    elif re.fullmatch(r'(\w+,)*\w+-\w+(,\w+)*', parsestr):
      lhs,rhs = parsestr.split('-')
      lidx = lhs.split(',')
      ridx = rhs.split(',')
      if len(lidx) != len(ridx):
        raise ValueError('Number of LHS, RHS indices provided different')
    else:
      raise ValueError('Argument \'%s\' not recognized'%parsestr)
    lrset = set(lidx)|set(ridx)
    if lrset != set(self._idxs):
      if len(lrset) < self.rank:
        raise ValueError('Index %s not accounted for'%(set(self._idxs)-lrset).pop())
      else:
        raise ValueError('Index %s missing from tensor'%(lrset-set(self._idxs)).pop())
    for ii in range(len(lidx)):
      if not self._dspace[lidx[ii]] ^ self._dspace[ridx[ii]]:
        raise ValueError('LHS,RHS indices %s-%s of endomorphism '
          'not compatible'%(lidx[ii],ridx[ii]))
    return func(self, lidx, ridx, *args, **kw_args)

  endo_wrap.__doc__ = func.__doc__
  endo_wrap.__name__ = func.__name__
  endo_wrap.__module__ = func.__module__
  return endo_wrap

def _endo_transformation(func):
  """Wrapper for functions that operate within a class of endomorphisms
  Function in question should, for first argument M, replace M._T with
  appropriate square matrix"""
  def endo_wrap2(self, lidx, ridx, *args, **kw_args):
    M, info = self._do_fuse((0,lidx),(1,ridx,0,True))
    func(M, *args, **kw_args)
    return M._do_unfuse({0:(lidx,info[0]), 1:(ridx,info[1])})

  endo_wrap2.__doc__ = func.__doc__
  endo_wrap2.__name__ = func.__name__
  endo_wrap2.__module__ = func.__module__
  return _endomorphic(endo_wrap2)

class Tensor:
  def __init__(self, T, idxs, spaces):
    """Pass numpy.ndarray with index names & identifiers as sequence
    (note: duplicates idxs, not spaces)"""
    self._idxs = tuple(idxs)
    self._T = T
    self._spaces = spaces
    rank = len(self._spaces)
    self._dspace = {idxs[i]: spaces[i] for i in range(rank)}
    self.shape = dictproperty(self.__shape_get, None, self.__shape_has,
      self.__shape_copy)

  @classmethod
  def tensor(cls, t, idxs):
    """Initializer for tensors provided by the user
    Takes list or np.ndarray t, list (as list or comma-separated string) of
    index names
    If string, may include matching (:L) or dual-matching (:L*)"""
    matchdata = {}
    if isinstance(idxs,str):
      if not re.fullmatch(r'\w+(,\w+(\:\w+\*?)?)*',idxs):
        raise ValueError('Invalid (not alphanumeric + _) index name in %s'%idxs)
      idxs = idxs.split(',')
      for ii in range(len(idxs)):
        if not re.fullmatch(r'\w+',idxs[ii]):
          m = re.fullmatch(r'(\w+)\:(\w+)(\*?)',idxs[ii])
          idxs[ii], l1, c = m.groups()
          if l1 not in idxs[:ii]:
            raise ValueError('Index %s referred to before definition'%l1)
          matchdata[idxs[ii]] = (l1, bool(c))
    else:
      for ll in idxs:
        if not isinstance(ll,str) or not re.fullmatch(r'\w+',ll):
          raise ValueError('Invalid (non-alphanumeric) index name %s'%ll)
    try:
      T = np.array(t,dtype=config.FIELD,copy=True)
    except:
      raise ValueError('type provided cannot be converted to numpy.ndarray')
    shape = T.shape
    rank = len(shape)
    if rank != len(idxs):
      raise ValueError('Tensor rank does not match index set')
    spaces = []
    dspace = {}
    for ii in range(rank):
      d = shape[ii]
      l0 = idxs[ii]
      if l0 in matchdata:
        l1,c = matchdata[l0]
        if c:
          V = dspace[l1].dual()
        else:
          V = dspace[l1]
        if V.dim != d:
          raise ValueError('Indices %s,%s with different dimensions matched' \
            %(l0,l1))
      else:
        V = links.VSpace(d)
      spaces.append(V)
      dspace[l0] = V
    return cls(T, idxs, tuple(spaces))

  def _tensorfactory(self, T, idxs, spaces):
    """For Tensor operates as alias for initializer
    For subclasses may be used to retain characteristics of element"""
    return Tensor(T, idxs, spaces)

  def __shape_get(self, key):
    try:
      return self._dspace[key].dim
    except IndexError as e:
      if isinstance(key, int):
        return self._spaces[key].dim
      else:
        raise e

  def __shape_has(self, key):
    return key in self._idxs

  def __shape_copy(self):
    return tuple(vs.dim for vs in self._spaces)

  def __getstate__(self):
    return self._T, self._idxs, self._spaces

  def __setstate__(self, initargs):
    #print(initargs[1])
    self.__init__(*initargs)

  @property
  def dshape(self):
    return {ll:self._dspace[ll].dim for ll in self._idxs}

  def __str__(self):
    return str(self._T)

  @property
  def rank(self):
    return len(self._spaces)

  @property
  def numel(self):
    return functools.reduce(int.__mul__, (vs.dim for vs in self._spaces))

  @property
  def idxset(self):
    return set(self._idxs)

  def __contains__(self, key):
    return key in self._idxs

  def _init_like(self, T):
    """Initialize tensor containing array T with index parameters of self"""
    return self._tensorfactory(T, tuple(self._idxs), tuple(self._spaces))

  def init_like(self, T, order):
    """Initialize tensor containing array T with index parameters of self,
    but permuted by order"""
    if isinstance(order,str):
      order = order.split(',')
    if set(self._idxs) != set(order):
      ll = list(set(self._idxs) ^ set(order))[0]
      if ll in self._idxs:
        raise ValueError('Index %s missing from order'%ll)
      else:
        raise ValueError('Unrecognized index %s in order'%ll)
    idxo = tuple(order.index(ll) for ll in self._idxs)
    if not isinstance(T,np.ndarray):
      T = np.array(T)
    T = T.transpose(idxo)
    return self._tensorfactory(T, tuple(self._idxs), tuple(self._spaces))

  @classmethod
  def init_from(cls, T, parsestr, *tensors, **settings):
    """Construct derived tensor given array provided
    Comma-separated list of indices, with identification either to
      previously-named index K, as in idx:K (or idx:K* for dual)
    or to index of argument, |#.K (optionally |# if the argument identifies a
      single vector space or |K if exactly one tensor is provided as argument"""
    T = np.array(T,dtype=config.FIELD)
    idxs = []
    clauses = parsestr.split(',')
    shape = T.shape
    vs = []
    if len(clauses) != len(shape):
      raise ValueError('Number of indices provided does not match tensor rank')
    for i in range(len(clauses)):
      m = re.fullmatch(r'(\w+)((\:(?P<rep>\w+)|'
        r'\|((?P<argnum>\d+)\.)?(?P<argidx>\w+))(?P<conj>\*)?)?', clauses[i])
      if not m:
        raise ValueError('Could not process argument \'%s\''%clauses[i])
      l1 = m.group(1)
      if l1 in idxs:
        raise ValueError('New index name %s repeated'%l1)
      idxs.append(l1)
      if m.group('rep'):
        # Repeat of previous index
        l0 = m.group('rep')
        if l0 not in idxs:
          raise ValueError('Unrecognized index %s referenced'%l0)
        i0 = idxs.index(l0)
        if shape[i] != shape[i0]:
          raise ValueError('Identified indices %s,%s do not have the same'
            ' dimension'%(l0,l1))
        V = vs[i0]
      elif m.group('argidx'):
        l0 = m.group('argidx')
        if not m.group('argnum'):
          # Solo argument or argument is solo index
          if len(tensors) == 1 and (isinstance(tensors[0],Tensor) \
              or isinstance(tensors[0], dict)):
            i0 = 0
          elif l0.isnumeric():
            i0 = int(l0)
            if i0 >= len(tensors):
              raise ValueError('Argument #%d referenced not provided'%i0)
            if not isinstance(tensors[i0],links.VSpace):
              raise NotImplementedError()
            V = tensors[i0]
            if m.group('conj'):
              vs.append(~V)
            else:
              vs.append(V)
            continue
          else:
            raise ValueError('Index %s improperly referenced'%l0)
        else:
          i0 = int(m.group('argnum'))
          if i0 >= len(tensors):
            raise ValueError('Argument #%d referenced not provided'%i0)
        T0 = tensors[i0]
        if isinstance(T0, dict):
          if l0 not in T0:
            raise ValueError('Fusion dictionary provided does not contain '
              'index %s'%l0)
          V = T0[l0][0]
        elif isinstance(T0, Tensor):
          if l0 not in T0._dspace:
            raise ValueError('Tensor provided does not contain index %s'%l0)
          V = T0._dspace[l0]
        if V.dim != shape[i]:
          raise ValueError('Dimension provided for index %d.%s does not match'
            ' index %s'%(i0,l0,l1))
      else:
        vs.append(links.VSpace(shape[i]))
        continue
      if m.group('conj'):
        vs.append(~V)
      else:
        vs.append(V)
    return cls(T, idxs, tuple(vs))

  def permuted(self, order):
    """Return tensor permuted according to order"""
    if set(self._idxs) != set(order):
      ll = list(set(self._idxs) ^ set(order))[0]
      if ll in self._idxs:
        raise ValueError('Index %s missing from order'%ll)
      else:
        raise ValueError('Unrecognized index %s in order'%ll)
    idxo = tuple(self._idxs.index(ll) for ll in order)
    return self._T.transpose(idxo)

  def matches(self, A, conj):
    """Verify that indices match
    conj indicates whether or not comparison is with dual spaces"""
    if set(self._idxs) != set(A._idxs):
      ll = list(set(self._idxs) ^ set(A._idxs))[0]
      raise ValueError('Failure to match indices including %s'%ll)
    for ll in self._idxs:
      if not self._dspace[ll].cmp(A._dspace[ll], conj):
        raise ValueError('tensors do not match at index %s'%ll)

  def __and__(self, A):
    """Verify that indices match - tests matches(A, False)"""
    try:
      self.matches(A, False)
      return True
    except ValueError:
      return False

  def __xor__(self, A):
    """Verify that indices match - tests matches(A, True)"""
    try:
      self.matches(A, True)
      return True
    except ValueError:
      return False

  def permute_to(self, A):
    """Return numpy.ndarray reshaped from A._T with the appropriate order
    Verifies that indices match"""
    self.matches(A, False)
    order = tuple(A._idxs.index(ll) for ll in self._idxs)
    return A._T.transpose(order)

  def __add__(self, A):
    return self._init_like(self._T + self.permute_to(A))

  def __sub__(self, A):
    return self._init_like(self._T - self.permute_to(A))

  def __truediv__(self, denom):
    return self._init_like(self._T/denom)

  def __mul__(self, fac):
    return self._init_like(self._T*fac)

  def __neg__(self):
    return self._init_like(-self._T)

  def __iadd__(self, A):
    self._T += self.permute_to(A)
    return self

  def __isub__(self, A):
    self._T -= self.permute_to(A)
    return self

  def __imul__(self, fac):
    self._T *= fac
    return self

  def __itruediv__(self, denom):
    self._T /= denom
    return self

  def __rmul__(self, fac):
    return self.__mul__(fac)

  def conj(self):
    return self.conjugate()

  def __invert__(self):
    return self.conjugate()

  def conjugate(self):
    return self._tensorfactory(self._T.conj(), self._idxs,
      tuple(~vs for vs in self._spaces))

  def norm(self):
    return linalg.norm(self._T)

  def __abs__(self):
    return self.norm()

  def __copy__(self):
    return self._init_like(self._T.copy())
    # __deepcopy__ using VSpace.__deepcopy__?

  def copy(self):
    return self.__copy__()

  def zeros_like(self):
    """Copy filled with zeros"""
    return self._init_like(np.zeros_like(self._T))

  @classmethod
  def zeros_from(cls, parsestr, *tensors):
    """Construct zero tensor based on indices of tensors
    as identified by parsestr
    parsestr is of the form -
      semicolon-separated list (entries corresponding to tensors)
      for each tensor, comma separated list of indices
      of the form 'old(:new)(*)'
      :new for change of name, * for dual index
      optionally some entries may be integers (dimensions) -
      comma-separated list of matching indices; :new group is omitted"""
    # Split into strings corresponding to tensors
    tensstr = parsestr.split(';')
    if len(tensors) != len(tensstr):
      raise ValueError('Additional arguments must match number of segments')
    idxs = []
    vs = []
    dims = []
    for i in range(len(tensstr)):
      istrs = tensstr[i].split(',')
      A = tensors[i]
      if not isinstance(A,cls): # A is dimension of new space
        if not isinstance(A,int) or A <= 0:
          raise ValueError('Argument must be tensor or (positive) dimension')
        V = links.VSpace(A)
        dims.extend(len(istrs)*[A])
        for s in istrs:
          m = re.fullmatch(r'(\w+)(\*)?',s)
          try:
            l1= m.group(1)
          except AttributeError:
            raise ValueError('Segment \'%s\' does not match valid pattern'
              ' for parsed string in zeros_from'%s)
          if l1 in idxs:
            raise ValueError('index %s duplicated in argument to zeros_from'%l1)
          idxs.append(l1)
          if m.group(2):
            vs.append(~V)
          else:
            vs.append(V)
      else:
        for s in istrs:
          m = re.fullmatch(r'(\w+)(\:(?P<rn>\w+))?(?P<conj>\*)?',s)
          try:
            l0 = m.group(1)
          except AttributeError:
            raise ValueError('Segment \'%s\' does not match valid pattern'
              ' for parsed string in zeros_from'%s)
          V = A._dspace[l0] # will raise KeyError if index is absent
          if m.group('rn') is None:
            l1 = l0
          else:
            l1 = m.group('rn')
          if l1 in idxs:
            raise ValueError('index %s duplicated in argument to zeros_from'%l0)
          idxs.append(l1)
          if m.group('conj') is None:
            vs.append(V)
          else:
            vs.append(~V)
          dims.append(V.dim)
    return cls(np.zeros(tuple(dims),dtype=config.FIELD), tuple(idxs), tuple(vs))

  @classmethod
  def _identity(cls, left, right, spaces):
    """identity constructor:
    left is a list of 'left' index names (length n)
    right is a list of 'right' index names (length n)
    spaces is a list of vector spaces (length n), corresponding to left"""
    dims = [v.dim for v in spaces]
    vright = tuple(~v for v in spaces)
    N = functools.reduce(int.__mul__,dims)
    T = np.identity(N,dtype=config.FIELD).reshape(tuple(2*dims))
    return cls(T, tuple(left+right), tuple(spaces)+vright)

  def id_like(self, parsestr):
    """Construct identity tensor within a copy of self
    parsestr identifies pairs of indices as l-r,t-b etc."""
    ipairs = parsestr.split(',')
    if len(ipairs)*2 != self.rank:
      raise ValueError('argument to id_like must identify all indices')
    idx0 = []
    idx1 = []
    iset = set()
    vs0 = []
    for ip in ipairs:
      try:
        l0,l1 = ip.split('-') 
      except ValueError:
        raise ValueError('list item %s formatted incorrectly'
          ' (should be l0-l1)'%ip)
      try:
        V = self._dspace[l0]
        V1 = self._dspace[l1]
      except KeyError:
        ll = l1 if l0 in self._idxs else l0
        raise ValueError('index %s invalid or absent'%ll)
      if l0 in iset:
        raise ValueError('index %s repeated'%l0)
      if l1 in iset:
        raise ValueError('index %s repeated'%l1)
      if not (V ^ V1):
        raise ValueError('indices %s, %s must be dual'%(l0,l1))
      vs0.append(V)
      idx0.append(l0)
      idx1.append(l1)
      iset.update({l0,l1})
    assert len(iset) == self.rank
    return self._identity(idx0,idx1,vs0)

  @classmethod
  def id_from(cls, parsestr, *tensors):
    """Construct identity tensor based on indices of tensors
    as identified by parsestr, of the form -
      semicolon-separated list (entries corresponding to tensors)
      for each tensor, comma separated list of indices
      of the form left-right or oldleft:left-right(&l2-r2&l3-r3...)
      optionally some entries may be integers (dimensions) -
      l1-r1,l2-r2,..."""
    # Split into strings corresponding to tensors
    tensstr = parsestr.split(';')
    if len(tensors) != len(tensstr):
      raise ValueError('Additional arguments must match number of segments')
    idx0 = []
    idx1 = []
    vs = []
    for i in range(len(tensstr)):
      istrs = tensstr[i].split(',')
      A = tensors[i]
      if not isinstance(A,cls): # A is dimension of new space
        if not isinstance(A,int) or A <= 0:
          raise ValueError('Argument must be tensor or (positive) dimension')
        if not re.fullmatch(r'\w+-\w+(,\w+-\w+)*',tensstr[i]):
          raise ValueError('%s contains invalid index pairs'%tensstr[i])
        V = links.VSpace(A)
        for s in istrs:
          l,r = s.split('-')
          idx0.append(l)
          idx1.append(r)
        vs.extend(len(istrs)*[V])
      else:
        for s in istrs:
          m = re.fullmatch(r'(\w+)-(\w+)',s)
          if m:
            l,r = m.groups()
            try:
              V = A._dspace[l]
              assert V^A._dspace[r]
            except:
              raise ValueError('Invalid index pair %s,%s of '
                'tensor argument %d'%(l,r,i))
            idx0.append(l)
            idx1.append(r)
            vs.append(V)
          else:
            m = re.fullmatch(r'(\w+)\:\w+-\w+(\&\w+-\w+)*',s)
            if not m:
              raise ValueError('Invalid segment %s'%s)
            l0 = m.group(1)
            if l0 not in A._idxs:
              raise ValueError('Index %s must belong to %dth tensor argument'%\
                (l0,i))
            V = A._dspace[l0]
            for l,r in re.findall(r'(?::|&)(\w+)-(\w+)\b',s):
              idx0.append(l)
              idx1.append(r)
              vs.append(V)
    r2 = len(idx0)
    if len(set(idx0+idx1)) != 2*r2:
      idxs = idx0+idx1
      for i in range(r2*2-1):
        if idxs[i] in idxs[i:]:
          raise ValueError('repeated index',idxs[i])
    assert r2 == len(idx1) and r2 == len(vs)
    return cls._identity(idx0,idx1,vs)

  def id_extend(self, parsestr, *tensors):
    """Extend self by multiplication with identity tensor based on indices
      of self and/or additional tensors (as in id_from)
    First clause of parsestr may optionally rename indices of self
    May have an(other) optional initial clause that identifies identity
      indices from self"""
    clauses = parsestr.split(';')
    if re.fullmatch(r'(\w+\>\w+\b,?)*\~?', clauses[0]):
      clause = clauses.pop(0)
      idxs = list(self._idxs)
      idxrep = set()
      for l0,l1 in re.findall(r'(\w+)\>(\w+)',clause):
        try:
          i0 = self._idxs.index(l0)
        except ValueError:
          raise KeyError('Index %s not found'%l0)
        if i0 in idxrep:
          raise ValueError('Index %s of original tensor repeated'%l0)
        idxs[i0] = l1
        idxrep.add(i0)
      if '~' not in clause and len(idxrep) != len(idxs):
        raise ValueError('Index %s not reassigned' \
          % idxs[(set(range(len(idxs))) - idxrep).pop()])
      if len(set(idxs)) != len(idxs):
        raise ValueError('Index %s repeated'%_findrep1(idxs))
      idxs = tuple(idxs)
    else:
      idxs = self._idxs
    if len(clauses) == len(tensors)+1:
      # Include self among tensors
      tensors = (self,)+tensors
    I = self.id_from(';'.join(clauses), *tensors)
    return self.__class__(np.tensordot(self._T, I._T, 0),
      idxs+I._idxs, self._spaces+I._spaces)

  def rand_like(self, **settings):
    """Copy with random tensor"""
    return self._init_like(_np_rand(copy(self.shape),**settings))

  @classmethod
  def rand_from(cls, parsestr, *tensors, **settings):
    """Construct random tensor, as in zeros_from"""
    A = cls.zeros_from(parsestr, *tensors)
    A._T = _np_rand(copy(A.shape), **settings)
    return A

  def _rename_dynamic(self, idxmap):
    """Rename indices of tensor (do not copy)"""
    self._idxs = tuple(idxmap[ll] for ll in self._idxs)
    self._dspace = {idxmap[ll]: self._dspace[ll] for ll in idxmap}

  def renamed(self, idxmap, view=True, strict=False):
    """Return tensor with indices renamed
    argument is dictionary, or string a0-b0,a1-b1,...
    if view (default), returns as TensorTransposedView instead of copy
    if strict, must identify all indices; otherwise indices left out will be
    preserved"""
    if not isinstance(idxmap, dict):
      if not isinstance(idxmap, str) or \
          not re.fullmatch(r'\w+-\w+(,\w+-\w+)*',idxmap):
        raise ValueError('argument not a valid mapping')
      idxstr = idxmap
      idxlist = re.findall(r'\b(\w+)-(\w+)\b',idxstr)
      idxmap = dict(idxlist)
      if len(idxlist) != len(idxmap):
        raise ValueError('repeated index of tensor in \'%s\''%idxstr)
    if len(idxmap) < self.rank:
      if strict: 
        raise ValueError('index \'%s\' missing from indices in strict renaming'\
          % list(set(self._idxs) - set(idxmap.keys())).pop())
      else:
        for ll in self._idxs:
          if ll not in idxmap:
            idxmap[ll] = ll
    if set(idxmap.keys()) != set(self._idxs):
      raise ValueError('index dictionary contains index \'%s\' not in tensor' \
        % list(set(idxmap.keys()) - set(self._idxs)).pop())
    if len(set(idxmap.values())) != self.rank:
      raise ValueError('repeated output index')
    if view:
      return TensorTransposedView(self, idxmap)
    else:
      return self.__class__(self._T, [idxmap[self._idxs[i]] \
          for i in range(self.rank)], tuple(self._spaces))

  def transpose(self, parsestr, view=True, strict=False, c=False):
    """Return Tensor with indices exchanged
    syntax a0-b0,a1-b1,... (alternatively dictionary);
      may contain terms of the form a-a
    if view (default), returns as TensorTransposedView instead of copy
    if strict, must name all indices; otherwise indices not named will be 
      preserved
    if c=True, will return conjugate transpose"""
    if c:
      return self.ctranspose(parsestr, strict)
    if isinstance(parsestr, dict):
      idxmap = parsestr
    else:
      if not isinstance(parsestr, str) or \
          not re.fullmatch(r'\w+-\w+(,\w+-\w+)*',parsestr):
        raise ValueError('argument not a valid mapping')
      idxlist = re.findall(r'\b(\w+)-(\w+)\b',parsestr)
      idxmap = dict(idxlist)
      if len(idxlist) != len(idxmap):
        raise ValueError('repeated index of tensor in \'%s\''%parsestr)
    for l0,l1 in tuple(idxmap.items()):
      if l0 not in self._idxs or l1 not in self._idxs:
        raise ValueError('pair \'%s-%s\' not tensor indices'%(l0,l1))
      if l0 != l1:
        if l1 in idxmap:
          raise ValueError('repeated index \'%s\' in argument'%l1) 
        if not self._dspace[l0] == self._dspace[l1]:
          raise ValueError('identified indices \'%s\' and \'%s\' do not match' \
            % (l0,l1))
      idxmap[l1] = l0
    return self.renamed(idxmap, view, strict)

  def cycle(self, parsestr, view=True):
    """Return Tensor with indices cycled
    syntax a0-b0,a1-b1-c1,... ;
    if view (default), returns as TensorTransposedView instead of copy
    indices not named will be preserved"""
    if not isinstance(parsestr, str) or \
        not re.fullmatch(r'(\w+(-\w+)*(,|$))*',parsestr):
      raise ValueError('argument not a valid mapping')
    alltens = set()
    idxmap = {}
    for cstr in parsestr.split(','):
      cycle = cstr.split('-')
      if not cycle:
        continue
      cset = set(cycle)
      if len(cset) != len(cycle):
        raise ValueError('index %s repeated within cycle'%_findrep1(cycle))
      if cset & alltens:
        raise ValueError('index %s repeated between cycles' \
          % (cset&alltens).pop())
      alltens.update(cset)
      for l in cycle:
        if l not in self._idxs:
          raise KeyError('index %s unrecognized'%l)
      l0 = cycle[0]
      V0 = self._dspace[l0]
      for idx in range(1,len(cycle)):
        l1 = cycle[idx]
        if not self._dspace[l1] == V0:
          raise ValueError('indices %s,%s do not match'%(l0,l1))
        idxmap[cycle[idx-1]] = l1
      idxmap[cycle[-1]] = l0
    return self.renamed(idxmap, view, strict=False)

  def ctranspose(self, parsestr, strict=True):
    """Return conjugate-transpose tensor; syntax as in transpose"""
    if isinstance(parsestr, dict):
      idxmap = parsestr
    else:
      if not isinstance(parsestr, str) or \
          not re.fullmatch(r'\w+-\w+(,\w+-\w+)*',parsestr):
        raise ValueError('argument not a valid mapping')
      idxlist = re.findall(r'\b(\w+)-(\w+)\b',parsestr)
      idxmap = dict(idxlist)
      if len(idxlist) != len(idxmap):
        raise ValueError('repeated index of tensor in \'%s\''%parsestr)
    for l0,l1 in tuple(idxmap.items()):
      if l0 not in self._idxs or l1 not in self._idxs:
        raise ValueError('pair \'%s-%s\' not tensor indices'%(l0,l1))
      if l0 != l1:
        if l1 in idxmap:
          raise ValueError('repeated index \'%s\' in argument'%l1) 
        idxmap[l1] = l0
      if not self._dspace[l0] ^ self._dspace[l1]:
        raise ValueError('identified indices \'%s\' and \'%s\' do not match' \
          % (l0,l1))
    if len(idxmap) < self.rank:
      if strict:
        raise ValueError('index \'%s\' missing from conjugate-transpose'\
          % list(set(self._idxs) - set(idxmap.keys())).pop())
      else:
        for ll in self._idxs:
          if ll not in idxmap:
            if not self._dspace[ll] ^ self._dspace[ll]:
              raise ValueError('index \'%s\' not self-dual'%ll)
    newidx = []
    newsp = []
    for l0 in self._idxs:
      l1 = idxmap[l0]
      newidx.append(l1)
      newsp.append(self._dspace[l1])
    return self._tensorfactory(self._T.conj(), tuple(newidx), tuple(newsp))

  def fuse(self, parsestr, *specs, **settings):
    """Fuse indices according to parsestr
    Substrings of the form l0,l1,l2>L, separated by semicolons
      :K indicates isomorphism with previously-named K (K* for dual)
      |#.K or |# (where # is the argument position, starting with 0)
        indicates that the space should be determined, by isomorphism, from
        argument # value 'K' (if dictionary) or simply argument # otherwise.
    final substring ~ indicates autocompletion
      (all unspecified indices are preserved)
    if incl_info (default True), returns second value which is a dictionary
      from index names to necessary information"""
    substrs = parsestr.split(';')
    if substrs[-1] == '~':
      substrs.pop(-1)
      auto = True
    else:
      auto = False
    newidxs = set()
    idxall = set()
    vall = {}
    args = []
    for sub in substrs:
      if len(specs) == 1:
        m = re.fullmatch(r'(\w+(?:,\w+)*)\>(\w+)((\:(?P<rep>\w+)|\|(?P<arg>0)|\|(?:0\.)?(?P<key>\w+))(?P<conj>\*)?)?',sub)
      else:
        m = re.fullmatch(r'(\w+(?:,\w+)*)\>(\w+)((\:(?P<rep>\w+)|\|(?P<arg>\d+)(\.(?P<key>\w+))?)(?P<conj>\*)?)?',sub)
      if m is None:
        raise ValueError('Invalid substring %s'%sub)
      l1 = m.group(2)
      if l1 in newidxs:
        raise ValueError('fused index %s repeated'%l1)
      newidxs.add(l1)
      subidxs = m.group(1).split(',')
      vs = []
      for ll in subidxs:
        if ll not in self._idxs:
          raise KeyError('index \'%s\' does not belong to original tensor'%ll) 
        if ll in idxall:
          raise ValueError('original index %s repeated'%ll)
        idxall.add(ll)
        vs.append(self._dspace[ll])
      vall[l1] = vs
      conj = bool(m.group('conj'))
      if m.group('rep'):
        l0 = m.group('rep')
        if l0 not in newidxs:
          raise KeyError('index \'%s\' referred to before specified'%l0)
        vs0 = vall[l0]
        for ii in range(len(subidxs)):
          if not vs[ii].cmp(vs0[ii], conj):
            raise ValueError('index \'%s\' paired with non-matching index' \
              % subidxs[ii])
        args.append((l1,subidxs,l0,conj))
      elif m.group('arg') or (len(specs) == 1 and m.group('key')):
        if len(specs) == 1:
          argi = 0
        else:
          argi = int(m.group('arg'))
          if argi >= len(specs):
            raise ValueError('Optional argument #%d not found'%argi)
        specinfo = specs[argi]
        if m.group('key'):
          l0 = m.group('key')
          try:
            specinfo = specinfo[l0]
          except:
            raise ValueError('argument #%d not dictionary with key %s' \
              % (argi,l0))
        if not isinstance(specinfo, FusionPrimitive):
          raise ValueError('argument #%d not a fusion primitive'%argi)
        specinfo.matchtensor(self, subidxs, conj)
        args.append((l1,subidxs,specinfo,conj))
      else:
        args.append((l1,subidxs))
    if auto: # Complete mapping with unfused indices
      for ll in set(self._idxs) - idxall:
        if ll in newidxs:
          raise ValueError('Autofilled index \'%s\' already assigned'%ll)
        args.append((ll,(ll,)))
    fused,info = self._do_fuse(*args)
    if 'incl_info' in settings and not settings['incl_info']:
      return fused
    return fused,info

  def _get_fuse_prim(self, *args):
    """Produce 'info' dictionary to be used within _do_fuse"""
    infos = {}
    for arg in args:
      if len(arg) == 2:
        l1,l0s = arg
        infos[l1] = FusionPrimitive(l0s,self)
      else:
        assert len(arg) == 4
        l1,l0s,fusion,conj = arg
        if not isinstance(fusion,FusionPrimitive):
          # Is previously-supplied index
          fusion = infos[fusion]
        assert len(l0s) == fusion.rank
        if conj:
          infos[l1] = fusion.conj()
        else:
          infos[l1] = fusion
    return infos

  def _do_fuse(self, *args, prims=None):
    """Fuse based on list of tuples
    (new idx, old idxs, [info/reference to previous, conj. bool])
    New index order will be explicitly as in args"""
    neworder = []
    newshape = []
    permutation = []
    newvs = []
    if prims is None: # Provided if computation is being repeated
      prims = self._get_fuse_prim(*args)
    for arg in args:
      l1,l0s = arg[:2]
      for ll in l0s:
        permutation.append(self._idxs.index(ll))
      neworder.append(l1)
      newvs.append(prims[l1].V)
      newshape.append(prims[l1].dim)
    T = self._T.transpose(permutation).reshape(newshape)
    return self._tensorfactory(T, tuple(neworder), tuple(newvs)), prims

  def unfuse(self, parsestr, *specs):
    """Unfuse indices according to parsestr
    Must include specifications (fusion primitives) from previous fusion
    Substrings of the form L>l0,l1,l2|#.K (.K optional) separated by semicolons
      L is original index; l0,... are the new unfused indices
      (must have order of original)
      # indicates argument number, 'K' indicates key if dictionary
      |# can be excluded only if there is a single output index
      May use |K if len(specs) = 1
    final substring ~ indicates autocompletion
      (all unspecified indices are preserved)"""
    substrs = parsestr.split(';')
    if substrs[-1] == '~':
      substrs.pop(-1)
      auto = True
    else:
      auto = False
    narg = len(specs)
    argd = {}
    newidxs = set()
    oldidxs = set()
    # First collect information from parsestr
    for sub in substrs:
      if narg == 1:
        m = re.fullmatch(r'(\w+)\>(\w+(?:,\w+)*)\|(0|((0\.)?(?P<key>\w+)))?(?P<conj>\*)?',sub)
      else:
        m = re.fullmatch(r'(\w+)\>(\w+(?:,\w+)*)\|(\d+)(?:\.(?P<key>\w+))?(?P<conj>\*)?',sub)
      if m:
        l0 = m.group(1)
        ls = m.group(2).split(',')
        if narg == 1:
          argi = 0
        else:
          argi = int(m.group(3))
        if argi >= narg:
          raise ValueError('expected more than %d specification arguments'%argi)
        if key:
          if key not in specs[argi]:
            raise KeyError('specification #%d missing key %d'%(argi,key))
          fusion = specs[argi][key]
        else:
          fusion = specs[argi]
        if not isinstance(fusion, FusionPrimitive):
          raise ValueError('specifications must be fusion primitives')
        if conj:
          fusion = fusion.conj()
        if not (fusion.V == self._dspace[l0]):
          raise ValueError('Index %s does not match original'%l0)
        if len(ls) != fusion.rank:
          raise ValueError('Index %s fused from %d indices, expanded into %d'%\
            (l0,fusion.rank,len(ls)))
        argd[l0] = (ls, fusion)
        if newidxs & set(ls):
          raise ValueError('New index %s repeated'%list(newidxs & set(ls))[0])
        newidxs.update(ls)
      elif re.fullmatch(r'\w+\>\w+',sub):
        l0,ll = sub.split('>')
        if ll in newidxs:
          raise ValueError('New index %s repeated'%ll)
        newidxs.add(ll)
        argd[l0] = ll
      else:
        raise ValueError('Invalid substring \'%s\''%sub)
      if l0 in oldidxs:
        raise ValueError('Index %s repeated'%l0)
      oldidxs.add(l0)
    # Fill in the gaps
    for l0 in set(self._idxs) - oldidxs:
      if auto:
        if l0 in newidxs:
          raise ValueError('Autocompleted index %s already given in expanded'
            ' indices'%l0)
        argd[l0] = l0
      else:
        raise ValueError('Index %s missing'%l0)
    return self._do_unfuse(argd)

  def _do_unfuse(self, fuseinfo):
    """Unfuse indices. fuseinfo is a complete dictionary of (current) indices
    to single new idx name, or otherwise (idx names, FusionPrimitive)"""
    newvs = []
    newshape = []
    newidxs = []
    for ll in self._idxs:
      if isinstance(fuseinfo[ll],tuple):
        names,fusion = fuseinfo[ll]
        vs = fusion._spaces_in
        newidxs.extend(names)
        newvs.extend(vs)
        newshape.extend((V.dim for V in vs))
      else:
        newidxs.append(fuseinfo[ll])
        V = self._dspace[ll]
        newvs.append(V)
        newshape.append(V.dim)
    T = self._T.reshape(newshape)
    return self._tensorfactory(T, tuple(newidxs), tuple(newvs))

  def contract(self, T2, parsestr):
    """Contract with tensor T2
    parsestr should be of the form
    a0-b0,a1-b1,a2-b2;a3>c0,a4>c1;b3>c2
    Including ~ in either of the last two substrings (or as a replacement for
      both) indicates remaining indices retain names
    Only ~ indicates tensor product with index names preserved
    Ending with * indicates contraction with conjugate"""
    if parsestr[-1] == '*':
      conj = True
      parsestr = parsestr[:-1]
    else:
      conj = False
    if len(parsestr) > 1 and ';' not in parsestr and parsestr[-1] == '~':
      substrs = [parsestr[:-1],'~']
    else:
      substrs = parsestr.split(';')
    if not conj and substrs[0] and substrs[0][-1] == '*':
      substrs[0] = substrs[0][:-1]
      conj = True
    auto1 = auto2 = False
    if len(substrs) == 1:
      # Either ending in ~, or else pairing all indices of two tensors
      if parsestr == '~':
        out1 = {l:l for l in self._idxs}
        out2 = {l:l for l in T2._idxs}
        contracted = []
      elif re.fullmatch(r'(?!,)((^|,)\w+-\w+)*',parsestr):
        if self.rank != T2.rank:
          raise ValueError('Inner-product-style contraction without '
            'matching rank')
        out1 = out2 = {}
        contracted = re.findall(r'\b(\w+)-(\w+)\b',parsestr)
        if self.rank != len(contracted):
          raise ValueError('%d index pairs provided for inner-product-style '
            'contraction of rank-%d tensors'%(len(contracted),self.rank))
      else:
        raise ValueError('Invalid argument %s'%parsestr)
      auto1 = auto2 = False
    elif len(substrs) == 2:
      if not substrs[1] == '~' or \
          not re.fullmatch(r'(?!,)((,|^)\w+-\w+)*',substrs[0]):
        raise ValueError('Invalid argument %s'%parsestr)
      auto1 = auto2 = ('','')
      contracted = re.findall(r'\b(\w+)-(\w+)\b',parsestr)
      out1 = {}
      out2 = {}
    elif len(substrs) == 3:
      if not re.fullmatch(r'(?!,)((,|^)\w+-\w+)*',substrs[0]):
        raise ValueError('Invalid substring %s'%substrs[0])
      contracted = re.findall(r'\b(\w+)-(\w+)\b',substrs[0])
      if not re.fullmatch(r'(?!,)((^|,)\w+\>\w+)*(\~|,\w*\~\w*)?',substrs[1]) \
          and not re.fullmatch(r'\w*\~\w*',substrs[1]):
        raise ValueError('Invalid substring \'%s\''%substrs[1])
      if not re.fullmatch(r'(?!,)((^|,)\w+\>\w+)*(\~|,\w*\~\w*)?',substrs[2]) \
          and not re.fullmatch(r'\w*\~\w*',substrs[1]):
        raise ValueError('Invalid substring \'%s\''%substrs[2])
      
      m = re.search(r'(\w*)\~(\w*)',substrs[1])
      if m:
        auto1 = m.groups()
      else:
        auto1 = False
      m = re.search(r'(\w*)\~(\w*)',substrs[2])
      if m:
        auto2 = m.groups()
      else:
        auto2 = False
      out1 = dict(re.findall(r'\b(\w+)\>(\w+)\b',substrs[1]))
      out2 = dict(re.findall(r'\b(\w+)\>(\w+)\b',substrs[2]))
    else:
      raise ValueError('Invalid argument %s'%parsestr)
    i1 = []
    i2 = []
    idxs1 = set()
    idxs2 = set()
    for l,r in contracted:
      if l not in self._idxs:
        raise KeyError('Contracted index %s missing'%l)
      if l in idxs1:
        raise ValueError('Contracted index %s repeated'%l)
      if l in out1:
        raise ValueError('Contracted index %s repeated in output'%l)
      if r not in T2._idxs:
        raise KeyError('Contracted index %s missing'%r)
      if r in idxs2:
        raise ValueError('Contracted index %s repeated'%r)
      if r in out2:
        raise ValueError('Contracted index %s repeated in output'%r)
      idxs1.add(l)
      idxs2.add(r)
    if auto1:
      for ll in set(self._idxs) - idxs1:
        if ll not in out1:
          out1[ll] = auto1[0]+ll+auto1[1]
    else:
      lx = set(self._idxs) - idxs1 - set(out1.keys())
      if lx:
        raise ValueError('Index \'%s\' from left absent'%lx.pop())
    if auto2:
      for ll in set(T2._idxs) - idxs2:
        if ll not in out2:
          out2[ll] = auto2[0]+ll+auto2[1]
    else:
      lx = set(T2._idxs) - idxs2 - set(out2.keys())
      if lx:
        raise ValueError('Index \'%s\' from right absent'%lx.pop())
    if len(set(out1.values())|set(out2.values())) < len(out1) + len(out2):
      l0 = _findrep1(list(out1.values())+list(out2.values()))
      raise ValueError('Output index \'%s\' repeated'%l0)
    return self._do_contract(T2, contracted, out1, out2, conj)

  def _do_contract(self, T2, contracted, out1, out2, conj):
    i1 = []
    i2 = []
    for l,r in contracted:
      i1.append(self._idxs.index(l))
      i2.append(T2._idxs.index(r))
      if not self._dspace[l].cmp(T2._dspace[r],not conj):
        raise ValueError('Contracted indices %s and %s do not match'%(l,r))
    lout1 = []
    lout2 = []
    vout1 = []
    vout2 = []
    for il in range(self.rank):
      if il not in i1:
        lout1.append(out1[self._idxs[il]])
        vout1.append(self._spaces[il])
    for il in range(T2.rank):
      if il not in i2:
        lout2.append(out2[T2._idxs[il]])
        vout2.append(T2._spaces[il])
    if conj:
      vout2 = [~V for V in vout2]
    arr2 = T2._T
    if conj:
      arr2 = arr2.conj()
    try:
      T = np.tensordot(self._T,arr2,[i1,i2])
    except MemoryError as me:
      print(copy(self.shape))
      print(copy(T2.shape))
      print('%dx%d->%d'%(self.numel,T2.numel,
        functools.reduce(int.__mul__,[v.dim for v in vout1+vout2])))
      raise me
    if len(lout1)+len(lout2) == 0: # Scalar
      return T[()]
    return self._tensorfactory(T, tuple(lout1+lout2), tuple(vout1+vout2))

  def trace(self, parsestr=''):
    """Trace over identified indices
    Contraction part of parsestr should take the form
      r0-l0,r1-l1,r2-l2
    Optionally followed by output of the form ;a0>b0,a1>b1
      otherwise all uncontracted indices keep their names
    parsestr may be ommitted iff tensor is rank-2"""
    if parsestr == '' and self.rank == 2:
      if not (self._spaces[0] ^ self._spaces[1]):
        raise ValueError('Matrix is not endomorphism-like')
      return self._do_trace([self._idxs],{})
    elif re.fullmatch(r'(?!,)((^|,)\w+-\w+)+(;|$)(\w+\>\w+(,|\~$|$))*',
        parsestr):
      tracel = re.findall(r'\b(\w+)-(\w+)\b',parsestr)
      idxtr = sum(tracel,())
      outl = re.findall(r'\b(\w+)\>(\w+)\b',parsestr)
      outd = dict(outl)
      outl0 = outd.keys()
      idxall = idxtr + tuple(outl0)
      if len(set(idxall)) < len(idxall):
        if set(idxtr) & set(outl0):
          raise ValueError('Traced index \'%s\' appears in output as well' \
            % list(set(idxtr) & set(outl0))[0])
        else:
          raise ValueError('Traced index \'%s\' repeated'%_findrep1(idxtr))
      for l,r in tracel:
        try:
          if not (self._dspace[r] ^ self._dspace[l]):
            raise ValueError('Non-matching indices %s-%s traced over'%(l,r))
        except KeyError:
          if l not in self._idxs:
            raise KeyError('Traced index \'%s\' missing'%l)
          else:
            raise KeyError('Traced index \'%s\' missing'%r)
      diffset = set(outl0) - set(self._idxs)
      if diffset:
        outl1 = outd.values()
        if parsestr[-1] == '~':
          for l0 in diffset:
            if l0 in outl1:
              raise ValueError('Index %s both explicitly & '
                'implicitly assigned'%l1)
            outd[l0] = l0
            outl.append((l0,l0))
        else:
          raise KeyError('Output index \'%s\' missing'%list(diffset)[0])
      if len(outd) < len(outl):
        raise ValueError('Index \'%s\' repeated in output' \
          % _findrep1([i for i,o in outl]))
      for l in set(self._idxs) - set(idxall):
        outd[l] = l
      ordered = sorted(tracel,key=lambda lr:-(self._dspace[lr[0]].dim))
      return self._do_trace(ordered, outd)
    else:
      raise ValueError('Invalid argument %s'%parsestr)

  def _do_trace(self, traced, output):
    """Helper function for trace operation: inputs list of tuples of
    indices to trace over, dictionary of output indices"""
    T = self._T
    idxs = list(self._idxs)
    for l,r in traced:
      T = T.trace(axis1=idxs.index(l),axis2=idxs.index(r))
      idxs.remove(l)
      idxs.remove(r)
    assert set(output.keys()) == set(idxs)
    if len(idxs) == 0:
      return T
    newvs = []
    for i in range(len(idxs)):
      l0 = idxs[i]
      idxs[i] = output[l0]
      newvs.append(self._dspace[l0])
    return self._tensorfactory(T, tuple(idxs), tuple(newvs))

  def svd(self, parsestr, chi=None, tolerance=1e-12, chi_ratio=2.,
      sv_tensor=False, discard=False):
    """Performs svd on self
    parsestr will be of the form
      l0,l1|cc|r0,r1,r2 (cc might be cl,cr)
      right indices may be replaced by autocomplete ~
      defining transformation into matrix
      cc (alternatively cl,cr) defines new center indices
    Returns U,s,V^dagger, where unitaries U and V are tensors and singular
      values s are by default a list
    Optional chi is the bond dimension to cut to
    Optional tolerance is the minimum singular value
    chi_ratio*chi will be the maximum acceptable final value of chi
      if float, else will be simply chi_ratio
    sv_tensor indicates whether the singular values are returned as a tensor
    if discard, delete array associated with tensor to clear space"""
    m = re.fullmatch(r'((?:\w+(?:,|(?=\|)))+)\|(\w+(?:,\w+)?)\|((?:(?:(?<=\|)|,)\w+)+|\~)',parsestr)
    if not m:
      raise ValueError('Invalid argument %s'%parsestr)
    ls = m.group(1).split(',')
    lset = set(ls)
    aset = set(self._idxs)
    if len(lset) < len(ls):
      raise ValueError('Left index %s repeated'%_findrep1(ls))
    if m.group(3) == '~':
      if lset - aset:
        raise KeyError('Left index %s does not '
          'belong to tensor'%list(lset-aset)[0])
      rset = aset - lset
      if not rset:
        raise ValueError('Cannot perform SVD with rank-0 codomain')
      rs = list(rset)
    else:
      rs = m.group(3).split(',')
      rset = set(rs)
      if lset | rset != aset:
        if rset - aset:
          raise ValueError('Right index %s does not '
            'belong to tensor'%list(rset-aset)[0])
        else:
          assert aset - (lset | rset)
          raise ValueError('Index %s not included'%list(rset^aset^lset)[0])
    cs = m.group(2).split(',')
    if len(cs) == 1:
      cl = cr = cs[0]
      if cl in aset:
        raise ValueError('%s also included in tensor'%cl)
    else:
      cl,cr = cs
      if cl in lset:
        raise ValueError('left index %s repeated as center'%cl)
      if cr in rset:
        raise ValueError('right index %s repeated as center'%cr)
    if cl == cr and sv_tensor:
      raise ValueError('Center indices must be distinct if singular values'
        ' are reported as tensor')
    M,info = self._do_fuse((0,ls),(1,rs))
    if discard:
      del self._T
    try:
      U,s,V = M._do_svd(chi, tolerance, chi_ratio, sv_tensor)
    except MemoryError as me:
      print(copy(self.shape))
      print('%dx%d'%(M.shape[0],M.shape[1]))
      raise me
    U = U._do_unfuse({0:(ls, info[0]), 1:cl})
    V = V._do_unfuse({0:cr, 1:(rs, info[1])})
    if sv_tensor:
      s = s.renamed({0:cl,1:cr})
    return U,s,V

  def _do_svd(self, chi, tolerance, chi_ratio, sv_tensor):
    """Helper function for svd; must provide rank-2 tensor indexed 0,1"""
    T = self.permuted((0,1))
    U,s,Vd = safesvd(T)
    # Truncate, as necessary
    D = len(s)
    snorm = linalg.norm(s)
    if chi and chi<D:
      # Truncate based on bond dimension
      if tolerance and s[chi-1]/snorm < tolerance:
        # Singular-value tolerance still calls for truncation below chi
        n = chi-1
        for nn in range(chi-2,0,-1):
          if s[nn]/snorm >= tolerance:
            n = nn+1
            break
      else:
        n = chi
    elif tolerance and s[-1]/snorm < tolerance:
      # Only truncate based on singular-value tolerance
      n = 1
      for nn in range(2,D): # Iterate from last singular value
        if s[-nn]/snorm >= tolerance:
          n = D-nn+1
          break
    else:
      n = D
    if n != D and config.degen_tolerance and \
        1-s[n]/s[n-1] < config.degen_tolerance:
      # Protecting degeneracies
      if chi_ratio: # Find maximum allowable value to follow degeneracies
        if isinstance(chi_ratio,int):
          chi_max = chi_ratio
        else:
          chi_max = int(np.floor(chi_ratio*chi))
        chi_max = min(chi_max,D)
      else:
        chi_max = D
      while n < chi_max and 1-s[n]/s[n-1] > config.degen_tolerance:
        n += 1
    if n != D:
      s = s[:n]
      U = U[:,:n]
      Vd = Vd[:n,:]
    V_l = self._dspace[0]
    V_r = self._dspace[1]
    V_cl = links.VSpace(n)
    V_cr = V_cl.dual()
    U = self.__class__(U, (0,1), (V_l,V_cr))
    Vd = self.__class__(Vd, (0,1), (V_cl,V_r))
    if sv_tensor:
      s = self.__class__(np.diag(s), (0,1), (V_cl,V_cr))
    return U,s,Vd

  def svd_split(self, parsestr, **settings):
    """Use SVD to provide a 'split' tensor, i.e. A = L.R
      where L = U.s^1/2, R = s^1/2.V^H
    parsestr and keyword arguments are as in svd"""
    U,s,Vd = self.svd(parsestr, sv_tensor=False, **settings)
    shalf = np.sqrt(s)
    l,c,r = parsestr.split('|')
    cs = c.split(',')
    if len(cs) == 2:
      cl,cr = cs
    else:
      cl = cr = cs[0]
    L = U.diag_mult(cl,shalf)
    R = Vd.diag_mult(cr,shalf)
    return L,R

  def diag_mult(self, ll, d):
    """Multiply a list/1D matrix, i.e. diagonal of a matrix,
        along a single index"""
    if isinstance(d, np.ndarray) and len(d.shape) == 2:
      if d.shape[0] != d.shape[1]:
        raise ValueError('Matrix supplied must be square')
      d = np.diag(d)
    if len(d) != self.shape[ll]:
      raise ValueError('Dimension does not match')
    ii = self._idxs.index(ll)
    Ttr = np.moveaxis(self._T,ii,-1)
    T1 = np.multiply(Ttr,d)
    idx1 = self._idxs[:ii] + self._idxs[ii+1:] + (ll,)
    space1 = self._spaces[:ii] + self._spaces[ii+1:] + (self._spaces[ii],)
    return self.__class__(T1, idx1, space1)

  def mat_mult(self, ll, M):
    """Multiply a matrix along a single index"""
    if ll[-1] == '*':
      ll = ll[:-1]
      M = M.conj()
    if not isinstance(M, np.ndarray):
      if isinstance(M, Tensor):
        if M.rank != 2:
          raise ValueError('tensor must be matrix-like')
        m = re.fullmatch(r'(\w+)-(\w+)', ll)
        if not m:
          raise ValueError('argument must be of the form index-index '
            'if Tensor is supplied')
        ll,lr = m.groups()
        if ll not in self._idxs:
          raise KeyError('unrecognized left index %s'%ll)
        if lr not in M._idxs:
          raise KeyError('unrecognized right index %s'%lr)
        if not self._dspace[ll] ^ M._dspace[lr]:
          raise ValueError('Cannot contract %s-%s'%(ll,lr))
        ld = {l:l for l in set(self._idxs)-{ll}}
        lb = (set(M._idxs)-{lr}).pop()
        return self._do_contract(M,((ll,lr),), ld, {lb:ll}, False)
    if len(M.shape) != 2 or M.shape[0] != M.shape[1]:
      raise ValueError('Matrix supplied must be square')
    if M.shape[0] != self.shape[ll]:
      raise ValueError('Dimension does not match')
    ii = self._idxs.index(ll)
    T1 = np.tensordot(self._T,M,(ii,0))
    idx1 = self._idxs[:ii] + self._idxs[ii+1:] + (ll,)
    space1 = self._spaces[:ii] + self._spaces[ii+1:] + (self._spaces[ii],)
    return self.__class__(T1, idx1, space1)

  @_endomorphic
  def eig(self, lidx, ridx, herm=True, selection=None, left=False, vecs=True,
      mat=False, discard=False, zero_tol=None):
    """Gets eigenvalue-eigenvector pairs of Tensor, with target space described
    by left indices
    herm determines whether or not the matrix being diagonalized is treated as
      if it has Hermitian symmetry.
    selection is the selection of eigenvalues. By default, will be all.
      May be integer (maximum #, or min if negative)
      or (min,max)
      If Hermitian flag is set, will order lowest to highest value
      (as in linalg.eigh). Otherwise will order highest to lowest magnitude.
    if zero_tol is set, will cut out eigenvectors with magnitude below the 
      given threshold
    Returns w, v. If mat, will return v as a Tensor (see e.g. svd);
      otherwise will return lists; if specified as string, will be index
      (otherwise 'c')
    If left, will return eigenvalues, right eigenvectors, left eigenvectors
    If vecs=False (and mat=False), will return eigenvalues only"""
    A, info = self._do_fuse((0,lidx),(1,ridx,0,True))
    if discard:
      del self._T
    N = A.shape[0]
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
    # Diagonalize
    if vecs:
      if herm:
        w, v = linalg.eigh(A._T, eigvals=selection)
        if zero_tol is not None:
          w0 = max(np.abs(w))
          s0 = sum(w < -w0*zero_tol)
          ncut = sum(w[s0:] < w0*zero_tol)
          w = np.delete(w, slice(s0,s0+ncut), 0)
          v = np.delete(v, slice(s0,s0+ncut), 1)
          nn -= ncut
        if left:
          vl = v
      else:
        if left:
          w, vl, v = linalg.eig(A._T, left=True)
          # Does not return dual basis of eigenvalues
          for i in range(vl.shape[1]):
            vl[:,i] /= vl[:,i].dot(v[:,i].conj())
        else:
          w, v = linalg.eig(A._T, left=False)
        if zero_tol is not None:
          w0 = max(np.abs(w))
          chi = sum(np.abs(w) > w0*zero_tol)
          if selection:
            selection = (selection[0], min(selection[1],chi-1))
            nn = selection[1] - selection[0] + 1
          else:
            selection = (0,chi-1)
            nn = chi
        if selection:
          idxw = np.argsort(-np.abs(w))
          # Sort eigenvalues, eigenvectors
          Msort = sparse.csc_matrix((nn*[1],idxw[selection[0]:selection[1]+1],
            range(nn+1)),dtype=int,shape=(N,nn))
          w = Msort.__rmul__(w)
          v = Msort.__rmul__(v)
          if left:
            vl = Msort.__rmul__(vl)
      if mat:
        # Process names for eigenvector index
        if isinstance(mat,tuple):
          cr,cl = mat
        elif isinstance(mat,str):
          cr = cl = mat
        else:
          cr = cl = 'c'
        if cr in lidx or not re.fullmatch(r'\w+',cr):
          raise ValueError('new center index %s invalid'%s)
        if left and (cl in ridx or not re.fullmatch(r'\w+',cl)):
          raise ValueError('new center index for left-eigenvectors'
            ' %s invalid'%s)
        RV = self.__class__(v, (0,'c'), (A._spaces[0], links.VSpace(nn)))
        RT = RV._do_unfuse({0:(lidx,info[0]),'c':cr})
        if left:
          LV = RV._init_like(vl).conj()
          LT = LV._do_unfuse({0:(ridx,info[1]),'c':cl})
          return w, RT, LT
        return w, RT
      else:
        vs = []
        init_args = ((0,), (A._spaces[0],))
        d_unfuse = {0:(lidx,info[0])}
        for i in range(nn):
          vs.append(self.__class__(v[:,i], *init_args)._do_unfuse(d_unfuse))
        if left: 
          vls = []
          init_args = ((1,), (A._spaces[1]))
          d_unfuse = {1:(lidx,info[1][1:])}
          for i in range(nn):
            vl = vl.conj()
            vls.append(self.__class__(vl[:,i], *init_args)._do_unfuse(d_unfuse))
            return w,vs,vls
        return w,vs
    else:
      if herm:
        return linalg.eigvalsh(A._T, eigvals=selection)
      elif selection:
        ws = linalg.eigvals(A._T)
        return sorted(ws, key=lambda z: -abs(z))[selection[0]:selection[1]+1]
      else:
        return linalg.eigvals(A._T)

  @_endomorphic
  def pow(self, lidx, ridx, n):
    """Raise to power n"""
    if not isinstance(n, int):
      raise NotImplementedError()
    if n < 0:
      raise NotImplementedError()
    if n == 0:
      return self.__class__._identity(lidx, ridx,  
        [self._dspace[l] for l in lidx])
    elif n == 1:
      return copy(self)
    else:
      M = self
      srank = self.rank//2
      contracted = [(ridx[i],lidx[i]) for i in range(srank)]
      outl = {l:l for l in lidx}
      outr = {r:r for r in ridx}
      for i in range(1,n):
        M = M._do_contract(self, contracted, outl, outr, False)
      return M

  @_endo_transformation
  def exp(M, coeff):
    M._T = linalg.expm(coeff*M._T)

  @_endo_transformation
  def log(M):
    M._T = linalg.logm(M._T)

  @_endo_transformation
  def sqrt(M):
    M._T = linalg.sqrtm(M._T)

  @_endo_transformation
  def inv(M):
    M._T = linalg.inv(M._T)
    

  @classmethod
  def identify_indices(cls, idxs, *tens):
    if not isinstance(idxs,str):
      raise ValueError('first argument must be string')
    if not all(isinstance(t, cls) for t in tens):
      raise TypeError('subsequent arguments must be Tensors')
    for clause in idxs.split(','):
      if re.fullmatch(r'\d+\.\w+(-\d+\.\w+\*?)*', clause):
        n_idxs = re.findall(r'(\d+)\.(\w+)')
      else:
        if not re.fullmatch(r'\w+(-\w+\*?)*', clause):
          raise ValueError('could not process argument \'%s\''%clause)
        idxl = clause.split('-')
        if len(tens) == 1:
          n_idxs = [(0,x) for x in idxl]
        elif len(tens) == len(idxl):
          n_idxs = [(i,idxl[i]) for i in range(len(tens))]
        else:
          raise ValueError('Expected length-%d clause without tensors'
            ' specified, got length-%d'%(len(tens),len(idxl)))
      t0,l0 = n_idxs[0]
      if t0 >= len(tens):
        raise ValueError('Got %d arguments, expected at least %d' \
          % (len(tens), t0+1))
      if l0 not in tens[t0]:
        raise ValueError('Tensor #%d does not have index %s'%(t0,l0))
      V0 = tens[t0]._dspace[l0]
      if not isinstance(V0, links.VectorSpaceTracked):
        raise ValueError('Only tracked vector spaces are to be identified '
          'with each other')
      for t1,l1 in n_idxs[1:]:
        if t1 >= len(tens):
          raise ValueError('Got %d arguments, expected at least %d' \
            % (len(tens), t1+1))
        c = (l1[-1] == '*')
        if c:
          ll = l1[:-1]
        else:
          ll = l1
        if ll not in tens[t1]:
          raise ValueError('Tensor #%d does not have index %s'%(t1,l1l))
        V1 = tens[t1]._dspace[ll]
        if V0.dim != V1.dim:
          raise ValueError('Can only identify vector spaces with the same'
            ' dimension')
        if c:
          if V0._VectorSpaceTracked__dual is None:
            V0._VectorSpaceTracked__dual = V1
            if V1._VectorSpaceTracked__dual is None:
              # Assign as each others' duals
              V1._VectorSpaceTracked__dual = V0
            else:
              # Just assign V1 as dual of V0
              cls._reassign_space_as(V1.dual(),V0)
          elif V1._VectorSpaceTracked__dual is None:
            cls._reassign_space_as(V1,V0.dual())
          elif V0 is V1:
            # Creating `real' vector space
            cls._reassign_space_as(V1,V0)
          else:
            cls._reassign_space_as(V1,V0.dual())
            cls._reassign_space_as(V1.dual(),V0)
        elif V0 is V1:
          continue
        else:
          if V0._VectorSpaceTracked__dual is None:
            if V1._VectorSpaceTracked__dual is not None:
              V0._VectorSpaceTracked__dual = V1.dual()
          elif V1._VectorSpaceTracked__dual is not None:
            if V1.dual() is V0:
              # Creating real space
              V0._VectorSpaceTracked__dual = V0
            else:
              cls._reassign_space_as(V1.dual(),V0.dual())
          cls._reassign_space_as(V1,V0)

  @classmethod
  def _reassign_space_as(cls, V1, V0):
    # Find all tensors that include vector space V1 and change it to V0
    import gc
    # Find tuples that refer to V1
    for r1 in gc.get_referrers(V1):
      if isinstance(r1,tuple):
        # Determine if it is the _spaces attribute of some Tensor
        tref = None
        for r2 in gc.get_referrers(r1):
          if isinstance(r2,dict) and '_spaces' in r2 and r2['_spaces'] is r1:
            for r3 in gc.get_referrers(r2):
              if isinstance(r3,Tensor) and r3.__dict__ is r2:
                tref = r3
                break
            if tref is not None:
              break
        if tref is None:
          continue
        # Find instances of V1 in tref._spaces
        i0 = 0
        newspaces = list(tref._spaces)
        for ii in range(r1.count(V1)):
          # Next instance
          it = r1.index(V1,i0)
          tref._dspace[tref._idxs[it]] = V0
          newspaces[it] = V0
        tref._spaces = tuple(newspaces)


class FusionPrimitive:
  """Object containing information about fusion operation"""
  def __init__(self, vs_in, v_out=None, idx_in=None):
    vs_in = tuple(vs_in)
    if isinstance(vs_in[0], links.VSpace):
      self._spaces_in = vs_in
      self._idxs = idx_in
    else:
      assert isinstance(v_out, Tensor)
      self._spaces_in = tuple(v_out._dspace[ll] for ll in vs_in)
      self._idxs = vs_in
      v_out = idx_in
    self._rank = len(vs_in)
    if v_out is None:
      if self._rank == 1:
        v_out = self._spaces_in[0]
      d = functools.reduce(int.__mul__, (vi.dim for vi in self._spaces_in))
      v_out = links.VSpace(d)
    self._out = v_out

  _tensor_class = Tensor

  @property
  def rank(self):
    return self._rank

  @property
  def dim(self):
    return self._out.dim
    
  def conj(self):
    return self.__class__([~v for v in self._spaces_in],~self._out,self._idxs)

  @property
  def V(self):
    return self._out

  @property
  def tensorclass(self):
    return self.__class__._tensor_class

  def matchtensor(self, T, idx2=None, c=False):
    """Compare with relevant indices of tensor T
    idx2, if provided, gives the indices to be fused, in the relevant order
    otherwise matches with self._idxs"""
    if idx2 is None:
      if self._idxs is None:
        raise ValueError('Must provide index list if not set internally')
      idx2 = self._idxs
    for i in range(self._rank):
      ll = idx2[i]
      if ll not in T:
        raise KeyError('Index %s provided does not belong to tensor'%ll)
      if not self._spaces_in[i].cmp(T._dspace[ll],c):
        raise ValueError('Index %s does not match fusion primitive'%ll)

  def cmp(self, p2, idx1=None, idx2=None, c=False):
    """Compare with indices of fusion primitive (given matching names)
    idx1 and idx2, if provided, take the place of self._idxs, p2._idxs
    Does not compare fused index"""
    assert isinstance(p2, FusionPrimitive)
    if self.rank != p2.rank:
      return False
    if idx1 is None:
      idx1 = self._idxs
    elif len(idx1) != self.rank:
      raise ValueError('Number of provided does not match rank')
    if idx2 is None:
      idx2 = p2._idxs
    elif len(idx2) != p2.rank:
      raise ValueError('Number of provided does not match rank')
    if idx1 is None or idx2 is None:
      raise ValueError('Must provide index list if not set internally')
    if set(idx1) != set(idx2):
      return False
    for i in range(self.rank):
      i2 = idx2.index(idx1[i])
      if not self._spaces_in[i].cmp(p2._spaces_in[i2],c):
        return False
    return True

  def zeros_like(self, idxs=None):
    """Create a zero tensor from the elements of the tensor as given"""
    if idxs is None:
      idxs = self._idxs
    shape = tuple(v.dim for v in self._spaces_in)
    return Tensor(np.zeros(shape,dtype=config.FIELD), idxs, self._spaces_in)

  def empty_like(self, idxs=None):
    """Create an empty tensor from the elements of the tensor as given"""
    if idxs is None:
      idxs = self._idxs
    shape = tuple(v.dim for v in self._spaces_in)
    return Tensor(np.empty(shape,dtype=config.FIELD), idxs, self._spaces_in)
 

class TensorTransposedView(Tensor):
  """View for tensor with indices renamed"""
  def __init__(self, T0, idxdict):
    self.__tensor = T0
    self.__idxdict = dict(idxdict)
    self._idxs = tuple(idxdict[l] for l in T0._idxs)
    self.shape = dictproperty(self._Tensor__shape_get, None, self._Tensor__shape_has, self._Tensor__shape_copy)

  @property
  def _T(self):
    return self.__tensor._T

  @property
  def _spaces(self):
    return self.__tensor._spaces

  @property
  def _dspace(self):
    return {self._idxs[i]:self.__tensor._spaces[i] for i in range(len(self._idxs))}

  def __copy__(self):
    return self._tensorfactory(copy(self._T), self._idxs, self._spaces)

  def __getstate__(self):
    return self.__tensor, self.__idxdict


class dictproperty:
  """Property-type 'descriptor' for accessing dictionary-esque properties
  For use particularly in SubNetwork attributes that refer to the parent
  network"""
  
  def __init__(self, fgetitem, fsetitem, fhas, fget, *argx):
    if callable(fhas):
      self.fhas = fhas
      self.__keys = None
    else:
      self.fhas = None
      self.__keys = fhas
    self.fgetitem = fgetitem
    self.fcopy = fget
    self.fsetitem = fsetitem
    self.argx = argx
    self.__doc__ = fgetitem.__doc__

  def __getitem__(self, key):
    if key not in self:
      raise AttributeError('%s missing'%key)
    return self.fgetitem(key, *self.argx)

  def __setitem__(self, key, value):
    if self.fsetitem is None:
      raise AttributeError('Cannot set item in inherited dictionary')
    if key not in self:
      raise KeyError('%s missing'%key)
    return self.fsetitem(key, value, *self.argx)

  def __contains__(self, key):
    if self.fhas is None:
      return key in self.__keys
    else:
      return self.fhas(key, *self.argx)

  def __set__(self, obj, value):
    raise AttributeError('Descriptor cannot be replaced')

  def __copy__(self):
    return self.fcopy(*self.argx)

  def __deepcopy__(self):
    cpy = self.__copy__()
    if cpy.__class__ != self.__class__:
      return deepcopy(cpy)

  def keys(self):
    if self.__keys is not None:
      return set(self.__keys)
    else:
      return self.__copy__().keys()

  def values(self):
    return self.__copy__().values()

  def __len__(self):
    if self.__keys is not None:
      return len(self.__keys)
    else:
      return len(self.__copy__())

  def __iter__(self):
    if self.__keys is not None:
      return iter(self.__keys)
    else:
      return iter(self.__copy__())

  def items(self):
    return self.__copy__().items()

