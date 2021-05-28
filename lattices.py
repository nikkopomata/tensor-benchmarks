# Abstract objects which need to be defined on lattices (infinite graphs)
from abc import ABC
from .networks import Network
from .tensors import dictproperty
from numpy import np
import collections

class PeriodicNetwork(Network,ABC):
  """Abstract class representing tensor network defined on periodic lattice
  (generically subclasses will not know whether the lattice is infinite or
  finite with periodic boundary conditions; objects may be re-applied in
  varying cases)
  Concrete subclasses should have a defined "simple" unit cell, which
  instances may enlarge into a full unit cell"""

  def __init__(self, tensors, unitdict, conjdict):
    assert hasattr(self,'_dimension')
    self._tlist = list(tensors)
    self._tunitcell = dict(unitdict)
    self._cunitcell = dict(conjdict)
    # Complete conjugacy dictionary - default is false
    for s in unitdict:
      if s not in conjdict:
        self._cunitcell[s] = False
    self._tset = [set() for t in tensors]
    for t,i in tensordict.items():
      self._tset[i].add(t)
    self._tdict = dictproperty(self.__tdictget,self.__tdictset,self.issite,None)
    self._conj = dictproperty(self.__cdictget,self.__cdictset,self.issite,None)
    self._tbonds = dictproperty(self.bondadjacency,None,self.issite,None)
    self._tout = dictproperty(self.getexternalindices,None,self.issite,None)
    self._out = dictproperty(self.externalindex,None,self.isexternalindex,None)
    self._tchained = dictproperty(self.__tchainget,None,self.issite,None)
    self._tree = None
    self._cache = False

  def verify_compatibility(self):
    # Verify indices in tensors of user-assigned instances as
    # compatible with structure of network
    tensors = self._tlist
    for i in range(len(tensors)):
      ts = set(self._tset[i])
      if not ts:
        continue
      T0 = tensors[i]
      s0 = ts.pop()
      ls = set(self.__tchainget(s0).keys())
      for s1 in ts:
        if ls != set(self.__tchainget(s1).keys()):
          raise ValueError('Incompatible sites %s, %s identified'%(s0,s1))
      idxs = T0.idxset
      if ls != idxs:
        if ls-idxs:
          raise ValueError('Tensor at site %s must have index "%s"' \
            %((ls-idxs).pop()))
        else:
          raise ValueError('Tensor at site %s has extranneous index "%s"' \
            %((idxs-ls).pop()))
      lpaired = set()
      for s0 in self._tset[i]:
        for l0,(s1a,l1) in self.bondadjacency(s0):
          s1 = self.unitsite(s1a)
          c = self._cunitcell[s0]^self._cunitcell[s1]
          i1 = self._tunitcell[s1]
          if (l0,i1,l1,c) not in lpaired:
            if not T0._dspace[l0].cmp(tensors[i1]._dspace[l1], c):
              raise ValueError('Bond %s.%s-%s.%s cannot be contracted' \
                %(s0,l0,s1a,l1))
            lpaired.add((l0,i1,l1,c))

  @property
  def unitcell:
    """Provide set of sites in unit cell of lattice"""
    return set(self._tunitcell.keys())

  @abstractmethod
  @classmethod
  def issite(cls, site):
    """Determine whether index indexes a valid site"""
    pass

  @abstractmethod
  def unitsite(self, site):
    """Return properly-indexed site within the unit cell"""
    pass

  @abstractmethod
  @classmethod
  def bondadjacency(cls, site):
    """Adjacency information, in terms of index:(site,index) dictionary"""
    pass

  @abstractmethod
  @classmethod
  def externalindex(cls, idx):
    """Returns tensor:original index if applicable"""
    pass

  @abstractmethod
  @classmethod
  def isexternalindex(cls, idx):
    pass

  @abstractmethod
  @classmethod
  def getexternalindices(cls, site):
    """Returns original index:external indices for given site""" 
    pass

  def __tdictget(self, site):
    return self._tunitcell[self.unitsite(site)]

  def __tdictset(self, site, value):
    self._tunitcell[self.unitsite(site)] = value

  def __cdictget(self, site):
    return self._cunitcell[self.unitsite(site)]

  def __cdictset(self, site, value):
    self._cunitcell[self.unitsite(site)] = value

  def __bondedget(self, site):
    return set(t for t,i in self.bondadjacency(site))

  def __tchainget(self, site):
    d = self.getexternalindices(site)
    d.update(self.bondadjacency(site))
    return d

  def updateto(self, *sites, c=False, rel=False):
    """Update tensor(s) sites[:-1] by identifying as sites[-1]"""
    for site in sites:
      if not self.issite(site):
        raise ValueError('arguments must identify site on lattice')
    s1 = self.unitsite(sites[-1])
    i1 = self._tunitcell[s1]
    c1 = self._cunitcell[s1]
    for s0a in sites[:-1]:
      s0 = self.unitsite(s0a)
      c0 = c
      if rel:
        c0 ^= self._cunitcell[s0] 
      i0 = self._tunitcell[t0]
      self._tlist[s0].matches(self._tlist[s1],c0^c1^self._cunitcell[s0])
      self._updateto(s0, i1, c0^c1)

  def replaceall(self, sitedict, *args, strict=False, c=False):
    """Replace tensors at site given with values of dictionary, for every node
      that points to that tensor
      (optionally series of arguments site, tensor, site, tensor, ...)
    if strict is True, compares exactly with vector spaces of current tensor;
      if False, only raise error if a conflict is caused
    c, if boolean, represents global conjugacy information;
      if a dictionary, represents conjugacy data for tensors indicated"""
    idxold = []
    conjs = []
    if isinstance(sitedict,dict):
      sites = list(sitedict.keys())
      tensors = list(sitedict.values())
    else:
      if len(args)%2 == 0:
        raise ValueError('Tensors not provided as dictionary or alternating '
            'list form')
      sites = [sitedict]+args[1::2]
      tensors = args[0::2]
    nten = len(sites)
    for ii in range(nten):
      T = tensors[ii]
      s0 = sites[ii]
      if not self.issite(s0):
        raise ValueError(str(s0)+'does not identify site on lattice')
      s0 = self.unitsite(s0)
      if isinstance(c,dict):
        conj = (s0 in c and c[s0])
      else:
        conj = bool(c)
      idx = self._tunitcell[s0]
      if idx in idxold:
        ifirst = idxold.index(idx)
        raise ValueError('Tensor at site %s already referenced as %s' \
          % (sites[ii], sites[ifirst]))
      idxold.append(idx)
      conj ^= self._cunitcell[s0]
      if strict:
        # Just check against existing tensor
        self._tlist[idx].matches(T, conj)
      conjs.append(conj)
    if not strict:
      # Test bonds individually
      for ii in range(nten):
        matchold = False
        matchnew = set()
        T = tensors[ii]
        si = idxold[ii]
        for l in T._idxs:
          for s0 in self._tset[ti]:
            sbond = self.bondadjacency(s0)
            if l in sbond:
              s1,l1 = sbond
              s1 = self.unitsite(s1)
              i1 = self._tunitcell(s1)
              if i1 in idxold:
                if i1 < ti or (i1 == si and l == l1 and conjs[ii]):
                  # Will have already been checked
                  continue
                c = self._cunitcell[s0]^self._cunitcell[s1]
                c ^= conjs[idxold.index(i1)]
                matchnew.add((i1, l1, c))
              else:
                matchold = True
          if matchold:
            if not T._dspace[l].cmp(self._tlist[ti]._dspace[l], conjs[ii]):
              raise ValueError('Index %s.%s changed improperly'%(ts[ti],l))
          for i1,l1,c in matchnew:
            if not T._dspace[l].cmp(tensors[i1]._dspace[l1], not c):
              s1 = self.bondadjacency(s0)[l1][0]
              raise ValueError('Bond %s.%s-%s.%s changed incompatibly' \
                % (s0,l,s1,l1))
    for ii in range(nten):
      # Change the network
      si = idxold[ii]
      self._tlist[si] = T
      if conjs[ii]:
        for s in self._tset[si]:
          self._cunitcell[s] = not self._cunitcell[s]

  def contract(self, **kw):
    raise NotImplementedError()

  def derive(self, *args):
    raise NotImplementedError()
  
  def freeindices(self, **kw):
    raise NotImplementedError()

  def setorder(self, p):
    raise NotImplementedError()
  
  def __copy__(self):
    cpy = self.__class__(self._tlist, self._tunitcell, self._cunitcell)
    return cpy

  def __getstate__(self):
    """Pickling procedure for when _dimension is the only necessary
    supplement to determine structure"""
    return self._dimension, self._tlist, self._tunitcell, self._cunitcell

  def __setstate__(self, initargs):
    self._dimension = initargs[0]
    PeriodicNetwork.__init__(self, *(initargs[1:]))

  @property
  def size(self):
    return self._dimension

  def conj(self):
    cpy = self.__copy__()
    for s,c in self._cunitcell.items():
      cpy._conj[s] = not c
    return cpy

  def optimize(self):
    raise NotImplementedError()

  def contraction_expense(self, *a, **kw):
    return np.inf

  def memexpense(self, *a, **kw):
    return np.inf


class PartitionFunction(PeriodicNetwork):
  """Abstract class representing partition function on an infinite lattice"""
  
  def freeindices(self, **kw):
    return ()

  def externalindex(self, idx):
    raise KeyError('Partition function has no external indices')

  def isexternalindex(self, idx):
    return False

  def getexternalindices(self, site):
    return {}
