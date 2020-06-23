"""Basic class structure for handling `vector space` primitives for
identifying indices with each other"""
from abc import ABC, abstractmethod
import collections

class VSAbstract(ABC):
  @property
  def dim(self):
    return self._dimension

  @abstractmethod
  def dual(self):
    pass

  def __invert__(self):
    return self.dual()

  @abstractmethod
  def __eq__(self, W):
    """Equivalent (i.e. contractible with dual)"""
    pass

  def __ne__(self, W):
    return not self.__eq__(W)

  @abstractmethod
  def __xor__(self, W):
    """Contractible with (equivalent to dual) - returns boolean"""
    pass

  def cmp(self, W, c):
    """Ternary comparison - c?V^W:V==W"""
    if c:
      return self ^ W
    else:
      return self == W


class VectorSpace(VSAbstract):
  """Basic class to represent vector spaces. Only attribute is dim"""
  
  def __init__(self, dim):
    self._dimension = int(dim)

  def dual(self):
    return self

  def __eq__(self, W):
    return self._dimension == W._dimension

  def __xor__(self, W):
    return self._dimension == W._dimension


class VectorSpaceTracked(VSAbstract):
  """Represent vector spaces in a way capable of sophisticated identification
  between spaces"""

  VSided = {}

  def __new__(cls, *args, **kw_args):
    if 'vsid' in kw_args:
      vsid = kw_args.pop('vsid')
      dualid = kw_args.pop('dual')
      #print('initializing',vsid,dualid)
      dim = args[0]
      if vsid in cls.VSided:
        if cls.VSided[vsid].__dual.__vsid != dualid:
          raise ValueError('vector-space ID of dual changed between pickling'
            ' (%s->%s)'%(cls.VSided[vsid].__dual.__vsid,dualid))
        if cls.VSided[vsid]._dimension != dim:
          raise ValueError('dimensions indicate vector-space collision')
        return cls.VSided[vsid]
      else:
        obj = super(VectorSpaceTracked,cls).__new__(cls, *args[1:], **kw_args)
        obj.__init__(dim)
        obj.__vsid = vsid
        cls.VSided[vsid] = obj
        obj.dual().__vsid = dualid
        cls.VSided[dualid] = obj.__dual
        #assert obj.__dual.__vsid == dualid
        #assert obj.__dual.__dual is obj
        #print(vsid,dualid)
        return obj
    return super(VectorSpaceTracked,cls).__new__(cls, *args[1:], **kw_args)

  def __init__(self, dim, dual=None):
    self._dimension = int(dim)
    self.__dual = dual
    self.__vsid = None

  def dual(self):
    if self.__dual is None:
      self.__dual = VectorSpaceTracked(self._dimension)
      self.__dual.__dual = self
    return self.__dual

  def __eq__(self,W):
    return self is W

  def __xor__(self,W):
    return self.__dual is W

  def __gen_id(self):
    from numpy import random
    vsid = random.randint(2**30)
    while vsid in self.__class__.VSided:
      vsid = random.randint(2**30)
    self.__vsid = vsid
    self.__class__.VSided[vsid] = self

  def __getnewargs_ex__(self):
    if self.__vsid is None:
      self.__gen_id()
    if self.dual().__vsid is None:
      self.__dual.__gen_id()
    return ((self._dimension,),{'vsid':self.__vsid, 'dual':self.__dual.__vsid})


class TensorIndexDict:
  """Property-type descriptor for accessing information contained in VSpace
  objects
  Largely borrowed from python-course.eu property implementation
  Not usable? """

  def __init__(self, fgetitem, fget=None):
    self.fgetitem = fgetitem
    self.fget = fget
    self.__doc__ = fgetitem.__doc__

  def __get__(self, obj, objtype=None):
    if obj is None:
      return self
    if self.fget is None:
      raise AttributeError('unreadable attribute')
    else:
      return self.fget(obj)

  def __getitem__(self, obj, key):
    return self.fgetitem(obj, key)

  def __set__(self, obj, value):
    raise AttributeError('TensorIndexDict descriptor cannot be set')

  def __delete__(self, obj):
    raise AttributeError('TensorIndexDict descriptor cannot be deleted')

  def getter(self, fget):
    return type(self)(self.fgetitem, self.fget)

