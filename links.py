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
    self._dimension = dim

  def dual(self):
    return self

  def __eq__(self, W):
    return self._dimension == W._dimension

  def __xor__(self, W):
    return self._dimension == W._dimension


class VectorSpaceTracked(VSAbstract):
  """Represent vector spaces in a way capable of sophisticated identification
  between spaces"""

  def __init__(self, dim, dual=None):
    self._dimension = dim
    self.__dual = dual

  def dual(self):
    if self.__dual is None:
      self.__dual = VectorSpaceTracked(self._dimension)
      self.__dual.__dual = self
    return self.__dual

  def __eq__(self,W):
    return self is W

  def __xor__(self,W):
    return self.__dual is W


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

