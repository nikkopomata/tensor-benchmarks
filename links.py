"""Basic class structure for handling `vector space` primitives for
identifying indices with each other"""
from abc import ABC, abstractmethod
import collections
import weakref,uuid

spacetype = 'gen'

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

  def trim(self, boolidx):
    assert len(boolidx) == self._dimension
    return self.__class__(sum(boolidx))


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
    if spacetype == 'weak':
      return VectorSpaceTrackedWeak.__new__(VectorSpaceTrackedWeak, *args, **kw_args)
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

class VectorSpaceTrackedWeak(VectorSpaceTracked):
  """Represent vector spaces in a way capable of sophisticated identification
  between spaces"""
  # TODO compatibility with SumRepresentation?

  VSided = weakref.WeakValueDictionary()

  def __new__(cls, *args, **kw_args):
    if 'vsid' in kw_args:
      vsid = kw_args.pop('vsid')
      dualid = kw_args.pop('dual')
      dim = args[0]
      if vsid in cls.VSided:
        if cls.VSided[vsid].__dualid != dualid:
          raise ValueError('vector-space ID of dual changed between pickling'
            ' (%s->%s)'%(cls.VSided[vsid].__dualid,dualid))
        if cls.VSided[vsid]._dimension != dim:
          raise ValueError('dimensions indicate vector-space collision')
        return cls.VSided[vsid]
      else:
        obj = super(VectorSpaceTracked,cls).__new__(cls, *args[1:], **kw_args)
        obj.__init__(dim)
        obj.__vsid = vsid
        obj.__dualid = dualid
        cls.VSided[vsid] = obj
        if dualid in cls.VSided:
          dual = cls.VSided[dualid]
          obj.__dual = weakref.ref(dual)
          dual.__dual = weakref.ref(obj)
          assert dual.__dualid == vsid
        #assert obj.__dual.__vsid == dualid
        #assert obj.__dual.__dual is obj
        #print(vsid,dualid)
        return obj
    return super(VectorSpaceTracked,cls).__new__(cls, *args[1:], **kw_args)

  def __init__(self, dim, dual=None):
    self._dimension = int(dim)
    if dual is not None:
      self.__dual = weakref.ref(dual)
    else:
      self.__dual = dual
    self.__vsid = None
    self.__dualid = None

  def dual(self):
    if self.__dual is None or self.__dual() is None:
      dual = VectorSpaceTrackedWeak(self._dimension)
      self.__dual = weakref.ref(dual)
      dual.__dual = weakref.ref(self)
      if self.__vsid is not None:
        dual.__dualid = self.__vsid
      if self.__dualid is not None:
        dual.__vsid = self.__dualid
      return dual
    else:
      return self.__dual()

  def __eq__(self,W):
    return self is W

  def __xor__(self,W):
    return self.__dual is not None and self.__dual() is W

  def __gen_id(self):
    from numpy import random
    vsid = random.randint(2**30)
    while vsid in self.__class__.VSided:
      vsid = random.randint(2**30)
    self.__vsid = vsid
    if self.__dual is not None and self.__dual() is not None:
      self.dual().__dualid = vsid
    self.__class__.VSided[vsid] = self

  def __getnewargs_ex__(self):
    if self.__vsid is None:
      self.__gen_id()
    dual = self.dual()
    if dual.__vsid is None:
      dual.__gen_id()
    return ((self._dimension,),{'vsid':self.__vsid, 'dual':self.__dualid})

  def __getstate__(self):
    return {}

  def __setstate__(self,stt):
    pass

class VectorSpaceIDed(VSAbstract):
  """Use UUID instead of direct tracking
  Does not require strict uniqueness of object
  Maintains reference to dual for convenience (but does not pickle it)"""
  # TODO is the not-pickling necessary?
  # TODO special unpickler?
  def __init__(self, dim, vsid=None):
    self._dimension = int(dim)
    if vsid is None:
      self.__vsid = uuid.uuid1()
    else:
      self.__vsid = vsid
    self.__dual = None
    self.__dualid = None

  def dual(self):
    if self.__dual is None:
      if self.__dualid is None:
        self.__dualid = uuid.uuid1()
      self.__dual = VectorSpaceIDed(self._dimension, vsid=self.__dualid)
      self.__dual.__dual = self
      self.__dual.__dualid = self.__vsid
    return self.__dual

  def __eq__(self, W):
    return isinstance(W, self.__class__) and self.__vsid == W.__vsid

  def __xor__(self, W):
    return isinstance(W, self.__class__) and self.__dualid == W.__vsid

  def __getstate__(self):
    state = self.__dict__.copy()
    del state['_VectorSpaceIDed__dual']
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self.__dual = None

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

