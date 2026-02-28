from fractions import Fraction
import numbers
import numpy as np
import numpy.lib.mixins
from math import gcd,lcm
import functools
import itertools

no_strict_checks = False
# Helper functions

def sqrt_reduce(x):
  """Separate integer or fraction x into perfect-square and not
  Returns a,b such that x = a**2 * b"""
  if isinstance(x,RationalArray):
    assert x.size == 1
    x = x.reshape(-1)[0]
  if isinstance(x,Fraction):
    pout,prad = sqrt_reduce(x.numerator)
    qout,qrad = sqrt_reduce(x.denominator)
    return Fraction(pout,qout), Fraction(prad,qrad)
  outside = 1
  radicand = 1
  from sympy import factorint
  for p,mult in factorint(x).items():
    if mult > 1:
      outside *= p**(mult//2)
    if mult % 2:
      radicand *= p
  return outside, radicand

def rational_gramschmidt_implicit(gram):
  # Rational "implicit" (from Gramian) Gram-Schmidt process, i.e. takes
  # array of Fractions & returns same plus list of norm-squared
  N = gram.shape[0]
  cob = rzeros((N,1))
  i0 = 0
  while gram[i0,i0] == 0:
    i0 += 1
    if i0 == N:
      raise ValueError('Expected nontrivial Gram matrix')
  norm = gram[i0,i0]
  nout,nrad = sqrt_reduce(norm)
  cob[i0,0] = Fraction(1,nout)
  norms = [nrad]
  ngram = 1
  # Left-inverse of change-of-basis matrix, for incorporating
  # division by norm-squared
  linv = rzeros((1,N))
  linv[0,i0] = Fraction(1,nout*nrad)
  for i in range(i0+1,N):
    if gram[i,i] == 0:
      continue
    coeff = rzeros((N,1))
    coeff[i] = 1
    coeff[:i,0] = -cob[:i,:].dot(linv[:,:i].dot(gram[:i,i]))
    norm = gram[i,i] + gram[i,:i].dot(coeff[:i,0])
    if norm == 0:
      continue
    nout,nrad = sqrt_reduce(norm)
    coeff /= nout
    coeff.rereduce()
    norms.append(nrad)
    cob = np.hstack((cob,coeff))
    linv = np.vstack((linv,coeff.T/nrad))
    ngram += 1
  norms = rarray(norms)
  cob.rereduce()
  assert no_strict_checks or np.all(linv*norms[:,None] == cob.T)
  assert no_strict_checks or np.all(linv.dot(gram.dot(cob)) == reye(ngram))
  return cob, norms

def complete_basis(vecs, normsq, target_normsq):
  # Target space has inner product given by diag(target_normsq);
  # vecs are orthogonal with norms^2 normsq
  # Returns a basis completing vecs, their norms^2

  # Change-of-basis matrix and its right-inverse
  cob = rarray(vecs)
  rinv = cob.T.copy()
  cob.raise_index(normsq,0,True)
  cob.lower_index(target_normsq,1,True)

  n,N = cob.shape
  assert n < N
  assert no_strict_checks or np.all(cob.dot(rinv) == reye(n))
  newvecs = []
  newnorms = []
  i0 = 0
  while len(newvecs) < N-n:
    vec = rzeros((N,))
    vec = -rinv.dot(cob[:,i0])
    vec[i0] += 1
    norm = vec.dot(vec*target_normsq)
    if norm:
      nout,nrad = sqrt_reduce(norm)
      vec /= nout
      newnorms.append(nrad)
      newvecs.append(vec)
      cob = np.vstack((cob,vec.T.lower_index(target_normsq,1)/nrad))
      rinv = np.hstack((rinv,vec[:,None]))
      assert no_strict_checks or np.all(cob.dot(rinv) == reye(n+len(newvecs)))
    i0 += 1
  assert no_strict_checks or np.all(rinv.dot(cob) == reye(N))
  return np.stack(newvecs,axis=0),rarray(newnorms)

def invert(M):
  reducer = rarray(M)
  N = reducer.shape[0]
  solver = reye(N)
  for k in range(N):
    for i in range(k):
      factor = reducer[k,i]
      solver[k] -= factor*solver[i]
      reducer[k] -= factor*reducer[i]
    solver.rereduce()
    reducer.rereduce()
    solver[k] /= reducer[k,k]
    reducer[k] /= reducer[k,k]
  for k in range(N):
    solver.rereduce()
    reducer.rereduce()
    for i in range(k+1,N):
      factor = reducer[k,i]
      solver[k] -= factor*solver[i]
      reducer[k] -= factor*reducer[i]
  return solver.rereduce()

def _integralize(arr):
  # Convert array into form p/q * integer array
  iarr = arr.copy()
  farr = iarr.reshape(-1)
  assert len(farr)
  p,q = _DC_integralize(farr,len(farr))
  if p == 0:
    return Fraction(1),np.zeros_like(iarr)
  else:
    return Fraction(p,q),iarr

def _DC_integralize(l,s):
  # Recursive divide-and-conquer approach to integralizing flat array l of
  # length s (in-place)
  if s == 1:
    f = l[0]
    if isinstance(f,int) or isinstance(f,np.integer):
      p,q = int(f),1
    else:
      if not isinstance(f,Fraction):
        raise TypeError('To convert to rational array, elements of array-like '
          'must be integer or rational, not %s'%type(f))
      p = l[0].numerator
      q = l[0].denominator
    if p < 0:
      l[0] = -1
      return -p,q
    else:
      l[0] = 1
      return p,q
  else:
    s1 = s//2
    l1 = l[:s1]
    l2 = l[s1:]
    p1,q1 = _DC_integralize(l1,s1)
    p2,q2 = _DC_integralize(l2,s-s1)
    p = gcd(p1,p2)
    if p != p1:
      l1 *= p1//p
    if p != p2:
      l2 *= p2//p
    q = lcm(q1,q2)
    if q != q1:
      l1 *= q//q1
    if q != q2:
      l2 *= q//q2
    return p,q

_HANDLED_FUNCTIONS = {}

# ufuncs comparing elements 
_COMPARE_UFUNCS = {np.equal,np.not_equal,np.less,np.less_equal,
  np.greater,np.greater_equal}
# binary ufuncs that can act on "compatibilized" (rescaled) integer arrays
_INT_COMPAT_UFUNCS = {np.add,np.subtract,np.maximum,np.minimum,np.amin,np.amax,np.fmax,np.fmin,np.remainder,np.divmod}
# binary ufuncs that require rescaling first 
_COMPAT_UFUNCS = _INT_COMPAT_UFUNCS | _COMPARE_UFUNCS
# ufuncs that return bools dependent only on integer arrays
_LOGICAL_UFUNCS = {np.logical_not,np.logical_and,np.logical_or,np.logical_xor,np.isfinite,np.isinf,np.isnan}
# unary ufuncs where ufunc action reduces to action on integer array
# (last few will take special treatment)
_INT_UNARY_UFUNCS = {np.absolute,np.conjugate,np.negative,np.positive,np.sign,np.square}
# ufuncs (mostly transcendental) that are best performed 
# by casting to floating-point
_FLOAT_UFUNCS = {np.sqrt,np.cbrt,np.hypot, np.deg2rad, np.modf,np.fabs,
  np.power,np.float_power, np.exp,np.expm1,np.exp2,
  np.log,np.log10,np.log1p,np.log2,np.logaddexp,np.logaddexp2,
  np.cos,np.sin,np.tan, np.arccos,np.arcsin,np.arctan,np.arctan2,
  np.cosh,np.arccosh,np.sinh,np.arcsinh,np.tanh,np.arctanh}
# TODO exp2,log2,power? (all rational-compatible in some cases)
# TODO modular-arithmetic-type ufuncs: divmod,trunc,floor,ceil
# TODO other cases that would require special attention for rationals:
# frexp/ldexp, copysign, clip, heaviside

def _compatibilize_inputs(*args, outputs=None, blanks=None,
    inplace_ok=False):
  """Mutually "compatibilize" with potentially more than one argument"""
  #print('Compatibilize %d arrays (%d outputs, %d blanks, %s)'%(len(args),len(outputs) if outputs else 0,len(blanks) if blanks else 0, 'inplace' if inplace_ok else ''))
  isdup = len(args)*[False]
  if outputs:
    outset = []
    for arr in outputs:
      if not any(arr._defaultbase is arr2 for arr2 in outset):
        outset.append(arr._defaultbase)
    if len(outset) == 1:
      p0,q0 = outset[0].prefactor.as_integer_ratio()
    else:
      # Need to compatibilize with each other
      ps = [arr.numerator for arr in outset]
      qs = [arr.denominator for arr in outset]
      p0 = gcd(*ps)
      q0 = lcm(*ps)
      new_factor = Fraction(p0,q0)
    for i,arr in enumerate(args):
      if isinstance(arr,RationalArray):
        if any(arr._defaultbase is arr2 for arr2 in outset):
          # Flag that these will be already covered
          isdup[i] = True
  else:
    outset = []
    p0 = 1
    q0 = 1
  r0 = Fraction(p0,q0)
  iargs = []
  ibase = []
  rbase = []
  rnew = {}
  if outputs:
    rfactors = [r0]
  else:
    rfactors = []
  scalars = {}
  # Collect arrays that need to be compatibilized
  for i,arg in enumerate(args):
    if isdup[i]:
      iargs.append(arg._int_array)
      continue
    elif isinstance(arg,RationalArray):
      iargs.append(arg._int_array)
      for i2,(rarg,idxs) in enumerate(rbase):
        if arg._defaultbase is rarg:
          idxs.add(i)
          isdup[i] = True
          break
      if isdup[i]:
        continue
      else:
        rfactors.append(arg.prefactor)
        rbase.append((arg._defaultbase,{i}))
    elif isinstance(arg,np.ndarray):
      if arg.dtype.kind == 'i' or (arg.dtype == np.dtype('O') and all(isinstance(ii,int) for ii in arg.reshape(-1))):
        iargs.append(arg)
        for i2,(iarg,idxs) in enumerate(ibase):
          if arg is iarg or arg.base is iarg:
            idxs.add(i)
            isdup[i] = True
            break
        if isdup[i]:
          continue
        else:
          ibase.append(((arg if arg.base is None else arg.base), {i}))
      else:
        arg = rarray(arg)
        rnew[i] = arg
        rfactors.append(arg.prefactor)
        iargs.append(arg._int_array)
    elif isinstance(arg,numbers.Number):
      if arg < 0:
        iargs.append(-1)
      elif arg == 0:
        iargs.append(0)
        # Don't want to add to factors
        continue
      else:
        iargs.append(1)
      if isinstance(arg,(int,np.integer)):
        arg = Fraction(arg,1)
      elif not isinstance(arg,Fraction):
        raise TypeError('Cannot compatibilize RationalArrays with number type %s'%type(arg))
      rfactors.append(abs(arg))
      scalars[i] = arg
    else:
      raise TypeError('Cannot compatibilize RationalArrays with type %s'%type(arg))
  # Now determine prefactor & compatibilize
  if ibase:
    # Numerator forced to 1
    p = 1
  else:
    p = gcd(*[r.numerator for r in rfactors])
  q = lcm(*[r.denominator for r in rfactors])
  r = Fraction(p,q)
  if r != r0:
    # Start with outputs
    for arr in outset:
      arr._compatibilize(r)
  # Next integer arrays
  for arr,idxs in ibase:
    # If numpy integer type, convert
    if arr.dtype.kind != 'O':
      for i in idxs:
        iargs[i] = q*iargs[i].astype(object)
        # TODO may need to convert int64->int?
    elif inplace_ok:
      arr *= q
    elif q != 1:
      for i in idxs:
        iargs[i] = q*iargs[i]
  # Next rational arrays
  for arr,idxs in rbase:
    if r != arr.prefactor:
      if inplace_ok:
        arr._compatibilize(r)
      else:
        for i in idxs:
          iargs[i] = int(arr.prefactor/r) * iargs[i]
  for arr in rnew.values():
    arr._compatibilize(r)
  # And scalars
  for i,x in scalars.items():
    iargs[i] = int(x/r)
  # Finally handle "blanks"
  if blanks:
    for arr in blanks:
      assert arr._fullview
      arr._defaultbase._RationalArray__prefactor = r
  return r, iargs

def _compatibilize_inputs0(*args, outputs=None, blanks=None,
    inplace_ok=False):
  """Mutually "compatibilize" with potentially more than one argument"""
  isdup = len(args)*[False]
  if outputs:
    outset = []
    for arr in outputs:
      if not any(arr._defaultbase is arr2 for arr2 in outset):
        outset.append(arr._defaultbase)
    if len(outset) == 1:
      p0,q0 = outset[0].prefactor.as_integer_ratio()
    else:
      # Need to compatibilize with each other
      ps = [arr.numerator for arr in outset]
      qs = [arr.denominator for arr in outset]
      p0 = gcd(*ps)
      q0 = lcm(*ps)
      new_factor = Fraction(p0,q0)
      for arr in outset:
        arr._compatibilize(new_factor)
    for i,arr in enumerate(args):
      if isinstance(arr,RationalArray):
        if any(arr._defaultbase is arr2 for arr2 in outset):
          # Flag that these will be already covered
          isdup[i] = True
  else:
    p0 = 1
    q0 = 1
  iargs = []
  ibase = []
  r0 = Fraction(p0,q0)
  r1 = r0
  for i,arg in enumerate(args):
    if isdup[i] or (inplace_ok and isinstance(arg,RationalArray) and \
        any(arg._defaultbase._int_array is iarg for iarg in ibase)):
      # Already compatibilized
      iargs.append(arg._int_array)
      isdup[i] = True
      continue
    elif inplace_ok and isinstance(arg,np.ndarray) and \
        any(arg is iarg or arg.base is iarg for iarg in ibase):
      # Likewise
      iargs.append(arg)
      isdup[i] = True
      continue
    r,iarr = _scalar_compatibilize(r1,arg,inplace_ok)
    if iargs and r1 != r:
      # Coefficient fraction has changed - need to rescale previous arrays
      factor = int(r1/r)
      for i2 in range(len(iargs)):
        if isdup[i2]:
          continue
        if not inplace_ok and iargs[i2] is args[i2]._int_array:
          # Avoided new allocation, but need one now
          for i3 in range(len(ibase)):
            if iargs[i2] is ibase[i3] or iargs[i2].base is ibase[i3]:
              ibase.pop(i3)
          iargs[i2] = iargs[i2] * factor
          # TODO this code block is probably rarely-enough encountered to want to test explicitly
        else:
          iargs[i2] *= factor
    r1 = r
    iargs.append(iarr)
    if iarr.base is None:
      ibase.append(iarr)
    else:
      ibase.append(iarr.base)
  if blanks:
    for arr in blanks:
      assert arr._fullview
      arr._defaultbase._RationalArray__prefactor = r1
  if outputs:
    # May need to re-compatibilize
    if r0 != r1:
      for arr in outset:
        arr._compatibilize(r1)
  return r1, iargs

def _scalar_compatibilize(r, array_or_rational, inplace_ok=False):
  # Helper function for _compatibilize and _compatibilize_inputs
  if isinstance(array_or_rational,(int,np.integer)):
    new_num = gcd(array_or_rational,r.numerator)
    return Fraction(new_num,r.denominator), int(array_or_rational)//new_num*r.denominator
  else:
    if not isinstance(array_or_rational,(Fraction,RationalArray)):
      # Convert to RationalArray; creates copy so allow inplace operations
      array_or_rational = rarray(array_or_rational)
      inplace_ok = True
    new_num = gcd(array_or_rational.numerator,r.numerator)
    new_denom = lcm(array_or_rational.denominator,r.denominator)
    scalar = (array_or_rational.numerator//new_num) * (new_denom//array_or_rational.denominator)
    newfactor = Fraction(new_num,new_denom)
    if isinstance(array_or_rational,Fraction):
      rv = scalar
    else:
      assert scalar > 0
      rv = array_or_rational._int_array
      if scalar != 1:
        if inplace_ok and array_or_rational._RationalArray__base is None:
          rv *= scalar
          array_or_rational._RationalArray__prefactor = newfactor
        else:
          rv = scalar * rv
      else:
        assert array_or_rational.prefactor == newfactor
    return newfactor, rv

def rarray(arr):
  """Create RationalArray from object or integer numpy array (or arraylike)"""
  if isinstance(arr,RationalArray):
    return arr.copy()
  if not isinstance(arr,np.ndarray):
    arr = np.array(arr,dtype=object)
  elif arr.dtype.kind != 'O':
    assert arr.dtype.kind in 'ui'
    arr = arr.astype('O')
  return RationalArray(*_integralize(arr))


class RationalArray(numpy.lib.mixins.NDArrayOperatorsMixin):
  def __new__(cls, factor, iarr):
    if isinstance(iarr,int) and isinstance(factor,Fraction):
      # Scalar, not even 0D array
      return iarr * factor
    else:
      return super().__new__(cls)

  def __init__(self, factor, iarr):
    if isinstance(factor,Fraction):
      assert factor > 0
      # Apparently numpy types sneaking into Fractions is something we have to
      # worry about?
      self.__prefactor = Fraction(int(factor.numerator),int(factor.denominator))
      self.__base = None
    else:
      self.__prefactor = None
      self.__base = factor._defaultbase
    self._int_array = iarr

  def __repr__(self):
    # Make it too long for pprint (why???)
    return 'RationalArray(%r, %r)'%(self.prefactor,self._int_array) + 80*' '

  def __str__(self):
    sarr = np.array([str(self.prefactor*ii) for ii in self._int_array.reshape(-1)])
    lens = np.array([len(s) for s in sarr]).reshape(self.shape)
    sarr = sarr.reshape(self.shape)
    collens = np.max(lens,axis=tuple(range(self.ndim-1)))
    lines = []
    for idx in itertools.product(*[range(l) for l in self.shape[:-1]]):
      row = '[' + ' '.join([s.rjust(l) for s,l in zip(sarr[idx],collens)]) + ']'
      nstart = 0
      while nstart < self.ndim-1 and idx[self.ndim-nstart-2] == 0:
        nstart += 1
      row = (nstart*'[').rjust(self.ndim-1) + row
      if lines and nstart:
        lines[-1] += nstart*']'
      lines.append(row)
    lines[-1] += (self.ndim-1)*']'
    return '\n'.join(lines)

  def rereduce(self):
    """Run through values of integer array to see if common factors can be
    pulled back out
    Returns self, for convenience"""
    if self.base is not None:
      self.base.rereduce()
    else:
      # TODO way to use nditer?
      reduction = gcd(*self._int_array.reshape(-1))
      if not reduction:
        # All zeros: simplifies to 1*0
        # TODO is this actually preferable to keeping the prefactor?
        self.__prefactor = Fraction(1)
      elif reduction != 1:
        self._int_array //= reduction
        self.__prefactor *= reduction
    return self

  def __array__(self, dtype=None, copy=None):
    if copy is False:
      raise ValueError('Copy must be created')
    elif dtype is None or dtype == np.dtype(object):
      # Fraction array
      frarr = self.prefactor * self._int_array
      if not isinstance(frarr,np.ndarray):
        # Zero-dimensional I guess
        frarr = np.array(frarr)
      return frarr
    elif dtype.kind == 'i':
      # Must be integer-valued
      if self.prefactor.denominator != 1:
        # Check again
        self.rereduce()
        if self.prefactor.denominator != 1:
          if not np.any(self._int_array):
            return np.zeros(self.shape,dtype=dtype)
          else:
            raise ValueError('Cannot cast array with non-integer values to integer type')
      return self._int_array.astype(dtype)
    elif dtype.kind in 'fc':
      # Convert to floating-point
      try:
        arr = dtype.type(self.prefactor) * self._int_array.astype(dtype)
      except OverflowError:
        # oops
        arr = (self.prefactor*self._int_array).astype(dtype)
      return arr
    else:
      raise TypeError('Cannot cast RationalArray to unsupported type %s'%dtype)

  def asfarray(self, dtype=float):
    return self.__array__(dtype=np.dtype(dtype))

  def __getitem__(self, index):
    # If applicable, construct view based on indexing operation applied to 
    # _int_array
    subint = self._int_array[index]
    if isinstance(subint,int):
      return self.prefactor * subint
    elif subint.ndim == 0:
      # None of this 0D array BS
      return self.prefactor * subint[()]
    subarr = RationalArray(self._defaultbase, self._int_array[index])

    #elif self.__base is not None:
      #assert subarr._int_array.base is self.__base._int_array
    return subarr

  def __copy__(self):
    return RationalArray(self.prefactor,self._int_array.copy())

  def copy(self):
    return self.__copy__()

  def _compatibilize(self, array_or_rational, inplace_ok=False):
    """For argument array_or_rational of the form f*X, where f=p2/q2 is a
    Fraction or integer and X is an integer array or 1,
    and for self.prefactor = p1/q1, rescales both to have prefactor
    gcd(p1,p2)/gcd(q1,q2), then returns new integer part
    This is always in-place for self; if inplace_ok, it is also done in-place
    for the argument"""
    assert self.__base is None
    r, iarr = _scalar_compatibilize(self.prefactor, array_or_rational, inplace_ok)
    if r != self.prefactor:
      adjust = r/self.prefactor
      assert adjust.numerator == 1
      self._int_array *= adjust.denominator
      self.__prefactor = r
    return iarr

  def __setitem__(self, index, values):
    if values is None:
      # TODO investigate behavior in conjunction with __iadd__ etc.
      return
    # Need to compatibilize
    iarr = self._defaultbase._compatibilize(values)
    self._int_array[index] = iarr

  def _float_ufunc(self, ufunc, method, dtype, *inputs, **kwargs):
    # Helper for __array_ufunc__ that converts self & any other RationalArray
    # arguments to floating-point
    if method == 'at' and isinstance(inputs[0],RationalArray):
      # Requires conversion, so can't do that
      return NotImplemented
    if 'out' in kwargs and any(isinstance(oarr,RationalArray) for oarr in kwargs['out']):
      return NotImplemented
    if dtype is None or dtype == np.dtype(object):
      # Called because of ufunc, not because of dtype
      dtype = np.dtype(float)
    in_fp = []
    for arg in inputs:
      if isinstance(arg,RationalArray):
        in_fp.append(np.array(arg,dtype=dtype))
      else:
        in_fp.append(arg)
    return getattr(ufunc,method)(*in_fp,**kwargs)
      
  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    if ufunc in _FLOAT_UFUNCS:
      # Requires conversion to floating-point
      if 'dtype' in kwargs:
        dtype = kwargs['dtype']
      else:
        dtype_options = []
        if 'out' in kwargs:
          dtype_options.extend(kwargs['out'])
        for arg in inputs:
          if isinstance(arg,np.ndarray) and arg.dtype.kind in 'fc':
            dtype_options.append(arg)
        if dtype_options:
          dtype = np.result_type(*dtype_options)
        else:
          dtype = np.dtype(float)
      if dtype.kind not in 'fc':
        dtype = np.dtype(float)
      return self._float_ufunc(ufunc,method,dtype,*inputs,**kwargs)
    # Check if anything should be floating-point, if so convert to
    # floating-point array
    # TODO any cases where float - rational interoperability may be useful?
    #   (e.g. float^rational power)
    # TODO does numpy do the dtype conversion itself?
    # TODO does numpy convert arraylike inputs and outputs itself?
    if 'dtype' in kwargs and np.dtype(kwargs['dtype']).kind in 'fc':
      return self._float_ufunc(ufunc,method,kwargs['dtype'],*inputs,**kwargs)
    if method == 'at' or method == 'reduceat':
      # TODO where are the indices actually?
      arr1,indices,*arr2 = inputs
      extra_inputs = [indices]
      array_inputs = [arr1,*arr2]
    else:
      extra_inputs = []
      array_inputs = inputs
    for arg in array_inputs:
      if isinstance(arg,numbers.Number):
        if np.dtype(type(arg)).kind in 'fc':
          return self._float_ufunc(ufunc,method,type(arg),*inputs,**kwargs)
      elif not isinstance(arg,RationalArray):
        if not isinstance(arg,np.ndarray):
          # Array-like
          arg = np.array(arg)
        if arg.dtype.kind in 'fc':
          return self._float_ufunc(ufunc,method,arg.dtype,*inputs,**kwargs)
    if 'out' in kwargs:
      for arr in kwargs['out']:
        if isinstance(arr,np.ndarray) and arr.dtype.kind in 'fc':
          return self._float_ufunc(ufunc,method,arr.dtype,*inputs,**kwargs)
    everywhere = 'where' not in kwargs or np.all(kwargs['where'])
    if ufunc in _COMPAT_UFUNCS:
      if ufunc in _COMPARE_UFUNCS and method == 'at':
        # Don't know how we got here, but we shouldn't be
        return NotImplemented
      # Need to compatibilize inputs & outputs
      out_compat = []
      out_nocompat = []
      if 'out' in kwargs:
        for i,oarr in enumerate(kwargs['out']):
          if oarr is None:
            continue
          elif (ufunc == np.floor_divide or ufunc == np.divmod) and i == 0:
            continue
          elif ufunc in _COMPARE_UFUNCS:
            continue
          elif not isinstance(oarr,RationalArray):
            return NotImplemented
          if oarr._fullview and everywhere:
            # Can just set the prefactor without worrying about the content
            out_nocompat.append(oarr)
          else:
            out_compat.append(oarr._defaultbase)
      if method == 'at':
        out_compat.append(self)
      newfactor,in_compat = _compatibilize_inputs(*array_inputs,outputs=out_compat,blanks=out_nocompat)
      iinputs = [in_compat[0],*extra_inputs,*in_compat[1:]]
      if ufunc in _COMPARE_UFUNCS:
        # Will produce boolean array on compatibilized integers
        return getattr(ufunc,method)(*iinputs, **kwargs)
      elif ufunc is np.divmod or ufunc is np.floor_divide:
        if method != '__call__' and method != 'outer':
          # TODO various options for floor_divide
          # (shouldn't be possible for divmod?)
          return NotImplemented
        if 'out' in kwargs:
          if ufunc is np.divmod:
            dout,rout = kwargs.pop('out')
          else:
            dout, = kwargs.pop('out')
            rout = None
        else:
          dout,rout = None,None
        kwdiv = dict(kwargs)
        idout = dout
        ifactor = None
        if dout is not None:
          if isinstance(idout,RationalArray):
            # why? but ok
            if dout.prefactor != 1:
              if dout._fullview and everywhere:
                dout._prefactor = Fraction(1)
              elif dout.numerator != 1:
                dout._compatibilize(1)
            if dout.denominator != 1:
              ifactor = dout.denominator
              # Have to use additional integer array
              idout = np.zeros_like(dout._int_array)
            else:
              idout = dout._int_array
          if idout.dtype.kind == 'i':
            # Won't let you with default casting
            kwdiv['casting'] = 'unsafe'
          kwdiv['out'] = (idout,)
        idout = getattr(np.floor_divide,method)(*in_compat,**kwdiv)
        if dout is None:
          dout = idout
        elif ifactor is not None:
          kwdiv['out'] = (dout._int_array,)
          np.multiply(idout,ifactor,**kwdiv)
        if ufunc is np.floor_divide:
          # Finished here
          return dout
        # Otherwise get remainder
        kwmod = dict(kwargs)
        if rout is not None:
          kwmod['out'] = rout._int_array
        irout = getattr(np.remainder,method)(*in_compat,**kwargs)
        if rout is not None:
          return rout
        else:
          return RationalArray(newfactor,irout)
      else:
        # Need to produce resulting integer array
        if 'out' in kwargs:
          iout = []
          rout = kwargs.pop('out')
          for oarr in rout:
            if oarr is None:
              iout.append(None)
            else:
              if not isinstance(oarr,RationalArray):
                # TODO any more cases with mixable outputs like divmod?
                return NotImplemented
              iout.append(oarr._int_array)
          kwargs['out'] = tuple(iout)
        else:
          rout = None
        iinputs = [in_compat[0],*extra_inputs,*in_compat[1:]]
        try:
          i_return = getattr(ufunc,method)(*iinputs,**kwargs)
        except OverflowError:
          # Known to happen at least when iinputs are scalars
          # (in v2.3.5 numpy attempts to convert to C long);
          # workaround is conversion to 0D object arrays
          iinputs = [np.array(iarr,dtype=object) for iarr in iinputs]
          i_return = getattr(ufunc,method)(*iinputs,**kwargs)
        if method == 'at':
          assert i_return is None
          return
        if isinstance(i_return,tuple):
          # But it shouldn't be, should it?
          r_return = []
          for i,iarr in enumerate(i_return):
            if rout is not None and rout[i] is not None:
              assert iarr is iout[i]
              r_return.append(rout[i])
            else:
              r_return.append(RationalArray(newfactor,iarr))
          return tuple(r_return)
        elif rout is not None:
          assert i_return is iout[0]
          return rout[0]
        else:
          return RationalArray(newfactor,i_return)
    elif ufunc in _INT_UNARY_UFUNCS:
      # inputs should just be operand
      compat, = inputs
      assert isinstance(compat,RationalArray) # TODO out provided
      if ufunc == np.square:
        f2 = compat.prefactor**2
      ifactor = None
      if 'out' in kwargs:
        rout, = kwargs.pop('out')
        if not isinstance(rout,RationalArray):
          if ufunc == np.sign:
            iout = rout
            rout = None
          else:
            return NotImplemented
        else:
          # May need to compatibilize
          if compat._defaultbase is rout._defaultbase:
            compat = compat.copy()
          iout = rout._int_array
          if not rout._fullview or not everywhere:
            if ufunc == np.sign:
              rout._compatibilize(1)
              ifactor = rout.denominator
              if ifactor != 1:
                iout = np.zeros_like(rout._int_array)
            elif ufunc == np.square:
              factor = rout._compatibilize(f2)
              ifactor = int(f2/factor)
            else:
              factor = rout._compatibilize(compat.prefactor)
              if factor != compat.prefactor:
                compat = compat.copy()
                compat._compatibilize(factor)
          else:
            if ufunc == np.sign:
              rout.__prefactor = Fraction(1)
              ifactor = 1
            elif ufunc == np.square:
              rout.__prefactor = f2
              ifactor = 1
            else:
              rout.__prefactor = compat.prefactor
        kwargs['out'] = iout
      else:
        rout = None
        iout = None
      i_return = getattr(ufunc,method)(compat._int_array,*extra_inputs,**kwargs)
      if method == 'at':
        assert i_return is None
        return
      if ufunc == np.sign:
        if ifactor is None:
          # Be normal & don't put the integers in the rational array
          return i_return
        else:
          if ifactor != 1:
            kwargs['out'] = (rout._int_array,)
            np.multiply(iout,ifactor,**kwargs)
          return rout
      elif ufunc == np.square:
        if ifactor is None:
          return RationalArray(f2,i_return)
        else:
          if ifactor != 1:
            kwargs['out'] = (rout._int_array,)
            np.multiply(iout,ifactor,**kwargs)
          return rout
      if rout is None:
        # Output array not yet defined
        rout = RationalArray(compat.prefactor,i_return)
      return rout
    elif ufunc in _LOGICAL_UFUNCS:
      if method == 'at':
        return NotImplemented
      # All can be done on integer arrays
      iinputs = []
      for arg in inputs:
        if isinstance(arg,RationalArray):
          iinputs.append(arg._int_array)
        else:
          iinputs.append(arg)
      iinputs = [iinputs[0],*extra_inputs,*iinputs[1:]]
      return getattr(ufunc,method)(*iinputs,**kwargs)
            
    elif ufunc == np.reciprocal:
      if method == 'at':
        # Convert to boolean "where" argument & supply to self.reciprocal
        operand,indices = inputs
        where = np.zeros(self.shape,dtype=bool)
        np.logical_not.at(where,indices,**kwargs)
        operand.reciprocal(where=where,inplace=True)
      elif method == '__call__':
        operand, = inputs
        if not isinstance(operand,RationalArray):
          operand = rarray(operand)
        return operand.reciprocal(**kwargs)
      else:
        return NotImplemented

    elif ufunc == np.divide:
      if method not in ('__call__','at','outer'):
        # TODO Can figure that out later
        return NotImplemented
      else:
        arr1,arr2 = array_inputs
        if isinstance(arr2,RationalArray):
          inv = arr2.reciprocal()
        elif isinstance(arr2,numbers.Number):
          inv = 1/Fraction(arr2)
        else:
          inv = rarray(arr2)
          inv.reciprocal(inplace=True)
        # Now we have a multiplication problem
        return self.__array_ufunc__(np.multiply,method,arr1,*extra_inputs,inv,**kwargs)
    elif ufunc == np.multiply:
      if method not in ('__call__','at','outer'):
        # TODO need a way of keeping track of factors
        return NotImplemented
      else:
        if 'out' in kwargs:
          oarr, = kwargs.pop('out')
          if isinstance(oarr,np.ndarray):
            if oarr.dtype.kind == 'O':
              oinputs = [np.array(arr,dtype=np.dtype('O')) for arr in inputs]
              oinputs = [oinputs[0],*extra_inputs,*oinputs[1:]]
              return getattr(ufunc,method)(*oinputs, **kwargs)
            elif oarr.dtype.kind == 'i':
              iout = oarr
              rout = None
            else:
              return NotImplemented
          elif isinstance(oarr,RationalArray):
            rout = oarr
            iout = oarr._int_array
          else:
            return NotImplemented
          kwargs['out'] = iout
        else:
          iout = None
          rout = None
              
        fparts = []
        iparts = []
        for in_arr in array_inputs:
          if isinstance(in_arr,RationalArray):
            fpart = in_arr.prefactor
            ipart = in_arr._int_array
          elif isinstance(in_arr,np.ndarray):
            if in_arr.dtype == np.dtype('O'):
              # Convert to RationalArray
              in_arr = rarray(in_arr)
              fpart = in_arr.prefactor
              ipart = in_arr._int_array
            elif in_arr.dtype.kind == 'i':
              fpart = Fraction(1)
              ipart = in_arr
            else:
              return NotImplemented
          elif isinstance(in_arr,numbers.Number):
            if isinstance(in_arr,Fraction):
              fpart = in_arr
              ipart = 1
            elif isinstance(in_arr,(int,np.integer)):
              fpart = Fraction(in_arr,1)
              ipart = 1
            else:
              return NotImplemented
            if not fpart:
              fpart = Fraction(1)
              ipart = 0
            elif fpart < 0:
              fpart = -fpart
              ipart = -ipart
          else:
            return NotImplemented
          # If ipart duplicates iout (will happen in __imul__ call),
          # replace with copy
          if iout is not None and isinstance(ipart,np.ndarray):
            ibase = ipart if ipart.base is None else ipart.base
            obase = iout if iout.base is None else iout.base
            if ibase is obase:
              ipart = ipart.copy()
          fparts.append(fpart)
          iparts.append(ipart)
        newfactor = functools.reduce(Fraction.__mul__,fparts,Fraction(1))
        extramult = 1
        if iout is not None:
          if rout is None:
            if newfactor.denominator != 1:
              raise ValueError('Cannot store non-integral multiplication output in integer array')
            else:
              extramult = newfactor.numerator
          else:
            if rout.__base is not None:
              extramult = rout.__base._compatibilize(newfactor)
            elif not everywhere:
              extramult = rout._compatibilize(newfactor)
            else:
              rout.__prefactor = newfactor
        if method == 'at':
          indices, = extra_inputs
          rout = inputs[0]
          if not isinstance(rout,RationalArray):
            assert self is array_inputs[1]
            if isinstance(rout,np.ndarray) and rout.dtype == np.dtype('O'):
              return np.multiply.at(rout,indices,np.array(array_inputs[1]))
            else:
              return NotImplemented
          if isinstance(indices,bool):
            if not indices:
              # No calculation -- why are we here?
              return
          elif isinstance(indices,np.ndarray) and \
              indices.dtype == np.dtype('bool'):
            everywhere = np.all(indices)
          else:
            counts = np.zeros(dout.shape,dtype=int)
            np.add.at(counts,indices,1)
            if np.max(counts) > 1:
              raise ValueError('np.multiply.at currently not supported for '
                'multiple applications on RationalArray elements')
            everywhere = np.all(counts)
          if everywhere and rout._fullview:
            rout._defaultbase.__prefactor = newfactor
            extramult = 1
          else:
            extramult = rout._defaultbase._compatibilize(newfactor)
            iout = rout._int_array

        iinputs = [iparts[0],*extra_inputs,*iparts[1:]]
        i_return = getattr(ufunc,method)(*iinputs,**kwargs)
        if extramult != 1:
          if 'where' in kwargs:
            where = kwargs['where']
          elif method == 'at':
            where = indices
          else:
            where = True
          np.multiply.at(iout,where,extramult)
        if method == 'at':
          assert i_return is None
          return
        if rout is None:
          if iout is not None:
            return iout
          else:
            return RationalArray(newfactor,i_return)
        else:
          return rout
    else:
      return NotImplemented

  def __array_function__(self, func, types, args, kwargs):
    # Copied lovingly from numpy example with minor changes
    if func not in _HANDLED_FUNCTIONS:
      return NotImplemented
    if not all(issubclass(t, (RationalArray,np.ndarray)) for t in types):
      return NotImplemented
    return _HANDLED_FUNCTIONS[func](*args, **kwargs)


  def reciprocal(self, where=True, inplace=False):
    """Implementation for numpy ufunc reciprocal"""
    # Test zeros (apparently 'all' calls logical_and.reduce but can't pass
    # necessary initial argument to handle "where" with object array
    if where is False:
      # Really just cover the bases
      if inplace:
        return self
      else:
        return self.copy()
    if not np.logical_and.reduce(self._int_array,dtype=bool,where=where,axis=None):
      # TODO support for infinities?
      if inplace:
        raise ZeroDivisionError()
      else:
        # Convert to object array
        return np.reciprocal(np.array(self),where=where)
      
    if np.all(where) and self._fullview:
      int_factor = lcm(*self._int_array.reshape(-1))
      newfactor = Fraction(self.denominator,self.numerator*int_factor)
      if inplace:
        np.floor_divide(int_factor,self._int_array,out=self._int_array)
        self.__prefactor = newfactor
        return
      else:
         iout = np.floor_divide(int_factor,self._int_array)
         return RationalArray(newfactor,iout)
    else:
      if where is True:
        inv_factor = lcm(*self._int_array.reshape(-1))
        nowhere = False
        reduce_factor = 1
      else:
        inv_factor = lcm(*self._int_array[where].reshape(-1))
        nowhere = np.logical_not(where)
        reduce_factor = gcd(*self._int_array[nowhere].reshape(-1))
      # p/q*f_r * [+]/f_r + q/p/f_i * f_i/[-]
      fplus = self.prefactor * reduce_factor
      fminus = Fraction(1)/(self.prefactor * inv_factor)
      p = gcd(fplus.numerator,fminus.numerator)
      q = lcm(fplus.denominator,fminus.denominator)
      newfactor = Fraction(p,q)
      if inplace:
        iarr = self._int_array
      else:
        iarr = np.zeros_like(self._int_array)
      f2minus = fminus/newfactor
      assert f2minus.denominator == 1
      np.floor_divide(inv_factor*f2minus.numerator,self._int_array,
        out=iarr,where=where)
      f2plus = self.prefactor/newfactor
      if nowhere is not False:
        np.multiply(f2plus.numerator,self._int_array,out=iarr,where=nowhere)
        if f2plus.denominator != 1:
          np.floor_divide.at(iarr,nowhere,f2plus.denominator)
      if inplace:
        self.__prefactor = newfactor
        return
      else:
        return RationalArray(newfactor,iarr)

  def lower_index(self, metric, axis, inplace=False):
    """'Lower' a 'contravariant' index, in the sense of contracting the index in
    question with a metric tensor (here assumed to be diagonal)"""
    xmetric = expand_dims(metric,
      tuple(range(axis))+tuple(range(axis+1,self.ndim)))
    if inplace:
      self *= xmetric
    else:
      return self * xmetric

  def raise_index(self, metric, axis, inplace=False):
    """'Raise' a 'covariant' index, in the sense of contracting the index in
    question with the inverse of a metric tensor
    (here assumed to be diagonal)"""
    xmetric = expand_dims(metric.reciprocal(),
      tuple(range(axis))+tuple(range(axis+1,self.ndim)))
    if inplace:
      self *= xmetric
    else:
      return self * xmetric

  
  def transpose(self, *axes):
    """Return a view of the array with axes transposed"""
    T = RationalArray(self._defaultbase,self._int_array.transpose(*axes))
    return T

  @property
  def T(self):
    return self.transpose()

  def reshape(self, *newshape, **kwargs):
    return RationalArray(self._defaultbase,
      self._int_array.reshape(*newshape,**kwargs))

  def dot(self, b, out=None):
    return dot(self, b, out=out)

  def trace(self, axis1=0, axis2=1, out=None):
    if out is not None:
      if not isinstance(out, RationalArray):
        return NotImplemented
      if out._fullview:
        out._defaultbase.__prefactor = self.prefactor
      else:
        out.__base._compatibilize(self.prefactor)
      self._int_array.trace(axis1, axis2, out=out._int_array)
      return out
    else:
      return RationalArray(self.prefactor,self._int_array.trace(axis1=axis1,axis2=axis2))

  def flatten(self, **kwargs):
    """Like ndarray.flatten, this does not return a view
    (for whatever reason)"""
    return RationalArray(self.prefactor,self._int_array.flatten(**kwargs))

  @property
  def shape(self):
    return self._int_array.shape

  @property
  def ndim(self):
    return self._int_array.ndim
  
  @property
  def size(self):
    return self._int_array.size

  def __len__(self):
    return len(self._int_array)

  @property
  def base(self):
    return self.__base

  @property
  def _defaultbase(self):
    if self.__base is None:
      return self
    else:
      return self.__base

  @property
  def _fullview(self):
    if self.__base is None:
      return True
    else:
      return self.size == self.__base.size

  @property
  def prefactor(self):
    if self.__base is None:
      return self.__prefactor
    else:
      return self.__base.__prefactor

  @property
  def numerator(self):
    return self.prefactor.numerator

  @property
  def denominator(self):
    return self.prefactor.denominator


def _implements(np_function):
  # See numpy "Writing custom array containers" docs
  def decorator(func):
    _HANDLED_FUNCTIONS[np_function] = func
    return func
  return decorator

@_implements(np.transpose)
def transpose(arr, *axes):
  return arr.transpose(*axes)

@_implements(np.reshape)
def reshape(arr, *newshape, **kwargs):
  return arr.reshape(*newshape,**kwargs)

def _convert_or_float(np_function,num_nonarray=0):
  # Variation on implements decorator which modifies function:
  # Check arguments; if any are floating-point numbers or arrays,
  # call np_function instead
  def decorator(func):
    def f(*args, **kwargs):
      args = list(args)
      num_array = len(args)-num_nonarray
      for i in range(num_array):
        if isinstance(args[i],np.ndarray) and args[i].dtype.kind in 'fc' \
            or isinstance(args[i],(float,complex,np.inexact)):
          if isinstance(args[i],np.ndarray):
            dtype = args[i].dtype
          else:
            dtype = np.dtype(type(args[i]))
          # Instead need to replace RationalArray arguments
          for j in range(num_array):
            if isinstance(args[j],RationalArray):
              args[j] = args[j].asfarray(dtype=dtype)
          return np_function(*args,**kwargs)
      # Now confirmed no float arrays -- instead convert to rational
      for i in range(num_array):
        if not isinstance(args[i],RationalArray):
          if isinstance(args[i],numbers.Number):
            args[i] = Fraction(args[i])
          else:
            args[i] = rarray(args[i])
      # Finally pass to originally-defined function
      return func(*args, **kwargs)

    _HANDLED_FUNCTIONS[np_function] = f
    return f
  return decorator

def _convert_or_float_iter(np_function):
  # Like convert_or_float, but for case when arrays are supplied as
  # iterable instead of getting positional arguments of their own
  def decorator(func):
    def f(arrays, *args, **kwargs):
      arrays = list(arrays)
      for i in range(len(arrays)):
        if isinstance(arrays[i],np.ndarray) and arrays[i].dtype.kind in 'fc' \
            or isinstance(arrays[i],(float,complex,np.inexact)):
          if isinstance(arrays[i],np.ndarray):
            dtype = arrays[i].dtype
          else:
            dtype = np.dtype(type(arrays[i]))
          # Instead need to replace RationalArray arguments
          for j in range(len(arrays)):
            if isinstance(arrays[j],RationalArray):
              arrays[j] = arrays[j].asfarray(dtype=dtype)
          return np_function(*args,**kwargs)
      # Now confirmed no float arrays -- instead convert to rational
      for i in range(len(arrays)):
        if not isinstance(arrays[i],RationalArray):
          if isinstance(arrays[i],numbers.Number):
            arrays[i] = Fraction(arrays[i])
          else:
            arrays[i] = RationalArray(arrays[i])
      # Finally pass to originally-defined function
      return func(arrays, *args, **kwargs)

    _HANDLED_FUNCTIONS[np_function] = f
    return f
  return decorator
  
@_convert_or_float(np.dot)
def dot(a, b, out=None):
  factor = a.prefactor * b.prefactor
  if out is not None:
    if not isinstance(out,RationalArray):
      if isinstance(out,np.ndarray):
        if out.dtype == np.dtype('O'):
          # Just convert a and b to object arrays
          return np.dot(np.array(a),np.array(b),out=out)
        elif out.dtype.kind == 'i':
          # TODO is this ok for types?
          np.dot(a._int_array,b._int_array,out=out)
          if factor.denominator != 1:
            if np.any(out % factor.denominator):
              raise ValueError('Cannot set non-integer dot product to integer array')
            out //= factor.denominator
          if factor.numerator != 1:
            out *= factor.numerator
          return out
        elif out.dtype.kind in 'fc':
          return np.dot(np.array(a,dtype=out.dtype),np.array(b,dtype=out.dtype),out=out)
        else:
          # ???
          return NotImplemented
      else:
        raise TypeError('Only RationalArray or ndarray outputs accepted for dot')
    else:
      if out._fullview:
        out._RationalArray__prefactor = factor
        extrafactor = 1
      else:
        out._RationalArray__base._compatibilize(factor)
        extrafactor = (out.prefactor/factor).denominator
      np.dot(a._int_array,b._int_array,out=out._int_array)
      if extrafactor != 1:
        out._int_array *= extrafactor
      return out
  else:
    return RationalArray(factor,np.dot(a._int_array,b._int_array))

@_convert_or_float(np.tensordot,1)
def tensordot(a, b, axes):
  if not isinstance(a,RationalArray):
    if isinstance(a,np.ndarray) and a.dtype.kind in 'fc':
      return np.tensordot(a,np.array(b,dtype=a.dtype),axes)
    else:
      a = rarray(a)
  if not isinstance(b,RationalArray):
    if isinstance(b,np.ndarray) and b.dtype.kind in 'fc':
      return np.tensordot(np.array(a,dtype=b.dtype),b,axes)
    else:
      b = rarray(b)
  factor = a.prefactor * b.prefactor
  tdot = np.tensordot(a._int_array,b._int_array,axes)
  return RationalArray(factor,tdot)

@_implements(np.isclose)
def isclose(a, b, **kwargs):
  if isinstance(a,RationalArray):
    dtype = b.dtype if isinstance(b,np.ndarray) else dtype(float)
    return np.isclose(np.array(a,dtype=dtype),b,**kwargs)
  else:
    dtype = a.dtype if isinstance(a,np.ndarray) else dtype(float)
    return np.isclose(a,np.array(b,dtype=dtype),**kwargs)

@_implements(np.allclose)
def allclose(a, b, **kwargs):
  return np.all(np.isclose(a,b,**kwargs))

@_implements(np.all)
def _all(a, **kwargs):
  if not isinstance(a,RationalArray):
    return NotImplemented
  return np.all(a._int_array,**kwargs)

@_implements(np.any)
def _any(a, **kwargs):
  if not isinstance(a,RationalArray):
    return NotImplemented
  return np.any(a._int_array,**kwargs)

@_convert_or_float(np.kron)
def kron(a, b):
  factor = a.prefactor * b.prefactor
  ikron = np.kron(a._int_array,b._int_array)
  return RationalArray(factor,ikron)

@_implements(np.outer)
def outer(a,b,**kwargs):
  if (isinstance(a,np.ndarray) or isinstance(a,RationalArray)) and a.ndim != 1:
    a = a.reshape(-1)
  if (isinstance(b,np.ndarray) or isinstance(b,RationalArray)) and b.ndim != 1:
    b = b.reshape(-1)
  return np.multiply.outer(a,b,**kwargs)

def _implements_catenator(np_function):
  # Another implements variation: takes out kwarg if there is one,
  # compatibilizes output & inputs, and passes compatibilized integer arrays,
  # compatible factor, and (if applicable) output array
  def decorator(func):
    def f(arrays, *args, **kwargs):
      if 'out' in kwargs:
        rout = kwargs.pop('out')
        if not isinstance(rout,RationalArray):
          return NotImplemented
        kwargs['out'] = rout._int_array
        outs = (rout,)
      else:
        rout = None
        outs = None
      factor,iarrays = _compatibilize_inputs(*arrays,outputs=outs)
      ireturn = func(iarrays, *args, **kwargs)
      if rout is None:
        return RationalArray(factor,ireturn)
      else:
        return rout

    _HANDLED_FUNCTIONS[np_function] = f
    return f
  return decorator

@_implements_catenator(np.stack)
def stack(iarrays, **kwargs):
  return np.stack(iarrays, **kwargs)

@_implements_catenator(np.hstack)
def hstack(iarrays, **kwargs):
  return np.hstack(iarrays, **kwargs)

@_implements_catenator(np.vstack)
def vstack(iarrays, **kwargs):
  return np.vstack(iarrays, **kwargs)

@_implements_catenator(np.concatenate)
def concatenate(iarrays, **kwargs):
  return np.concatenate(iarrays, **kwargs)

@_implements(np.append)
def append(arr,values,axis=None):
  factor, (iarr0,ivals) = _compatibilize_inputs(arr, values)
  iarr = np.append(iarr0,ivals,axis=axis)
  return RationalArray(factor, iarr)

@_implements(np.zeros)
def rzeros(shape, **kwargs):
  # Note: kwargs ignored, mostly accepted bc numpy interoperability uses the
  # 'like' argument
  return RationalArray(Fraction(1),np.zeros(shape,dtype=object))

@_implements(np.ones)
def rones(shape,**kwargs):
  return RationalArray(Fraction(1),np.ones(shape,dtype=object))

@_implements(np.zeros_like)
def zeros_like(a, **kwargs):
  return RationalArray(Fraction(1),np.zeros_like(a._int_array,**kwargs))

@_implements(np.ones_like)
def ones_like(a, **kwargs):
  return RationalArray(Fraction(1),np.ones_like(a._int_array,**kwargs))

@_implements(np.eye)
def reye(N, M=None, k=0, **kwargs):
  return RationalArray(Fraction(1),np.eye(N,M=M,k=k,dtype=object))

@_implements(np.linalg.norm)
def fnorm(arr):
  return np.linalg.norm(arr.asfarray())

@_implements(np.diag)
def diag(v,k=0):
  if not isinstance(v,RationalArray):
    v = rarray(v)
  return RationalArray(v.prefactor,np.diag(v._int_array,k=k))

@_implements(np.expand_dims)
def expand_dims(arr, axis):
  return RationalArray(arr, np.expand_dims(arr._int_array,axis))

@_implements(np.squeeze)
def squeeze(arr, axis=None):
  return RationalArray(arr, np.squeeze(arr._int_array,axis=axis))

@_implements(np.ravel)
def ravel(arr, **kwargs):
  return RationalArray(arr, np.ravel(arr._int_array,**kwargs))

@_implements(np.trace)
def trace(arr, **kwargs):
  return arr.trace(**kwargs)

@_implements(np.nonzero)
def nonzero(arr):
  return np.nonzero(arr._int_array)

@_implements(np.argwhere)
def argwhere(arr):
  return np.argwhere(arr._int_array)

# see: https://stackoverflow.com/questions/40828173/how-can-i-make-my-class-pretty-printable-in-python
from pprint import PrettyPrinter
def pprint_rational_array(printer, arr, stream, indent, allowance, context, level):
  # TODO pay attention to other arguments?
  lines = str(arr).split('\n')
  stream.write('RA'+lines[0])
  indent += 2
  delim = '\n' + indent*' '
  for line in lines[1:]:
    stream.write(delim)
    stream.write(line)
PrettyPrinter._dispatch[RationalArray.__repr__] = pprint_rational_array

