# Objects and operators for the square lattice
from . import tensors
from . import networks
from . import operators
from .lattices import * 

class SquarePartitionFunction(PartitionFunction):
  """Partition function on square lattice
  Indices are labeled t,l,r,b, with a simple unit cell of size 1
  May provide nested list of tensors, or
  (x,y) dimensions of unit cell + nested list (which may have dimensions that
  divide the unit cell dimension) or + single tensor
  Nested list elements may be replaced with (x,y) coordinate index of
  previously referenced tensor or ((x,y),c) adding conjugation data"""
  
  def __init__(self, size, tens=None):
    tlist = []
    tdict = {}
    cdict = {}
    if isinstance(size, tuple):
      if len(size) != 2:
        raise ValueError('size must be provided as (x,y) tuple')
      Nx,Ny = size
      if not (isinstance(x,int) and isinstance(y,int) and x > 0 and y > 0):
        raise ValueError('size must be provided as pair of integers')
      if not tens:
        raise ValueError('Tensors must be provided along with size')
      if isinstance(tens, tensors.Tensor):
        tlist = [tens]
        tdict = {(x,y):0 for x in range(Nx) for y in range(Ny)}
      elif isinstance(tens, list):
        if not isinstance(tens[0], list):
          raise ValueError('Nested list must be two-dimensional')
        lx = len(tens)
        ly = len(tens[0])
        if any(not isinstance(tl, list) or len(tl) != ly for tl in tens[1:]):
          raise ValueError('Nested list does not have consistent dimensions')
        if Nx%lx != 0 or Ny%ly != 0:
          raise ValueError('Dimensions (%d,%d) of list do not divide '
            'size (%d,%d) provided'%(lx,ly,Nx,Ny))
        px = Nx//lx
        py = Ny//ly
        forward_ref = {}
        for x0 in range(lx):
          for y0 in range(ly):
            T = tens[x0][y0]
            if isinstance(T,tensors.Tensor):
              if (x0,y0) in forward_ref:
                # Referred to at previous site
                n = forward_ref[x0,y0]
              else:
                n = len(tlist)
                tlist.append(T)
              tdict.update({(x0+i*lx,y0+j*ly):n \
                for i in range(px) for j in range(py)})
            elif isinstance(T, tuple):
              if len(T) != 2:
                raise ValueError('Reference to other site must be '
                  '(x,y) or ((x,y),c)')
              x,y = T
              if isinstance(x,tuple):
                if len(x) != 2:
                  raise ValueError('Reference to other site must be '
                    '(x,y) or ((x,y),c)')
                (x,y),c = x,y
                cdict.update({(x0+i*lx,y0+j*ly):c \
                  for i in range(px) for j in range(py)})
              if not isinstance(x,int) or not isinstance(y,int):
                raise ValueError('Reference coordinates must be integer')
              x %= lx
              y %= ly
              if not isinstance(tens[x][y],Tensor):
                raise ValueError('References are not resolved past depth 1')
              if x > x0 or (x == x0 and y > y0):
                # May be new reference
                if (x,y) in forward_ref:
                  n = forward_ref[x,y]
                else:
                  n = len(tlist)
                  tlist.append(tens[x][y])
                  forward_ref[x,y] = n
              else:
                n = tdict[x,y]
              tdict.update({(x0+i*lx,y0+j*ly):n \
                for i in range(px) for j in range(py)})
            else:
              raise ValueError('Nested list elements must be tensors or '
                'coordinate-reference tuples')
    elif tens:
      raise ValueError('size must be provided as (x,y) tuple')
    else:
      tens = size
      if not isinstance(tens,list):
        raise ValueError('Tensors must be provided as nested list')
      if not isinstance(tens[0], list):
        raise ValueError('Nested list must be two-dimensional')
      Nx = len(tens)
      Ny = len(tens[0])
      if any(not isinstance(tl, list) or len(tl) != Ny for tl in tens[1:]):
        raise ValueError('Nested list does not have consistent dimensions')
      forward_ref = {}
      for x0 in range(Nx):
        for y0 in range(Ny):
          T = tens[x0][y0]
          if isinstance(T,tensors.Tensor):
            if (x0,y0) in forward_ref:
              # Referred to at previous site
              n = forward_ref[x0,y0]
            else:
              n = len(tlist)
              tlist.append(T)
            tdict[x0,y0] = n
          elif isinstance(T, tuple):
            if len(T) != 2:
              raise ValueError('Reference to other site must be '
                '(x,y) or ((x,y),c)')
            x,y = T
            if isinstance(x,tuple):
              if len(x) != 2:
                raise ValueError('Reference to other site must be '
                  '(x,y) or ((x,y),c)')
              (x,y),c = x,y
              cdict[x0,y0] = c
            if not isinstance(x,int) or not isinstance(y,int):
              raise ValueError('Reference coordinates must be integer')
            x %= Nx
            y %= Ny
            if not isinstance(tens[x][y],Tensor):
              raise ValueError('References are not resolved past depth 1')
            if x > x0 or (x == x0 and y > y0):
              # May be new reference
              if (x,y) in forward_ref:
                n = forward_ref[x,y]
              else:
                n = len(tlist)
                tlist.append(tens[x][y])
                forward_ref[x,y] = n
            else:
              n = tdict[x,y]
            tdict[x0,y0] = n
          else:
            raise ValueError('Nested list elements must be tensors or '
              'coordinate-reference tuples')
    self._dimension = (Nx,Ny)
    super().__init__(tlist, tdict, cdict)
    self.verify_compatibility()

  @property
  def Nx(self):
    return self._dimension[0]

  @property
  def Ny(self):
    return self._dimension[1]

  @classmethod
  def issite(cls,site):
    if not isinstance(site,tuple) or len(site) != 2:
      return False
    x,y = site
    return 

  def unitsite(self, site):
    return site[0]%self._dimension[0], site[1]%self._dimension[1]

  @classmethod
  def bondadjacency(cls, site):
    x,y = site
    return {'t':((x,y+1),'b'),'l':((x+1,y),'r'),
            'b':((x,y-1),'t'),'r':((x-1,y),'l')}

# NOTE: make subclass of RG ABC? 
class HOTRGsquarevert:
  def __init__(self, base, chi, xselect=None, yselect=None):
    """Optionally select which elements of the unit cell to coarsegrain
    (if (x,y) in xselect x yselect, will coarse-grain (x,y)-(x,y+1))"""
    if xselect is None:
      xselect = tuple(range(base.Nx))
      mx = 1
    else:
      assert all(isinstance(x,int) and x >= 0 for x in xselect)
      xselect = tuple(sorted(xselect))
      mx = max(xselect)//base.Nx+1
    if yselect is None:
      yselect = tuple(range(0,base.Ny,2))
      my = base.Ny%2+1
    else:
      assert all(isinstance(y,int) and y >= 0 for y in yselect)
      yselect = tuple(sorted(yselect))
      my = (max(yselect)+1)//base.Ny+1
    self._base = base
    self._chi = chi
    self._multiplicity = (mx,my)
    self._xselect = xselect
    self._yselect = yselect
    self._next = None
    self.wL = {}

  def initializeHOSVD(self):
    """Initialize using original (HOSVD) method"""
    Lnet = networks.network('B;T;T.b-B.t;T.r>t,B.r>b',
      self._base[0,0],self._base[0,1])
    Lnet = Lnet.derive('|T.l,B.l,T.t,B.b|l,r>~')
    Rnet = networks.network('B;T;T.b-B.t;T.l>t,B.l>b',
      self._base[1,0],self._base[1,1])
    Rnet = Lnet.derive('|T.r,B.r,T.t,B.b|l,r>~')
    # Row-by-row
    for y in self._yselect:
      for x in self._xselect:
        Lnet.replaceall('B,T',self._base[x,y],self._base[x,y+1])
        Rnet.replaceall('B,T',self._base[x+1,y],self._base[x+1,y+1])
        ML = Lnet.contract()
        MR = Rnet.contract()
        nL = ML.norm()
        nR = MR.norm()
        wL,UL = ML.eig('tr-tl,br-bl',selection=-self.chi,mat='c')
        wR,UR = MR.eig('tl-tr,bl-br',selection=-self.chi,mat='c')
        epsL = nL**2-sum(wL)
        epsR = nR**2-sum(wR)
        if epsL>epsR:
          self.wL[x,y] = UR.renamed('tr-t,br-b')
        else:
          self.wL[x,y] = UL.renamed('tl-t,bl-b')

  def initializemixed(self):
    """Initialize with sum of environment tensors"""
    Lnet = networks.network('B;T;T.b-B.t;T.r>t,B.r>b',
      self._base[0,0],self._base[0,1])
    Lnet = Lnet.derive('|T.l,B.l,T.t,B.b|l,r>~')
    Rnet = networks.network('B*;T*;T.b-B.t;T.l>t,B.l>b',
      self._base[1,0],self._base[1,1])
    Rnet = Lnet.derive('|T.r,B.r,T.t,B.b|l,r>~')
    # Row-by-row
    for y in self._yselect:
      for x in self._xselect:
        Lnet.replaceall('B,T',self._base[x,y],self._base[x,y+1])
        Rnet.replaceall('B,T',self._base[x+1,y],self._base[x+1,y+1])
        ML = Lnet.contract()
        MR = Rnet.contract()
        M = ML/ML.norm() + MR/MR.norm()
        w,U = ML.eig('tr-tl,br-bl',selection=-self.chi,mat=True)
        self.wL[x,y] = 
        wR,UR = MR.eig('tl-tr,bl-br',selection=-self.chi,mat=True)
        epsL = nL**2-sum(wL)
        epsR = nR**2-sum(wR)
        if epsL>epsR:
          self.wL[x,y] = UR.renamed('tr-t,br-b')
        else:
          self.wL[x,y] = UL.renamed('tl-t,bl-b')

  def optimizelin(self, delta, niter):
