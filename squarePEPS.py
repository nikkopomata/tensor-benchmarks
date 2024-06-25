from quantrada import config,networks,tensors
import numpy as np
import re
from quantrada.MPS.DMRGalgo import PseudoShelf
# TODO error in expectation values with nontrivial pseudo-schmidt coefficients?

logger = config.logger.getChild('PEPS')
stdlog = config.streamlog
gcollect = False
showfidelity = False
ztol = 1e-6

class PEPSComplex:
  def __init__(self, tensors, pschmidth=None,pschmidtv=None):
    # Set PEPS tensors
    self._Nx = len(tensors)
    self._Ny = len(tensors[0])
    self._tensors = []
    self._pseudoschmidt = {}
    for x in range(self._Nx):
      assert len(tensors[x]) == self._Ny
      self._tensors.append(list(tensors[x]))
      for y,T in enumerate(tensors[x]):
        # Perform checks
        lcmp = {'q'}
        if x!=0:
          lcmp.add('l')
        if x!=self._Nx-1:
          lcmp.add('r')
        if y!=0:
          lcmp.add('b')
        if y!=self._Ny-1:
          lcmp.add('t')
        assert T.idxset == lcmp
        if x:
          assert T.getspace('l') ^ tensors[x-1][y].getspace('r')
          dl = T.dshape['l']
          if pschmidth is not None:
            s = pschmidth[x-1][y]
            assert len(s) == dl
            self._pseudoschmidt[(x-1,x),y] = list(s)
          else:
            self._pseudoschmidt[(x-1,x),y] = dl*[1.]
        if y:
          assert T.getspace('b') ^ tensors[x][y-1].getspace('t')
          db = T.dshape['b']
          if pschmidtv is not None:
            s = pschmidtv[x][y-1]
            assert len(s) == db
            self._pseudoschmidt[x,(y-1,y)] = list(s)
          else:
            self._pseudoschmidt[x,(y-1,y)] = db*[1.]
    # Edge and corner matrices
    self._matrices = {}
    self._reducers = {}

  @property
  def size(self):
    return self._Nx,self._Ny

  def gettensor(self,x,y,with_ps=(),with_ps_in=(),half_ps=False):
    # If with_ps, apply pseudoschmidt on indices given
    # If with_ps_in, do not raise error for invalid indices
    # If half_ps, incorporate square root
    T = self._tensors[x][y]
    if with_ps_in:
      assert not with_ps
      with_ps = set(with_ps_in) & T.idxset
    pss = {}
    if with_ps:
      if 'l' in with_ps:
        pss['l'] = self._pseudoschmidt[(x-1,x),y]
      if 'r' in with_ps:
        pss['r'] = self._pseudoschmidt[(x,x+1),y]
      if 't' in with_ps:
        pss['t'] = self._pseudoschmidt[x,(y,y+1)]
      if 'b' in with_ps:
        pss['b'] = self._pseudoschmidt[x,(y-1,y)]
      if half_ps:
        pss = {l:np.sqrt(s) for l,s in pss.items()}
      for l in pss:
        T = T.diag_mult(l,pss[l])
    return T

  def getbond(self, x, y, direction):
    """Get virtual space from x,y to direction (one of l,r,t,b)"""
    return self._tensors[x][y].getspace(direction)

  def getchi(self, x, y, direction):
    return self.getbond(x,y,direction).dim

  # TODO negative indexing on _matrices?
  def getTmatrix(self,x,y,usecorner=False):
    """Upper edge transfer matrix
    Return corner if x is on left or right edge and usecorner=True"""
    assert y
    if usecorner and x == 0:
      return self._matrices['TL',x,y]
    if usecorner and x == self._Nx-1:
      return self._matrices['TR',x,y]
    assert x and x!=self._Nx-1
    return self._matrices['T',x,y]

  def getBmatrix(self,x,y,usecorner=False):
    """Lower edge transfer matrix"""
    assert y != self._Ny-1
    if usecorner and x == 0:
      return self._matrices['BL',x,y]
    if usecorner and x == self._Nx-1:
      return self._matrices['BR',x,y]
    assert x and x!=self._Nx-1
    return self._matrices['B',x,y]

  def getLmatrix(self,x,y,usecorner=False):
    """Left edge transfer matrix"""
    assert x != self._Nx-1
    if usecorner and y == 0:
      return self._matrices['BL',x,y]
    if usecorner and y == self._Ny-1:
      return self._matrices['TL',x,y]
    assert y and y!=self._Ny-1
    return self._matrices['L',x,y]

  def getRmatrix(self,x,y,usecorner=False):
    """Right edge transfer matrix"""
    assert x
    if usecorner and y == 0:
      return self._matrices['BR',x,y]
    if usecorner and y == self._Ny-1:
      return self._matrices['TR',x,y]
    assert y and y!=self._Ny-1
    return self._matrices['R',x,y]

  def getTLmatrix(self,x,y):
    """Upper-left corner transfer matrix"""
    assert x!=self._Nx-1 and y
    return self._matrices['TL',x,y]
  
  def getTRmatrix(self,x,y):
    """Upper-right corner transfer matrix"""
    assert x and y
    return self._matrices['TR',x,y]

  def getBLmatrix(self,x,y):
    """Lower-left corner transfer matrix"""
    assert x!=self._Nx-1 and y!=self._Ny-1
    return self._matrices['BL',x,y]

  def getBRmatrix(self,x,y):
    """Lower-right corner transfer matrix"""
    assert x and y!=self._Ny-1
    return self._matrices['BR',x,y]

  def settensor(self,x,y,T,reset_matrices=True):
    # TODO multiple at once and/or no checks
    if x:
      assert T.getspace('l') ^ self._tensors[x-1][y].getspace('r')
    if x!=self._Nx-1 and x!=-1:
      assert T.getspace('r') ^ self._tensors[x+1][y].getspace('l')
    if y:
      assert T.getspace('b') ^ self._tensors[x][y-1].getspace('t')
    if y!=self._Ny-1 and y!=-1:
      assert T.getspace('t') ^ self._tensors[x][y+1].getspace('b')
    self._tensors[x][y] = T
    if reset_matrices:
      # Remove transfer matrices (TODO; & supporting isometries?)
      # dependent on (x,y)
      # Upper edge (below tensor)
      for y1 in range(y,0,-1):
        if ('T',x,y1) in self._matrices:
          del self._matrices['T',x,y1]
      # Lower edge (above tensor)
      for y1 in range(y,self._Ny-1):
        if ('B',x,y1) in self._matrices:
          del self._matrices['B',x,y1]
      # Left edge (to right of tensor)
      for x1 in range(x,self._Nx-1):
        if ('L',x1,y) in self._matrices:
          del self._matrices['L',x1,y]
      # Right edge
      for x1 in range(x,0,-1):
        if ('R',x1,y) in self._matrices:
          del self._matrices['R',x1,y]
      for x1 in range(x,self._Nx-1):
        # Upper-left corner
        for y1 in range(y,0,-1):
          if ('TL',x1,y1) in self._matrices:
            del self._matrices['TL',x1,y1]
        # Lower-left corner
        for y1 in range(y,self._Ny-1):
          if ('BL',x1,y1) in self._matrices:
            del self._matrices['BL',x1,y1]
      for x1 in range(x,0,-1):
        # Upper-right corner
        for y1 in range(y,0,-1):
          if ('TR',x1,y1) in self._matrices:
            del self._matrices['TR',x1,y1]
        # Lower-right corner
        for y1 in range(y,self._Ny-1):
          if ('BR',x1,y1) in self._matrices:
            del self._matrices['BR',x1,y1]

  def reduce_simple(self, edge, x0, y0, chiCTM):
    # Basic spectral decomposition like in zipper (but contracts free indices)
    # x0,y0 correspond to the smaller of the coordinates for the edge matrices
    # this is used to construct
    # TODO move to "plaquette subcomplex"?
    bonds = ['bra0.q-ket0.q','bra1.q-ket1.q']
    T0 = self.gettensor(x0,y0,with_ps_in=('l','r','t','b'))
    if edge == 'T' or edge == 'B':
      T1 = self.gettensor(x0+1,y0,with_ps_in=('r','t','b'))
      bonds.extend(['bra0.r-bra1.l','E0.rb-E1.lb'])
      if edge == 'T':
        logger.info('Reducing (%d,%d)-(%d,%d) bond with edge bond above',
          x0,y0,x0+1,y0)
        E0 = self.getTmatrix(x0,y0+1,usecorner=True)
        E1 = self.getTmatrix(x0+1,y0+1,usecorner=True)
        bonds.extend(['E0.bb-bra0.t','E0.bk-ket0.t',
          'E1.bb-bra1.t','E1.bk-ket1.t','bra0.b-ket0.b','bra1.b-ket1.b'])
      else:
        logger.info('Reducing (%d,%d)-(%d,%d) bond with edge bond below',
          x0,y0,x0+1,y0)
        E0 = self.getBmatrix(x0,y0-1,usecorner=True)
        E1 = self.getBmatrix(x0+1,y0-1,usecorner=True)
        bonds.extend(['E0.tb-bra0.b','E0.tk-ket0.b',
          'E1.tb-bra1.b','E1.tk-ket1.b','bra0.t-ket0.t','bra1.t-ket1.t'])
      if x0:
        E0 = E0.trace('lb-lk')
        bonds.append('bra0.l-ket0.l')
      if x0 != self._Nx-2:
        E1 = E1.trace('rb-rk')
        bonds.append('bra1.r-ket1.r')
      out0 = {('E0','rk'):'e0',('ket0','r'):'v0'}
      out1 = {('E1','lk'):'e1',('ket1','l'):'v1'}
    else:
      T1 = self.gettensor(x0,y0+1,with_ps_in=('l','r','t'))
      bonds.extend(['bra0.t-bra1.b','E0.tb-E1.bb'])
      if edge == 'L':
        logger.info('Reducing (%d,%d)-(%d,%d) bond with edge bond to left',
          x0,y0,x0,y0+1)
        E0 = self.getLmatrix(x0-1,y0,usecorner=True)
        E1 = self.getLmatrix(x0-1,y0+1,usecorner=True)
        bonds.extend(['E0.rb-bra0.l','E0.rk-ket0.l',
          'E1.rb-bra1.l','E1.rk-ket1.l','bra0.r-ket0.r','bra1.r-ket1.r'])
      else:
        logger.info('Reducing (%d,%d)-(%d,%d) bond with edge bond to right',
          x0,y0,x0,y0+1)
        E0 = self.getRmatrix(x0+1,y0,usecorner=True)
        E1 = self.getRmatrix(x0+1,y0+1,usecorner=True)
        bonds.extend(['E0.lb-bra0.r','E0.lk-ket0.r',
          'E1.lb-bra1.r','E1.lk-ket1.r','bra0.l-ket0.l','bra1.l-ket1.l'])
      if y0:
        E0 = E0.trace('bb-bk')
        bonds.append('bra0.b-ket0.b')
      if y0 != self._Ny-2:
        E1 = E1.trace('tb-tk')
        bonds.append('bra1.t-ket1.t')
      out0 = {('E0','tk'):'e0',('ket0','t'):'v0'}
      out1 = {('E1','bk'):'e1',('ket1','b'):'v1'}
    #logger.debug('(%d,%d) edge %s: %s',x0,y0,edge,','.join(bonds))
    net = networks.Network.network('E0;E1;bra0*,ket0;bra1*,ket1;' + \
      ','.join(bonds),E0,E1,T0,T1)
    #net.optimize()
    #print(list(net.freeindices()))
    #net.tree.pprint()
    M = net.contract({**out0,**out1})
    logger.debug('Obtaining chi-%d reduction tensors via spectral decomposition',chiCTM)
    w,vR,vL = M.eig('e1-e0,v1-v0',herm=False,selection=chiCTM,left=True,
      mat=('1','0'), zero_tol=ztol)
    logger.debug('Returning reduction tensors')
    if edge == 'T':
      vR = vR.renamed('e1-t,v1-b,1-r')
      vL = vL.renamed('e0-t,v0-b,0-l')
      self._reducers['Tr',(x0,x0+1),y0] = vL
      self._reducers['Tl',(x0,x0+1),y0] = vR
    if edge == 'B':
      vR = vR.renamed('e1-b,v1-t,1-r')
      vL = vL.renamed('e0-b,v0-t,0-l')
      self._reducers['Br',(x0,x0+1),y0] = vL
      self._reducers['Bl',(x0,x0+1),y0] = vR
    if edge == 'L':
      vR = vR.renamed('e1-l,v1-r,1-t')
      vL = vL.renamed('e0-l,v0-r,0-b')
      self._reducers['Lt',x0,(y0,y0+1)] = vL
      self._reducers['Lb',x0,(y0,y0+1)] = vR
    if edge == 'R':
      vR = vR.renamed('e1-r,v1-l,1-t')
      vL = vL.renamed('e0-r,v0-l,0-b')
      self._reducers['Rt',x0,(y0,y0+1)] = vL
      self._reducers['Rb',x0,(y0,y0+1)] = vR
    return vL,vR

  def renormalize_edge(self, edge, x0, y0, chiCTM=None, fromtensor=None):
    """Contract with reducers
    (default: assume reducers already present; if chiCTM provided, check)
    May provide (spatially) prior edge tensor as fromtensor"""
    # NOTE edge tensors do not include "external" (bulk-facing) pseudo-schmidts
    # (therefore must always include boundary-facing in renormalization!)
    # TODO operator insertion
    if edge == 'T':
      T = self.gettensor(x0,y0,with_ps=('t','r'))
      logger.info('Reducing (%d,%d) tensor with edge above', x0,y0)
      if fromtensor is None:
        E = self.getTmatrix(x0,y0+1)
      else:
        E = fromtensor
      if chiCTM and ('Tr',(x0-1,x0),y0) not in self._reducers:
        self.reduce_simple('T',x0-1,y0,chiCTM)
      if chiCTM and ('Tl',(x0,x0+1),y0) not in self._reducers:
        self.reduce_simple('T',x0,y0,chiCTM)
      wL = self._reducers['Tr',(x0-1,x0),y0]
      wR = self._reducers['Tl',(x0,x0+1),y0]
      net = networks.Network.network('E;bra*,ket;Lb*,Lk;Rb*,Rk;'
        'E.bk-ket.t,E.bb-bra.t,E.lk-Lk.t,E.lb-Lb.t,E.rk-Rk.t,E.rb-Rb.t,'
        'ket.l-Lk.b,ket.r-Rk.b,bra.l-Lb.b,bra.r-Rb.b,ket.q-bra.q;'
        'Lk.l>lk,Lb.l>lb,Rk.r>rk,Rb.r>rb,ket.b>bk,bra.b>bb', E,T,wL,wR)
    elif edge == 'B':
      T = self.gettensor(x0,y0,with_ps=('b','r'))
      logger.info('Reducing (%d,%d) tensor with edge below', x0,y0)
      if fromtensor is None:
        E = self.getBmatrix(x0,y0-1)
      else:
        E = fromtensor
      if chiCTM and ('Br',(x0-1,x0),y0) not in self._reducers:
        self.reduce_simple('B',x0-1,y0,chiCTM)
      if chiCTM and ('Bl',(x0,x0+1),y0) not in self._reducers:
        self.reduce_simple('B',x0,y0,chiCTM)
      wL = self._reducers['Br',(x0-1,x0),y0]
      wR = self._reducers['Bl',(x0,x0+1),y0]
      net = networks.Network.network('E;bra*,ket;Lb*,Lk;Rb*,Rk;'
        'E.tk-ket.b,E.tb-bra.b,E.lk-Lk.b,E.lb-Lb.b,E.rk-Rk.b,E.rb-Rb.b,'
        'ket.l-Lk.t,ket.r-Rk.t,bra.l-Lb.t,bra.r-Rb.t,ket.q-bra.q;'
        'Lk.l>lk,Lb.l>lb,Rk.r>rk,Rb.r>rb,ket.t>tk,bra.t>tb', E,T,wL,wR)
    elif edge == 'L':
      T = self.gettensor(x0,y0,with_ps=('l','t'))
      logger.info('Reducing (%d,%d) tensor with edge to left', x0,y0)
      if fromtensor is None:
        E = self.getLmatrix(x0-1,y0)
      else:
        E = fromtensor
      if chiCTM and ('Lt',x0,(y0-1,y0)) not in self._reducers:
        self.reduce_simple('L',x0,y0-1,chiCTM)
      if chiCTM and ('Lb',x0,(y0,y0+1)) not in self._reducers:
        self.reduce_simple('L',x0,y0,chiCTM)
      wB = self._reducers['Lt',x0,(y0-1,y0)]
      wT = self._reducers['Lb',x0,(y0,y0+1)]
      net = networks.Network.network('E;bra*,ket;Tb*,Tk;Bb*,Bk;'
        'E.rk-ket.l,E.rb-bra.l,E.tk-Tk.l,E.tb-Tb.l,E.bk-Bk.l,E.bb-Bb.l,'
        'ket.t-Tk.r,ket.b-Bk.r,bra.t-Tb.r,bra.b-Bb.r,ket.q-bra.q;'
        'Tk.t>tk,Tb.t>tb,Bk.b>bk,Bb.b>bb,ket.r>rk,bra.r>rb', E,T,wT,wB)
    elif edge == 'R':
      T = self.gettensor(x0,y0,with_ps=('r','t'))
      logger.info('Reducing (%d,%d) tensor with edge to right', x0,y0)
      if fromtensor is None:
        E = self.getRmatrix(x0+1,y0)
      else:
        E = fromtensor
      if chiCTM and ('Rt',x0,(y0-1,y0)) not in self._reducers:
        self.reduce_simple('R',x0,y0-1,chiCTM)
      if chiCTM and ('Rb',x0,(y0,y0+1)) not in self._reducers:
        self.reduce_simple('R',x0,y0,chiCTM)
      wB = self._reducers['Rt',x0,(y0-1,y0)]
      wT = self._reducers['Rb',x0,(y0,y0+1)]
      net = networks.Network.network('E;bra*,ket;Tb*,Tk;Bb*,Bk;'
        'E.lk-ket.r,E.lb-bra.r,E.tk-Tk.r,E.tb-Tb.r,E.bk-Bk.r,E.bb-Bb.r,'
        'ket.t-Tk.l,ket.b-Bk.l,bra.t-Tb.l,bra.b-Bb.l,ket.q-bra.q;'
        'Tk.t>tk,Tb.t>tb,Bk.b>bk,Bb.b>bb,ket.l>lk,bra.l>lb', E,T,wT,wB)
    T1 = net.contract()
    if fromtensor is None:
      # Otherwise do not store
      self._matrices[edge,x0,y0] = T1
    return T1

  def renormalizeTL(self, x0,y0, chiCTM=None, fromdir=None,fromtensor=None):
    """Contract with reducers to get new corner matrices
    If chiCTM is required reducers may be calculated if necessary
    fromdir may be 'l' (use existing corner matrix from left) or 't'
      default behavior: preference to take fastest path to boundary
      (if only one exists, or if only one reducer exists even if chiCTM is
      provided, will use that one)
    fromtensor may optionally be said existing corner tensor (in which case
      fromdir must be provided)
      Setting fromtensor to False will proceed by default without overwriting
        existing corner matrix"""
    # TODO operator insertion
    if fromdir is None:
      assert fromtensor is None
      if x0 == 0 or ('TL',x0-1,y0) not in self._matrices or \
          ('T',x0,y0) not in self._matrices:
        fromdir = 't'
      elif y0 == self._Ny-1 or ('TL',x0,y0+1) not in self._matrices or \
          ('L',x0,y0) not in self._matrices:
        fromdir = 'l'
      else:
        # Both exist, check reducers
        lred = ('Lt',x0,(y0-1,y0)) in self._reducers 
        tred = ('Tl',(x0,x0+1),y0) in self._reducers 
        if lred ^ tred:
          fromdir = 'l' if lred else tred
        else:
          assert lred or chiCTM
          if self._Ny-y0-1 < x0:
            fromdir = 't'
          else:
            fromdir = 'l'
    if fromdir == 'l':
      if fromtensor:
        Cl = fromtensor
      else:
        Cl = self.getTLmatrix(x0-1,y0)
      if chiCTM and ('Lt',x0,(y0-1,y0)) not in self._reducers:
        self.reduce_simple('L',x0,y0-1,chiCTM)
      E = self.getTmatrix(x0,y0)
      wB = self._reducers['Lt',x0,(y0-1,y0)]
      net = networks.Network.network('C;E;Bk,Bb*;C.rk-E.lk,C.rb-E.lb,C.bk-Bk.l,'
        'C.bb-Bb.l,E.bk-Bk.r,E.bb-Bb.r;E.rk>rk,E.rb>rb,Bk.b>bk,Bb.b>bb', Cl,E,wB)
    else:
      if fromtensor:
        Ct = fromtensor
      else:
        Ct = self.getTLmatrix(x0,y0+1)
      if chiCTM and ('Tl',(x0,x0+1),y0) not in self._reducers:
        self.reduce_simple('T',x0,y0,chiCTM)
      E = self.getLmatrix(x0,y0)
      wR = self._reducers['Tl',(x0,x0+1),y0]
      net = networks.Network.network('C;E;Rk,Rb*;C.bk-E.tk,C.bb-E.tb,C.rk-Rk.t,'
        'C.rb-Rb.t,E.rk-Rk.b,E.rb-Rb.b;E.bk>bk,E.bb>bb,Rk.r>rk,Rb.r>rb', Ct,E,wR)
    C = net.contract()
    if fromtensor is None:
      self._matrices['TL',x0,y0] = C
    return C

  def renormalizeTR(self, x0,y0, chiCTM=None, fromdir=None,fromtensor=None):
    """Contract with reducers to get new corner matrices
    If chiCTM is required reducers may be calculated if necessary
    fromdir may be 'r' (use existing corner matrix from right) or 't'
      default behavior: preference to take fastest path to boundary
      (if only one exists, or if only one reducer exists even if chiCTM is
      provided, will use that one)
    fromtensor may optionally be said existing corner tensor (in which case
      fromdir must be provided)"""
    # TODO operator insertion
    if fromdir is None:
      assert fromtensor is None
      if x0 == self._Nx-1 or ('TR',x0+1,y0) not in self._matrices or \
          ('T',x0,y0) not in self._matrices:
        fromdir = 't'
      elif y0 == self._Ny-1 or ('TR',x0,y0+1) not in self._matrices or \
          ('R',x0,y0) not in self._matrices:
        fromdir = 'r'
      else:
        # Both exist, check reducers
        rred = ('Rt',x0,(y0-1,y0)) in self._reducers 
        tred = ('Tr',(x0-1,x0),y0) in self._reducers 
        if rred ^ tred:
          fromdir = 'r' if rred else tred
        else:
          assert rred or chiCTM
          if self._Ny-y0-1 < self._Nx-x0-1:
            fromdir = 't'
          else:
            fromdir = 'r'
    if fromdir == 'r':
      if fromtensor:
        Cr = fromtensor
      else:
        Cr = self.getTRmatrix(x0+1,y0)
      if chiCTM and ('Rt',x0,(y0-1,y0)) not in self._reducers:
        self.reduce_simple('R',x0,y0-1,chiCTM)
      E = self.getTmatrix(x0,y0)
      wB = self._reducers['Rt',x0,(y0-1,y0)]
      net = networks.Network.network('C;E;Bk,Bb*;C.lk-E.rk,C.lb-E.rb,C.bk-Bk.r,'
        'C.bb-Bb.r,E.bk-Bk.l,E.bb-Bb.l;E.lk>lk,E.lb>lb,Bk.b>bk,Bb.b>bb', Cr,E,wB)
    else:
      if fromtensor:
        Ct = fromtensor
      else:
        Ct = self.getTRmatrix(x0,y0+1)
      if chiCTM and ('Tr',(x0-1,x0),y0) not in self._reducers:
        self.reduce_simple('T',x0-1,y0,chiCTM)
      E = self.getRmatrix(x0,y0)
      wL = self._reducers['Tr',(x0-1,x0),y0]
      net = networks.Network.network('C;E;Lk,Lb*;C.bk-E.tk,C.bb-E.tb,C.lk-Lk.t,'
        'C.lb-Lb.t,E.lk-Lk.b,E.lb-Lb.b;E.bk>bk,E.bb>bb,Lk.l>lk,Lb.l>lb', Ct,E,wL)
    C = net.contract()
    if fromtensor is None:
      self._matrices['TR',x0,y0] = C
    return C

  def renormalizeBL(self, x0,y0, chiCTM=None, fromdir=None,fromtensor=None):
    """Contract with reducers to get new corner matrices
    If chiCTM is required reducers may be calculated if necessary
    fromdir may be 'l' (use existing corner matrix from left) or 'b'
      default behavior: preference to take fastest path to boundary
      (if only one exists, or if only one reducer exists even if chiCTM is
      provided, will use that one)
    fromtensor may optionally be said existing corner tensor (in which case
      fromdir must be provided)"""
    # TODO operator insertion
    if fromdir is None:
      assert fromtensor is None
      if x0 == 0 or ('BL',x0-1,y0) not in self._matrices or \
          ('B',x0,y0) not in self._matrices:
        fromdir = 'b'
      elif y0 == 0 or ('BL',x0,y0-1) not in self._matrices or \
          ('L',x0,y0) not in self._matrices:
        fromdir = 'l'
      else:
        # Both exist, check reducers
        lred = ('Lb',x0,(y0,y0+1)) in self._reducers 
        bred = ('Bl',(x0,x0+1),y0) in self._reducers 
        if lred ^ bred:
          fromdir = 'l' if lred else bred
        else:
          assert lred or chiCTM
          if y0 < x0:
            fromdir = 'b'
          else:
            fromdir = 'l'
    if fromdir == 'l':
      if fromtensor:
        Cl = fromtensor
      else:
        Cl = self.getBLmatrix(x0-1,y0)
      if chiCTM and ('Lb',x0,(y0,y0+1)) not in self._reducers:
        self.reduce_simple('L',x0,y0,chiCTM)
      E = self.getBmatrix(x0,y0)
      wT = self._reducers['Lb',x0,(y0,y0+1)]
      net = networks.Network.network('C;E;Tk,Tb*;C.rk-E.lk,C.rb-E.lb,C.tk-Tk.l,'
        'C.tb-Tb.l,E.tk-Tk.r,E.tb-Tb.r;E.rk>rk,E.rb>rb,Tk.t>tk,Tb.t>tb', Cl,E,wT)
    else:
      if fromtensor:
        Cb = fromtensor
      else:
        Cb = self.getBLmatrix(x0,y0-1)
      if chiCTM and ('Bl',(x0,x0+1),y0) not in self._reducers:
        self.reduce_simple('B',x0,y0,chiCTM)
      E = self.getLmatrix(x0,y0)
      wR = self._reducers['Bl',(x0,x0+1),y0]
      net = networks.Network.network('C;E;Rk,Rb*;C.tk-E.bk,C.tb-E.bb,C.rk-Rk.b,'
        'C.rb-Rb.b,E.rk-Rk.t,E.rb-Rb.t;E.tk>tk,E.tb>tb,Rk.r>rk,Rb.r>rb', Cb,E,wR)
    C = net.contract()
    if fromtensor is None:
      self._matrices['BL',x0,y0] = C
    return C

  def renormalizeBR(self, x0,y0, chiCTM=None, fromdir=None,fromtensor=None):
    """Contract with reducers to get new corner matrices
    If chiCTM is required reducers may be calculated if necessary
    fromdir may be 'l' (use existing corner matrix from left) or 'b'
      default behavior: preference to take fastest path to boundary
      (if only one exists, or if only one reducer exists even if chiCTM is
      provided, will use that one)
    fromtensor may optionally be said existing corner tensor (in which case
      fromdir must be provided)"""
    # TODO operator insertion
    if fromdir is None:
      assert fromtensor is None
      if x0 == self._Nx-1 or ('BR',x0+1,y0) not in self._matrices or \
          ('B',x0,y0) not in self._matrices:
        fromdir = 'b'
      elif y0 == 0 or ('BR',x0,y0-1) not in self._matrices or \
          ('R',x0,y0) not in self._matrices:
        fromdir = 'r'
      else:
        # Both exist, check reducers
        rred = ('Rb',x0,(y0,y0+1)) in self._reducers 
        bred = ('Br',(x0-1,x0),y0) in self._reducers 
        if rred ^ bred:
          fromdir = 'r' if rred else bred
        else:
          assert rred or chiCTM
          if y0 < self._Nx-x0-1:
            fromdir = 'b'
          else:
            fromdir = 'r'
    if fromdir == 'r':
      if fromtensor:
        Cr = fromtensor
      else:
        Cr = self.getBRmatrix(x0+1,y0)
      if chiCTM and ('Rb',x0,(y0,y0+1)) not in self._reducers:
        self.reduce_simple('R',x0,y0,chiCTM)
      E = self.getBmatrix(x0,y0)
      wT = self._reducers['Rb',x0,(y0,y0+1)]
      net = networks.Network.network('C;E;Tk,Tb*;C.lk-E.rk,C.lb-E.rb,C.tk-Tk.r,'
        'C.tb-Tb.r,E.tk-Tk.l,E.tb-Tb.l;E.lk>lk,E.lb>lb,Tk.t>tk,Tb.t>tb', Cr,E,wT)
    else:
      if fromtensor:
        Cb = fromtensor
      else:
        Cb = self.getBRmatrix(x0,y0-1)
      if chiCTM and ('Br',(x0-1,x0),y0) not in self._reducers:
        self.reduce_simple('B',x0-1,y0,chiCTM)
      E = self.getRmatrix(x0,y0)
      wL = self._reducers['Br',(x0-1,x0),y0]
      net = networks.Network.network('C;E;Lk,Lb*;C.tk-E.bk,C.tb-E.bb,C.lk-Lk.b,'
        'C.lb-Lb.b,E.lk-Lk.t,E.lb-Lb.t;E.tk>tk,E.tb>tb,Lk.l>lk,Lb.l>lb', Cb,E,wL)
    C = net.contract()
    if fromtensor is None:
      self._matrices['BR',x0,y0] = C
    return C

  def update_pseudoschmidt(self, s, key):
    s = np.array(s)
    norm = np.linalg.norm(s)
    logger.debug('Normalizing pseudo-schmidt coefficients at %s from %0.8f',key,
      norm)
    s /= norm
    ssort = np.sort(s)
    s0sort = np.sort(self._pseudoschmidt[key])
    dold = len(s0sort)
    dnew = len(s)
    if dold != dnew:
      logger.warning('Updating bond %s from dimension %d to dimension %d',
        key,dold,dnew)
      if dold > dnew:
        s0sort[-dnew:] -= ssort
        diff = np.linalg.norm(s0sort)
      else:
        ssort[-dold:] -= s0sort
        diff = np.linalg.norm(ssort)
    else:
      diff = np.linalg.norm(ssort-s0sort)
    logger.info('Pseudo-schmidt coefficients have diff %0.4g',diff)
    self._pseudoschmidt[key] = s
    return diff

  def pseudoschmidt_incorporated(self, dirs=('t','r')):
    """Return a copy with pseudo-schmidt coefficients incorporated into tensors
    along directions specified (or half in all directions)"""
    if dirs == 'half':
      psargs = dict(with_ps_in=('t','l','b','r'),half_ps=True)
    else:
      assert set(dirs) in [{'t','r'},{'t','l'},{'b','r'},{'b','l'}]
      psargs = dict(with_ps_in=dirs)
    tensors = []
    for x in range(self._Nx):
      xtens = []
      tensors.append(xtens)
      for y in range(self._Ny):
        xtens.append(self.gettensor(x,y,**psargs))
    return PEPSComplex(tensors)

  def useshelf(self, path=None):
    matrices = self._matrices
    self._matrices = CTMShelf(dirname=path)
    for key in list(matrices):
      self._matrices = matrices.pop(key)

  def xsimpleupdate(self, U, x0, y, chi, svd_tol=1e-12, idxparse='lk-lb,rk-rb'):
    """Perform simple update with bond dimension chi and tolerance
    svd_tol between sites (x0,y) and (x0+1,y) using tensor U
      (default indices lb,rb to contract, lk,rk free, modified with idxparse)"""
    Tl = self.gettensor(x0,y,with_ps_in=('l','t','b','r'))
    Tr = self.gettensor(x0+1,y,with_ps_in=('t','b','r'))
    Tl,*auxl = svd_or_fuse(Tl,'r','fl')
    Tr,*auxr = svd_or_fuse(Tr,'l','fr')
    lk,lb,rk,rb = re.fullmatch(r'(\w+)-(\w+),(\w+)-(\w+)',idxparse).groups()
    logger.info('Performing simple update between (%d,%d) and (%d,%d)',
      x0,y,x0+1,y)
    net = networks.Network.network(f'L;R;U;L.q-U.{lb},R.q-U.{rb},L.r-R.l;'
      f'U.{lk}>ql,U.{rk}>qr,~', Tl, Tr, U)
    T = net.contract()
    logger.debug('Splitting site tensors')
    U,s,V = T.svd(f'fl,ql|r,l|fr,qr',chi, svd_tol)
    Tl = contract_or_unfuse(U.renamed('ql-q'),auxl,'r','fl')
    Tr = contract_or_unfuse(V.renamed('qr-q'),auxr,'l','fr')
    # Remove pseudo-schmidt coefficients
    if x0:
      Tl = Tl.diag_mult('l',np.power(self._pseudoschmidt[(x0-1,x0),y],-1))
    if y:
      Tl = Tl.diag_mult('b',np.power(self._pseudoschmidt[x0,(y-1,y)],-1))
      Tr = Tr.diag_mult('b',np.power(self._pseudoschmidt[x0+1,(y-1,y)],-1))
    if x0 < self._Nx-2:
      Tr = Tr.diag_mult('r',np.power(self._pseudoschmidt[(x0+1,x0+2),y],-1))
    if y < self._Ny-1:
      Tl = Tl.diag_mult('t',np.power(self._pseudoschmidt[x0,(y,y+1)],-1))
      Tr = Tr.diag_mult('t',np.power(self._pseudoschmidt[x0+1,(y,y+1)],-1))
    self._tensors[x0][y] = Tl
    self._tensors[x0+1][y] = Tr
    # Return difference in pseudo-schmidt coefficients
    return self.update_pseudoschmidt(s,((x0,x0+1),y))

  def ysimpleupdate(self, U, x, y0, chi, svd_tol=1e-12, idxparse='bk-bb,tk-tb'):
    """Perform simple update with bond dimension chi and tolerance
    svd_tol between sites (x0,y) and (x0+1,y) using tensor U
      (default indices lb,rb to contract, lk,rk free, modified with idxparse)"""
    Tb = self.gettensor(x,y0,with_ps_in=('l','t','b','r'))
    Tt = self.gettensor(x,y0+1,with_ps_in=('t','l','r'))
    Tb,*auxb = svd_or_fuse(Tb,'t','fb')
    Tt,*auxt = svd_or_fuse(Tt,'b','ft')
    bk,bb,tk,tb = re.fullmatch(r'(\w+)-(\w+),(\w+)-(\w+)',idxparse).groups()
    logger.info('Performing simple update between (%d,%d) and (%d,%d)',
      x,y0,x,y0+1)
    net = networks.Network.network(f'B;T;U;B.q-U.{bb},T.q-U.{tb},B.t-T.b;'
      f'U.{bk}>qb,U.{tk}>qt,~', Tb, Tt, U)
    T = net.contract()
    logger.debug('Splitting site tensors')
    U,s,V = T.svd(f'fb,qb|t,b|ft,qt',chi, svd_tol)
    Tb = contract_or_unfuse(U.renamed('qb-q'),auxb,'t','fb')
    Tt = contract_or_unfuse(V.renamed('qt-q'),auxt,'b','ft')
    # Remove pseudo-schmidt coefficients
    if y0:
      Tb = Tb.diag_mult('b',np.power(self._pseudoschmidt[x,(y0-1,y0)],-1))
    if x:
      Tb = Tb.diag_mult('l',np.power(self._pseudoschmidt[(x-1,x),y0],-1))
      Tt = Tt.diag_mult('l',np.power(self._pseudoschmidt[(x-1,x),y0+1],-1))
    if y0 < self._Ny-2:
      Tt = Tt.diag_mult('t',np.power(self._pseudoschmidt[x,(y0+1,y0+2)],-1))
    if x < self._Nx-1:
      Tb = Tb.diag_mult('r',np.power(self._pseudoschmidt[(x,x+1),y0],-1))
      Tt = Tt.diag_mult('r',np.power(self._pseudoschmidt[(x,x+1),y0+1],-1))
    self._tensors[x][y0] = Tb
    self._tensors[x][y0+1] = Tt
    # Return difference in pseudo-schmidt coefficients
    return self.update_pseudoschmidt(s,(x,(y0,y0+1)))

  def trotterupdate_h(self, parity, chi, trotterkeys, Udict, **su_args):
    """Horizontal parts of Trotterized update
    trotterkeys is nested list of (key,idxparse) where key is to Udict"""
    for x in range(parity,self._Nx-1,2):
      for y in range(self._Ny):
        key,ip = trotterkeys[x][y]
        self.xsimpleupdate(Udict[key],x,y,chi,idxparse=ip,**su_args)

  def trotterupdate_v(self, parity, chi, trotterkeys, Udict, **su_args):
    """Vertical parts of Trotterized update
    trotterkeys is nested list of (key,idxparse) where key is to Udict"""
    for y in range(parity, self._Ny-1,2):
      for x in range(self._Nx):
        key,ip = trotterkeys[x][y]
        self.ysimpleupdate(Udict[key],x,y,chi,idxparse=ip,**su_args)

  def trotterupdate(self, chi, delta, order, trotterkeys, Hdict, **su_args):
    """Full Trotter (sub)-step
    trotterkeys is nested list of (key,idxparse) where key is to Hdict
    order may be provided as "-" or "+", or as (direction,parity) pairs"""
    logger.info('Trotter step %s in order %s',delta,order)
    # Exponentiate
    Udict = {}
    for xkeys in trotterkeys[0]+trotterkeys[1]:
      for key,pstr in xkeys:
        if key not in Udict:
          logger.debug('Exponentiating Hamiltonian term with key "%s"',key)
          Udict[key] = Hdict[key].exp(pstr,delta)
    if order == '+':
      order = [('h',0),('v',1),('h',1),('v',0)]
    elif order == '-':
      order = [('h',0),('v',1),('h',1),('v',0)][::-1]
    for direction,parity in order:
      if direction in 'xh':
        self.trotterupdate_h(parity,chi,trotterkeys[0],Udict,**su_args)
      else:
        assert direction in 'yv'
        self.trotterupdate_v(parity,chi,trotterkeys[1],Udict,**su_args)

  def trotterevolve(self, chi, delta, ksuzuki, trotterkeys, Hdict,
      **su_args):
    """Suzuki-expanded Trotter evolution according to given order ksuzuki
    (yes that's 2k in the standard definitions, so what)
    trotterkeys is triple list ([horizontal matrix, vertical matrix])
    of (key,idxparse) where key is to Hdict"""
    assert isinstance(ksuzuki,int) and ksuzuki>0
    if ksuzuki > 2:
      assert ksuzuki%2 == 0
      # Recursive form
      uk = 1/(4 - np.power(4,1/(ksuzuki-1)))
      self.trotterevolve(chi,uk*delta,ksuzuki-2,trotterkeys,Hdict,**su_args)
      self.trotterevolve(chi,uk*delta,ksuzuki-2,trotterkeys,Hdict,**su_args)
      self.trotterevolve(chi,(1-4*uk)*delta,ksuzuki-2,trotterkeys,Hdict,**su_args)
      self.trotterevolve(chi,uk*delta,ksuzuki-2,trotterkeys,Hdict,**su_args)
      self.trotterevolve(chi,uk*delta,ksuzuki-2,trotterkeys,Hdict,**su_args)
    elif ksuzuki == 2:
      self.trotterupdate(chi, delta/2, '+', trotterkeys, Hdict, **su_args)
      self.trotterupdate(chi, delta/2, '-', trotterkeys, Hdict, **su_args)
    else:
      self.trotterupdate(chi, delta, '+', trotterkeys, Hdict, **su_args)

  def compare_pseudoschmidt(self, ps2, which='max'):
    if isinstance(ps2,PEPSComplex):
      ps2 = ps2._pseudoschmidt
    diffs = []
    def getdiff(s1,s2):
      s1 = np.sort(s1)
      s2 = np.sort(s2)
      d1 = len(s1)
      d2 = len(s2)
      if d1 > d2:
        s1[-d2:] -= s2
        diffs.append(np.linalg.norm(s1))
      elif d2 > d1:
        s2[-d1:] -= s1
        diffs.append(np.linalg.norm(s2))
      else:
        diffs.append(np.linalg.norm(s1-s2))
    for x in range(self._Nx-1):
      for y in range(self._Ny):
        getdiff(self._pseudoschmidt[(x,x+1),y],ps2[(x,x+1),y])
    for x in range(self._Nx):
      for y in range(self._Ny-1):
        getdiff(self._pseudoschmidt[x,(y,y+1)],ps2[x,(y,y+1)])
    if which == 'max':
      return max(diffs)
    elif which == 'min':
      return min(diffs)
    elif which == 'sum':
      return sum(diffs)
    elif which == 'norm':
      return np.linalg.norm(diffs)
    else:
      return diffs

  def produce_complex(self, chiCTM):
    """Compute transfer matrices for CTM complex"""
    # TODO more limited parts
    # Edges: contraction of PEPS tensors
    # TODO case chiCTM < chiPEPS?
    Tbl = self.gettensor(0,0,with_ps=('t','r'))
    self._matrices['BL',0,0] = Tbl.contract(Tbl,'q-q;r>rk,t>tk;r>rb,t>tb*')
    Ttl = self.gettensor(0,self._Ny-1,with_ps=('r',))
    self._matrices['TL',0,self._Ny-1] = Ttl.contract(Ttl,
      'q-q;r>rk,b>bk;r>rb,b>bb*')
    Tbr = self.gettensor(self._Nx-1,0,with_ps=('t',))
    self._matrices['BR',self._Nx-1,0] = Tbr.contract(Tbr,
      'q-q;l>lk,t>tk;l>lb,t>tb*')
    Ttr = self.gettensor(self._Nx-1,self._Ny-1)
    self._matrices['TR',self._Nx-1,self._Ny-1] = Ttr.contract(Ttr,
      'q-q;l>lk,b>bk;l>lb,b>bb*')
    # Left & right edges
    for y in range(1,self._Ny-1):
      Tl = self.gettensor(0,y,with_ps=('t',))
      self._matrices['L',0,y] = Tl.contract(Tl,
        'q-q;t>tk,r>rk,b>bk;t>tb,r>rb,b>bb*')
      Tr = self.gettensor(self._Nx-1,y,with_ps=('t',))
      self._matrices['R',self._Nx-1,y] = Tr.contract(Tr,
        'q-q;t>tk,l>lk,b>bk;t>tb,l>lb,b>bb*')
    # Top & bottom edges
    for x in range(1,self._Nx-1):
      Tb = self.gettensor(x,0,with_ps=('r',))
      self._matrices['B',x,0] = Tb.contract(Tb,
        'q-q;t>tk,l>lk,r>rk;t>tb,l>lb,r>rb*')
      Tt = self.gettensor(x,self._Ny-1,with_ps=('r',))
      self._matrices['T',x,self._Ny-1] = Tt.contract(Tt,
        'q-q;b>bk,l>lk,r>rk;b>bb,l>lb,r>rb*')
    # All left edge transfers
    for x in range(1,self._Nx-1):
      for y in range(self._Ny-1):
        self.reduce_simple('L',x,y,chiCTM)
      self.renormalizeBL(x,0)
      for y in range(1,self._Ny-1):
        self.renormalize_edge('L',x,y)
      self.renormalizeTL(x,self._Ny-1)
    # All right edge transfers
    for x in range(self._Nx-2,0,-1):
      for y in range(self._Ny-1):
        self.reduce_simple('R',x,y,chiCTM)
      self.renormalizeBR(x,0)
      for y in range(1,self._Ny-1):
        self.renormalize_edge('R',x,y)
      self.renormalizeTR(x,self._Ny-1)
    # All bottom transfers
    for y in range(1,self._Ny-1):
      for x in range(self._Nx-1):
        self.reduce_simple('B',x,y,chiCTM)
      self.renormalizeBL(0,y)
      for x in range(1,self._Nx-1):
        self.renormalize_edge('B',x,y)
      self.renormalizeBR(self._Nx-1,y)
    # All top transfers
    for y in range(self._Ny-2,0,-1):
      for x in range(self._Nx-1):
        self.reduce_simple('T',x,y,chiCTM)
      self.renormalizeTL(0,y)
      for x in range(1,self._Nx-1):
        self.renormalize_edge('T',x,y)
      self.renormalizeTR(self._Nx-1,y)
    # Corner transfers in bulk
    for dx in range(self._Nx-2):
      for dy in range(self._Ny-2):
        self.renormalizeBL(1+dx,1+dy)
        self.renormalizeBR(self._Nx-dx-2,1+dy)
        self.renormalizeTL(1+dx,self._Ny-dy-2)
        self.renormalizeTR(self._Nx-dx-2,self._Ny-dy-2)

  def expv_single(self, O, x, y, idxparse='k-b'):
    """Expectation value of one single-site operator"""
    lk,lb = idxparse.split('-')
    T = self.gettensor(x,y,with_ps_in=('t','l','r','b'))
    TO = T.mat_mult('q-'+lb,O)
    logger.info('Computing expectation value of single-site operator at '
      '%d,%d',x,y)
    # TODO cache norm
    net = networks.Network.network('ket,bra*;ket.q-bra.q',T)
    # Add edge matrices
    if x > 0:
      net = net.derive('+L;L.rk-ket.l,L.rb-bra.l',self.getLmatrix(x-1,y,True))
    if x < self._Nx-1:
      net = net.derive('+R;R.lk-ket.r,R.lb-bra.r',self.getRmatrix(x+1,y,True))
    if y > 0:
      net = net.derive('+B;B.tk-ket.b,B.tb-bra.b',self.getBmatrix(x,y-1,True))
    if y < self._Ny-1:
      net = net.derive('+T;T.bk-ket.t,T.bb-bra.t',self.getTmatrix(x,y+1,True))
    # Add corner matrices
    if x > 0 and y > 0:
      net = net.derive('+BL;L.bk-BL.tk,L.bb-BL.tb,B.lk-BL.rk,B.lb-BL.rb',
        self.getBLmatrix(x-1,y-1))
    if x < self._Nx-1 and y > 0:
      net = net.derive('+BR;R.bk-BR.tk,R.bb-BR.tb,B.rk-BR.lk,B.rb-BR.lb',
        self.getBRmatrix(x+1,y-1))
    if x > 0 and y < self._Ny-1:
      net = net.derive('+TL;L.tk-TL.bk,L.tb-TL.bb,T.lk-TL.rk,T.lb-TL.rb',
        self.getTLmatrix(x-1,y+1))
    if x < self._Nx-1 and y<self._Ny-1:
      net = net.derive('+TR;R.tk-TR.bk,R.tb-TR.bb,T.rk-TR.lk,T.rb-TR.lb',
        self.getTRmatrix(x+1,y+1))
    normsq = net.contract()
    logger.debug('Norm-squared with center (%d,%d) is %0.8g',x,y,np.real(normsq))
    # Add impurity
    net['ket'] = TO
    return net.contract()/normsq

  def expv_hneighbor(self, O0, O1, x0, y, idxparse='k-b'):
    """Expectation value of two single-site operators
    on horizontally adjacent sites (x0,y), (x0+1,y)"""
    lk,lb = idxparse.split('-')
    T0 = self.gettensor(x0,y,with_ps_in=('t','l','r','b'))
    T0O = T0.mat_mult('q-'+lb,O0)
    T1 = self.gettensor(x0+1,y,with_ps_in=('t','r','b'))
    T1O = T1.mat_mult('q-'+lb,O1)
    logger.info('Computing correlator of single-site operators between '
      '(%d,%d) and (%d,%d)',x0,y,x0+1,y)
    # TODO cache norm
    net = networks.Network.network('ket0,bra0*;ket1,bra1*;ket0.q-bra0.q,'
      'ket1.q-bra1.q,ket0.r-ket1.l,bra0.r-bra1.l',T0,T1)
    # Add edge matrices
    if x0 > 0:
      net = net.derive('+L;L.rk-ket0.l,L.rb-bra0.l',self.getLmatrix(x0-1,y,True))
    if x0 < self._Nx-2:
      net = net.derive('+R;R.lk-ket1.r,R.lb-bra1.r',self.getRmatrix(x0+2,y,True))
    if y > 0:
      net = net.derive('+B0;+B1;B0.tk-ket0.b,B0.tb-bra0.b,'
        'B1.tk-ket1.b,B1.tb-bra1.b,B0.rk-B1.lk,B0.rb-B1.lb',
        self.getBmatrix(x0,y-1,True),self.getBmatrix(x0+1,y-1,True))
    if y < self._Ny-1:
      net = net.derive('+T0;+T1;T0.bk-ket0.t,T0.bb-bra0.t,'
        'T1.bk-ket1.t,T1.bb-bra1.t,T0.rk-T1.lk,T0.rb-T1.lb',
        self.getTmatrix(x0,y+1,True),self.getTmatrix(x0+1,y+1,True))
    # Add corner matrices
    if x0 > 0 and y > 0:
      net = net.derive('+BL;L.bk-BL.tk,L.bb-BL.tb,B0.lk-BL.rk,B0.lb-BL.rb',
        self.getBLmatrix(x0-1,y-1))
    if x0 < self._Nx-2 and y > 0:
      net = net.derive('+BR;R.bk-BR.tk,R.bb-BR.tb,B1.rk-BR.lk,B1.rb-BR.lb',
        self.getBRmatrix(x0+2,y-1))
    if x0 > 0 and y < self._Ny-1:
      net = net.derive('+TL;L.tk-TL.bk,L.tb-TL.bb,T0.lk-TL.rk,T0.lb-TL.rb',
        self.getTLmatrix(x0-1,y+1))
    if x0 < self._Nx-2 and y<self._Ny-1:
      net = net.derive('+TR;R.tk-TR.bk,R.tb-TR.bb,T1.rk-TR.lk,T1.rb-TR.lb',
        self.getTRmatrix(x0+2,y+1))
    normsq = net.contract()
    logger.debug('Norm-squared with center (%d+1/2,%d) is %0.8g',x0,y,np.real(normsq))
    # Add impurities
    net['ket0'] = T0O
    net['ket1'] = T1O
    return net.contract()/normsq

  def expv_vneighbor(self, O0, O1, x, y0, idxparse='k-b'):
    """Expectation value of two single-site operators
    on vertically adjacent sites (x,y0), (x,y0+1)"""
    lk,lb = idxparse.split('-')
    T0 = self.gettensor(x,y0,with_ps_in=('t','l','r','b'))
    T0O = T0.mat_mult('q-'+lb,O0)
    T1 = self.gettensor(x,y0+1,with_ps_in=('t','l','r'))
    T1O = T1.mat_mult('q-'+lb,O1)
    logger.info('Computing correlator of single-site operators between '
      '(%d,%d) and (%d,%d)',x,y0,x,y0+1)
    # TODO cache norm
    net = networks.Network.network('ket0,bra0*;ket1,bra1*;ket0.q-bra0.q,'
      'ket1.q-bra1.q,ket0.t-ket1.b,bra0.t-bra1.b',T0,T1)
    # Add edge matrices
    if y0 > 0:
      net = net.derive('+B;B.tk-ket0.b,B.tb-bra0.b',self.getBmatrix(x,y0-1,True))
    if y0 < self._Ny-2:
      net = net.derive('+T;T.bk-ket1.t,T.bb-bra1.t',self.getTmatrix(x,y0+2,True))
    if x > 0:
      net = net.derive('+L0;+L1;L0.rk-ket0.l,L0.rb-bra0.l,'
        'L1.rk-ket1.l,L1.rb-bra1.l,L0.tk-L1.bk,L0.tb-L1.bb',
        self.getLmatrix(x-1,y0,True),self.getLmatrix(x-1,y0+1,True))
    if x < self._Nx-1:
      net = net.derive('+R0;+R1;R0.lk-ket0.r,R0.lb-bra0.r,'
        'R1.lk-ket1.r,R1.lb-bra1.r,R0.tk-R1.bk,R0.tb-R1.bb',
        self.getRmatrix(x+1,y0,True),self.getRmatrix(x+1,y0+1,True))
    # Add corner matrices
    if x > 0 and y0 > 0:
      net = net.derive('+BL;L0.bk-BL.tk,L0.bb-BL.tb,B.lk-BL.rk,B.lb-BL.rb',
        self.getBLmatrix(x-1,y0-1))
    if x < self._Nx-1 and y0 > 0:
      net = net.derive('+BR;R0.bk-BR.tk,R0.bb-BR.tb,B.rk-BR.lk,B.rb-BR.lb',
        self.getBRmatrix(x+1,y0-1))
    if x > 0 and y0 < self._Ny-2:
      net = net.derive('+TL;L1.tk-TL.bk,L1.tb-TL.bb,T.lk-TL.rk,T.lb-TL.rb',
        self.getTLmatrix(x-1,y0+2))
    if x < self._Nx-1 and y0<self._Ny-2:
      net = net.derive('+TR;R1.tk-TR.bk,R1.tb-TR.bb,T.rk-TR.lk,T.rb-TR.lb',
        self.getTRmatrix(x+1,y0+2))
    normsq = net.contract()
    logger.debug('Norm-squared with center (%d,%d+1/2) is %0.8g',x,y0,np.real(normsq))
    # Add impurities
    net['ket0'] = T0O
    net['ket1'] = T1O
    return net.contract()/normsq


def svd_or_fuse(tensor, direction, fuseidx):
  """Determine if there is an advantage to performing SVD before operation
  that uses only index "direction" and physical index 'q'
  If yes, perform SVD with central index fuseidx; if no, fuse other indices
    into fuseidx
  Return partial tensor, auxiliary data
  Inverse operation is contract_or_unfuse"""
  ds = tensor.dshape
  Lsize = ds.pop(direction) * ds.pop('q')
  Rsize = np.multiply.reduce(list(ds.values()))
  lfused = list(ds)
  if Lsize < Rsize:
    # SVD
    U,s,V = tensor.svd(f'{direction},q|{fuseidx}|'+','.join(ds))
    return U.diag_mult(fuseidx,s), V, lfused
  elif len(lfused) == 1:
    # Just rename
    l = lfused[0]
    return tensor.renamed({l:fuseidx}),l
  else:
    return tensor.fuse(','.join(ds)+f'>{fuseidx};~')+(lfused,)
    
def contract_or_unfuse(tensor, aux, direction, fuseidx):
  """Inverse operation to svd_or_fuse
  aux is as in T,*aux = svd_or_fuse(tensor,direction,fuseidx) """
  if len(aux) == 1:
    # Singleton index
    l, = aux
    return tensor.renamed({fuseidx:l})
  auxobj,lfused = aux
  if isinstance(auxobj,tensors.Tensor):
    # SVD
    return tensor.contract(auxobj,f'{fuseidx}-{fuseidx};~')
  else:
    # Fused
    fjoin = ','.join(lfused)
    return tensor.unfuse(f'{fuseidx}>{fjoin}|{fuseidx};~',auxobj)

import os.path,tempfile
class CTMShelf(PseudoShelf):
  def __init__(self, dirname=None, tmpdir=None):
    if dirname is None:
      self.tempdir = tempfile.TemporaryDirectory(dir=tmpdir)
      dirname = self.tempdir.name
    super().__init__(dirname)

  def fname(self, key):
    return os.path.join(self.path,'%s_%d_%d.p'%key)
