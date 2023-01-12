import quantrada
from quantrada import tensors, networks, operators, links, config
from copy import copy
import numpy as np
from numbers import Number
from .mpsabstract import *

def randMPS(indphys, indvirt, N=None):
  if not N:
    N = len(indphys)
  elif not isinstance(indphys, list):
    indphys = N*[indphys]
  elif N != len(indphys):
    indphys = (N//len(indphys))*indphys
    indphys = indphys[:N]
  if isinstance(indvirt, int):
    chi0 = indvirt
    indvirt = (N-1)*[chi0]
    D = indphys[0].dim
    n = 0
    while D < chi0:
      indvirt[n] = D
      n += 1
      D *= indphys[n].dim
    n = N-1
    D = indphys[n].dim
    while D < chi0:
      n -= 1
      indvirt[n] = D
      D *= indphys[n].dim
    for n in range(N-1):
      indvirt[n] = links.VSpace(indvirt[n])
  elif isinstance(indvirt, links.VSAbstract):
    indvirt = (N-1)*[indvirt]
  elif N-1 != len(indvirt):
    indvirt = ((N-1)//len(indvirt))*indvirt
    indvirt = indvirt[:N-1]
  Ms = [tensors.Tensor(None,('b','r'),(indphys[0],indvirt[0]))]
  for n in range(1,N-1):
    #Ms.append(tensors.Tensor.rand_from('b;l*;r',indphys[n],indvirt[n-1],indvirt[n]))
    Ms.append(tensors.Tensor(None,('b','l','r'),(indphys[n],indvirt[n-1].dual(),indvirt[n])))
  #Ms.append(tensors.Tensor.rand_from('b;l*',indphys[-1],indvirt[-1]))
  Ms.append(tensors.Tensor(None,('b','l'),(indphys[-1],indvirt[-1].dual())))
  return MPS([M.rand_like() for M in Ms])


class MPS(MPSgeneric):
  def __init__(self,Ms,schmidt=None):
    self._matrices = list(Ms)
    N = len(Ms)
    self._Nsites = N
    self._leftcanon = (N-1)*[False] + [None]
    self._rightcanon = [None] + (N-1)*[False]
    if schmidt:
      assert len(schmidt) == N-1
      self._schmidt = list(schmidt)
    else:
      self._schmidt = []
      for n in range(self._Nsites-1):
        if not (Ms[n]._dspace['r'] ^ Ms[n+1]._dspace['l']):
          raise ValueError(f'Failure in bond matching at sites {n}-{n+1}')
        d = Ms[n].dshape['r']
        self._schmidt.append(d*[1/d])
      self.restore_canonical()

  def getTc(self, n):
    return self._matrices[n]

  def getTL(self, n):
    if n == 0:
      return self._matrices[0]
    return self._matrices[n].diag_mult('l',self._schmidt[n-1])

  def getTR(self, n):
    if n == self._Nsites-1:
      return self._matrices[n]
    return self._matrices[n].diag_mult('r',self._schmidt[n])

  def setTL(self, M, n, schmidt=None, unitary=False):
    if n == 0:
      assert schmidt is None
      self.setTc(M, n)
      if unitary:
        self._leftcanon[0] = True
    else:
      MPSgeneric.setTL(self, M, n, schmidt, unitary)

  def setTR(self, M, n, schmidt=None, unitary=False):
    if n == self._Nsites-1:
      assert schmidt is None
      self.setTc(M, n)
      if unitary:
        self._rightcanon[n] = True
    else:
      MPSgeneric.setTR(self, M, n, schmidt, unitary)

  def resetcanon(self, n, bondright=False):
    # Set canonical-form flags to false
    # If bondright, corresponds to a change in the bond to the right of site n,
    # otherwise to a change in the tensor at site n
    if bondright:
      self._leftcanon[n+1] = False
      self._rightcanon[n] = False
    else:
      if n != 0:
        self._rightcanon[n] = False
      if n != self.N-1:
        self._leftcanon[n] = False

  def restore_canonical(self, tol=None, almost_canon=False):
    # TODO stabilized ("almost-canonical") version
    Ms = self._matrices
    schmidt = self._schmidt
    # Left-canonical form
    T = Ms[0].diag_mult('r',schmidt[0])
    U,s,V = T.svd('b|r,l|r',tolerance=tol)
    Ms[0] = U
    T = V.diag_mult('l',s)
    for n in range(1,self.N-1):
      T = T.contract(Ms[n],'r-l;~').diag_mult('r',schmidt[n])
      U,s,V = T.svd('l,b|r,l|r',tolerance=tol)
      Ms[n] = U
      T = V.diag_mult('l',s)
    T = T.contract(Ms[-1],'r-l;~')
    U,s,V = T.svd('l|r,l|b',tolerance=tol)
    Ms[-1] = V
    self._rightcanon[-1] = True
    Ntot = np.linalg.norm(s)
    schmidt[-1] = s/Ntot
    # Right-canonical sweep
    T = U.diag_mult('r',s)
    for n in range(self.N-2,0,-1):
      T = T.contract(Ms[n],'l-r;~')
      V,s,U = T.svd('b,r|l,r|l',tolerance=tol)
      Ni = np.linalg.norm(s)
      schmidt[n-1] = s/Ni
      Ntot *= Ni
      Ms[n] = V.diag_mult('r',np.power(schmidt[n],-1))
      T = U.diag_mult('r',s)
      self._leftcanon[n] = True
      self._rightcanon[n] = True
    Ms[0] = U.contract(Ms[0],'l-r;~')
    self._leftcanon[0] = True
    if config.verbose >= config.VDEBUG:
      print('normalized to',self.normsq())
    return Ntot

  def printcanon(self,compare=False):
    # Ends up displaying arrows for directions that transfer matrices can be
    # "pushed"
    for n in range(self.N-1):
      print('>' if self._leftcanon[n] else '-',end='')
      print(' ' if self.getbond(n,'r')^self.getbond(n,'l') else '|',end='')
      print('<' if self._rightcanon[n+1] else '-',end='')
    print()
    if compare:
      TL = self.getTL(0)
      IL = TL.id_from('r:t-b',TL)
      lc = abs(TL.contract(TL,'b-b;r>t;r>b*')-IL) < 1e-8
      print('>' if lc else '-',end='')
      print(' ' if self.getbond(0,'r')^self.getbond(0,'l') else '|',end='')
      for n in range(1,self.N-1):
        TL = self.getTL(n)
        TR = self.getTR(n)
        IL = TL.id_from('r:t-b',TL)
        IR = TR.id_from('l:t-b',TR)
        lc = abs(TL.contract(TL,'l-l,b-b;r>t;r>b*')-IL) < 1e-8
        rc = abs(TR.contract(TR,'r-r,b-b;l>t;l>b*')-IR) < 1e-8
        print('<' if rc else '-',end='')
        print('>' if lc else '-',end='')
        print(' ' if self.getbond(n,'r')^self.getbond(n,'l') else '|',end='')
      TR = self.getTR(self.N-1)
      IR = TR.id_from('l:t-b',TR)
      rc = abs(TR.contract(TR,'b-b;l>t;l>b*')-IR) < 1e-8
      print('<' if rc else '-')

  def normsq(self):
    return self.dot(self)

  def dot(self, phi):
    assert self.N == phi.N
    T = self._matrices[0].diag_mult('r',self._schmidt[0])
    T = T.contract(phi._matrices[0].diag_mult('r',phi._schmidt[0]),
      'b-b;r>tr;r>br*')
    for n in range(1,self.N-1):
      T = T.contract(self._matrices[n].diag_mult('r',self._schmidt[n]),
        'tr-l;br>br;r>tr,b>q')
      T = T.contract(phi._matrices[n].diag_mult('r',phi._schmidt[n]),
        'br-l,q-b;~;r>br*')
    T = T.contract(self._matrices[-1],'tr-l;~;b>q')
    return T.contract(phi._matrices[-1],'br-l,q-b*')

  # TODO integrate expv1, expv subroutines
  def expv1(self, O, parse, n):
    # parse: t1-b1,t2-b2,...
    subs = parse.split(',')
    nO = len(subs)
    t1,b1 = subs[0].split('-')
    if n == self.N-1:
      assert nO == 1
      T = self._matrices[-1].diag_mult('l',self._schmidt[-1])
      return T.contract(T,'l-l;b>b;b>t*').contract(O,f'b-{t1},t-{b1}')
    TL = self._matrices[n].diag_mult('r',self._schmidt[n])
    if n != 0:
      TL = TL.diag_mult('l',self._schmidt[n-1])
    if nO == 1:
      if n == 0:
        T = TL.contract(TL,'r-r;b>b;b>t*')
      else:
        T = TL.contract(TL,'r-r,l-l;b>b;b>t*')
      return T.contract(O,f'b-{t1},t-{b1}')
    if n != 0:
      T = TL.contract(TL,'l-l;b>b,r>rb;b>t,r>rt*').contract(O,f'b-{t1},t-{b1};~')
    else:
      T = O.contract(TL,f'{t1}-b;~;r>rb').contract(TL,f'{b1}-b;~;r>rt*')
    for m in range(1,nO-1):
      tm,bm = subs[m].split('-')
      Tc = self._matrices[m+n].diag_mult('r',self._schmidt[m+n])
      T = T.contract(Tc,f'{tm}-b,rb-l;~;r>rb').contract(Tc,f'{bm}-b,rt-l;~;r>rt*')
    nlast = n+nO-1
    tn,bn = subs[-1].split('-')
    TR = self._matrices[nlast]
    if nlast != self.N-1:
      TR = TR.diag_mult('r',self._schmidt[nlast])
    T = T.contract(TR, f'{tn}-b,rb-l;~')
    if nlast == self.N-1:
      return T.contract(TR, f'{bn}-b,rt-l*')
    return T.contract(TR, f'{bn}-b,rt-l,r-r*')

  def expv(self, parse, *Oxs):
    """Expectation value of multiple operators
    parse: t1-b1,t2-b2;t1-b1,...
    args alternates sites & operators: x1, then O1 at site x1,
      then x2, then O2 at site x2, etc.
    fermionic case: include ^ at beginning or end of substring for any parity -1     operator; next argument should be parities of physical indices
      (1d array/list if uniform, 2d array/list of lists/etc otherwise)"""
    N = self.N
    subs = parse.split(';')
    nOs = len(subs)
    fermionic = '^' in parse
    Opar = nOs*[False] # Can just refer to this
    Oxs = list(Oxs)
    if fermionic:
      # Process parity argument
      parities = Oxs.pop(0)
      if isinstance(parities[0],Number):
        # Uniform
        parities = N*[parities]
      elif len(parities) < self._Nsites:
        assert N % len(parities) == 0
        nrep = N//len(parities)
        if isinstance(parities, np.ndarray):
          parities = np.tile(parities,(nrep,1))
        else:
          parities = nrep*parities
      # Trim substrings, identify operator parities
      for i,s in enumerate(subs):
        if '^' in s:
          Opar[i] = True
          subs[i] = s.strip('^')
    assert len(Oxs) == 2*nOs
    Os = []
    xs = []
    for i in range(nOs):
      xs.append(Oxs[2*i])
      Os.append(Oxs[2*i+1])
    if nOs == 1:
      return self.expv1(Os[0], subs[0], xs[0])
    # Iterate over operators
    partot = False # Determination of whether JW string is needed
    for iO, O in enumerate(Os):
      # Pre-processing
      x0 = xs[iO]
      subsubs = subs[iO].split(',')
      assert O.rank == 2*len(subsubs)
      Orank = len(subsubs)
      if partot:
        # Apply parity operation to operator
        O = O.diag_mult(subsubs[0].split('-')[1],parities[x0])
      if iO:
        # Include extant transfer matrix
        tstr = 'M;O'
        bstr = f'M.b-B{x0}.l,M.t-T{x0}.l'
        Ts = [transf,O]
      else:
        tstr = 'O'
        Ts = [O]
        if x0:
          # Contract left indices
          bstr = f'B{x0}.l-T{x0}.l'
        else:
          # No left indices
          bstr = ''
      for dx in range(Orank):
        if x0:
          bstr += ','
        # Add info pertaining to site #x
        x = dx+x0
        # Prepare tensors
        A = self.getTL(x)
        it,ib = subsubs[dx].split('-')
        tstr += f';T{x},B{x}*'
        bstr += f'T{x}.b-O.{it},B{x}.b-O.{ib}'
        if dx < Orank-1:
          bstr += f',T{x}.r-T{x+1}.l,B{x}.r-B{x+1}.l'
        elif iO == nOs-1 and x < N-1:
          # Will trace right index
          bstr += f',T{x}.r-B{x}.r'
          A = A.diag_mult('r',self.getschmidt(x))
        Ts.append(A)
      xf = x0+Orank-1
      if iO < nOs-1:
        # Output indices
        ostr = f'T{xf}.r>t,B{xf}.r>b'
      else:
        ostr = ''
      net = networks.Network.network(';'.join((tstr,bstr,ostr)),*Ts)
      transf = net.contract()
      # Change parity as necessary
      if Opar[iO]:
        partot = not partot
      if iO < nOs-1:
        # Contract with intermediate sites
        x0 += Orank
        assert x0 <= xs[iO+1]
        for x in range(x0,xs[iO+1]):
          A = self.getTL(x)
          transf = transf.contract((A.diag_mult('b',parities[x]) if partot else A),'b-l;~;b>q,r>b*')
          transf = transf.contract(A,'t-l,q-b;~;r>t')
    return transf
   
  def tebd_left(self, U, chi):
    ML = self.getTR(0)
    MR = self.getTR(1)
    M = ML.contract(MR,'r-l;b>bl;b>br,r>r')
    T = U.contract(M, 'tl-bl,tr-br;~')
    ML,s,MR = T.svd('bl|r,l|br,r',chi=chi)
    norm = np.linalg.norm(s)
    self.setschmidt(s/norm,0,strict=False)
    self.setTL(ML.renamed('bl-b'),0,unitary=True)
    self.setTR(MR.renamed('br-b'),1,unitary=True)
    return norm

  def tebd_right(self, U, chi, double=True):
    ML = self.getTL(self.N-2)
    MR = self.getTL(self.N-1)
    M = ML.contract(MR,'r-l;b>bl,l>l;b>br')
    T = U.contract(M, 'tl-bl,tr-br;~')
    if double:
      T = T.contract(U, 'bl-tl,br-tr;~')
    ML,s,MR = T.svd('bl,l|r,l|br',chi=chi)
    norm = np.linalg.norm(s)
    self.setschmidt(s/norm,self.N-2,strict=False)
    self.setTL(ML.renamed('bl-b'),self.N-2,unitary=True)
    self.setTR(MR.renamed('br-b'),self.N-1,unitary=True)
    return norm

  def tebd_sweep(self, Us, chi):
    norm = self.tebd_left(Us[0], chi)
    for n in range(1,self.N-2):
      norm *= self.tebd_bulk(Us[n], chi, n)
    norm *= self.tebd_right(Us[-1], chi, double=True)
    for n in range(self.N-3,0,-1):
      norm *= self.tebd_bulk(Us[n], chi, n)
    norm *= self.tebd_left(Us[0], chi)
    return norm

  def tebd_sweeps(self, Us, chi, niter=1000, ncheck=10, delta=1e-5):
    # Expand Us if necessary
    if not isinstance(Us, list):
      Us = [Us]
    if len(Us) == 1:
      Us = (self.N-1)*Us
    elif len(Us) < self.N-1:
      Us = ((self.N-1)//2+1)*Us
      Us = Us[:self.N]
    # Iterate
    for ni in range(niter):
      if ni % ncheck == 0:
        # Restore double-canonical form and compare Schmidt lists
        self.restore_canonical()
        if ni and delta:
          # Comparison
          diff = self.schmidtdiff(psi0)
          print(f'[{ni}]\t{diff}')
          if diff/self.N < delta:
            return
        psi0 = copy(self)
      self.tebd_sweep(Us, chi)

  def do_tebd(self, H, chis, taus, niters, deltas):
    Nstep = len(taus)
    if isinstance(chis, int):
      chis = Nstep*[chis]
    if isinstance(niters, int):
      niters = Nstep*[niters]
    if isinstance(deltas, float):
      deltas = Nstep*[deltas]

    if not isinstance(H, list):
      H = [H]
    for itau,tau in enumerate(taus):
      print(f'tau={tau} (dim {chis[itau]}, iter {niters[itau]}, tol {deltas[itau]:0.4g})')
      Us = [Hi.exp('tl-bl,tr-br',-tau) for Hi in H]
      self.tebd_sweeps(Us, chis[itau], niter=niters[itau], delta=deltas[itau])


class MPO(MPOgeneric):
  def expandunitcell(self, M):
    raise NotImplementedError()

  @classmethod
  def _construct_FSM(cls, Ms, initial, final, transition, idx_state,
      state_lists, statedim, pinleft={}, pinright={}):
    N = len(Ms)
    # TODO case N % nproc != 0
    if hasattr(Ms[0],'group'):
      sym = True
      group = Ms[0].group
    else:
      sym = False
    idx0 = idx_state[0]
    # Boundary + "pinning" states
    d0 = Ms[0].shape['l']
    BCL = np.zeros(d0,dtype=config.FIELD)
    BCL[idx0[initial]] = 1
    if sym: # TODO will be untested
      for idx,(s,k) in enumerate(state_lists[0]):
        if k == group.triv and s in pinleft:
          pin = pinleft.pop(s)
          n = 1 if isinstance(pin,Number) else len(pin)
          assert state_lists[0][idx+n-1] == (s,k)
          BCL[idx:idx+n] = pin
      assert len(pinleft) == 0
    else:
      for s,pin in pinleft.items():
        assert (isinstance(pin,Number) and statedim[s] == 1) or len(pin) == statedim[s]
        BCL[idx0[s]:idx0[s]+statedim[s]] = pin
    BCR = np.zeros(d0,dtype=config.FIELD)
    BCR[idx0[final]] = 1
    if sym: # TODO will be untested
      for idx,(s,k) in enumerate(state_lists[0]):
        if k == group.triv and s in pinright:
          pin = pinright.pop(s)
          n = 1 if isinstance(pin,Number) else len(pin)
          assert state_lists[0][idx+n-1] == (s,k)
          BCR[idx:idx+n] = pin
      assert len(pinright) == 0
    else:
      for s,pin in pinright.items():
        assert (isinstance(pin,Number) and statedim[s] == 1) or len(pin) == statedim[s]
        BCR[idx0[s]:idx0[s]+statedim[s]] = pin
    # "Trim" inaccessible virtual states
    trimL = []
    # Indicator for accessible states
    ind = (BCL != 0)
    for n in range(N):
      ind = ind.dot(transition[n])
      if np.all(ind):
        break
      trimL.append(ind != 0)
    trimR = []
    ind = (BCR != 0)
    for n in range(N-1,-1,-1):
      ind = transition[n].dot(ind)
      if np.all(ind):
        break
      trimR.append(ind != 0)
    assert len(trimL)+len(trimR) < N-1
    # Use BCL, BCR to terminate MPO
    BCL = Ms[-1].init_fromT(BCL,'r|r')
    BCR = Ms[0].init_fromT(BCR, 'l|l')
    Ms[0] = Ms[0].contract(BCL, 'l-r;~')
    Ms[-1] = Ms[-1].contract(BCR, 'r-l;~')
    # New virtual spaces
    for nl,trim in enumerate(trimL):
      V0 = Ms[nl].getspace('r')
      V1 = V0.trim(trim)
      assert V1.dim == sum(trim)
      I = np.identity(V0.dim,dtype=int)
      U = BCL.init_from(I[trim,:],'r|0,l|1*',V1,V0)
      Ms[nl] = Ms[nl].contract(U,'r-l;~')
      Ms[nl+1] = Ms[nl+1].contract(U,'l-l;~;r>l*')
      state_lists[nl+1] = [state_lists[nl+1][i] for i,t in enumerate(trim) if t]
      #state_lists[nl+1] = np.array(state_lists[nl+1])[trim]
    for nr,trim in enumerate(trimR):
      nn = N-nr-1
      V0 = Ms[nn].getspace('l')
      V1 = V0.trim(trim)
      assert V1.dim == sum(trim)
      I = np.identity(V0.dim,dtype=int)
      U = BCL.init_from(I[trim,:],'l|0,r|1*',V1,V0)
      Ms[nn] = Ms[nn].contract(U,'l-r;~')
      Ms[nn-1] = Ms[nn-1].contract(U,'r-r;~;l>r*')
      state_lists[nn] = [state_lists[nn][i] for i,t in enumerate(trim) if t]
      #state_lists[nn] = np.array(state_lists[nn])[trim]
    return MPO(Ms), state_lists[1:]

  def rand_MPS(self, bonds=None, bond=None, charge=None):
    if bond:
      assert bonds is None
      bonds = (self.N-1)*[bond]
    else:
      assert isinstance(bonds,list) and len(bonds) == self.N-1
    T = self._matrices[0]
    Ms = [T.rand_from('b;r',self._matrices[0],bonds[0])]
    charged = (charge is not None and charge != T.group.triv)
    xc = self.N//2 # Location of charge
    for n in range(1,self.N-1):
      if charged and n == xc:
        Ms.append(T.rand_from('b;r:l*;r',self._matrices[n],Ms[-1],bonds[n],irrep=charge))
      else:
        Ms.append(T.rand_from('b;r:l*;r',self._matrices[n],Ms[-1],bonds[n]))
    Ms.append(T.rand_from('b;r:l*',self._matrices[-1],Ms[-1]))
    if charged:
      return MPSirrep(Ms, charge, xc)
    return MPS(Ms)

  def dot(self, O2):
    assert self.N == O2.N
    M0 = self[0].contract(O2[0],'t-b;b>b,r>br;t>t,r>tr')
    M0, data = M0.fuse('br,tr>r;~')
    Ms = [M0]
    for n in range(1,self.N-1):
      M = self[n].contract(O2[n],'t-b;b>b,l>bl,r>br;t>t,l>tl,r>tr')
      M, data = M.fuse('bl,tl>l|r*;br,tr>r;~', data)
      Ms.append(M)
    M = self[-1].contract(O2[-1],'t-b;b>b,l>bl;t>t,l>tl')
    M, data = M.fuse('bl,tl>l|r*;~',data)
    Ms.append(M)
    return MPO(Ms)

  def getboundarytransfleft(self, psi, lr='l', strict=True):
    T = LeftTransfer(psi, 0, lr, self)
    if lr == 'r':
      T0 = psi.getTR(0)
    else:
      if strict:
        T.setstrict()
      T0 = psi.getTc(0)
    TV = self.getT(0).contract(T0,'t-b;r>c,b>b;r>t')
    TV = TV.contract(T0,'b-b;~;r>b*')
    T.setvalue(TV)
    return T
  
  def getboundarytransfright(self, psi, lr='r', strict=True):
    n = self.N-1
    T = RightTransfer(psi, n, lr, self)
    if lr == 'l':
      T0 = psi.getTL(n)
    else:
      if strict:
        T.setstrict()
      T0 = psi.getTc(n)
    TV = self.getT(n).contract(T0,'t-b;l>c,b>b;l>t')
    TV = TV.contract(T0,'b-b;~;l>b*')
    T.setvalue(TV)
    return T

  def right_transfer(self, psi, n, collect=False, strict=True):
    # Transfer matrix of expectation with psi from site n on (inclusive)
    rts = self.getboundarytransfright(psi,strict=strict).moveby(self.N-1-n,collect=collect)
    if collect:
      rts = rts[::-1]
    return rts

  def left_transfer(self, psi, n, collect=False):
    # Transfer matrix of expectation with psi from site n on (inclusive)
    return self.getboundarytransfleft(psi,strict=strict).moveby(n,collect=collect)

  def expv(self, psi, strict=False):
    T = self.right_transfer(psi, 1, strict=strict)
    return T.left(terminal=True)

  def DMRG_opt_single_left(self, psi, TR, tol=None):
    #TODO Option to use NetworkOperator
    assert TR.site == 1
    Heff = TR.T.contract(self.getT(0),'c-r;t>rt,b>r;~')
    w,v = Heff.eig('b-t,r-rt')
    # ML will be new tensor, MR is gauge-fixing (moving orthogonality center)
    ML,s,MR = v[0].svd('b|r,l|r')
    psi.setTL(ML,0,unitary=True)
    psi.setschmidt(s/np.linalg.norm(s),0)
    return MR
    #self.getboundarytransfleft(psi)
  
  def DMRG_opt_single_right(self, psi, TL, gauge=None, tol=None):
    # Bond dimension is small - no need to proceed iteratively
    n = self.N-1
    assert TL.site == n-1
    Heff = TL.T.contract(self.getT(n),'c-l;t>lt,b>l;~')
    w,v = Heff.eig('b-t,l-lt')
    # MR will be new tensor, ML is gauge-fixing (moving orthogonality center)
    ML,s,MR = v[0].svd('l|r,l|b')
    psi.setTR(MR, n, unitary=True)
    psi.setschmidt(s/np.linalg.norm(s),n-1)
    return ML

  def DMRG_opt_double_left(self, psi, chi, TR, tol=None):
    # Get starting tensor
    assert TR.site == 2
    M0 = psi.getTR(0).contract(psi.getTR(1),'r-l;b>bl;b>br,~')
    # Don't (?) need to change orthogonality center 
    Heff = operators.NetworkOperator('(T);Ol;Or;R;R.t-T.r,Ol.r-Or.l,'
      'Or.r-R.c,Ol.t-T.bl,Or.t-T.br;R.b>r,Ol.b>bl,Or.b>br',
      self.getT(0),self.getT(1),TR.T)
    # Get leading eigenvector (smallest `algebraic' part, start w/current)
    #if Heff.shape[0] <= keig+1: # TODO remove
    if False:
      # TODO efficient dense calculation, perform eig in operators
      Heff = Heff.asdense(inprefix='t',outprefix='')
      w,v = Heff.eig('bl-tbl,br-tbr,r-tr')
    else:
      w,v = Heff.eigs(keig,which='SA',guess=M0)
    # Get two-site ``matrix''; use SVD for true result
    ML,s,MR = v[np.argmin(w)].svd('bl|r,l|br,r',chi=chi,tolerance=tol)
    ML = ML.renamed('bl-b')
    MR = MR.renamed('br-b')
    psi.setTL(ML,0,unitary=True)
    psi.setTR(MR,1,unitary=True)
    psi.setschmidt(s/np.linalg.norm(s),0)

  def DMRG_opt_double_right(self, psi, chi, TL, tol=None):
    n = self.N-2
    assert TL.site == n-1
    # Starting tensor guess
    M0 = psi.getTL(n).contract(psi.getTL(n+1),'r-l;b>bl,~;b>br')
    Heff = operators.NetworkOperator('L;(T);Ol;Or;L.t-T.l,L.c-Ol.l,'
      'Ol.r-Or.l,Ol.t-T.bl,Or.t-T.br;L.b>l,Ol.b>bl,Or.b>br',
      TL.T,self.getT(n),self.getT(n+1))
    # Get leading eigenvector
    #if Heff.shape[0] <= keig+1: #TODO remove
    if False:
      # TODO efficient dense calculation, perform eig in operators
      Heff = Heff.asdense(inprefix='t',outprefix='')
      w,v = Heff.eig('bl-tbl,br-tbr,l-tl')
    else:
      w,v = Heff.eigs(keig,which='SA',guess=M0)
    ML,s,MR = v[np.argmin(w)].svd('l,bl|r,l|br',chi=chi,tolerance=tol)
    ML = ML.renamed('bl-b')
    MR = MR.renamed('br-b')
    psi.setTL(ML,n,unitary=True)
    psi.setTR(MR,n+1,unitary=True)
    psi.setschmidt(s/np.linalg.norm(s), n)

  def DMRGsweep_single(self, psi, TRs, tol=None):
    # Start on left side
    mat = self.DMRG_opt_single_left(psi, TRs.pop(0), tol)
    TLs = []
    TL = self.getboundarytransfleft(psi)
    # T -- => T -- s(new) -- gauge unitary -- s(old)^-1 --
    # -s- Tr -s- <= -- s(new) -- gauge -- s(old)^-1 -- s(old) -- Tr -s-
    gauge = mat.diag_mult('l',psi.getschmidt(0))
    # Sweep right
    for n in range(1,self.N-1):
      if config.verbose > config.VDEBUG:
        print(f'site {n}, transfers {len(TLs)} .. {len(TRs)}')
      TLs.append(TL)
      mat = self.DMRG_opt_single(psi, n, TL, TRs.pop(0), True, mat, None, tol)
      TL = TL.right()
      # -- T -- => -- T -- s(new) -- gauge unitary -- s(old)^-1 --
      gauge = mat.diag_mult('l',psi.getschmidt(n))
    # Right side
    mat = self.DMRG_opt_single_right(psi, TL, gauge, tol)
    TR = self.getboundarytransfright(psi)
    TRs = []
    # -- T => -- s(old)^-1 -- gauge unitary -- s(new) -- T
    # Sweep left
    for n in range(self.N-2,0,-1):
      TRs.insert(0,TR)
      if config.verbose > config.VDEBUG:
        print(f'site {n}, transfers {len(TLs)} .. {len(TRs)}')
      mat = self.DMRG_opt_single(psi, n, TLs.pop(), TR, False, None, mat, tol)
      TR = TR.left()
      # -- T -- => -- s(old)^-1 -- gauge unitary -- s(new) -- T --
      #gauge = mat.diag_mult('r',psi._schmidt[n-1])
    #assert TR.T.getspace('t') ^ psi[0].getspace('r')
    #TR.gauge(mat)
    TRs.insert(0,TR)
    # Correct with gauge unitary
    psi.rgauge(mat,0)
    assert TR.T.getspace('t') ^ psi[0].getspace('r')
    #psi._matrices[0] = psi._matrices[0].contract(mat,'r-l;~')
    if config.verbose >= config.VDEBUG:
      psi.printcanon() #DEBUG
    return TRs

  def DMRGsweep_double(self, psi, chi, tol=None):
    # Should start out as canonical
    # Get right transfer matrices before starting
    if config.verbose >= config.VDEBUG:
      psi.printcanon(compare=True) #DEBUG
    TRs = self.right_transfer(psi, 2, collect=True)
    # Optimize on left site
    self.DMRG_opt_double_left(psi, chi, TRs.pop(0), tol)
    TLs = []
    TL = self.getboundarytransfleft(psi)
    # Sweep rightward
    for n in range(1,self.N-2):
      TLs.append(TL)
      if config.verbose > config.VDEBUG:
        print(f'site {n}/{n+1}, transfers {len(TLs)} .. {len(TRs)}, schmidt rank {len(psi._schmidt[n])}')
      self.DMRG_opt_double(psi, n, chi, TL, TRs.pop(0), None, True, tol)
      TL = TL.right()
    # Optimize on right side
    self.DMRG_opt_double_right(psi, chi, TL, tol)
    TRs = []
    TR = self.getboundarytransfright(psi)
    # Sweep leftward
    for n in range(self.N-3,0,-1):
      if config.verbose > config.VDEBUG:
        print(f'site {n}/{n+1}, transfers {len(TLs)} .. {len(TRs)}, schmidt rank {len(psi._schmidt[n])}')
      TRs.insert(0,TR)
      self.DMRG_opt_double(psi, n, chi, TLs.pop(), TR, None, False, tol)
      TR = TR.left()
    self.DMRG_opt_double_left(psi, chi, TR, tol)
    TRs.insert(0,TR)
    return TRs

  def do_dmrg(self, psi0, chi, Edelta=1e-10, delta2=1e-8,ncanon=10,ncanon2=10,
              nsweep=1000,nsweep2=100,tol=1e-12,tol1=None,tol0=1e-12,Eshift=0,
              transfinit=100,savefile=None,saveprefix=None,cont=False):
    # Instead of alternating single + double update perform double update
    # until reaching threshold delta2 (or nsweep2 iterations) then
    # perform single update until reaching energy-difference threshold
    # Use tol for error threshold in chi^2 truncations (two-site update),
    # tol1 for error threshold in chi^1 svd (one-site update + canonical form)
    # tol0 is threshold for initialization
    # Optional: parameters nsweep, nsweep2, Edelta, delta2, ncanon, ncanon2,
    #   tol, tol1 may be made chi-dependent by passing a callable, list, or dict
    if isinstance(chi,list):
      chis,chi = chi,chi[0]
    else:
      chis = [chi]
    nsweep,nsweep2,Edelta,delta2,ncanon,ncanon2,tol,tol1,i1,i2 = chidependent(chis,nsweep,nsweep2,Edelta,delta2,ncanon,ncanon2,tol,tol1,0,0)
    # TODO allow adjustment of nsweeps, deltas based on chi
    ic0 = 0
    sweep = 0
    if savefile and cont and os.path.isfile(savefile):
      print('reloading from',savefile)
      psi, sv = pickle.load(open(savefile,'rb'))
      if isinstance(sv, tuple):
        chi, sd, sweep = sv
        print(f'Start at: chi={chi}, {sd}x, sweep #{sweep}')
        if sd == 1:
          # Skip double
          nsweep2[chi] = 0
          i1[chi] = sweep
          TRs = self.right_transfer(psi, 1, collect=True)
        else:
          i2[chi] = sweep
        ic0 = chis.index(chi)
      else:
        ic0 = -1
        # Check if result represents now-intermediate bond dimension
        if saveprefix and os.path.isfile(f'{saveprefix}c{chis[0]}.p'):
          # Find first uncomputed bond dimension
          ic0 = 1
          while ic0<len(chis) and os.path.isfile(f'{saveprefix}c{chis[ic0]}.p'):
            print('Found state with chi =',chis[ic0])
            ic0 += 1
        elif len(chis) > 1:
          # Get bond dimension from state
          chi = max(psi.getchi(n) for n in range(psi.N))
          ic0 = 0
          while chis[ic0] <= chi and ic0 < len(chis):
            ic0 += 1
          print('Last bond dimension presumed equal to',chis[ic0])
        if ic0 == -1 or ic0 == len(chis):
          print('Already completed: E =',sv)
          return psi,sv
        else:
          chi = chis[ic0]
          if ic0:
            print(f'Completed with chi={chis[ic0-1]} (E={sv:0.8f})')
      psi0 = copy(psi)
    else:
      if isinstance(psi0,str):
        print('loading from',psi0)
        psi0, sv = pickle.load(open(psi0,'rb'))
      elif not isinstance(psi0,MPS):
        bond = psi0 if psi0 else chi
        psi0 = self.rand_MPS(bond=bond)
        psi0.restore_canonical(tol=tol0)
      psi = copy(psi0)
    E = self.expv(psi)
    for ic,chi in enumerate(chis[ic0:]):
      print(f'{chi=:>3} (#{ic})')
      for niter in range(i2[chi],nsweep2[chi]):
        # Perform two-site DMRG
        E0 = E
        TRs = self.DMRGsweep_double(psi,chi,tol=tol[chi])
        E = TRs[0].left().left(terminal=True)
        if config.verbose or (niter%ncanon2[chi] == 0):
          print(f'    \t{np.real(E-E0)}')
        #if (niter+1)%ncanon2 == 0: #TODO
        psi.restore_canonical(almost_canon=True,tol=tol1[chi])
        # Difference in Schmidt indices
        Ldiff = psi.schmidtdiff(psi0)
        ip = abs(psi.dot(psi0)) # Don't take per-site fidelity?
        pdiff = 1-ip
        print(f'[{niter}]\t{Ldiff:0.6g}\t{pdiff:10.6g}\tE={np.real(E)+Eshift:0.10g}')
        if Ldiff < delta2[chi]:
          if savefile:
            pickle.dump((psi,(chi,1,0)),open(savefile,'wb'))
          break
        psi0 = copy(psi)
        if savefile:
          pickle.dump((psi,(chi,2,niter+1)),open(savefile,'wb'))
      for niter in range(i1[chi],nsweep[chi]):
        E0 = E
        if niter % ncanon[chi] == 0:
          psi.restore_canonical(almost_canon=True,tol=tol1[chi])
          # TODO gauge transfers instead?
          TRs = self.right_transfer(psi, 1, collect=True)
        TRs = self.DMRGsweep_single(psi, TRs)
        E = np.real(TRs[0].left(terminal=True))
        if config.verbose >= config.VDEBUG:
          print(f'[{niter}] (E={E+Eshift:0.3f}, norm {np.real(psi.normsq()):0.3f})=>{np.real(E/psi.normsq())} or {np.real(self.expv(psi)/psi.normsq())}')
        elif config.verbose or (niter%ncanon[chi] == 0):
          print(f'[{niter}] E={np.real(E+Eshift):0.8f} ({np.real(E0-E):+0.4g})')
        if abs(E0-E)<Edelta[chi]:
          if savefile and ic < len(chis)-1:
            pickle.dump((psi,(chis[ic+1],2,0)),open(savefile,'wb'))
          break
        if savefile:
          pickle.dump((psi,(chi,1,niter+1)),open(savefile,'wb'))
      Efin = np.real(E)
      # Restore canonical form
      psi.restore_canonical(tol=tol1[chi],almost_canon=True)
      if config.verbose >= config.VDEBUG:
        psi.printcanon(compare=True)
      if saveprefix:
        pickle.dump((psi,Efin), open(f'{saveprefix}c{chi}.p','wb'))
    if savefile:
      pickle.dump((psi,Efin),open(savefile,'wb'))
    return psi,Efin


class MPSirrep(MPS):
  """MPS "charged" (transforms under irrep) at site charge_site"""
  def __init__(self, Ms, irrep, chargesite, schmidt=None):
    super().__init__(Ms, schmidt)
    self._irrep = irrep
    self._site = chargesite

  @property
  def irrep(self):
    return self._irrep

  def charged_at(self,n):
    return self._site == n

  @property
  def charge_site(self):
    return self._site

  @property
  def group(self):
    return self._matrices[0].group

  def __copy__(self):
    psi = self.__class__(self._matrices,self._irrep,self._site,self._schmidt)
    psi._leftcanon = list(self._leftcanon)
    psi._rightcanon = list(self._rightcanon)
    return psi

  def printcanon(self, compare=False):
    e = self.group.triv
    # Verify symmetry
    for n in range(self.N):
      if n != self._site and self._matrices[n]._irrep != e:
        print(f'matrix at {n} transforms under {self._matrices[n]._irrep} instead of trivial {e} expected')
    n = self._site
    if self._matrices[n]._irrep != self._irrep:
      print(f'charged matrix at {n} transforms under {self._matrices[n]._irrep} instead of expected (non-trivial) {self._irrep}')
    super().printcanon(compare)
