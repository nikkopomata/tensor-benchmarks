import quantrada
from quantrada import tensors, networks, operators, links, config, invariant
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
  def __init__(self,Ms,schmidt=None,tol=1e-15):
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
        self._schmidt.append(d*[1])
      if tol is not None:
        self.restore_canonical(tol=tol)

  def iscanon(self):
    # Just check boolean flags
    return all(self._leftcanon[:-1]) and all(self._rightcanon[1:])

  def issite(self, n):
    return (n >= 0) and (n < self._Nsites)

  def getTc(self, n):
    return self._matrices[n]

  def getTL(self, n):
    if n == 0:
      return self._matrices[0]
    if n < 0:
      n = self.N+n
    return self._matrices[n].diag_mult('l',self._schmidt[n-1])

  def getTR(self, n):
    if n == self._Nsites-1:
      return self._matrices[n]
    if n < 0:
      n = self.N+n
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
      config.logger.debug('Imposing left-canonical form at site %d',n)
      U,s,V = T.svd('l,b|r,l|r',tolerance=tol)
      Ms[n] = U
      T = V.diag_mult('l',s)
    T = T.contract(Ms[-1],'r-l;~')
    U,s,V = T.svd('l|r,l|b',tolerance=tol)
    Ms[-1] = V
    self._rightcanon[-1] = True
    Ntot = np.linalg.norm(s)
    config.logger.log(5,'Left-canonical form imposed; pulling out norm %0.10g',
      Ntot)
    schmidt[-1] = s/Ntot
    # Right-canonical sweep
    T = U.diag_mult('r',schmidt[-1])
    for n in range(self.N-2,0,-1):
      T = T.contract(Ms[n],'l-r;~')
      config.logger.debug('Imposing full canonical form at site %d',n)
      V,s,U = T.svd('b,r|l,r|l',tolerance=tol)
      Ni = np.linalg.norm(s)
      schmidt[n-1] = s/Ni
      config.logger.log(5,'Pulling out factor %0.10g',Ni)
      Ntot *= Ni
      Ms[n] = V.diag_mult('r',np.power(schmidt[n],-1))
      T = U.diag_mult('r',schmidt[n-1])
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

  def logcanon(self,printlevel=10,complevel=5,thresh=1e-10):
    rc = map('-<'.__getitem__, self._rightcanon[1:])
    lc = map('->'.__getitem__, self._rightcanon[:-1])
    mid = [' |'[self.getbond(n,'r')^self.getbond(n,'l')] for n in range(self.N-1)]
    canon_logger.log(printlevel, 'canon flags: '+(self.N-1)*'%s%s%s',
      *sum(zip(lc[1:],mid,rc[:-1]),()))
    if canon_logger.getEffectiveLevel() <= complevel:
      rc = []
      TL = self.getTL(0)
      IL = TL.id_from('r:t-b',TL)
      lc = [abs(TL.contract(TL,'b-b;r>t;r>b*')-IL) < thresh]
      for n in range(1,self.N-1):
        TL = self.getTL(n)
        IL = TL.id_from('r:t-b',TL)
        lc.append(abs(TL.contract(TL,'l-l,b-b;r>t;r>b*')-IL) < thresh)
        TR = self.getTR(n)
        IR = TR.id_from('l:t-b',TR)
        rc.append(abs(TR.contract(TR,'r-r,b-b;l>t;l>b*')-IR) < thresh)
      TR = self.getTR(self.N-1)
      IR = TR.id_from('l:t-b',TR)
      rc.append(abs(TR.contract(TR,'b-b;l>t;l>b*')-IR) < thresh)
      lc = map('->'.__getitem__,lc)
      rc = map('-<'.__getitem__,rc)
      canon_logger.log(complevel, 'canon eval: '+(self.N-1)*'%s%s%s',
        *sum(zip(rc,lc,mid),()))
  def normsq(self):
    return self.dot(self)

  def dot(self, phi):
    assert self.N == phi.N
    T = self.getTR(0)
    T = T.contract(phi.getTR(0), 'b-b;r>tr;r>br*')
    for n in range(1,self.N-1):
      T = T.contract(self.getTR(n), 'tr-l;br>br;r>tr,b>q')
      T = T.contract(phi.getTR(n), 'br-l,q-b;~;r>br*')
    T = T.contract(self.getTc(self.N-1),'tr-l;~;b>q')
    return T.contract(phi.getTc(self.N-1),'br-l,q-b*')

  def transfer_expv(self, *Ops, normalize=False):
    """Get expectation value of one or more MPO operators
    (optionally with different state)
    Ops as in call to TransferMatrix
    if normalize, divide by norm(s)"""
    transf = LeftTransfer(self,0,'l',*Ops)
    transf.compute(None)
    val = transf.moveby(self.N-2).right(terminal=True)
    if normalize:
      nm = self.normsq()
      if isinstance(Ops[-1],MPS):
        nm = np.sqrt(nm*Ops[-1].normsq())
      val /= nm
    return val

  def truncate(self, cutoff_tol, verbose=False):
    # Truncate bonds to specified value of Schmidt coefficients
    # Only bother with interior sites
    # TODO fix
    for l in range(1,self.N-2):
      aschmidt = np.array(self._schmidt[l])
      keep = aschmidt > cutoff_tol
      if not np.all(keep):
        if verbose:
          print(f'{l}: {len(keep)}->{sum(keep)}')
        T1 = self._matrices[l].truncate_bond('r', keep)
        self._matrices[l] = T1
        self._matrices[l+1] = self._matrices[l+1].truncate_bond('l*',keep,V=T1.getspace('r'))
        self._schmidt[l] = list(aschmidt[keep])

  def truncate_tochi(self, cutoff_chi, verbose=False):
    # Truncate bonds to specified value of Schmidt coefficients
    # Only bother with interior sites
    # TODO fix for nonabelian symmetries
    for l in range(1,self.N-2):
      if len(self._schmidt[l]) <= cutoff_chi:
        continue
      aschmidt = np.array(self._schmidt[l])
      cutoff = sorted(aschmidt)[-cutoff_chi]
      keep = aschmidt >= cutoff
      if not np.all(keep):
        if verbose:
          print(f'{l}: {len(keep)}->{sum(keep)}')
        T1 = self._matrices[l].truncate_bond('r', keep)
        self._matrices[l] = T1
        self._matrices[l+1] = self._matrices[l+1].truncate_bond('l*',keep,V=T1.getspace('r'))
        self._schmidt[l] = list(aschmidt[keep])
   
  def todense(self,sitenames=None,suffix=''):
    if not sitenames:
      sitenames = [str(n) for n in range(self.N)]
    pnames = [s+suffix for s in sitenames]
    assert 'virt' not in pnames
    # Left half
    L = self.getTL(0).renamed({'b':pnames[0],'r':'virt'})
    for n in range(1,self.N//2+1):
      L = L.contract(self.getTL(n),f'virt-l;~;b>{pnames[n]},r>virt')
    # Right half
    R = self.getTL(-1).renamed({'b':pnames[-1],'l':'virt'})
    for n in range(self.N-2,self.N//2,-1):
      R = R.contract(self.getTL(n),f'virt-r;~;b>{pnames[n]},l>virt')
    return L.contract(R,'virt-virt;~')

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
    df = Ms[-1].shape['r']
    BCR = np.zeros(df,dtype=config.FIELD)
    if len(idx_state) == N+1:
      idxf = idx_state[-1]
    else:
      idxf = idx0
    BCR[idxf[final]] = 1
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
        BCR[idxf[s]:idxf[s]+statedim[s]] = pin
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
    BCL = Ms[0].init_fromT(BCL,'r|l*')
    BCR = Ms[-1].init_fromT(BCR, 'l|r*')
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

  @classmethod
  def FSMinit_restricted(cls, processes, N=None, parities_phys=None,
      parities_virt=None, init_fin=('i','f'), irreps_virt=None, sreps_virt=None,
      missing_phys={}, auto_initfin=True, pinleft={},pinright={},**kwargs):
    """Create MPO from finite-state machine
    Assumes machine may act trivially at ends and in particular behaves
      in a way impossible for a periodic machine
    Processes gives list of statexstate:matrix dictionaries
      Instead of pair may give just one state (key s is equivalent to (s,s))
      Instead of matrix may give array (=> diag()) or scalar (=> id,
        or diag(parity) if fermionic)
    States may be multidimensional, in which case matrices should be
      provided with additional nontrivial dimensions
    If init_fin is provided (default 'i','f'), that is the pair initial, final
      Otherwise (not init_fin) first element is label for initial state,
      last element is label for final
    If auto_initfin (default false) use initial & final states at all sites
      (at least until trimming step)--similar to periodic behavior
    If N is provided, will repeat "processes" as necessary
    If fermionic, will perform Jordan-Wigner automatically:
      provide parities_phys to list parities (list of lists)
      for physical indices
      & parities_virt to identify parities for virtual states
      (makes JW strings extending to the right; to get correct transformation
       make sure fermionic operators are written in ascending order)
    If symmetric, associate each state either with an irrep (in
      irreps_virt) or lists irreps associated with each state
      (either as SumRepresentation object or as decomposition list)
    For finite MPO may provide pinleft and pinright giving boundary ("pinning")
      value of virtual states
    missing_phys is either dictionary to physical spaces not otherwise
      specified or (if all the same) simply the spaces (or dimensions)
      themselves (if dictionary may also include "default" option)
    """
    # TODO collect state transformation info from tensors provided
    # TODO substantiate from charged tensors
    # TODO recombine with main method
    
    # Identify labels for initial & final states
    if init_fin:
      initial,final = init_fin
    else:
      initial = processes.pop(0)
      final = processes.pop()
    fermionic = (parities_phys is not None)
    if fermionic:
      assert parities_virt is not None
      assert len(processes) % len(parities_phys) == 0
    nproc = len(processes)
    if N is None:
      N = nproc
    else:
      assert N%nproc == 0
    assert N > 1
    Ms = []
    states_left = [] # List of enumerated states on the left side of a site
    states_right = [] # List of enumerated states on the right side
    state_lists = [] # states_left but without initial & final
    allstates = {initial,final} # All bond states encountered
    statedims = {} # States with dimension > 1
    Vphys = [] # VSpaces for physical indices
    if not isinstance(missing_phys,dict):
      missing_phys = {n:missing_phys for n in range(nproc)}
    # First iteration over processes:
    # identify information about physical spaces & collect virtual states
    for n,p in enumerate(processes):
      # Initialize states_left & states_right entries with initial & final
      if auto_initfin:
        left = {initial,final}
        right = {initial,final}
      else:
        left = set()
        right = set()
      Vp = None
      if fermionic:
        # Know physical dimension from parities_phys
        d = len(parities_phys[n])
      else:
        d = None
      for ss,M in p.items():
        if isinstance(ss,tuple):
          # Event which changes state
          sl,sr = ss
        else:
          # Event which keeps state the same
          sl,sr = ss,ss
        left.add(sl)
        right.add(sr)
        # Look at action of event to get/verify information about physical space
        if isinstance(M,tensors.Tensor):
          idxs = M.idxset
          assert idxs.issuperset({'t','b'}) and idxs.issubset({'t','b','l','r'})
          # Directly identifies space 
          V = M._dspace['b']
          if Vp is not None:
            assert V == Vp
          else:
            if d is not None:
              assert d == V.dim
            else:
              d = V.dim
            Vp = V
          if 'l' in idxs:
            vdl = M.shape['l']
            if sl in statedims:
              assert vdl == statedims[sl]
            else:
              statedims[sl] = vdl
          if 'r' in idxs:
            vdr = M.shape['r']
            if sr in statedims:
              assert vdr == statedims[sr]
            else:
              statedims[sr] = vdr
        elif isinstance(M,np.ndarray) or isinstance(M,list):
          # Only identifies dimension
          if d is not None:
            assert d == len(M)
          else:
            d = len(M)
      allstates.update(left,right) # Add sets of states
      states_left.append(left)
      states_right.append(right)
      if auto_initfin:
        state_lists.append(list(left-{initial,final}))
      else:
        state_lists.append(list(left))
      # If information is not complete:
      if d is None:
        # Get from missing_phys
        if n in missing_phys:
          mp = missing_phys[n]
        elif 'default' in missing_phys:
          mp = missing_phys['default']
        else:
          raise KeyError(f'No physical space or dimension provided for site {n}')
        if isinstance(mp,links.VSAbstract):
          Vp = mp
        else:
          d = mp
      if Vp is None:
        # Have dimension but not space: make one
        if n in missing_phys and isinstance(missing_phys[n],links.VSAbstract):
          Vp = missing_phys[n]
          assert d == Vp.dim
        else:
          Vp = links.VSpace(d)
      Vphys.append(Vp)
    if auto_initfin:
      state_lists.append(list(states_right[-1]-{initial,final}))
    else:
      state_lists.append(list(states_right[-1]))
    # Check that states processed at a given site are the same as those
    # output by the previous one
    if not states_left[0].issubset({initial} | pinleft.keys()):
      diff = states_left[0] - {initial} - pinleft.keys()
      raise ValueError(f'States {diff} at left edge unaccounted for')
    if not states_right[N-1].issubset({final} | pinright.keys()):
      diff = states_right[N-1] - {final} - pinright.keys()
      raise ValueError(f'States {diff} at right edge unaccounted for')

    for n in range(1,nproc):
      if states_left[n] != states_right[n-1]:
        diffr = states_left[n]-states_right[n-1]
        diffl = states_right[n-1]-states_left[n]
        if diffl and diffr:
          raise ValueError(f'Virtual states on bond between {n-1}-{n} differ: {diffl} on left only, {diffr} on right only')
        elif diffl:
          raise ValueError(f'Virtual states on bond between {n-1}-{n} differ: {diffl} on left only')
        else:
          raise ValueError(f'Virtual states on bond between {n-1}-{n} differ: {diffr} on right only')
    if fermionic:
      # Parities have been identified for all states given
      assert allstates == (set(parities_virt.keys()) | {initial,final})
      if initial not in parities_virt:
        parities_virt[initial] = False
      if final not in parities_virt:
        parities_virt[final] = False
    symmetric = ((irreps_virt is not None) or (sreps_virt is not None))
    if symmetric:
      group = Vphys[0].group
      if sreps_virt is None:
        sreps_virt = {}
      for s,rep in sreps_virt.items():
        if isinstance(rep,links.VSAbstract):
          sreps_virt[s] = list(rep._decomp)
      triv = [(group.triv,1)]
      sreps_virt.update({initial:triv,final:triv})
      if irreps_virt is not None:
        for s,k in irreps_virt.items():
          if s in sreps_virt:
            assert sreps_virt[s] == [(k,1)]
          else:
            sreps_virt[s] = [(k,1)]
      for s,rep in sreps_virt.items():
        d = group.sumdims(rep)
        if s in statedims and s not in {initial,final}:
          assert d == statedims[s]
        else:
          statedims[s] = d
      assert set(sreps_virt.keys()) == allstates
    # Assign indices to states 
    # initial & final are first & last respectively
    slfull = []
    idx_state = []
    # Current length of state_lists does not account for multidimensional states
    for n in range(nproc+1):
      if auto_initfin or initial in state_lists[n]:
        slf = [initial]
        inv = {initial:0}
      for s in state_lists[n]:
        if s in {initial,final}:
          continue
        inv[s] = len(slf)
        if s in statedims:
          slf.extend(statedims[s]*[s])
        else:
          slf.append(s)
      if auto_initfin or final in state_lists[n]:
        inv[final] = len(slf)
        slf.append(final)
      slfull.append(slf)
      idx_state.append(inv)
    statedimfull = {s:1 for s in allstates}
    statedimfull.update(statedims)
    dvirt = [len(slf) for slf in slfull]
    # Transition/adjacency matrices
    transition = []
    # Create matrices
    for n,proc in enumerate(processes):
      d = Vphys[n].dim
      # l t b r
      M = np.zeros((dvirt[n],dvirt[n+1],d,d),dtype=config.FIELD)
      # Transition matrix
      trans = np.zeros((dvirt[n],dvirt[n+1]),dtype=int)
      # Indexing states on left & right bonds
      idxl = idx_state[n]
      idxr = idx_state[n+1]
      # Initial/final entries
      if auto_initfin:
        M[idxl[initial],idxr[initial],:,:] = np.identity(d)
        M[idxl[final],idxr[final],:,:] = np.identity(d)
        trans[idxl[initial],idxr[initial]] = 1
        trans[idxl[final],idxr[final]] = 1
      # Iterate over events
      for ss,mat in proc.items():
        if isinstance(ss,tuple):
          sl,sr = ss
        else:
          sl,sr = ss,ss
        dl,dr = statedimfull[sl],statedimfull[sr]
        # Not 1D: Need multi-dimensional tensor *or* maintaned state
        assert dl == dr or (isinstance(mat,tensors.Tensor) and mat.rank > 2)
        if fermionic and not isinstance(mat,tensors.Tensor):
          # Check: w/ trivial, parity, or generally diagonal transformation,
          # left & right states must have same parity
          assert parities_virt[sr] == parities_virt[sl]
        if symmetric and not isinstance(mat,tensors.Tensor):
          assert sreps_virt[sl] == sreps_virt[sr]
        if mat is None or mat is False:
          # Just identity or parity
          if fermionic and parities_virt[sl]:
            mat = np.diag(parities_phys[n])
          else:
            mat = np.identity(d,dtype=config.FIELD)
        elif isinstance(mat, Number):
          # Identity or parity, with coefficient
          # (may be passed as True, which is 1)
          if fermionic and parities_virt[sl]:
            mat = np.diag(mat*parities_phys[n])
          else:
            mat = mat*np.identity(d,dtype=config.FIELD)
        elif not isinstance(mat,tensors.Tensor):
          if fermionic and parities_virt[sl]:
            # If fermionic should be parity-conserving
            mat = np.multiply(mat,parities_phys[n])
          # Turn into matrix
          mat = np.diag(np.array(mat,dtype=config.FIELD))
          if len(mat.shape) != 2:
            raise ValueError('bare arrays provided must be 1D')
        else:
          if fermionic and parities_virt[sl]:
            # Preceded by odd fermionic terms
            mat = mat.diag_mult('b',parities_phys[n])
          if mat.rank > 2:
            if 'l' in mat:
              assert mat.shape['l'] == dl
            else:
              assert dl == 1
              matp = np.expand_dims(mat.permuted(('r','t','b')),0)
            if 'r' in mat:
              assert mat.shape['r'] == dr
            else:
              assert dr == 1
              matp = np.expand_dims(mat.permuted(('l','t','b')),1)
            if mat.rank == 4:
              matp = mat.permuted(('l','r','t','b'))
            mat = matp
          else:
            mat = mat.permuted(('t','b'))
        if mat.ndim == 2 and (dl != 1 or dr != 1):
          assert dl == dr
          mat = np.tensordot(np.identity(dl),mat,0)
        if mat.ndim == 2:
          M[idxl[sl],idxr[sr],:,:] = mat
          trans[idxl[sl],idxr[sr]] = 1
        else:
          M[idxl[sl]:idxl[sl]+dl,idxr[sr]:idxr[sr]+dr,:,:] = mat
          trans[idxl[sl]:idxl[sl]+dl,idxr[sr]:idxr[sr]+dr] = np.any(mat,axis=(2,3))
      Ms.append(M)
      transition.append(trans)
    if symmetric: # Re-order
      Vvirt = []
      for n in range(nproc+1):
        # Build rep list
        ksegments = defaultdict(list)
        repall = []
        idx0 = 0
        while idx0 < dvirt[n]:
          s = slfull[n][idx0]
          for k,m in sreps_virt[s]:
            dm = m*group.dim(k)
            ksegments[k].append((s,idx0,idx0+dm))
            idx0 += dm
        assert idx0 == dvirt[n]
        # Collect
        idx1 = 0
        idxst2 = {} # Replace inverse dictionary idx_state
        sl2 = [] # Replace state_lists (will have somewhat different form)
        lrep = []
        permuted = np.zeros(dvirt[n],dtype=int)
        for k in ksegments:
          idx0 = idx1
          for s,i0,i1 in ksegments[k]:
            if s not in idxst2:
              idxst2[s] = idx1
            sl2.extend((i1-i0)*[(s,k)])
            idx2 = idx1+i1-i0
            permuted[idx1:idx2] = range(i0,i1)
            idx1 = idx2
          lrep.append((k,(idx1-idx0)//group.dim(k)))
        Vvirt.append(group.SumRep(lrep))
        idx_state[n] = idxst2
        state_lists[n] = sl2
        if n:
          Ms[n-1] = Ms[n-1][:,permuted,:,:]
          transition[n-1] = transition[n-1][:,permuted]
        if n<nproc:
          Ms[n] = Ms[n][permuted,:,:,:]
          transition[n] = transition[n][permuted,:]
      tclass = group.Tensor
    else:
      Vvirt = [links.VSpace(d) for d in dvirt]
      state_lists = slfull
      tclass = tensors.Tensor
    for n in range(N):
      Vp = Vphys[n]
      Ms[n] = tclass.init_from(Ms[n],'l|0*,r|1,t|2*,b|2', Vvirt[n],Vvirt[n+1],Vp)
    return cls._construct_FSM(Ms, initial, final, transition, idx_state, state_lists, statedimfull, pinleft,pinright,**kwargs)

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

  def getboundarytransfleft(self, psi, lr='l', strict=True, manager=None):
    if manager:
      T = LeftTransferManaged(psi, 0, lr, self, manager=manager)
    else:
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
  
  def getboundarytransfright(self, psi, lr='r', strict=True, manager=None):
    n = self.N-1
    if manager:
      T = RightTransferManaged(psi, n, lr, self, manager=manager)
    else:
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

  def right_transfer(self, psi, n, collect=False, strict=True, manager=None):
    # Transfer matrix of expectation with psi from site n on (inclusive)
    rts = self.getboundarytransfright(psi,strict=strict,manager=manager).moveby(self.N-1-n,collect=collect)
    if collect:
      rts = rts[::-1]
    return rts

  def left_transfer(self, psi, n, collect=False,manager=None):
    # Transfer matrix of expectation with psi from site n on (inclusive)
    return self.getboundarytransfleft(psi,manager=manager).moveby(n,collect=collect)

  def expv(self, psi, strict=False):
    T = self.right_transfer(psi, 1, strict=strict)
    return T.left(terminal=True)

  def dagger(self):
    return self.__class__([M.conj().renamed('t-b,b-t') for M in self._matrices])

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

  def DMRG_opt_double_left(self, psi, chi, TR, tol=None, eigtol=None):
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
      kw = dict(tol=eigtol) if eigtol is not None else {}
      w,v = Heff.eigs(keig,which='SA',guess=M0,**kw)
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

  def DMRGsweep_single(self, psi, TRs, tol=None, eigtol=None):
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
      elif config.verbose >= 3:
        print(f'single-update at site',n,'->')
      TLs.append(TL)
      mat = self.DMRG_opt_single(psi, n, TL, TRs.pop(0), True, mat, None, tol, eigtol)
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
      elif config.verbose >= 3:
        print(f'single-update at site',n,'<-')
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

  def DMRGsweep_double(self, psi, chi, tol=None, eigtol=None):
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
      elif config.verbose >= 2:
        print(f'double update at sites {n}->{n+1}')
      self.DMRG_opt_double(psi, n, chi, TL, TRs.pop(0), None, True, tol, eigtol)
      TL = TL.right()
    # Optimize on right side
    self.DMRG_opt_double_right(psi, chi, TL, tol)
    TRs = []
    TR = self.getboundarytransfright(psi)
    # Sweep leftward
    for n in range(self.N-3,0,-1):
      if config.verbose > config.VDEBUG:
        print(f'site {n}/{n+1}, transfers {len(TLs)} .. {len(TRs)}, schmidt rank {len(psi._schmidt[n])}')
      elif config.verbose >= 2:
        print(f'double update at sites {n}<-{n+1}')
      TRs.insert(0,TR)
      self.DMRG_opt_double(psi, n, chi, TLs.pop(), TR, None, False, tol, eigtol)
      TR = TR.left()
    self.DMRG_opt_double_left(psi, chi, TR, tol)
    TRs.insert(0,TR)
    return TRs

  def do_dmrg(self, psi0, chi, Edelta=1e-10, delta2=1e-8,ncanon=10,ncanon2=10,
              nsweep=1000,nsweep2=100,tol=1e-12,tol1=None,tol0=1e-12,Eshift=0,
              eigtol=None, eigtol_rel=None,
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
    nsweep,nsweep2,Edelta,delta2,ncanon,ncanon2,tol,tol1,eigtol,eigtol_rel,i1,i2 = chidependent(chis,nsweep,nsweep2,Edelta,delta2,ncanon,ncanon2,tol,tol1,eigtol,eigtol_rel,0,0)
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
    for ic,chi in enumerate(chis[ic0:],start=ic0):
      print(f'{chi=:>3} (#{ic})')
      etol = eigtol[chi] if (eigtol_rel[chi] is None) else None
      for niter in range(i2[chi],nsweep2[chi]):
        # Perform two-site DMRG
        if config.haltsig:
          print('Exiting due to halt signal...')
          import sys
          sys.exit()
        if etol is None and eigtol_rel[chi]:
          # TODO better initial tolerance value 
          etol = eigtol_rel[chi]*abs(E)
          if etol < max(Edelta[chi],delta2[chi]):
            # Presume artificially low initial energy
            etol = eigtol_rel[chi]
        E0 = E
        TRs = self.DMRGsweep_double(psi,chi,tol=tol[chi],eigtol=etol)
        E = TRs[0].left().left(terminal=True)
        if eigtol_rel[chi]:
          etol = eigtol_rel[chi]*abs(E-E0)
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
        if config.haltsig:
          print('Exiting due to halt signal...')
          import sys
          sys.exit()
        E0 = E
        if niter % ncanon[chi] == 0:
          psi.restore_canonical(almost_canon=True,tol=tol1[chi])
          # TODO gauge transfers instead?
          TRs = self.right_transfer(psi, 1, collect=True)
        TRs = self.DMRGsweep_single(psi, TRs)
        E = np.real(TRs[0].left(terminal=True))
        if eigtol_rel[chi]:
          etol = eigtol_rel[chi]*abs(E-E0)
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
  def __init__(self, Ms, irrep, chargesite, schmidt=None, tol=1e-15):
    super().__init__(Ms, schmidt, tol=tol)
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

# TODO transfer-matrix-fetching methods

class MPSReflected(MPS):
  """'View' of MPS with spatial reflection"""
  def __init__(self, base):
    self.__base = base
    self._Nsites = base._Nsites

  def iscanon(self):
    return self.__base.iscanon()

  def issite(self,n):
    return self.__base.issite(n)

  @property
  def _leftcanon(self):
    return self.__base._rightcanon[::-1]

  @property
  def _rightcanon(self):
    return self.__base._leftcanon[::-1]

  def getTc(self, n):
    T = self.__base.getTc(self.N-n-1)
    return T.renamed({'l':'r','r':'l'},overcomplete=True)

  def getTL(self, n):
    T = self.__base.getTR(self.N-n-1)
    return T.renamed({'l':'r','r':'l'},overcomplete=True)

  def getTR(self, n):
    T = self.__base.getTL(self.N-n-1)
    return T.renamed({'l':'r','r':'l'},overcomplete=True)

  def getschmidt(self, n):
    return self.__base._schmidt[self.N-n-2]

  def setTL(self, *args, **kw_args):
    raise NotImplementedError()

  def setTR(self, *args, **kw_args):
    raise NotImplementedError()

  def restore_canonical(self, **kw_args):
    raise NotImplementedError()

  def __copy__(self):
    # TODO distinguish between copy of view & copy of state
    matrices = [self.getTc(n) for n in range(self.N)]
    schmidts = self.__base._schmidt[::-1]
    if isinstance(self.__base,MPSirrep):
      return MPSirrep(matrices, self.__base._irrep, self.charge_site,
        schmidts)
    else:
      return MPS(matrices, schmidts)
    #return MPSReflected(self.__base)

  def charged_at(self, n):
    return self.__base.charged_at(self.N-n-1)

  def lgauge(self, *args, **kw_args):
    raise NotImplementedError()

  def rgauge(self, *args, **kw_args):
    raise NotImplementedError()

  def regauge_left(self, *args, **kw_args):
    raise NotImplementedError()

  def regauge_right(self, *args, **kw_args):
    raise NotImplementedError()

  # Methods for reflected rep MPS
  @property
  def irrep(self):
    return self.__base._irrep

  @property
  def charge_site(self):
    return self.N-self.__base._site-1

  @property
  def group(self):
    return self.__base.group
