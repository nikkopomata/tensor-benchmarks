# General methods for MPS
import quantrada
from quantrada import tensors, networks, operators, links, config
from copy import copy
import numpy as np
from numbers import Number
from abc import ABC, abstractmethod
from itertools import compress
from collections import defaultdict
import os.path, pickle,sys,uuid, re
import traceback

keig = 5

def chidependent(chis,*parameters):
  """Convert optionally-chi-dependent parameters into dict
  May be scalar, callable, list, or dict"""
  # TODO move to general-purpose toolbox
  # TODO use in methods other than MPO.do_dmrg
  params = list(parameters)
  for ip,p in enumerate(parameters):
    if callable(p):
      params[ip] = {c:p(c) for c in chis}
    elif isinstance(p,list):
      if len(p) < len(chis):
        # Repeat final element
        p.extend((len(chis)-len(p))*[p[-1]])
      params[ip] = {c:p[ic] for ic,c in enumerate(chis)}
    elif isinstance(p,dict):
      # Convert in a way to raise KeyError here
      params[ip] = {c:p[c] for c in chis}
    else:
      # "Scalar"
      params[ip] = {c:p for c in chis}
  return params

# TODO randMPS prep, include invariant tensors

def positive_contribution(As, tolerance):
  # Given leading eigenvectors of a CP map extract a positive matrix
  # First: Separate Hermitian & anti-Hermitian parts
  AH = []
  for A in As:
    Ad = A.ctranspose('t-b')
    AH.extend([A+Ad,1j*(A-Ad)])
  # Orthogonalize
  Hs = []
  for H in AH:
    coeffs = [H0.contract(H,'t-b,b-t') for H0 in Hs]
    if abs(H)**2 - np.linalg.norm(coeffs)**2 < tolerance:
      # Effectively linearly dependent on previous entries
      continue
    assert np.linalg.norm(np.imag(coeffs)) < tolerance
    for i,H0 in enumerate(Hs):
      H -= np.real(coeffs[i])*H0
    Hs.append(H/abs(H))
  # Combine
  # TODO Check claim that this is "effectively linear least squares"
  Ptot = Hs[0].zeros_like()
  for H in Hs:
    Ptot += H.trace('t-b')*H
  return Ptot


class MPSgeneric(ABC):
  # TODO: Compare various possible forms for stability
  # Note: Index Schmidt such that it is to the right of matrix with same
  #   index -- matches finite MPS but conflicts with old version

  @property
  def N(self):
    return self._Nsites

  def __getitem__(self,n):
    return self.getTc(n)

  def __copy__(self):
    psi = self.__class__(self._matrices,self._schmidt)
    psi._leftcanon = list(self._leftcanon)
    psi._rightcanon = list(self._rightcanon)
    return psi

  def getbond(self, n, right=True):
    # Bond vector space between sites n & n+1
    # Default (right=True): use n->n+1 space rather than dual
    if right == True or right == 'r':
      return self.getTc(n).getspace('r')
    else:
      return self.getTc(n+1).getspace('l')

  def getchi(self, n):
    # Bond between site n-n+1
    return len(self.getschmidt(n))

  def getTc(self, n):
    return self._matrices[n%self._Nsites]

  def getTL(self, n):
    n = n%self._Nsites
    return self._matrices[n].diag_mult('l',self._schmidt[n-1])

  def getTR(self, n):
    n = n%self._Nsites
    return self._matrices[n].diag_mult('r',self._schmidt[n])

  def getschmidt(self, n):
    return self._schmidt[n%self._Nsites]

  def charged_at(self, n):
    return False

  def setTc(self, M, n):
    self._matrices[n] = M
    self.resetcanon(n)

  @abstractmethod
  def setTL(self, M, n, schmidt=None, unitary=False):
    if schmidt is not None:
      self._schmidt[n-1] = list(schmidt)
    self._matrices[n] = M.diag_mult('l',np.power(self._schmidt[n-1],-1))
    if unitary:
      self._rightcanon[n] = False
      self._leftcanon[n] = True
    else:
      self.resetcanon(n)

  @abstractmethod
  def setTR(self, M, n, schmidt=None, unitary=False):
    if schmidt is not None:
      self._schmidt[n] = list(schmidt)
    self._matrices[n] = M.diag_mult('r',np.power(self._schmidt[n],-1))
    if unitary:
      self._leftcanon[n] = False
      self._rightcanon[n] = True
    else:
      self.resetcanon(n)

  def setschmidt(self, s, n, strict=True):
    if strict:
      assert len(s) == self._matrices[n].shape['r']
    # TODO address degeneracy w/ non-Abelian symmetries?
    self._schmidt[n] = list(s)
    self.resetcanon(n, bondright=True)

  @abstractmethod
  def resetcanon(self, n, bondright=False):
    # Set canonical-form flags to false
    # If bondright, corresponds to a change in the bond to the right of site n,
    # otherwise to a change in the tensor at site n
    if bondright:
      self._leftcanon[n+1] = False
      self._rightcanon[n] = False
    else:
      self._leftcanon[n] = False
      self._rightcanon[n] = False

  def lgauge(self, U, n, unitary=True):
    # Only transform site at n
    # If unitary, preserves right-canonical form
    self._matrices[n] = self._matrices[n].mat_mult('l-r',U)
    self._leftcanon[n] = False
    if not unitary:
      self._rightcanon[n] = False

  def rgauge(self, U, n, unitary=True):
    # Only transform site at n
    # If unitary, preserves left-canonical form
    self._matrices[n] = self._matrices[n].mat_mult('r-l',U)
    self._rightcanon[n] = False
    if not unitary:
      self._leftcanon[n] = False

  def regauge_left(self, n, U, s, sameside=True, unitary=True):
    # Apply gauge transformation to the left of the Schmidt matrix
    # sameside=True:
    # T[n-1]--old_s--T[n] => T[-]*U^d -- s -- s^-1*U*old_s*T[+]
    # sameside=False:
    # T[n-1]--old_s--T[n] => T[-]*U^d*s^-1 -- s -- U*old_s*T[+]
    if unitary:
      Ud = U.conj().renamed('l-r,r-l')
    else:
      Ud = U.inv('l-r')
    old_schmidt = self.getschmidt(n-1)
    if sameside:
      Ul = Ud
      self.setTc(self.getTc(n-1).mat_mult('r-l',Ud), n-1)
      self.setschmidt(s, n-1)
      Ur = U.diag_mult('r',old_schmidt).diag_mult('l',np.power(s,-1))
    else:
      # T[-]*U^d defines "right-canonical" form at n-1
      Ul = Ud.diag_mult('r',np.power(s,-1))
      self.setTR(self.getTc(n-1).mat_mult('r-l',Ud), n-1, s)
      Ur = U.diag_mult('r',old_schmidt)
    self.setTc(self.getTc(n).mat_mult('l-r',Ur), n)
    # TODO check these are the right matrices
    return Ul,Ur

  def regauge_right(self, n, U, s, sameside=True, unitary=True):
    # Apply gauge transformation to the right of the Schmidt matrix
    # sameside=True:
    # T[n]--old_s--T[n+1] => T[-]*old_s*U*s^-1 -- s -- U^d*T[+]
    # sameside=False
    # T[n]--old_s--T[n+1] => T[-]*old_s*U -- s -- s^-1*U^d*T[+]
    # TODO this function is now untested
    if unitary:
      Ud = U.conj().renamed('l-r,r-l')
    else:
      Ud = U.inv('l-r')
    old_schmidt = self.getschmidt(n)
    if sameside:
      Ur = Ud
      self.setTc(self.getTc(n+1).mat_mult('r-l',Ud), n+1)
      self.setschmidt(s, n)
      Ul = U.diag_mult('l',old_schmidt).diag_mult('r',np.power(s,-1))
    else:
      Ur = Ud.diag_mult('l',np.power(s,-1))
      self.setTL(self.getTc(n+1).mat_mult('l-r',Ud), n+1, s)
      Ul = U.diag_mult('l',old_schmidt)
    self.setTc(self.getTc(n).mat_mult('r-l',Ul), n)
    # TODO check these are the right matrices
    return Ul,Ur

  def lefttransfers(self, T0, n0, lr, *Os):
    # Collect left transfer matrices starting with T0
    # n0 scalar: go n0 -> full unit cell
    # Otherwise tuple (n0, distance)
    #   (here n0 means we start between sites n0-1,n0)
    # lr is 'l' or 'r' for which side Schmidt indices are applied to
    # TODO default values for T0?
    # TODO default behavior breaks for finite
    if isinstance(n0, tuple):
      n0,N = n0
    else:
      N = self._Nsites
    LTs = [self.getlefttransfer(T0, n0-1, lr, *Os)]
    for dn in range(1,N):
      LTs.append(LTs[-1].right())
    return LTs

  def getlefttransfer(self, T0, n, lr, *Os):
    # Left transfer with given value at specified site
    T = LeftTransfer(self, n, lr, *Os)
    T.setvalue(T0)
    return T

  def righttransfers(self, T0, n0, lr, *Os):
    # Collect right transfer matrices starting with T0
    # n0 scalar: go n0-1 -> full unit cell
    # Otherwise tuple (n0, distance)
    #   (here n0 means we start between sites n0-1,n0)
    # lr is 'l' or 'r' for which side Schmidt indices are applied to
    # TODO default values for T0?
    # TODO default behavior breaks for finite
    if isinstance(n0, tuple):
      n0,N = n0
    else:
      N = self._Nsites
    RTs = [self.getrighttransfer(T0, n0, lr, *Os)]
    for dn in range(1,N):
      RTs.append(RTs[-1].left())
    return RTs

  def getrighttransfer(self, T0, n, lr, *Os):
    # Left transfer with given value at specified site
    T = RightTransfer(self, n, lr, *Os)
    T.setvalue(T0)
    return T

  def getentropy(self, n):
    # Bipartite entropy at n
    p = np.power(self.getschmidt(n),2)
    return np.sum(-p*np.log(p))

  def rectifyleft(self, n0, nf, tol=None):
    N = self.N
    if config.verbose >= config.VDEBUG:
      self.printcanon() #DEBUG
    # Convert to left-canonical form
    # Already partially canonical:
    while self._leftcanon[n0%N] and n0<nf:
      n0 += 1
    U0 = None
    for n in range(n0,nf):
      if config.verbose == config.VDEBUG:
        print('->',n%N) #DEBUG
      TR = self.getTR(n)
      if U0 is not None:
        TR = TR.mat_mult('l-r',U0)
      TR = TR.diag_mult('l',self.getschmidt(n-1))
      TL,s,U0 = TR.svd('l,b|r,l|r',tolerance=tol)
      if config.verbose > config.VDEBUG:
        print('->',n%N,abs(U0-U0.init_like(np.diag(np.diag(U0._T)),'l,r')))
      self.setschmidt(s/np.linalg.norm(s),n,strict=False)
      self.setTL(TL,n,unitary=True)
    # Leaves last site "uncorrected"
    return U0

  def rectifyright(self, n0, nf, tol=None):
    N = self.N
    if config.verbose >= config.VDEBUG:
      self.printcanon() #DEBUG
    # Convert to right-canonical form
    # Already partially canonical:
    while self._rightcanon[(n0-1)%N] and n0>nf:
      n0 -= 1
    U0 = None
    for n in range(n0-1,nf-1,-1):
      TL = self.getTL(n)
      if config.verbose >= config.VDEBUG:
        print(n%N,'<-',end=' ') #DEBUG
        if config.verbose > config.VDEBUG:
          TR = self.getTR(n)
          TT = TR.contract(TR,'r-r,b-b;l>t;l>b*')
          print('deviation',abs(TT-TT.id_like('t-b')))
        else:
          print()
      if U0 is not None:
        TL = TL.mat_mult('r-l',U0)
      TL = TL.diag_mult('r',self.getschmidt(n))
      TR,s,U0 = TL.svd('b,r|l,r|l',tolerance=tol)
      if config.verbose > config.VDEBUG:
        #print('<-',n%N,abs(U0-U0.init_like(np.diag(np.diag(U0._T)),'l,r')),abs(U0.diag_mult('r',s)-U0.diag_mult('l',s)))
        UT = U0.permuted(('l','r'))
        UTf = np.flip(UT,1)
        D2 = np.diag(np.power(s,2))
        print('<-',n%N,np.linalg.norm(UT - np.diag(np.diag(UT))),np.linalg.norm(UTf - np.diag(np.diag(UTf))))
        print('\t',np.linalg.norm(UT.dot(D2)-D2.dot(UT)),np.linalg.norm(UTf.dot(D2)-D2.dot(UTf)))
      self.setschmidt(s/np.linalg.norm(s),n-1,strict=False)
      self.setTR(TR,n,unitary=True)
      if config.verbose > config.VDEBUG:
        self.printcanon(True)
    return U0

  def tebd_bulk(self, U, chi, n):
    ML = self.getTL(n).diag_mult('r',self.getschmidt(n))
    MR = self.getTR(n+1)
    M = ML.contract(MR,'r-l;b>bl,l>l;b>br,r>r')
    T = U.contract(M, 'tl-bl,tr-br;~')
    if self.charged_at(n+1):
      # Need to put right first
      MR,s,ML = T.svd('br,r|l,r|bl,l',chi=chi)
    else:
      ML,s,MR = T.svd('bl,l|r,l|br,r',chi=chi)
    norm = np.linalg.norm(s)
    self.setschmidt(s/norm,n,strict=False)
    self.setTL(ML.renamed('bl-b'),n,unitary=True)
    self.setTR(MR.renamed('br-b'),n+1,unitary=True)
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
          diff = 0
          for n in range(self.N-1):
            s0 = np.sort(-schmidts[n])
            s1 = np.sort(-self._schmidt[n])
            slen = min(len(s0),len(s1))
            diff += np.linalg.norm(s0[:slen]-s1[:slen])
            if len(s0) > slen:
              diff += np.linalg.norm(s0[slen:])
            elif len(s1) > slen:
              diff += np.linalg.norm(s1[slen:])
          print(f'[{ni}]\t{diff/self.N}')
          if diff/self.N < delta:
            return
        schmidts = self._schmidt.copy()
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
      print(f'tau={tau} (dim {chis[itau]}, iter {niters[itau]}, tol {deltas[itau]})')
      Us = [Hi.exp('tl-bl,tr-br',-tau) for Hi in H]
      self.tebd_sweeps(Us, chis[itau], niter=niters[itau], delta=deltas[itau])

  def schmidtdiff(self, psi0):
    Ldiff = 0
    for n,l in enumerate(self._schmidt):
      la = np.array(psi0._schmidt[n])
      lb = np.array(l)
      lmin = min(len(la),len(lb))
      Ld2 = np.linalg.norm(la[:lmin]-lb[:lmin])**2
      # Corrections for extra indices may not be necessary?
      if len(la) > lmin:
        Ld2 += np.linalg.norm(la[lmin:])**2
      elif len(lb) > lmin:
        Ld2 += np.linalg.norm(lb[lmin:])**2
      Ldiff += np.sqrt(Ld2)
    return Ldiff/self.N

  def ancillaryat(self, site):
   """For compatibility among subtypes with ancillary/purification/thermal
   indices: if site is an integer, return set containing name(s) of ancillary
   indices at site; otherwise (site is a list, set, etc.) return
   set of (site,index) tuples"""
   return {}

  @abstractmethod
  def issite(self, n):
    """For compatibility among finite & infinite subtypes
    Expects integer"""
    return NotImplemented

  def local_operator_expv(self, subparse, O, x0, lterm=False, rterm=False):
    # Processing for expectation values of local operators
    # subparse parses t1-b1,t2-b2,...
    # O is the operator
    # x0 is the initial site
    # lterm/rterm are either boolean to indicate closed/open boundary or
    #   otherwise are transfer matrices

    # Pre-processing
    subsubs = subparse.split(',')
    Orank = len(subsubs)
    assert O.rank == 2*Orank
    # Tensors & parse string for network
    # TODO dictionary-based initialization
    tstr = 'O'
    Ts = [O]
    bstrs = []
    ostr = ''
    if isinstance(lterm,tensors.Tensor):
      # Include extant transfer matrix
      tstr += ';L'
      bstrs = ['L.b-B0.l','L.t-T0.l']
      Ts.append(lterm)
    elif lterm:
      assert lterm == True
      if self.issite(x0-1):
        #print('left term at',x0)#DEBUG
        # Contract left indices
        bstrs = ['B0.l-T0.l']
    for dx in range(Orank):
      # Add info pertaining to site #x
      x = dx+x0
      # Prepare tensors
      A = self.getTL(x)
      it,ib = subsubs[dx].split('-')
      tstr += f';T{dx},B{dx}*'
      bstrs.extend([f'T{dx}.b-O.{it}',f'B{dx}.b-O.{ib}'])
      if dx < Orank-1:
        bstrs.extend([f'T{dx}.r-T{dx+1}.l',f'B{dx}.r-B{dx+1}.l'])
      Ts.append(A)
      # Any additional indices
      for anc in self.ancillaryat(x):
        bstrs.append(f'T{dx}.{anc}-B{dx}.{anc}')
    xf = x0+Orank-1
    assert dx == Orank-1
    if isinstance(rterm,tensors.Tensor):
      # Include extant transfer matrix
      tstr += ';R'
      bstrs.extend([f'R.b-B{dx}.r',f'R.t-T{dx}.r'])
      Ts.append(rterm)
    elif rterm:
      assert rterm == True
      if self.issite(xf+1):
        bstrs.append(f'B{dx}.r-T{dx}.r')
        Ts[-1] = Ts[-1].diag_mult('r',self.getschmidt(xf))
    if lterm is False:
      if rterm is False:
        ostr = f'T0.l>lt,B0.l>lb,T{dx}.r>rt,B{dx}.r>rb'
      else:
        ostr = 'T0.l>t,B0.l>b'
    elif rterm is False:
      ostr = f'T{dx}.r>t,B{dx}.r>b'
    net = networks.Network.network(';'.join((tstr,','.join(bstrs),ostr)),*Ts)
    transf = net.contract()
    return transf

  # TODO integrate expv1, expv subroutines
  def expv1(self, O, parse, n):
    # parse: t1-b1,t2-b2,...
    return self.local_operator_expv(parse, O, n, lterm=True, rterm=True)

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
    transf = True
    for iO, O in enumerate(Os):
      x0 = xs[iO]
      if partot:
        # Apply parity operation
        # TODO should not have "normal" behavior for width > 1?
        ll = re.search(r'-(\w+)\b',subs[iO]).group(1)
        O = O.diag_mult(ll, parities[x0])
      transf = self.local_operator_expv(subs[iO], O, x0, lterm=transf,
        rterm=(iO==nOs-1))
      if iO == nOs-1:
        # Completed
        return transf
      if Opar[iO]:
        # Change parity as necessary
        partot = not partot
      x0 += O.rank//2
      if x0 < xs[iO+1]:
        # Contract with intermediate sites
        T = self.getlefttransfer(transf, x0-1, 'l') 
        for x in range(x0,xs[iO+1]):
          T = T.right(fparity=(parities[x%self.N] if partot else None))
        transf = T.T
      else:
        assert x0 == xs[iO+1]

class MPOgeneric:
  """MPO for a finite segment -- neither explicitly PBC nor OBC"""
  def __init__(self,Os):
    self._matrices = tuple(Os)
    self._Nsites = len(Os)

  @property
  def N(self):
    return self._Nsites

  def __getitem__(self,n):
    return self._matrices[n]

  def expandunitcell(self, M):
    return self.__class__(M*self._matrices)

  @classmethod
  def FSMinit(cls, processes, N=None, parities_phys=None, parities_virt=None,
      init_fin=('i','f'), irreps_virt=None, sreps_virt=None, missing_phys={},
      **kw_args):
    """Create MPO from finite-state machine
    Processes gives list of statexstate:matrix dictionaries
      Instead of pair may give just one state (key s is equivalent to (s,s))
      Instead of matrix may give array (=> diag()) or scalar (=> id,
        or diag(parity) if fermionic)
    States may be multidimensional, in which case matrices should be
      provided with additional nontrivial dimensions
    If init_fin is provided (default 'i','f'), that is the pair initial, final
      Otherwise (not init_fin) first element is label for initial state,
      last element is label for final
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
    """
    # TODO collect state transformation info from tensors provided
    # TODO substantiate from charged tensors
    
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
    periodic = (nproc < N) # Just repeating unit cell
    if isinstance(missing_phys,links.VSAbstract):
      # Only one space
      missing_phys = [missing_phys]
    if isinstance(missing_phys,list):
      # Repeating pattern of physical indices
      nphys = len(missing_phys)
      missing_phys = {n:missing_phys[n%nphys] for n in range(N)}
    Ms = []
    states_left = [] # List of enumerated states on the left side of a site
    states_right = [] # List of enumerated states on the right side
    state_lists = [] # states_left but without initial & final
    allstates = {initial,final} # All bond states encountered
    statedims = {} # States with dimension > 1
    Vphys = [] # VSpaces for physical indices
    # First iteration over processes:
    # identify information about physical spaces & collect virtual states
    for n,p in enumerate(processes):
      # Initialize states_left & states_right entries with initial & final
      left = {initial,final}
      right = {initial,final}
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
      state_lists.append(list(left-{initial,final}))
      # If information is not complete:
      if d is None:
        # Get from missing_phys
        assert n in missing_phys
        if isinstance(missing_phys[n],links.VSAbstract):
          Vp = missing_phys[n]
        else:
          d = missing_phys[n]
      if Vp is None:
        # Have dimension but not space: make one
        if n in missing_phys and isinstance(missing_phys[n],links.VSAbstract):
          Vp = missing_phys[n]
          assert d == Vp.dim
        else:
          Vp = links.VSpace(d)
      Vphys.append(Vp)
    # Check that states processed at a given site are the same as those
    # output by the previous one
    for n in range(nproc):
      if states_left[n] != states_right[n-1]:
        # TODO non-periodic case of n=0
        diffl = states_left[n]-states_right[n-1]
        diffr = states_right[n-1]-states_left[n]
        if diffl and diffr:
          raise ValueError(f'Virtual states on bond between {n-1}-{n} differ: {diffl} on left only, {diffr} on right only')
        elif diffl:
          raise ValueError(f'Virtual states on bond between {n-1}-{n} differ: {diffl} on left only')
        else:
          raise ValueError(f'Virtual states on bond between {n-1}-{n} differ: {diffr} on right only')
    if fermionic:
      # Parities have been identified for all states given
      assert allstates == (set(parities_virt.keys()) | {initial,final})
      parities_virt[initial] = False
      parities_virt[final] = False
    symmetric = ((irreps_virt is not None) or (sreps_virt is not None))
    if symmetric:
      group = Vphys[0].group
      if sreps_virt is None:
        sreps_virt = {}
      for s,rep in sreps_virt.items():
        if isinstance(rep,links.VSAbstract):
          sreps_virt[s] = list(rep._decomp)
      if irreps_virt is not None:
        for s,k in irreps_virt.items():
          if s in sreps_virt:
            assert sreps_virt[s] == [(k,1)]
          else:
            sreps_virt[s] = [(k,1)]
      triv = [(group.triv,1)]
      sreps_virt.update({initial:triv,final:triv})
      for s,rep in sreps_virt.items():
        d = group.sumdims(rep)
        if s in statedims:
          assert d == statedims[s]
        else:
          statedims[s] = d
      assert set(sreps_virt.keys()) == allstates
    # Assign indices to states 
    # initial & final are first & last respectively
    slfull = []
    idx_state = []
    # Current length of state_lists does not account for multidimensional states
    for n in range(nproc):
      slf = [initial]
      inv = {initial:0}
      for s in state_lists[n]:
        inv[s] = len(slf)
        if s in statedims:
          slf.extend(statedims[s]*[s])
        else:
          slf.append(s)
      inv[final] = len(slf)
      slfull.append(slf+[final])
      idx_state.append(inv)
    statedimfull = {s:1 for s in allstates}
    statedimfull.update(statedims)
    #slfull = [[initial]+sl+[final] for sl in state_lists]
    #idx_state = [{s:i for i,s in enumerate(sl)} for sl in slfull]
    dvirt = [len(slf) for slf in slfull]
    # Transition/adjacency matrices
    transition = []
    # Create matrices
    for n,proc in enumerate(processes):
      d = Vphys[n].dim
      # l t b r
      M = np.zeros((dvirt[n],dvirt[(n+1)%nproc],d,d),dtype=config.FIELD)
      trans = np.zeros((dvirt[n],dvirt[(n+1)%nproc]),dtype=int)
      # Indexing states on left & right bonds
      idxl = idx_state[n]
      idxr = idx_state[(n+1)%nproc]
      # Initial/final entries
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
      for n in range(nproc):
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
        Ms[n-1] = Ms[n-1][:,permuted,:,:]
        Ms[n] = Ms[n][permuted,:,:,:]
        transition[n-1] = transition[n-1][:,permuted]
        transition[n] = transition[n][permuted,:]
      tclass = group.Tensor
    else:
      Vvirt = [links.VSpace(d) for d in dvirt]
      state_lists = slfull
      tclass = tensors.Tensor
    if periodic: # Expand as necessary
      Ms = (N//nproc)*Ms
      state_lists = (N//nproc)*state_lists
      transition = (N//nproc)*transition
    for n in range(N):
      Vp = Vphys[n%nproc]
      Ms[n] = tclass.init_from(Ms[n],'l|0*,r|1,t|2*,b|2', Vvirt[n%nproc],Vvirt[(n+1)%nproc],Vp)
    return cls._construct_FSM(Ms, initial, final, transition, idx_state, state_lists, statedimfull, **kw_args)

  def dot(self, O2):
    assert self.N == O2.N
    M0 = self[0].contract(O2[0],'t-b;b>b,l>bl;r>br;t>t,r>tr')
    M0, data0 = M0.fuse('br,tr>r;~')
    data = data0
    Ms = [M0]
    for n in range(1,self.N-1):
      M = self[n].contract(O2[n],'t-b;b>b,l>bl,r>br;t>t,l>tl,r>tr')
      M, data = M.fuse('bl,tl>l|r*;br,tr>r;~', data)
      Ms.append(M)
    M = self[-1].contract(O2[-1],'t-b;b>b,l>bl;t>t,l>tl')
    M, data = M.fuse('bl,tl>l|0.r*;br,tr>r|1.l*;~',data,data0)
    Ms.append(M)
    return self.__class__(Ms)

  def getT(self, n):
    return self._matrices[n]

  def DMRG_opt_single(self, psi, n, TL, TR, right, gL=None, gR=None, tol=None,
      eigtol=None):
    assert (TL.site - (n-1))%self.N == 0
    assert (TR.site - (n+1))%self.N == 0
    if config.verbose > 3:
      print('Optimizing site',n%self.N)
    if gL is not None and gR is None:
      M0 = psi.getTR(n).mat_mult('l-r',gL).diag_mult('l',psi.getschmidt(n-1))
    elif gL is None and gR is not None:
      M0 = psi.getTL(n).mat_mult('r-l',gR).diag_mult('r',psi.getschmidt(n))
    elif gL is not None and gR is not None:
      M0 = psi.getTc(n).mat_mult('l-r',gL).mat_mult('r-l',gR)
      M0 = M0.diag_mult('l',psi.getschmidt(n-1)).diag_mult('r',psi.getschmidt(n))
    else:
      M0 = psi.getTL(n).diag_mult('r',psi.getschmidt(n))
    Heff = operators.NetworkOperator('L;(T);O;R;L.t-T.l,R.t-T.r,'
      'L.c-O.l,O.r-R.c,O.t-T.b;L.b>l,R.b>r,O.b>b', TL.T,self.getT(n),TR.T)
    #if Heff.shape[0] <= keig+1: #TODO remove
    kw = dict(tol=eigtol) if eigtol is not None else {}
    if False:
      Heff = Heff.asdense(inprefix='t',outprefix='')
      if psi.charged_at(n):
        w,v = Heff.eig('b-tb,l-tl,r-tr',irrep=psi.irrep)
      else:
        w,v = Heff.eig('b-tb,l-tl,r-tr')
    else:
      if psi.charged_at(n):
        Heff.charge(psi.irrep)
      w,v = Heff.eigs(keig,which='SA',guess=M0,**kw)
    v = v[np.argmin(w)]
    if config.verbose > config.VDEBUG:
      print('deviation',1-abs(M0.contract(v,'l-l,r-r,b-b*'))/abs(M0)/abs(v))
    if right:
      # ML will be new tensor, MR is gauge-fixing
      # New orthogonality center is on right
      ML,s,MR = v.svd('l,b|r,l|r',tolerance=tol)
      psi.setTL(ML,n,unitary=True)
      psi.setschmidt(s/np.linalg.norm(s),n)
      return MR
    else:
      # MR will be new tensor, ML is gauge-fixing
      # New orthogonality center is on left
      MR,s,ML = v.svd('r,b|l,r|l',tolerance=tol)
      psi.setTR(MR,n,unitary=True)
      psi.setschmidt(s/np.linalg.norm(s),n-1)
      return ML

  def DMRG_opt_double(self, psi, n, chi, TL, TR, Ulr, right, tol=None,
      eigtol=None):
    # TODO can we optimize the truncation better than naive svd?
    # Starting tensor guess
    # Incorporate Schmidt coefficients on either side
    assert (TL.site - (n-1))%self.N == 0
    assert (TR.site - (n+2))%self.N == 0
    if config.verbose > 3:
      print(f'Optimizing sites {n%self.N}-{(n+1)%self.N}')
    if (Ulr is None) or right:
      ML = psi.getTL(n)
    else:
      ML = psi.getTc(n).mat_mult('l-r',Ulr).diag_mult('l',psi.getschmidt(n-1))
    if (Ulr is None) or (not right):
      MR = psi.getTR(n+1)
    else:
      MR = psi.getTc(n+1).mat_mult('r-l',Ulr).diag_mult('r',psi.getschmidt(n+1))
    M0 = ML.diag_mult('r',psi.getschmidt(n)).contract(MR,'r-l;b>bl,~;b>br,~')
    # Don't (?) need to shift orthogonality center regardless of direction of
    # sweep because that is only required for intermediate transfer operators
    Heff = operators.NetworkOperator('L;(T);Ol;Or;R;L.t-T.l,R.t-T.r,'
      'L.c-Ol.l,Ol.r-Or.l,Or.r-R.c,Ol.t-T.bl,Or.t-T.br;'
      'L.b>l,R.b>r,Ol.b>bl,Or.b>br', TL.T,self.getT(n),self.getT(n+1),TR.T)
    kw = dict(tol=eigtol) if eigtol is not None else {}
    #if Heff.shape[0] <= keig+1: #TODO remove
    if False:
      # TODO efficient dense calculation, perform eig in operators
      Heff = Heff.asdense(inprefix='t',outprefix='')
      #print(f'Heff ({n}-{n+1}): {Heff._irrep}',Heff.getspace('l')._decomp,Heff.getspace('r')._decomp,'charge?',psi.charged_at(n) or psi.charged_at(n+1))
      if psi.charged_at(n) or psi.charged_at(n+1):
        w,v = Heff.eig('bl-tbl,br-tbr,l-tl,r-tr',irrep=psi.irrep)
      else:
        w,v = Heff.eig('bl-tbl,br-tbr,l-tl,r-tr')
    else:
      if psi.charged_at(n) or psi.charged_at(n+1):
        Heff.charge(psi.irrep)
      # Leading eigenvector (smallest `algebraic' part)
      w,v = Heff.eigs(keig,which='SA',guess=M0,**kw)
    if psi.charged_at(n+1):
      MR,s,ML = v[np.argmin(w)].svd('r,br|l,r|bl,l',chi=chi, tolerance=tol)
    else:
      ML,s,MR = v[np.argmin(w)].svd('l,bl|r,l|br,r',chi=chi, tolerance=tol)
    config.logger.debug('SVD for two-site DMRG completed')
    ML = ML.renamed('bl-b')
    MR = MR.renamed('br-b')
    # Separate Schmidt coefficients on either side
    psi.setTL(ML, n, unitary=True)
    psi.setTR(MR, n+1, unitary=True)
    psi.setschmidt(s/np.linalg.norm(s), n)

  # TODO factorize DMRG driver to integrate finite & infinite


class TransferMatrix(ABC):
  # site indicates site used in calculating this particular tensor
  # TODO trace, contract methods
  def __init__(self, psi, site, lr, *Ops, **kw_args):
    # lr specfiies which side schmidt vectors go
    # if there is a distinct bra vector, pass as final positional argument 
    self.psi = psi
    self.site = site
    if len(Ops) and isinstance(Ops[-1], MPSgeneric):
      self._psi2,Ops = Ops[-1],Ops[:-1]
    else:
      self._psi2 = None
    self.operators = Ops
    if len(Ops) == 1:
      self.opidx = ('c',)
    else:
      self.opidx = tuple(f'c{n}' for n in range(len(Ops)))
    assert lr in {'l','r'}
    self._schmidtdir = lr
    self._strict = False
    if 'manager' in kw_args:
      self.manager = kw_args['manager']
      self.id = uuid.uuid1()
    else:
      self.T = None

  def conj(self):
    # Call constructor
    # TODO wouldn't work for more than 1/non-hermitian operators
    if self._psi2 is None:
      args = (self.psi,self.site,self._schmidtdir)
    else:
      args = (self._psi2,self.site,self._schmidtdir,self.psi)
    kw_args = dict(manager=self.manager) if hasattr(self,'manager') else {}
    cpy = self.__class__(*(args+self.operators),**kw_args)
    if self.T is not None:
      T = self.T
      legflip = {'t':'b','b':'t'}
      legflip.update({self.opidxs[i]:self.opidxs[self.depth-i-1] for i in range(self.depth)})
      cpy.T = T.renamed(legflip).conj()
    return cpy
      
  def sitetensors(self, fparity=None):
    if self._schmidtdir == 'l':
      psiTs = (self.ket.getTL(self.site),self.bra.getTL(self.site))
    else:
      psiTs = (self.ket.getTR(self.site),self.bra.getTR(self.site))
    return psiTs + tuple(Op.getT(self.site) for Op in self.operators)

  def compute(self, T0, fparity=None):
    # T0 is previous tensor; if boundary pass None
    # fparity is diagonal of fermionic parity matrix; applied to bra
    self.checkcanon()
    if T0 is None:
      net = networks.Network.network(self.netconstructor(starter=True),*self.sitetensors(fparity=fparity))
    else:
      assert set(self.idxs) == T0.idxset
      net = networks.Network.network(self.netconstructor(),T0,*self.sitetensors(fparity=fparity))
    if fparity is not None:
      net['bra'] = net['bra'].diag_mult('b',fparity)
    config.logger.log(5,'computing transfer matrix at site %d (i.e. %d) depth %d',self.site,self.site%self.psi.N,self.depth)
    self.T = net.contract()
    return self.T

  def setvalue(self, T):
    self.T = T
    assert T.rank == self.depth+2
    assert T.idxset == set(self.idxs)

  @property
  def idxs(self):
    return ('t',) + self.opidx + ('b',)

  @abstractmethod
  def gauge(self, mat):
    pass

  @property
  def depth(self):
    return len(self.operators)

  @abstractmethod
  def initnext(self):
    # Initialize next transfer vector (without tensor)
    pass

  def __setstate__(self,state):
    # Inclusion of separate bra MPS
    self.__dict__.update(state)
    if '_psi2' not in state:
      self._psi2 = None

  @property
  def bra(self):
    if self._psi2 is None:
      return self.psi
    else:
      return self._psi2

  @property
  def ket(self):
    return self.psi

  @bra.setter
  def bra(self, phi):
    self._psi2 = phi

  @ket.setter
  def ket(self, phi):
    self.psi = phi

class LeftTransfer(TransferMatrix):
  def netconstructor(self, terminal=False, starter=False):
    tens = ['T','ket','bra*']+[f'O{i}' for i in range(self.depth)]
    outstr = 'ket.r>t,bra.r>b'
    bondh = 'ket.l-T.t,bra.l-T.b'
    if self.depth == 0:
      bondv = 'ket.b-bra.b'
    else:
      bondh += f',O0.l-T.{self.opidx[0]}'
      bondv = f'ket.b-O0.t,bra.b-O{self.depth-1}.b'
      outstr += ',O0.r>'+self.opidx[0]
      for i in range(1,self.depth):
        outstr += f',O{i}.r>{self.opidx[i]}'
        bondh += f',T.{self.opidx[i]}-O{i}.l'
        bondv += f',O{i-1}.b-O{i}.t'
    if terminal:
      return ';'.join(tens+[bondv+','+bondh])
    elif starter:
      return ';'.join(tens[1:]+[bondv,outstr])
    else:
      return ';'.join(tens+[bondv+','+bondh,outstr])

  def checkcanon(self):
    config.logger.log(8,'Applying transfer matrix to left of site %d',self.site)
    if self._strict and not (self.bra._leftcanon[self.site%self.psi.N] and self.ket._leftcanon[self.site%self.psi.N]):
      raise ValueError(f'strictly-enforced transfer matrix requires left-canonical: site {self.site%self.psi.N}')

  def setstrict(self):
    assert self._schmidtdir == 'l'
    self._strict = True

  def right(self, terminal=False, fparity=None, unstrict=False):
    Tnext = self.initnext()
    if terminal:
      assert self.site == self.psi.N-2
      net = networks.Network.network(Tnext.netconstructor(terminal=True),
        self.T, *Tnext.sitetensors(fparity=fparity))
      return net.contract()
    if not unstrict:
      Tnext._strict = self._strict
    Tnext.compute(self.T, fparity=fparity)
    return Tnext

  def initnext(self):
    ops = list(self.operators)
    if self._psi2 is not None:
      ops.append(self._psi2)
    return LeftTransfer(self.psi,self.site+1,self._schmidtdir, *ops)

  def moveby(self, dn, collect=False, unstrict=False):
    if self._strict and config.verbose >= config.VDEBUG: #DEBUG
      self.psi.printcanon()
    Tl = self
    if collect:
      Ts = [self]
      for n in range(dn):
        Tl = Tl.right(unstrict=unstrict)
        Ts.append(Tl)
      return Ts
    else:
      for n in range(dn):
        Tl = Tl.right(unstrict=unstrict)
        # Temporary--no need to keep
        Tl.discard = True
      return Tl

  def Ereduce(self):
    # Subtract out accrued multiples of terminal condition
    assert self._psi2 is None
    assert self.depth == 1 and self._psi2 is None
    O = self.operators[0]
    assert (self.site+1)%O.N == 0
    if self._schmidtdir == 'r':
      T = self.T
    else:
      schmidt = self.psi.getschmidt(self.site)
      T = self.T.diag_mult('t',schmidt).diag_mult('b',schmidt)
    bid = T.id_from('t-b',T)
    dE = T.trace('t-b').contract(O.Rboundary,'c-l')
    if self._schmidtdir == 'r':
      bid = bid.diag_mult('t',np.power(schmidt,2))
    dT = bid.contract(O.Lterm, f';~;r>c')
    self.T -= dE*dT
    if config.verbose >= config.VDEBUG:
      # Check calculation
      if self._schmidtdir == 'r':
        T = self.T
      else:
        schmidt = self.psi.getschmidt(self.site)
        T = self.T.diag_mult('t',schmidt).diag_mult('b',schmidt)
      dE0 = T.trace('t-b').contract(O.Rboundary,f'c-l')
      print('check E reduction: (l)',dE0)#DEBUG
    return dE
    
  def initializer(self):
    # Check normalization of "initializer"
    assert self._psi2 is None
    assert self.depth == 1 and self._schmidtdir == 'l'
    T = self.T.diag_mult('b',np.power(self.psi.getschmidt(self.site),2))
    return T.trace(f't-b').contract(self.operators[0].Rterm,f'c-l')

  def gauge(self,M):
    assert self._schmidtdir == 'l'
    self.T = self.T.mat_mult('t-l',M).mat_mult('b-l*',M)

# TODO multiple inheritance from a `ManagedTransfer'?
class LeftTransferManaged(LeftTransfer):
  def initnext(self):
    ops = list(self.operators)
    if self._psi2 is not None:
      ops.append(self._psi2)
    return LeftTransferManaged(self.psi,self.site+1,self._schmidtdir,
      *ops, manager=self.manager)

  @property
  def T(self):
    if self.id.hex in self.manager.database:
      return self.manager.database[self.id.hex]
    else:
      return None

  @T.setter
  def T(self, value):
    # Delete previous & regenerate ID
    if self.id.hex in self.manager.database:
      del self.manager.database[self.id.hex]
    self.id = uuid.uuid1()
    self.manager.database[self.id.hex] = value

  def discard(self):
    del self.manager.database[self.id.hex]

class RightTransfer(TransferMatrix):
  def netconstructor(self, terminal=False, starter=False):
    tens = ['T','ket','bra*']+[f'O{i}' for i in range(self.depth)]
    outstr = 'ket.l>t,bra.l>b'
    bondh = 'ket.r-T.t,bra.r-T.b'
    if self.depth == 0:
      bondv = 'ket.b-bra.b'
    else:
      bondh += f',O0.r-T.{self.opidx[0]}'
      bondv = f'ket.b-O0.t,bra.b-O{self.depth-1}.b'
      outstr += ',O0.l>'+self.opidx[0]
      for i in range(1,self.depth):
        outstr += f',O{i}.l>{self.opidx[i]}'
        bondh += f',T.{self.opidx[i]}-O{i}.r'
        bondv += f',O{i-1}.b-O{i}.t'
    if terminal:
      return ';'.join(tens+[bondv+','+bondh])
    elif starter:
      return ';'.join(tens[1:]+[bondv,outstr])
    else:
      return ';'.join(tens+[bondv+','+bondh,outstr])

  def checkcanon(self):
    config.logger.log(8,'Applying transfer matrix to right of site %d',self.site)
    if self._strict and not (self.bra._rightcanon[self.site%self.psi.N] and self.ket._rightcanon[self.site%self.psi.N]):
      raise ValueError(f'strictly-enforced transfer matrix requires left-canonical: site {self.site%self.psi.N}')

  def setstrict(self):
    assert self._schmidtdir == 'r'
    self._strict = True

  def left(self, terminal=False, fparity=None, unstrict=False):
    # terminal: state and/or operators are finite, yield number or lower-rank
    # transfer matrix
    # TODO case of one or more operators terminating
    Tnext = self.initnext()
    if terminal:
      assert self.site == 1
      net = networks.Network.network(Tnext.netconstructor(terminal=True),
        self.T, *Tnext.sitetensors(fparity=fparity))
      return net.contract()
    if not unstrict:
      Tnext._strict = self._strict
    Tnext.compute(self.T)
    return Tnext

  def initnext(self):
    ops = list(self.operators)
    if self._psi2 is not None:
      ops.append(self._psi2)
    return RightTransfer(self.psi,self.site-1,self._schmidtdir, *ops)

  def moveby(self, dn, collect=False, unstrict=False):
    if self._strict and config.verbose >= config.VDEBUG: #DEBUG
      self.psi.printcanon()
    Tl = self
    if collect:
      Ts = [self]
      for n in range(dn):
        Tl = Tl.left(unstrict=unstrict)
        Ts.append(Tl)
      return Ts
    else:
      for n in range(dn):
        Tl = Tl.left(unstrict=unstrict)
        # Temporary--no need to keep
        Tl.discard = True
      return Tl

  def Ereduce(self):
    # Subtract out accrued multiples of terminal condition
    assert self.depth == 1
    O = self.operators[0]
    assert self.site%O.N == 0
    if self._schmidtdir == 'l':
      T = self.T
    else:
      schmidt = self.psi.getschmidt(self.site-1)
      T = self.T.diag_mult('t',schmidt).diag_mult('b',schmidt)
    bid = T.id_from('t-b',T)
    dE = T.trace(f't-b').contract(O.Lboundary,f'c-r')
    if self._schmidtdir == 'l':
      bid = bid.diag_mult(self.idxs[0],np.power(schmidt,2))
    dT = bid.contract(O.Rterm, f';~;l>c')
    self.T -= dE*dT
    if config.verbose >= config.VDEBUG:
      # Check calculation
      if self._schmidtdir == 'l':
        T = self.T
      else:
        schmidt = self.psi.getschmidt(self.site-1)
        T = self.T.diag_mult('t',schmidt).diag_mult('b',schmidt)
      print('Check E reduction (r)',T.trace(f't-b').contract(O.Lboundary,f'c-r'))
    return dE
    
  def initializer(self):
    # Check normalization of "initializer"
    assert self.depth == 1 and self._schmidtdir == 'r'
    T = self.T.diag_mult(self.idxs[0],np.power(self.psi.getschmidt(self.site-1),2))
    return T.trace(f'{self.idxs[0]}-{self.idxs[2]}').contract(self.operators[0].Lterm,f'{self.idxs[1]}-r')

  def gauge(self, M):
    # TODO conditions other than dir r, possibly other index names
    assert self._schmidtdir == 'r'
    self.T = self.T.mat_mult('t-r',M).mat_mult('b-r*',M)

class RightTransferManaged(RightTransfer):
  def initnext(self):
    ops = list(self.operators)
    if self._psi2 is not None:
      ops.append(self._psi2)
    return RightTransferManaged(self.psi,self.site-1,self._schmidtdir,
      *ops, manager=self.manager)

  @property
  def T(self):
    #with self.manager.shelfcontext() as shelf:
    if self.id.hex in self.manager.database:
      return self.manager.database[self.id.hex]
    else:
      return None

  @T.setter
  def T(self, value):
    # Delete previous & regenerate ID
    if self.id.hex in self.manager.database:
      del self.manager.database[self.id.hex]
    self.id = uuid.uuid1()
    self.manager.database[self.id.hex] = value

  def discard(self):
    del self.manager.database[self.id.hex]

  #def __del__(self):
  #  if hasattr(self,'discard') and self.discard == True:
  #    with self.manager.shelfcontext() as shelf:
  #      del shelf[self.id.hex]
