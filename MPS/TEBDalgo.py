from .mpsbasic import *
from .. import config,links,tensors
from ..management import BondOptimizer,GeneralManager,PseudoShelf,safedump
import os,os.path
import pickle
import numpy as np
from copy import copy,deepcopy
from abc import abstractmethod

def open_finite_tebd(Hamiltonian, chis=None, file_prefix=None, savefile='auto',
    override_chis=True, override=True, use_shelf=False, **tebd_kw):
  """Initialize optimization manager
  (unpickle if saved, create otherwise)
  'override' tells manager to replace settings if different in restored manager
    (except where to save, will need different function for that)
  'override_chis' tells manager to replace bond dimensions
    (if False, list of chis must be equal)
    If current bond dimension might not be in list, can pass
      'lower' to make the algorithm select the next-lowest, or
      'higher' to make the algorithm select the next-highest
  remaining arguments as in TEBDManager()"""
  if savefile and (savefile != 'auto' or file_prefix):
    # Check existing
    filename = (file_prefix+'.p') if savefile=='auto' else savefile
    if os.path.isfile(filename):
      if isinstance(chis, int):
        chis = [chis]
      config.streamlog.warn('Reloading manager from %s',filename)
      try:
        Mngr = pickle.load(open(filename,'rb'))
      except:
        config.logger.exception('Manager file apparently corrupted') 
        if 'savesafe' in tebd_kw and tebd_kw['savesafe']:
          config.logger.warning('Reloading from backup')
          Mngr = pickle.load(open(filename+'.old','rb'))
        else:
          raise
      if override_chis:
        # Reset bond dimension list
        # TODO better way of doing this
        if chis is not None:
          useas = override_chis if isinstance(override_chis,str) else None
          Mngr.resetchis(chis, use_as=useas)
          chi_from_settings = False
        else:
          chi_from_settings = True
      else:
        assert chis is None or chis == Mngr.chis
        chi_from_settings = False
      if Mngr.filename != filename:
        config.streamlog.warning('Updating save destination from %s to %s',
          Mngr.filename, filename)
        Mngr.filename = filename
        Mngr.saveprefix = file_prefix
      if override:
        # Additionally reset general settings
        tebd_kw.pop('psi0',None)
        Mngr.resetsettings(stage_override=chi_from_settings,**tebd_kw)
        if use_shelf:
          Mngr.setdbon()
        else:
          Mngr.setdboff(False)
      return Mngr
  if chis is not None:
    tebd_kw['chis'] = chis
  return TEBDManager(Hamiltonian, use_shelf=use_shelf, file_prefix=file_prefix, savefile=savefile, **tebd_kw)

# TODO shared MPS optimizer? Or Evolver superclass?
class TEBDManager(BondOptimizer): 
  """Optimizer for TEBD"""
  _default_file = 'MPS/tebddefaults.toml'
  _logname = 'TEBD'
  def __init__(self, Hamiltonian, *args, psi0=None, **kw_args):
    """*Hamiltonian as list of two-site operators
       *list of bond dimensions (or single bond dimension) (optional, may be
          provided with settings)
       *(optional) list of timesteps (optional, may be provided with settings)
       *psi0 may be provided as 4th positional arg
      psi0 if provided is the initial state
      file_prefix: optional prefix to save after bond-dimension steps
      savefile: location for autosave
        default value 'auto' gives file_prefix.p if provided, nothing otherwise
        use None or null string to avoid autosave when saving steps
      Edelta & schmidtdelta are tolerance levels for energy & schmidt-index
        comparisons, respectively
      ncanon gives how frequetly to restore canonical form
      nE gives how frequently to compute energy (by default equal to ncanon,
        to reduce overhead)
      nsweep gives (maximum) numbers of sweeps during each stage
      tolc & tol are tolerances (for SVD) in restore_canonical and
        update, respectively"""
    self.__version = '0.0'
    self.Hs = Hamiltonian
    self.Us = None
    # Initialize state & its properties
    self.psi = psi0
    self.N = len(Hamiltonian)+1
    self.stageindex = None
    self.chi = None
    self.timestep = None
    self.chis = None
    self.timesteps = None
    self.schmidt0 = None
    self.E = None
    self.E0 = None
    self._reldiff = None

    if args:
      if isinstance(args[0],int):
        self.chis = [args[0]]
      else:
        self.chis = list(args[0])
      if len(args) > 1:
        if isinstance(args[1],int) or isinstance(args[1],float):
          self.timesteps = [args[1]]
        else:
          self.timesteps = list(args[1])
        if len(args) > 2:
          assert psi0 is None
          self.psi = args[2]

    super().__init__(**kw_args)   
  
  def _configure_stage_settings(self, override=False):
    self._attr_from_settings('chis','stages',lambda l:[c for c,t in l],
      'chis',None,'chi',lambda c: [c],override=override)
    self._attr_from_settings('timesteps','stages',lambda l:[t for c,t in l],
      'timesteps',None,'timestep',lambda t: [t],override=override)

    nchi = len(self.chis)
    nt = len(self.timesteps)
    if nchi > nt:
      self.timesteps = list(self.timesteps) + (nchi-nt)*[self.timesteps[-1]]
    elif nchi < nt:
      self.chis = list(self.chis) + (nt-nchi)*[self.chis[-1]]


  def getmultiples(self):
    # Multiples (in label-value dictionary) of timestep to exponentiate with
    if self.settings['order'] == 1:
      return {1:1}
    elif self.settings['order'] == 2:
      if self.settings['sequence'] == 'sweep':
        return {1:.5}
      else:
        return {1:.5,(1,2):1}
    else:
      k = self.settings['order']//2
      assert self.settings['order'] == 2*k
      uks = 1/(4-np.power(4,1/(2*np.arange(k,1,-1) - 1)))
      uk2 = np.stack((uks,1-4*uks))
      d = {}
      import itertools
      for idxs in itertools.product((0,1),repeat=(k-1)):
        d[idxs] = np.prod([uk2[idx,n] for n,idx in enumerate(idxs)])
        # Doubled
        d[idxs,2] = 2*d[idxs]
        # Other possible adjacent value takes last 1 and turns it to 0
        # TODO this is only relevant for some terms anyway?
        if any(idxs):
          ilast = k-2 - idxs[::-1].index(1)
          alt = idxs[:ilast]+(0,)+idxs[ilast+1:]
          d[idxs,-1] = d[idxs] + d[alt]
      return d

  def _initfuncs(self):
    # Labels for supervisors
    self.supfunctions = {'base':baseSequentialSupervisor,
                         'stage':TEBDOptSupervisor,
                         'highorder':trotterOrderkSupervisor,
                         'altsweep':alternatingSweep,
                         'seqsweep':sequentialSweep}
    # Labels for bound methods
    self.supcommands = {'init':self.initstate,
                        'setstage':self.setstage,
                        'savestep':self.savestep,
                        'canonical':self.restorecanonical,
                        'compare1':self.compareschmidt,
                        'compare2':self.compareboth,
                        'saveschmidt':self.saveschmidt,
                        'updateleft':self.updateleft,
                        'update':self.update,
                        'updateright':self.updateright,
                        'updategen':self.update_unspec}
    self.callables = {}

  def _update_state_to_version(self, state):
    super()._update_state_to_version(state)
    if self.__version != state['_TEBDManager__version']:
      self.logger.info('TEBDManager version updated from %s to %s',
        state['_TEBDManager__version'],self.__version)

  # No registered objects (for now)

  def query_system(self):
    return {'N':self.N}

  def getE(self, savelast=True, updaterel=True):
    if savelast:
      self.E0 = self.E
    args = sum((['tl-bl,tr-br',self.Hs[n],n] for n in range(self.N-1)),[])
    # TODO handle different index names?
    self.E = np.real(self.psi.expvsum(*args))
    self.logger.log(20,'Energy calculated as %s',self.E) 
    if self.E0 is not None and updaterel:
      # TODO update eigtol if changed on reload?
      dE = abs(self.E - self.E0)
      self._reldiff = dE
    return self.E

  def savestep(self):
    """Save output after optimizing a single bond dimension, or otherwise
    as settings dictate"""
    idx = self.stageindex
    if self.settings['saveat'] == 'chi':
      if idx == len(self.chis)-1 or self.chis[idx+1] != self.chi:
        self.logger.log(20,'Saving completed bond dimension %s',self.chi)
        safedump((self.psi,self.E),f'{self.saveprefix}c{self.chi}.p',newsave=True)
    elif self.settings['saveat'] == 'timestep':
      if idx == len(self.timesteps)-1 or self.timesteps[idx+1] != self.timestep:
        self.logger.log(20,'Saving completed timestep %g',self.timestep)
        safedump((self.psi,self.E),f'{self.saveprefix}t{self.timestep:g}.p',newsave=True)
    elif self.settings['saveat'] == 'all':
      self.logger.log(20,'Saving completed stage chi=%d, tau=%g',
        self.chi,self.timestep)
      safedump((self.psi,self.E),
        f'{self.saveprefix}t{self.timestep:g}c{self.chi}.p',newsave=True)
    elif self.settings['saveat'] == 'final':
      if idx == len(self.chis):
        self.logger.log(20,'Saving optimized state')
        safedump((self.psi,self.E),f'{self.saveprefix}c{self.chi}.p',newsave=True)

  def initstate(self):
    # Options for passing MPS
    # Copied from DMRG
    if isinstance(self.psi,str):
      self.logger.log(30, 'Loading initial state from %s',self.psi)
      # TODO context to avoid uneccesary updates?
      rv = pickle.load(open(self.psi,'rb'))
      if isinstance(rv,tuple):
        rv, *sv = rv
      if isinstance(rv,DMRGManager):
        self.logger.log(15, 'Loaded from DMRG manager')
        self.psi = rv.psi
        if len(rv.suplabels) == 3 and rv.suplabels[2] == 'singlesweep':
          site,direction,gauge = rv.supstatus[2]
          self.logger.info('Gauging (%s) at site %d',direction,site)
          if direction == 'r' or site == self.N-1:
            self.psi.lgauge(gauge,site)
          else:
            self.psi.rgauge(gauge,site)
      elif isinstance(rv,TEBDManager):
        self.logger.log(15, 'Loaded from TEBD manager')
        self.psi = rv.psi
      else:
        self.logger.log(15, 'Loaded as state')
        assert isinstance(rv, MPS)
        self.psi = rv
    elif isinstance(self.psi,MPS):
      self.psi = copy(self.psi)
    else:
      self.logger.log(30, 'Initializing from random')
      bond = self.psi if self.psi else self.chis[0]
      self.logger.log(10, 'Using bond %s',bond)
      indphys = [Hi.getspace('bl') for Hi in self.Hs]
      indphys.append(self.Hs[-1].getspace('br'))
      self.psi = randMPS(indphys, bond)
      self.restorecanonical()
    if 'process_psi0' in self.callables:
      self.psi = self.callables['process_psi0'](self.psi)
    self._initializeROs()
    if not self.psi.iscanon():
      self.restorecanonical()
    if not self.E:
      self.getE()

  def nextstage(self):
    """Increment stage (next chi/tau pair)"""
    self.setstage(self.stageindex+1)

  def _update_stage(self, idx):
    self.stageindex = idx
    self.chi = self.chis[idx]
    told,self.timestep = self.timestep,self.timesteps[idx]
    config.streamlog.log(30, 'chi = % 3d, tau = % 5g (#%d)',
      self.chi, self.timestep, self.stageindex)
    if self.timestep != told or self.Us is None:
      # Will need to recalculate Us (after determining which are needed)
      self.Us = {}

  def _evaluate_with_settings(self):
    self._multiples = self.getmultiples()

  def query_stage(self, idx):
    chi = self.chis[idx]
    timestep = self.timesteps[idx]
    bidx = idx - len(self.chis)
    return dict(chi=chi, timestep=timestep, index=idx, back_index=bidx)

  def _get_stage_data(self):
    return (self.stageindex,)

  def resume_message(self):
    config.streamlog.log(30, 'Resuming: chi = % 3d, tau = % 5g (#%d)',
      self.chi, self.timestep, self.stageindex)

  def restorecanonical(self, almost=False):
    self.logger.log(12,'Restoring to canonical form')
    # TODO almost-canonical
    # TODO transfer gauging
    self.psi.restore_canonical(almost_canon=True)

  def compareschmidt(self, niter, announce=True):
    Ldiff = 0
    for n,l0 in enumerate(self.schmidt0):
      l1 = self.psi._schmidt[n]
      if isinstance(l0, dict):
        # Dictionary by charge sector
        V1 = self.psi.getbond(n)
        viter = V1.full_iter()
        l1 = {k:l1[idx:idx+d*degen:d] for k,degen,d,idx in viter}
        G = self.psi[0].group
        if V1.qprimary and any(G.indicate(k)==-2 for k in l0):
          for k in list(l1):
            if G.indicate(k) == -1:
              kd = G.dual(k)
              l1[kd] = l1.pop(k)
        elif V1.qsecond and any(G.indicate(k)==-1 for k in l0):
          for k in list(l1):
            if G.indicate(k) == -2:
              kd = G.dual(k)
              l1[kd] = l1.pop(k)
        for k in set(l1).intersection(set(l0)):
          Ldiff += adiff(l0[k],l1[k])
        for k in set(l1) - set(l0):
          Ldiff += np.linalg.norm(l1[k])
        for k in set(l0) - set(l1):
          Ldiff += np.linalg.norm(l0[k])
      else:
        Ldiff += adiff(l0,l1)
        l1 = copy(l1)
      # Update history
      self.schmidt0[n] = l1
    Ldiff /= self.N
    self.logger.log(20, '[%s] schmidt diff %s', niter, Ldiff)
    if announce:
      config.streamlog.log(30,f'[% 4d] %#10.4g',niter,Ldiff)
    return Ldiff

  def saveschmidt(self):
    self.schmidt0 = []
    for n,ll in enumerate(self.psi._schmidt):
      V = self.psi.getbond(n)
      if hasattr(V,'full_iter'):
        # Dictionary by charge sector
        l1 = {k:ll[idx:idx+d*degen:d] for k,degen,d,idx in V.full_iter()}
        # Flip 'secondary' quaternionic irreps as necessary
        for k,nn in V:
          if V.group.indicate(k) == -2:
            l1[V.group.dual(k)] = l1.pop(k)
      else:
        l1 = copy(ll)
      # Update history
      self.schmidt0.append(l1)
    
  def compareboth(self, niter):
    # Schmidt-coefficient difference
    Ldiff = self.compareschmidt(niter, announce=False)
    self.getE()
    self.logger.log(20, '[%s]  energy diff %s', niter, self.E-self.E0)
    config.streamlog.log(30,f'[% 4d] %#10.4g %#+10.4g E=%#0.10f',niter,Ldiff,self.E-self.E0,self.E)
    return Ldiff,abs(self.E-self.E0)/self.settings['nE']

  def updateleft(self, tindex, double=False):
    self.logger.log(12, 'left edge TEBD update')
    self.psi.tebd_left(self.getU(tindex,0,double=double),self.chi)

  def updateright(self, tindex, double=False):
    self.logger.log(12, 'right edge TEBD update')
    self.psi.tebd_right(self.getU(tindex,self.N-2,double=double),self.chi,
      double=False)

  def update(self, n, tindex):
    self.logger.log(15, 'site %s TEBD update', n)
    self.psi.tebd_bulk(self.getU(tindex,n),self.chi,n,
      tolerance=self.settings['tol'])

  def update_unspec(self, n, tindex):
    # Update method that sorts out boundary from bulk
    if n == 0:
      self.updateleft(tindex)
    elif n == self.N-2:
      self.updateright(tindex)
    else:
      self.update(n,tindex)

  def getU(self, tindex, n, double=False):
    # Fetch e^(t_iH_n) for appropriate multiple of timestep
    # TODO incorporate possibly identical terms?
    key = (tindex, double, n)
    mul = self._multiples[tindex]
    if double:
      mul = 2*mul
    self.logger.log(8,'Fetching site %d-%d term at index %s '
      '(exponentiated with timestep %g)', n,n+1,tindex,self.timestep*mul)
    if key not in self.Us:
      self.Us[key] = self.Hs[n].exp('tl-bl,tr-br',-self.timestep*mul)
    return self.Us[key]

  def returnvalue(self):
    """Define return value for completed run()"""
    return self.psi, self.E

  def baseargs(self):
    """Define arguments for base supervisor"""
    return len(self.chis), self.N

def baseSequentialSupervisor(nstage, N, settings, state=None):
  # Very basic progression through identified stages
  if state is None:
    sidx = 0
    yield 'setstage',(sidx,)
    yield 'init', ()
  else:
    sidx, = state
  while sidx < nstage:
    yield 'runsub', ('stage',N),(sidx,)
    sidx += 1
    if sidx < nstage:
      yield 'savestep',()
      yield 'setstage',(sidx,)
  yield 'complete',()

def TEBDOptSupervisor(N, settings, state=None):
  if settings['sequence'] == 'sweep':
    sweeper = 'seqsweep'
  elif settings['sequence'] == 'alternate':
    sweeper = 'altsweep'
  if settings['order'] > 2:
    sweeper, sub_or_order = 'highorder', sweeper
  else:
    sub_or_order = settings['order']
  if state is None:
    i0 = 0
    yield 'saveschmidt', ()
  else:
    i0, = state
    if i0 >= settings['nsweep']:
      yield 'announce', 'Completing sweep then ending cycle'
      i0 = settings['nsweep']-1
  for niter in range(i0, settings['nsweep']):
    yield 'runsub', (sweeper, sub_or_order, N), (niter,)
    if settings['ncanon'] != 0 and (niter+1) % settings['ncanon'] == 0:
      yield 'canonical',(True,)
    if (niter+1) % settings['nE'] == 0:
      diff,diffE = yield 'compare2',(niter,)
    else:
      diff = yield 'compare1',(niter,)
      diffE = 1
    if diff < settings['schmidtdelta'] or diffE < settings['Edelta']:
      if settings['ncanon'] != 0 and (niter+1) % settings['ncanon'] != 0:
        yield 'canonical', (True,)
      break
  yield 'complete', ()

def trotterOrderkSupervisor(subsweeper, N, settings, state=None):
  k = settings['order']//2
  def getkey(idx):
    flags = []
    for ik in range(k-1):
      idx,n = divmod(idx,5)
      flags.insert(0,int(n==2))
    return tuple(flags)
  if state is None:
    i0 = 0
    lastkey = (k-1)*(0,)
  else:
    i0, = state
    lastkey = getkey(i0-1)
  nsubsweeps = 5**(k-1)
  for index in range(i0,nsubsweeps):
    key = getkey(index)
    if index == 0:
      first = key
    elif lastkey == key:
      first = (key,2)
    elif lastkey < key:
      first = (key,-1)
    else:
      first = (lastkey,-1)
    yield 'runsub',(subsweeper,(key,first,index==nsubsweeps-1),N),(index,)
    lastkey = key
  yield 'complete',()

def alternatingSweep(trotterinfo, N, settings, state=None):
  if isinstance(trotterinfo, int):
    # Trotter order -- 1 or 2
    order = trotterinfo
    key = 1
    first = key
    islast = True
  else:
    order = 2
    key,first,islast = trotterinfo
  if state is None:
    site0 = 0
    step0 = 0
  else:
    site0,step0 = state
  # Start with possibly-consolidated even step
  if step0 == 0:
    assert site0%2==0
    for site in range(site0,N-1,2):
      yield 'updategen',(site,first),(site,0)
    if settings['ncanon'] == 0:
      yield 'canonical', (True,), (N+site0%2,0)
    site0 = 1
  # Then possibly-doubled odd step
  if step0 <= 1:
    assert site0%2==1
    if order == 1:
      key2 = key
    else:
      key2 = (key,2)
    for site in range(site0,N-1,2):
      yield 'updategen',(site,key2),(site,1)
    if settings['ncanon'] == 0:
      yield 'canonical',(True,), (N+(site0+1)%2,1)
    site0 = 0
  # Final even step *if* order>1 and not consolidated away 
  if order == 2 and islast:
    assert site0%2==0
    for site in range(site0,N-1,2):
      yield 'updategen',(site,key),(site,2)
    if settings['ncanon'] == 0:
      yield 'canonical',(True,), (N+site0%2,2)
  yield 'complete', ()

def sequentialSweep(trotterinfo, N, settings, state=None):
  if isinstance(trotterinfo, int):
    # Trotter order -- 1 or 2
    order = trotterinfo
    key = 1
    first = key
    islast = True
  else:
    order = 2
    key,first,islast = trotterinfo
  if state is None:
    site0 = 0
    diridx = 0
  else:
    site0,diridx = state
  # Rightwards sweep
  if diridx == 0:
    if site0 == 0:
      yield 'updateleft',(first,),(0,0)
      site0 = 1
    for site in range(site0,N-2):
      yield 'update',(site,key),(site,0)
    # Double last if order==2
    yield 'updateright',(key,order==2),(N-2,0)
    site0 = N-3
  # Leftwards sweep
  if order == 2:
    for site in range(site0,0,-1):
      yield 'update',(site,key),(site,1)
    if islast:
      yield 'updateleft',(key,),(0,1)
  if settings['ncanon'] == 0:
    yield 'canonical', (True,), (-1,1)
  yield 'complete',()


class MPS_POVM:
  """Object which simulates a `low-entanglement'/MPS-compatible,
  rank-1 (at least for now) POVM: measure() takes in MPS, yields pair of
  MPS (often bond-dimension 1, in typical cases at least lower bond dimension)
  and measurement probability (or probability density)"""
  @abstractmethod
  def measure(psi):
    pass

  @property
  def Nsites:
    return self._Nsites

class ProductPOVM(MPS_POVM):
  """Rank-1 POVM which measures every site independently in its own basis
  (and therefore returns a product state)"""
  def __init__(self, states, spaces=None, weights=None, normalize=True):
    """Provide states to measure in as list of lists of Tensors, arrays,
      or integers (indicating basis element)
    If Tensor not used must supply spaces, either as list of vector spaces
      or as MPS/MPO with corresponding physical indices
    May provide probability weights corresponding to each element of each
      on-site POVM (default is uniform, dimension/# elements)
    May also provide weights as relative normalization of arguments, by setting
      normalize=False; otherwise normalizes states to 1 & ignores norm"""
    self._states = []
    self._Nsites = len(states)
    if isinstance(spaces, MPSgeneric) or isinstance(spaces,MPOgeneric):
      spaces = [M.getspace('b') for M in spaces._matrices]
    elif spaces is None:
      spaces = []
      for i,istates in enumerate(states):
        for state in istates:
          if isinstance(state,tensors.Tensor):
            spaces.append(state._spaces[0])
            break
        assert len(spaces) == i+1
    if weights is None:
      weights = [len(istates)*[1] for istates in states]
    else:
      weights = deepcopy(weights)
    for i,istates in enumerate(states):
      self._states.append([])
      V = spaces[i]
      d = V.dim
      for j,state in enumerate(istates):
        if isinstance(state,int):
          state,bidx = d*[0],state
          state[bidx] = 1
        if not isinstance(state,tensors.Tensor) or isinstance(state,np.ndarray):
          state = np.ndarray(state)
        if isinstance(state,tensors.Tensor):
          assert state.rank == 1
          assert state._spaces[0] == V
          state = state.renamed({state._idxs[0]:'b'})
        else:
          state = state.squeeze()
          assert state.size == (d,)
          state = tensors.Tensor(state,('b',),(V,))
        norm = state.norm()
        if not normalize:
          weights[i][j] *= norm**2
        state /= norm
        self._states[i].append(state)


class UniformPOVM(ProductPOVM):
  """ProductPOVM which is the same on every site"""

class BasisPOVM(ProductPOVM):
  """Most basic POVM for a given state space:
  measures in the computational basis"""
  def __init__(self, spec, N=None):
    """spec specifies on-site vector space, may be:
    -MPS or MPO
    -vector space (if N provided) or list of vector spaces
    -dimension (if N provided) or list of dimensions"""
    if isinstance(spec, MPSgeneric) or isinstance(spec, MPOgeneric):
      assert N is None or N==spec.N
      N = spec.N
      spaces = [M.getspace('b') for M in spec._matrices]
    elif isinstance(spec,list):
      spaces = list(spec)
      for n,V in range(len(spec)):
        if isinstance(V,int):
          spaces[n] = links.VSpace(V)
      assert N is None or N==len(spec)
      N = len(spec)
    else:
      if not isinstance(spec,links.VSAbstract):
        spec = links.VSpace(spec)
      spaces = N*[spec]
    # TODO any reason to explicitly instantiate projectors?
    #super().__init__([list(range(V.dim)) for V in spaces], spaces)
    self._spaces = spaces
    self._Nsites = N



class METTSManager(GeneralManager):
  # No registered objects (for now)
  # TODO ``basis'' ensemble states should be cheap to store in memory but
  # may be worth revisiting
  def __init__(self, Hamiltonian, *measurement_args, psi0=None, **kw_args):
    """Arguments are as in TEBDManager, except:
      +ngen is number of states to generate
      +bakein will be number of states to generate before saving
      -Edelta, schmidtdelta will be 0 (no convergence expected)
      -saveat will be none
    measurement_args specify a POVM"""
