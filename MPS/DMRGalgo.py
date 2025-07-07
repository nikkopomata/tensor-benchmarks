from .mpsbasic import *
from .. import config
from ..management import BondOptimizer,GeneralManager,PseudoShelf,safedump
import os,os.path,shutil
import pickle,shelve,contextlib,itertools
import logging
import numpy as np
from copy import copy,deepcopy
from collections.abc import MutableMapping

# Commands that can be used within supervisor to initialize a state

def open_manager(Hamiltonian, chis, file_prefix=None, savefile='auto',
    override=True, override_chis=True, resume_from_old=False, use_shelf=False,
    **dmrg_kw):
  """Initialize optimization manager
  (unpickle if saved, create otherwise)
  'override' tells manager to replace settings if different in restored manager
    (except where to save, will need different function for that)
  'override_chis' tells manager to replace bond dimensions
    (if False, list of chis must be equal)
    If current bond dimension might not be in list, can pass
      'lower' to make the algorithm select the next-lowest, or
      'higher' to make the algorithm select the next-highest
  'resume_from_old' indicates that the filename may point to a saved status
    consisting of (psi, (chi, single_or_double, niter))
  remaining arguments  as in DMRGManager()"""
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
        if 'savesafe' in dmrg_kw and dmrg_kw['savesafe']:
          config.logger.warning('Reloading from backup')
          Mngr = pickle.load(open(filename+'.old','rb'))
        else:
          raise
      if resume_from_old and isinstance(Mngr, tuple):
        config.streamlog.info('Converting old savepoint')
        psi,sv = Mngr
        if isinstance(sv,tuple):
          chi,ds,niter = sv
          assert chi in chis
          dmrg_kw['psi0'] = psi
          Mngr = DMRGManager(Hamiltonian, chis, file_prefix=file_prefix,
            savefile=savefile, use_shelf=use_shelf, **dmrg_kw)
          Mngr.restorecanonical()
          idx = chis.index(chi)
          Mngr.setstage(chi)
          Mngr.supervisors = 'paused'
          Mngr.supstatus = [(chi,ds),(niter,)]
          if ds == 1:
            Mngr.getrighttransfers(1)
            Mngr.suplabels = ['base','single']
          else:
            Mngr.saveschmidt()
            Mngr.getrighttransfers(2)
            Mngr.suplabels = ['base','double']
          Mngr.getE()
        else:
          assert isinstance(sv,float)
          config.logger.debug('Optimization previously completed')
          import glob
          cfiles = glob.glob(f'{file_prefix}c*.p')
          cmax = 0
          for f in cfiles:
            m = re.search(r'c(\d+)\.p$',f)
            cmax = max(cmax,int(m.group(1)))
          assert cmax
          if cmax == max(chis):
            config.logger.info('At final bond dimension')
            Mngr = DMRGManager(Hamiltonian, chis, file_prefix=file_prefix,
              savefile=savefile, use_shelf=use_shelf, **dmrg_kw)
            Mngr.psi = psi
            Mngr.supstatus = 'complete'
            Mngr.E = sv
          else:
            chi = min(c for c in chis if c > cmax)
            dmrg_kw['psi0'] = psi
            Mngr = DMRGManager(Hamiltonian, chis, file_prefix=file_prefix,
              savefile=savefile, use_shelf=use_shelf, **dmrg_kw)
            Mngr.logger.info('Resuming from bond dimension determined as %d',chi)
            # State should already be canonical
            idx = chis.index(chi)
            Mngr.setstage(chi)
            Mngr.supervisors = 'paused'
            Mngr.supstatus = [(chi,2)]
            Mngr.suplabels = ['base']
            Mngr.E = sv
        return Mngr
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
        dmrg_kw.pop('psi0',None)
        Mngr.resetsettings(stage_override=chi_from_settings,**dmrg_kw)
        if use_shelf:
          Mngr.setdbon()
        else:
          Mngr.setdboff(False)
      return Mngr
  return DMRGManager(Hamiltonian, chis, use_shelf=use_shelf, file_prefix=file_prefix, savefile=savefile, **dmrg_kw)


class DMRGManager(BondOptimizer):
  """Optimizer for DMRG"""
  _default_file = 'MPS/dmrgdefaults.toml'
  def __init__(self, Hamiltonian, *args, psi0=None, **kw_args):
    """MPO Hamiltonian; list of bond dimensions (or single bond dimension)
      psi0 if provided is the initial state
      file_prefix: optional prefix to save after bond-dimension steps
      savefile: location for autosave
        default value 'auto' gives file_prefix.p if provided, nothing otherwise
        use None or null string to avoid autosave when saving steps
      Edelta & schmidtdelta are tolerance levels for energy & schmidt-index
        comparisons, respectively
      ncanon1 & ncanon2 give how frequetly to restore canonical form during
        single & double update portion respectively
      nsweep1 & nsweep2 give (maximum) numbers of sweeps during
        single & double updates respectively
      tol0, tol1, & tol2 are tolerances (for SVD) in restore_canonical,
        single update, & double update respectively
      eigtol, if provided, is the tolerance for iterative eigenvalue solver
      eigtol_rel instead gives eigtol relative to change in E
        (initial value tolerance is still eigtol)
      tolratio gives increase in tolerance for double update"""
    self.__version = '1.1'
    self.H = Hamiltonian
    # Initialize state & its properties
    self.psi = psi0
    self.N = Hamiltonian.N
    self.chiindex = None
    self.chi = None
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
        assert psi0 is None
        self.psi = args[1]
    else:
      self.chis = None

    super().__init__(**kw_args)   
    if self.chis is None:
      if 'chis' in self.settings_man.global_settings:
        self.chis = list(self.settings_man.global_settings['chis'])
      else:
        self.chis = [self.settings_man.global_settings['chi']]
    else:
      # TODO feels redundant
      self.set_setting('chis',self.chis)


  @property
  def nlefttransf(self):
    return self._ROstatespec[0]

  @property
  def nrighttransf(self):
    return self._ROstatespec[1]

  def _initfuncs(self):
    # Labels for supervisors
    self.supfunctions = {'base':baseSupervisor,
                         'double':doubleOptSupervisor,
                         'single':singleOptSupervisor,
                         'doublesweep':doubleSweepSupervisor,
                         'singlesweep':singleSweepSupervisor}
    # Labels for bound methods
    self.supcommands = {'init':self.initstate,
                        'setchi':self.setstage,
                        'savestep':self.savestep,
                        'canonical':self.restorecanonical,
                        'righttransf':self.getrighttransfers,
                        'compare1':self.compare1,
                        'compare2':self.compare2,
                        'saveschmidt':self.saveschmidt,
                        'gaugeat':self.gauge_at,
                        'singleupdateleft':self.singleupdateleft,
                        'singleupdate':self.singleupdate,
                        'singleupdateright':self.singleupdateright,
                        'doubleupdateleft':self.doubleupdateleft,
                        'doubleupdate':self.doubleupdate,
                        'doubleupdateright':self.doubleupdateright,
                        'resettol':self.resettol}
    self.callables = {}

  def _initlog(self,level=None):
    # TODO was there a reason for defaulting to 10 instead of parent level?
    # TODO move to superclass w/ name constant?
    self.logger = config.logger.getChild('DMRG')
    if level is not None:
      self.logger.setLevel(level)

  def _update_state_to_version(self, state):
    if '_DMRGManager__version' not in state:
      state['_DMRGManager__version'] = 'void'
    if state['_DMRGManager__version'] == 'void' or ('transfRs' in state and 'ntransfR' not in state):
      if 'ntransfR' not in state:
        self.ntransfR = len(state['transfRs'])
        self.ntransfL = len(state['transfLs'])
        self.dbpath = None
        self.database = {}
        # Convert to managed
        for n in range(self.ntransfR):
          T0 = self.transfRs[n]
          self.transfRs[n] = RightTransferManaged(self.psi,T0.site,'r',
            self.H, manager=self)
          self.transfRs[n].T = T0.T
        for n in range(self.ntransfL):
          T0 = self.transfLs[n]
          self.transfLs[n] = LeftTransferManaged(self.psi,T0.site,'l',
            self.H, manager=self)
          self.transfLs[n].T = T0.T
      elif 'database' not in state:
        # TODO depricate
        if self.dbfile:
          # Need to re-load before unloading
          self.transfRs = []
          self.transfLs = []
          assert self.dbfile[-3:] == '.db'
          self.database = None
          with shelve.open(self.dbfile) as shelf:
            if self.ntransfR and (f'{self.N-self.ntransfR}R' not in shelf):
              self.transfRs = self.H.right_transfer(self.psi,self.N-self.ntransfR,collect=True)
            else:
              for n in range(self.N-self.ntransfR,self.N):
                self.transfRs.append(shelf.pop(f'{n}R'))
            if self.ntransfL and (f'{self.ntransfL-1}L' not in shelf):
              self.transfLs = self.H.left_transfer(self.psi,self.ntransfL-1,collect=True)
            else:
              for n in range(self.ntransfL):
                self.transfLs.append(shelf.pop(f'{n}L'))
        else:
          self.dbpath = None
        del self.dbfile
        self.database = {}
        # Convert to managed
        for n in range(self.ntransfR):
          T0 = self.transfRs[n]
          self.transfRs[n] = RightTransferManaged(self.psi,T0.site,'r',
            self.H, manager=self)
          self.transfRs[n].T = T0.T
        for n in range(self.ntransfL):
          T0 = self.transfLs[n]
          self.transfLs[n] = LeftTransferManaged(self.psi,T0.site,'l',
            self.H, manager=self)
          self.transfLs[n].T = T0.T
      elif 'dbfile' in state:
        if self.dbfile:
          self.database = {}
          if os.path.isfile(self.dbfile):
            with shelve.open(self.dbfile) as shelf:
              for k in shelf:
                self.database[k] = shelf.pop(k)
            os.remove(self.dbfile)
          else:
            self.logger.error('Database file not found, regenerating')
            self.regenerate()
        del self.dbfile
        self.dbpath = None
      elif 'dbpath' not in state:
        self.logger.critical('Unexpected pickle state')
        self.dbpath = None
        self.database = {}
      self.__version = '0.0'# check
    self.logger.debug('Unpickling DMRGManager from version %s',self.__version)
    if self.__version[0] == '0':
      # Registered-object form
      self._ROstatespec = [self.ntransfL,self.ntransfR]
      self._registry = {}
      self._activeRO = set()
      self._inactiveRO = set()
      for n in range(self.ntransfL):
        self._registry[n,'l'] = self.transfLs[n]
        self._activeRO.add((n,'l'))
      for n in range(self.ntransfL,self.N-1):
        self._registry[n,'l'] = LeftTransferManaged(self.psi,n,'l',
          self.H,manager=self)
        self._inactiveRO.add((n,'l'))
      for n in range(self.N-self.ntransfR,self.N):
        self._registry[n,'r'] = self.transfRs[n-self.N+self.ntransfR]
        self._activeRO.add((n,'r'))
      for n in range(1,self.N-self.ntransfR):
        self._registry[n,'r'] = RightTransferManaged(self.psi,n,'r',
          self.H,manager=self)
        self._inactiveRO.add((n,'r'))
      delattr(self, 'ntransfR')
      delattr(self, 'ntransfL')
      delattr(self, 'transfRs')
      delattr(self, 'transfLs')
      self.__version = '1.0'
    # TODO recovery options if directory is expected but does not exist
    if self.__version == '1.0':
      self.settings['savesafe'] = False
      self.__version = '1.0.1'
    if self.__version == '1.0.1':
      self.settings['tolratio'] = 1
      self.__version = '1.0.2'
    if self.__version == '1.0.2':
      if isinstance(self.database,PseudoShelf):
        self.database._backup_dict = {}
      self.__version = '1.0.3'
    # TODO any necessary updates from pre-GM
    if self.__version == '1.0.3':
      if isinstance(state['eigtol'],float) and self.settings.get('eigtol_rel',None):
        self._reldiff = state['eigtol']/self.settings['eigtol_rel']
      else:
        self._reldiff = None
    # Reached point of general manager superclass existing, should be safe
    # to invoke
    super()._update_state_to_version(state)
    self.__version = '1.1'
    self.chirules = {}
    self._initfuncs()
    if self.__version != state['_DMRGManager__version']:
      self.logger.info('DMRGManager version updated from %s to %s',
        state['_DMRGManager__version'],self.__version)
    self.supervisors = 'paused'

  # "Registered objects" (here transfer vectors)

  def _computeRO(self, key):
    site, direction = key
    transf = self._registry[key]
    if direction == 'r':
      self.logger.debug('Computing right transfer %d',site) 
      if site == self.N-1:
        transf.compute(None)
      else:
        assert (site+1,'r') in self._activeRO
        transf.compute(self._registry[site+1,'r'].T)
    else:
      self.logger.debug('Computing left transfer %d',site)
      if site == 0:
        transf.compute(None)
      else:
        assert (site-1,'l') in self._activeRO
        transf.compute(self._registry[site-1,'l'].T)

  @property
  def _nullROstate(self):
    return (0,0)

  def _initializeROs(self,statespec=None):
    for site in range(self.N-1):
      self._registry[site,'l'] = LeftTransferManaged(self.psi,site,'l',self.H,
        manager=self)
    for site in range(1,self.N):
      self._registry[site,'r'] = RightTransferManaged(self.psi,site,'r',self.H,
        manager=self)
    super()._initializeROs(statespec)

  def _validateROstate(self, spec):
    ntr,ntl = spec
    assert isinstance(ntl,int) and isinstance(ntr,int)
    assert ntr >= 0
    assert ntl >= 0
    assert ntr < self.N
    assert ntl < self.N

  def _validateRO(self, key):
    site,lr = key
    transf = self._registry[key]
    if lr == 'l':
      if not isinstance(transf,LeftTransferManaged):
        if isinstance(transf,LeftTransfer):
          self.logger.error('Left transfer at %d unconverted',transf.site)
          self._registry[key] = LeftTransferManaged(self.psi,transf.site,'l',
            self.H, manager=self)
          self._registry[key].T = transf.T
        else:
          return False
    else:
      assert lr == 'r'
      if not isinstance(transf,RightTransferManaged):
        tR = self._registry[key]
        if isinstance(transf,RightTransfer):
          self.logger.error('Right transfer at %d unconverted',transf.site)
          self._registry[key] = RightTransferManaged(self.psi,transf.site,'r',
            self.H, manager=self)
          self._registry[key].T = transf.T
        else:
          return False
    return site == self._registry[key].site

  def _ROcheckdbentries(self, key):
    """Confirm existence of 'database' entry/entries"""
    transf = self._registry[key]
    if transf.id.hex not in self.database:
      # Re-compute
      site,lr,*arg = key
      self.logger.warning('%s %stransfer vector at site %d missing; restoring',
        'Right' if lr == 'r' else 'Left', arg if arg else '', site)
      if site == 0 or site == self.N-1:
        transf.compute(None)
      else:
        site0 = site + (1 if lr == 'r' else -1)
        transf.compute(self._registry[(site0,lr)+tuple(arg)].T)
    return {transf.id.hex}

  def ROstatelist(self, statespec):
    ntl,ntr = statespec
    return [(n,'l') for n in range(ntl)]+[(self.N-n-1,'r') for n in range(ntr)]

  def query_system(self):
    return {'N':self.N}

  def shiftleftupto(self, site):
    """Align left transfer vectors up to (not including) site
    If not at edge, return (new) rightmost left transfer matrix"""
    self._updateROstate((site,self._ROstatespec[1]))
    if site:
      return self.fetchRO((site-1,'l'))

  def shiftrightupto(self, site):
    """Align right transfer vectors up to (not including) site
    If not at edge, return (new) leftmost right transfer matrix"""
    self._updateROstate((self._ROstatespec[0],self.N-site-1))
    if site != self.N-1:
      return self.fetchRO((site+1,'r'))

  def getE(self, savelast=True, updaterel=True):
    if savelast:
      self.E0 = self.E
    if self.nlefttransf == 0 and self.nrighttransf == 0:
      assert not self.E
      self.E = np.real(self.H.expv(self.psi))
      self.logger.log(25,'Initial energy %s',self.E)
      return self.E
    # TODO efficient outside of assumed case of no left transfers
    tR0 = self.gettransfR(self.N-self.nrighttransf)
    transf = tR0.moveby(self.N-self.nrighttransf-1,unstrict=True)
    self.E = np.real(transf.left(terminal=True))
    self.logger.log(20,'Energy calculated as %s',self.E) 
    if self.E0 is not None and updaterel:
      # TODO update eigtol if changed on reload?
      dE = abs(self.E - self.E0)
      self._reldiff = dE
      if self.settings['eigtol_rel']:
        #self.eigtol = self.settings['eigtol_rel']*dE
        self.logger.log(10,'eigtol_rel set to %s',self.eigtol)
    return self.E

  @property
  def eigtol(self):
    if self.settings['eigtol_rel'] and self._reldiff:
      return self.settings['eigtol_rel'] * self._reldiff
    else:
      return self.settings['eigtol']

  def savestep(self):
    """Save output after optimizing a single bond dimension"""
    self.logger.log(20,'Saving completed bond dimension %s',self.chi)
    safedump((self.psi,self.E),f'{self.saveprefix}c{self.chi}.p',newsave=True)

  def resettol(self):
    """Reset eigtol_rel to eigtol"""
    if self.settings['eigtol'] is not None:
      self.logger.log(10,'Resetting eigtol to %s',self.settings['eigtol'])
      self._reldiff = None

  def initstate(self):
    # Options for passing MPS
    if isinstance(self.psi,str):
      self.logger.log(30, 'Loading initial state from %s',self.psi)
      # TODO context to avoid uneccesary updates?
      rv = pickle.load(open(self.psi,'rb'))
      if isinstance(rv,tuple):
        rv, *sv = rv
      if isinstance(rv,DMRGManager):
        self.logger.log(15, 'Loaded from manager')
        self.psi = rv.psi
        if len(rv.suplabels) == 3 and rv.suplabels[2] == 'singlesweep':
          site,direction,gauge = rv.supstatus[2]
          self.logger.info('Gauging (%s) at site %d',direction,site)
          if direction == 'r' or site == self.N-1:
            self.psi.lgauge(gauge,site)
          else:
            self.psi.rgauge(gauge,site)
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
      self.psi = self.H.rand_MPS(bond=bond)
      self.restorecanonical()
    if 'process_psi0' in self.callables:
      self.psi = self.callables['process_psi0'](self.psi)
    self._initializeROs()
    if not self.psi.iscanon():
      self.restorecanonical()
    if not self.E:
      self.getE()

  def _update_stage(self, chi):
    super()._update_stage(chi)
    # Reset eigenvalue tolerance
    # TODO what about resume?
    self._reldiff = None

  def getrighttransfers(self, n0):
    self.logger.log(10, 'Collecting right transfer matrices')
    # TODO gauge from canonical
    self.clearall()
    self._updateROstate((0,self.N-n0))
          
  def gettransfR(self, n):
    self.logger.debug('Fetching right transfer %d',n)
    return self.fetchRO((n,'r'))

  def gettransfL(self, n):
    self.logger.debug('Fetching left transfer %d',n)
    return self.fetchRO((n,'l'))

  def restorecanonical(self, almost=False):
    self.logger.log(12,'Restoring to canonical form')
    # TODO almost-canonical
    # TODO transfer gauging
    self.psi.restore_canonical(almost_canon=True)

  def compare1(self, niter):
    if niter == 0:
      config.streamlog.log(30, 'Single update')
    self.getE()
    Ediff = self.E-self.E0
    self.logger.log(20, '[%s]  energy diff %s', niter, Ediff)
    config.streamlog.log(30,f'[% 4d] %#+12.4g E=%#0.10f',niter,Ediff,self.E)
    return abs(Ediff)

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
    
  def compare2(self, niter, return_Ediff=False):
    # Schmidt-coefficient difference
    if niter == 0:
      config.streamlog.log(30, 'Double update')
    # TODO full RMS option?
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
    self.getE()
    self.logger.log(20, '[%s]  energy diff %s', niter, self.E-self.E0)
    config.streamlog.log(30,f'[% 4d] %#10.4g %#+10.4g E=%#0.10f',niter,Ldiff,self.E-self.E0,self.E)
    if return_Ediff:
      return Ldiff,abs(self.E-self.E0)
    return Ldiff

  def doubleupdateleft(self):
    self.shiftrightupto(1)
    self.shiftleftupto(0)
    tR = self.gettransfR(2)
    self.logger.log(12, 'left edge double update')
    self.H.DMRG_opt_double_left(self.psi, self.chi, tR, self.settings['tol2'])

  def doubleupdateright(self):
    # TODO better way of handling transfer
    self.shiftleftupto(self.N-2)
    self.shiftrightupto(self.N-1)
    self.logger.log(12, 'right edge double update')
    tL = self.gettransfL(self.N-3)
    self.H.DMRG_opt_double_right(self.psi, self.chi, tL, self.settings['tol2'])

  def doubleupdate(self, n, direction):
    self.logger.log(15, 'site %s double update (%s)', n, direction)
    self.shiftleftupto(n)
    self.shiftrightupto(n+1)
    tR = self.gettransfR(n+2)
    tL = self.gettransfL(n-1)
    #eigtol = min(self.settings['eigtol'],self.eigtol*self.settings['tolratio'])
    eigtol = self.eigtol
    if eigtol is not None and self.settings['tolratio'] != 1:
      eigtol *= self.settings['tolratio']
    self.H.DMRG_opt_double(self.psi, n, self.chi,
        tL,tR,None,direction=='r', self.settings['tol2'], eigtol)

  def singleupdateleft(self):
    self.shiftleftupto(0)
    self.shiftrightupto(0)
    tR = self.gettransfR(1)
    self.logger.log(10, 'left edge single update')
    mat = self.H.DMRG_opt_single_left(self.psi, tR, self.settings['tol1'])
    return mat.diag_mult('l', self.psi.getschmidt(0))

  def singleupdateright(self, gauge):
    self.shiftleftupto(self.N-1)
    self.shiftrightupto(self.N-1)
    tL = self.gettransfL(self.N-2)
    self.logger.log(10, 'right edge single update')
    mat = self.H.DMRG_opt_single_right(self.psi, tL, self.settings['tol2'])
    return mat

  def singleupdate(self, n, direction, gauge):
    self.logger.log(12, 'site %s single update (%s)', n, direction)
    right = direction == 'r'
    tL = self.shiftleftupto(n)
    tR = self.shiftrightupto(n)
    # TODO gauge within this step
    gL,gR = (gauge, None) if right else (None, gauge)
    mat = self.H.DMRG_opt_single(self.psi, n, tL, tR,
        right, gL, gR, self.settings['tol1'], self.eigtol)
    if right:
      #TODO is this the right way to do it?
      #mat = mat.diag_mult('l',self.psi.getschmidt(n))
      pass
    return mat

  def gauge_at(self, site, gauge, direction='r'):
    self.logger.log(10, 'gauging tensor at site %s', site)
    if direction == 'r':
      self.psi.rgauge(gauge,site)
    else:
      self.psi.lgauge(gauge,site)

  def returnvalue(self):
    """Define return value for completed run()"""
    return self.psi, self.E

  def baseargs(self):
    """Define arguments for base supervisor"""
    return self.chis, self.N, self.settings

def baseSupervisor(chis, N, settings, state=None):
  if state is None:
    cidx = 0
    #yield 'setchi',(0,chis[0])
    yield 'setchi',(chis[0],)
    yield 'init',()
  else:
    chi,nupdate = state
    cidx = chis.index(chi)
    if nupdate == 1:
      # Complete bond-dimension cycle
      yield 'runsub',('single',N),(chi,1)
      cidx += 1
      if cidx == len(chis):
        yield 'complete',()
        return
      else:
        yield 'savestep', ()
        #yield 'setchi',(cidx,chis[cidx])
        yield 'setchi',(chis[cidx],)
  while cidx < len(chis):
    yield 'runsub',('double',N),(chis[cidx],2)
    yield 'runsub',('single',N),(chis[cidx],1)
    cidx += 1
    if cidx < len(chis):
      yield 'savestep', ()
      #yield 'setchi',(cidx,chis[cidx])
      yield 'setchi',(chis[cidx],)
  yield 'complete',()

def doubleOptSupervisor(N, settings, state=None):
  if state is None:
    i0 = 0
    yield 'righttransf', (2,)
    yield 'saveschmidt', ()
  else:
    i0, = state
    if i0 >= settings['nsweep2']:
      yield 'announce', 'Completing sweep then ending cycle'
      i0 = settings['nsweep2']-1
  for niter in range(i0,settings['nsweep2']):
    E = yield 'runsub',('doublesweep',N),(niter,)
    if niter % settings['ncanon2'] == 0:
      yield 'canonical', (True,)
      yield 'righttransf', (2,) #TODO use gauge?
    diff,diffE = yield 'compare2', (niter,True)
    if diff < settings['schmidtdelta'] or diffE < settings['Edelta']:
      if niter % settings['ncanon2'] != 0:
        yield 'canonical', (True,)
      break
  yield 'complete', ()

def singleOptSupervisor(N, settings, state=None):
  if state is None:
    i0 = 0
    yield 'righttransf',(1,)
  else:
    i0, = state
  for niter in range(i0,settings['nsweep1']):
    E = yield 'runsub',('singlesweep',N),(niter,)
    if niter % settings['ncanon1'] == 0:
      yield 'canonical', (True,)
      yield 'righttransf',(1,) # TODO use gauge?
    diff = yield 'compare1', (niter,)
    if diff < settings['Edelta']:
      if niter % settings['ncanon1'] != 0:
        yield 'canonical', (True,)
      break
  yield 'complete', ()

def doubleSweepSupervisor(N, settings, state=None):
  if state is None:
    state = (0,'r')
  n, direction = state
  if direction == 'r':
    # Sweep right
    if n == 0:
      yield 'doubleupdateleft', (), (0,'r')
      n = 1
    while n < N-2:
      yield 'doubleupdate', (n,'r'), (n,'r')
      n += 1
  if n == N-2:
    yield 'doubleupdateright', (), (N-2,'l')
    n -= 1
  while n > 0:
    yield 'doubleupdate', (n,'l'), (n,'l')
    n -= 1
  yield 'complete', ()

def singleSweepSupervisor(N, settings, state=None):
  if state is None:
    state = (0,'r',None)
  n, direction, gauge = state
  if direction == 'r':
    if n == 0:
      gauge = yield 'singleupdateleft', (), (0,'r',gauge)
      n = 1
    while n < N-1:
      gauge = yield 'singleupdate', (n,'r',gauge), (n,'r',gauge)
      n += 1
  if n == N-1:
    gauge = yield 'singleupdateright', (gauge,), (N-1,'l',gauge)
    n -= 1
  while n > 0:
    gauge = yield 'singleupdate', (n,'l',gauge), (n,'l',gauge)
    n -= 1
  yield 'gaugeat', (0,gauge)
  yield 'complete', ()

def adiff(v1,v2):
  # RMS difference between two (sorted) lists of potentially different size 
  numel = max(len(v1),len(v2))
  a1 = np.pad(sorted(v1),(numel-len(v1),0))
  a2 = np.pad(sorted(v2),(numel-len(v2),0))
  return np.linalg.norm(a1-a2)

def ortho_manager(Hamiltonian, chis, npsis, file_prefix=None, savefile='auto',
    override=True, override_chis=True, resume_from_old=False, use_shelf=False,
    groundstate=None, **dmrg_kw):
  """Initialize optimization manager
  (unpickle if saved, create otherwise)
  'override' tells manager to replace settings if different in restored manager
    (except where to save, will need different function for that)
  'override_chis' tells manager to replace bond dimensions
    (if False, list of chis must be equal)
    If current bond dimension might not be in list, can pass
      'lower' to make the algorithm select the next-lowest, or
      'higher' to make the algorithm select the next-highest
  'resume_from_old' indicates that the filename may point to a saved status
    consisting of (psi, (chi, single_or_double, niter))
  'groundstate' is a state to use as an initial fixed state
  remaining arguments  as in DMRGManager()"""
  if savefile and (savefile != 'auto' or file_prefix):
    # Check existing
    filename = (file_prefix+'.p') if savefile=='auto' else savefile
    if os.path.isfile(filename):
      if isinstance(chis, int):
        chis = [chis]
      config.streamlog.warn('Reloading manager from %s',filename)
      Mngr = pickle.load(open(filename,'rb'))
      if override_chis:
        # Reset bond dimension list
        # TODO options for resetting npsis
        useas = override_chis if isinstance(override_chis,str) else None
        assert npsis >= Mngr.npsis
        stage_from_settings = False
        if chis is not None:
          Mngr.resetchis(chis, use_as=useas)
        elif npsis != Mngr.npsi_tot:
          Mngr.npsi_tot = npsis
          Mngr.resetchis(Mngr.chis)
        else:
          stage_from_settings = True
      else:
        assert chis is None or chis == Mngr.chis
        assert npsis == Mngr.npsi_tot
        stage_from_settings = False
      if Mngr.filename != filename:
        config.streamlog.warn('Updating save destination from %s to %s',
          Mngr.filename, filename)
        Mngr.filename = filename
        Mngr.saveprefix = file_prefix
      if override:
        # Additionally reset general settings
        dmrg_kw.pop('psi0',None)
        dmrg_kw.pop('persiteshift',None)
        Mngr.resetsettings(stage_override=stage_from_settings,**dmrg_kw)
        if use_shelf:
          Mngr.setdbon()
        else:
          Mngr.setdboff(False)
      if groundstate is not None:
        Mngr.supcommands['init'] = Mngr.initstate_from_fixed
      return Mngr
  if groundstate is not None:
    Mngr = DMRGOrthoManager(Hamiltonian, chis, npsis, use_shelf=use_shelf, file_prefix=file_prefix, savefile=savefile, fixed_states=[groundstate], **dmrg_kw)
    Mngr.supcommands['init'] = Mngr.initstate_from_fixed
    return Mngr
  else:
    return DMRGOrthoManager(Hamiltonian, chis, npsis, use_shelf=use_shelf, file_prefix=file_prefix, savefile=savefile, **dmrg_kw)


class DMRGOrthoManager(DMRGManager):
  """Set of low-lying orthogonal states"""
  def __init__(self, Hamiltonian, chis, npsis, fixed_states=None,
      persiteshift=0, **kw_args):
    self.psiindex = 0
    self.npsi_tot = npsis
    self.npsis = 1
    self.npsi_stage = self.npsis
    self.psilist = [None]
    self.Es = [None]
    self.s0s = [None]
    self.E0s = [None]
    self.dschmidts = [0]
    # States which are not actively updated
    # (either fixed or dependent on other states)
    if fixed_states is not None:
      self.passive_states = list(fixed_states)
    else:
      self.passive_states = []
    # Map 'active' states to dependent 'passive' states
    self.passive_dependent = {0:set()}
    # Inverse of above
    self.passive_dependency = {j:[] for j in range(self.npassive)}
    super().__init__(Hamiltonian, chis, **kw_args)
    if persiteshift <= 0:
      raise ValueError('persiteshift must be provided')
    self.Eshift = self.N*persiteshift
    # Override useguess from settings
    self._guess_override = None
    self.__version = '1.1.1'
    # State specification:
    # (i,i,<0/1>) -> number of <left/right> Hamiltonian transfers for state i
    # (i,j,<0/1>) -> number of transfers between states i and j
    # (i,-1,:) -> dummy for if transfers w/ passive states dependent on i
    #   are kept

  def _update_state_to_version(self, state):
    super()._update_state_to_version(state)
    if self.__version == '0':
      self.dschmidts = self.npsis*[0]
      self.__version = '0.1'
    if self.__version == '0.1':
      # Update status
      if self.npsi_tot > self.npsis:
        # Put (back) in add-state mode
        if self.supstatus == 'complete':
          self.supstatus = []
          self.suplabels = ['base']
        basestat = self.supstatus[0]
        labels = list(self.suplabels)
        self.supstatus[0] = (self.chi,self.npsis-1,0)
        self.suplabels[1] = 'addstate'
        if len(labels) > 1:
          if labels[1] == 'double':
            # Directly translates
            niter,psi0,npsi = self.supstatus[1]
            #print(f'{psi0}/{npsi} {self.npsis}/{self.npsi_tot} states')
            if psi0 == 'bakein':
              self.supstatus[1] = (niter,'bakein',self.npsis)
            else:
              niter1 = min(niter,self.settings['nsweepadd']-1)
              self.supstatus[1] = (niter1,psi0,self.npsis)
          else:
            assert labels[1] == 'single'
            # Back to the start of the addstate supervisor
            if len(labels) == 3:
              assert self.suplabels.pop() == 'singlesweep'
              # Need to re-gauge state
              n, d, g = self.supstatus.pop()
              if g is not None:
                if d == 'r' or n == self.N-1:
                  self.psi.lgauge(g,n)
                else:
                  self.psi.rgauge(g,n)
            self.suplabels.pop()
            self.supstatus.pop()
            # Put in canonical form
            self.clearcache()
            for ipsi in range(self.npsis):
              self.selectpsi(ipsi,None)
              self.restorecanonical()
      else:
        # Add npsis to base status
        c,sd = self.supstatus[0]
        self.supstatus[0] = (c,self.npsis,sd)
      # With SettingsManager will already have been set to default
      #self.settings['nsweepadd'] = 10
      self.__version = '0.2'
    if self.__version == '0.2':
      if state['eigtol'] is not None:
        state['eigtol'] = abs(state['eigtol'])
      # transfLs/transfRs will have been converted from old form,
      # correct registered objects as appropriate for ortho form
      ntL,ntR = self._ROstatespec
      active,reg = self._activeRO,self._registry

      self._ROstatespec = np.zeros((self.npsis,self.npsis+1,2),dtype=int)
      self._activeRO = set()
      self._inactiveRO = set()
      self._registry = {}
      ipsi = self.psiindex
      for n,lr in reg:
        key = (n,lr,ipsi)
        assert reg[n,lr].psi is self.psilist[ipsi] and reg[n,lr]._psi2 is None
        self._registry[key] = reg[n,lr]
        if (n,lr) in active:
          self._activeRO.add(key)
        else:
          self._inactiveRO.add(key)
      self._ROstatespec[ipsi,ipsi,:] = ntL,ntR
      self._ROstatespec[:,-1,:] = 1
      self._ROstatespec[self.psiindex,-1,:] = 0

      for i,j in self.dottransferR:
        assert len(self.dottransferR[i,j]) == ntR
        phi,psi = self.psilist[i],self.psilist[j]
        for n,tR in enumerate(self.dottransferR[i,j]):
          assert tR.psi is psi
          assert tR._psi2 is phi
          key = (self.N-ntR+n,'r',i,j)
          self._registry[key] = tR
          self._activeRO.add(key)
        self._ROstatespec[i,j,:] = (ntL,ntR)
        tLs = self.dottransferL.pop((i,j))
        assert len(tLs) == ntL
        for n,tL in enumerate(tLs):
          assert tL.psi is psi
          assert tL._psi2 is phi
          key = (n,'l',i,j)
          self._registry[key] = tL
          self._activeRO.add(key)
      # Should have gone through everything
      assert not self.dottransferL
      for kcache in list(self.transfcacheR):
        if isinstance(kcache,tuple):
          i,j = kcache
          tRs = self.transfcacheR.pop((i,j))
          tLs = self.transfcacheL.pop((i,j))
          phi,psi = self.psilist[i],self.psilist[j]
          nL,nR = len(tLs),len(tRs)
          self._ROstatespec[i,j,:] = (nL,nR)
          assert not np.any(self._ROstatespec[j,i,:])
          for n,tR in enumerate(tRs):
            key = (self.N-len(tRs)+n,'r',i,j)
            assert tR.psi is psi
            assert tR._psi2 is phi
            self._registry[key] = tR
            self._activeRO.add(key)
          for n,tL in enumerate(tLs):
            key = (n,'l',i,j)
            assert tL.psi is psi
            assert tL._psi2 is phi
            self._registry[key] = tL
            self._activeRO.add(key)
        else:
          i = kcache
          tRs = self.transfcacheR.pop(i)
          tLs = self.transfcacheL.pop(i)
          psi = self.psilist[i]
          nL,nR = len(tLs),len(tRs)
          self._ROstatespec[i,i,:] = (nL,nR)
          for n,tR in enumerate(tRs):
            key = (self.N-len(tRs)+n,'r',i)
            assert tR.psi is psi
            assert tR._psi2 is None
            self._registry[key] = tR
            self._activeRO.add(key)
          for n,tL in enumerate(tLs):
            key = (n,'l',i)
            assert tL.psi is psi
            assert tL._psi2 is None
            self._registry[key] = tL
            self._activeRO.add(key)
      assert not self.transfcacheL
      # Generate empty transfers
      for i in range(self.npsis):
        for j in range(self.npsis):
          phi,psi = self.psilist[i],self.psilist[j]
          nL,nR = self._ROstatespec[i,j,:]
          if i == j:
            k0 = (i,)
            phi = self.H
          else:
            k0 = (i,j)
          for n in range(1,self.N-nR):
            key = (n,'r') + k0
            self._inactiveRO.add(key)
            self._registry[key] = RightTransferManaged(psi, n, 'r', phi,
              manager=self)
          for n in range(nL,self.N-1):
            key = (n,'l')+k0
            self._inactiveRO.add(key)
            self._registry[key] = LeftTransferManaged(psi, n, 'l', phi,
              manager=self)

      delattr(self, 'dottransferR')
      delattr(self, 'dottransferL')
      delattr(self, 'transfcacheL')
      delattr(self, 'transfcacheR')
      self.passive_states = []
      self.passive_dependent = {i:set() for i in range(self.npsis)}
      self.passive_dependency = {j:[] for j in range(self.npassive)}
      # With SettingsManager will have already been set to default
      #self.settings['usereflection'] = False
      self._updateROstate(self._ROstatespec)
      self.__version = '1.0'
    if self.__version == '1.0':
      if not hasattr(self,'passive_states'):
        self.passive_states = []
        self.passive_dependent = {i:set() for i in range(self.npsis)}
        self.passive_dependency = {j:[] for j in range(self.npassive)}
        self._ROstatespec = np.pad(self._ROstatespec,[(0,0),(0,1),(0,0)])
      # With SettingsManager will have already been set to default
      #self.settings_allchi['addupdatedouble'] = True
      #self.settings['addupdatedouble'] = True
      self.__version = '1.0.1'
      self.__version = '1.1'
    if self.__version == '1.1':
      self._guess_override = None
      self.npsi_stage = self.npsis
    self.__version = '1.1.1'
    # TODO any necessary updates for GM
    if self.__version != state['_DMRGOrthoManager__version']:
      self.logger.info('DMRGOrthoManager version updated from %s to %s',
        state['_DMRGOrthoManager__version'],self.__version)

  @property
  def useguess(self):
    # TODO could be a job for a context manager?
    if self._guess_override is not None:
      return self._guess_override
    else:
      return self.settings['useguess']

  def query_system(self):
    # TODO is nfixed intended to be different from npassive?
    return {'N':self.N, 'nfixed':self.npassive}

  @property
  def nlefttransf(self):
    return self._ROstatespec[self.psiindex,self.psiindex,0]

  @property
  def nrighttransf(self):
    return self._ROstatespec[self.psiindex,self.psiindex,1]

  @property
  def npassive(self):
    return len(self.passive_states)

  @property
  def psi(self):
    return self.psilist[self.psiindex]

  @psi.setter
  def psi(self, value):
    self.psilist[self.psiindex] = value

  @property
  def E(self):
    return self.Es[self.psiindex]

  @E.setter
  def E(self, value):
    self.Es[self.psiindex] = value

  @property
  def E0(self):
    return self.E0s[self.psiindex]

  @E0.setter
  def E0(self, value):
    self.E0s[self.psiindex] = value

  @property
  def schmidt0(self):
    return self.s0s[self.psiindex]

  @schmidt0.setter
  def schmidt0(self, value):
    self.s0s[self.psiindex] = value

  def _initfuncs(self):
    super()._initfuncs()
    # Labels for supervisors
    self.supfunctions['base'] = orthoBaseSupervisor
    self.supfunctions['addstate'] = addStateSupervisor
    self.supfunctions['double'] = doubleOrthoSupervisor
    self.supfunctions['single'] = singleOrthoSupervisor
    # Labels for bound methods
    self.supcommands['addstatecond'] = self.check_add_state
    self.supcommands['newstate'] = self.newstate_center
    self.supcommands['select'] = self.selectpsi
    self.supcommands['clearall'] = self.clearall
    self.supcommands['shiftbench'] = self.shiftbenchmark
    self.supcommands['benchmark1'] = self.benchmark1
    self.supcommands['benchmark2'] = self.benchmark2
    self.supcommands['getdiff'] = self.get_schmidt_diff
    self.supcommands['setchi'] = self.setchi

  def setchi(self, chi):
    self.setstage(chi, self.npsi_stage)

  def savestep(self):
    """Save output after completing bond-dimension optimization"""
    self.logger.log(20,'Saving states with bond dimension %d',self.chi)
    for i in range(self.npsis):
      safedump((self.psilist[i],self.Es[i]),f'{self.saveprefix}c{self.chi}n{i}.p',newsave=True)

  def check_add_state(self, niter, delta):
    if self.npsis == 1 and delta is None:
      # TODO setting for not-already-initialized state?
      return 1,True
    if self.npsis < self.npsi_tot and niter == self.settings['nsweep2']-1:
      return self.npsis,True
    return self.npsis, (self.npsi_tot > self.npsis and delta is not None and delta < self.settings['newthresh'])

  # Stage-related methods: incorporate stage variable npsi
  def _configure_stage_settings(self, override=False):
    if override or self.npsi_tot is None:
      self.npsi_tot = self.settings_man.global_settings['npsis']
    BondOptimizer._configure_stage_settings(self, override=override)

  def _update_stage(self, chi, npsis):
    if chi != self.chi: # TODO or npsis constant for resume? 
      BondOptimizer._update_stage(self, chi)
    self.npsi_stage = npsis
    self.logger.info('Stage set to chi=%d, # states=%d',chi,npsis)
    self._reldiff = None
  
  def query_stage(self, chi, npsis):
    sd = BondOptimizer.query_stage(self, chi)
    sd['npsis'] = npsis
    return sd

  def _get_stage_data(self):
    return (self.chi, self.npsi_stage)

  def resume_message(self):
    config.streamlog.log(30, 'Resuming: chi = % 3d (#%s), %d states', self.chi, self.chiindex, self.npsis)

  def _computeRO(self,key,flip=False):
    site,direction,*spec = key
    transf = self._registry[key]
    if flip and len(spec) == 2:
      i,j = spec
      if (site,direction,j,i) in self._activeRO:
        transf.T = self._registry[site,direction,j,i].T.renamed('t-b,b-t').conj()
        return
    if direction == 'r':
      self.logger.debug('Computing right transfer %d for %s',site,spec) 
      if site == self.N-1:
        transf.compute(None)
      else:
        assert (site+1,'r')+tuple(spec) in self._activeRO
        transf.compute(self._registry[(site+1,'r')+tuple(spec)].T)
    else:
      self.logger.debug('Computing left transfer %d for %s',site,spec)
      if site == 0:
        transf.compute(None)
      else:
        assert (site-1,'l')+tuple(spec) in self._activeRO
        transf.compute(self._registry[(site-1,'l')+tuple(spec)].T)

  @property
  def _nullROstate(self):
    ROnull = np.zeros((self.npsis,self.npsis+1,2),dtype=int)
    ROnull[:,-1,:] = 1
    ROnull[self.psiindex,-1,:] = 0
    return ROnull

  def _initializeROs(self,statespec=None):
    for ipsi in range(self.npsis):
      psi = self.psilist[ipsi]
      # Hamiltonian expectation
      for site in range(self.N-1):
        self._registry[site,'l',ipsi] = LeftTransferManaged(psi,
          site,'l', self.H, manager=self)
      for site in range(1,self.N):
        self._registry[site,'r',ipsi] = RightTransferManaged(psi,
          site,'r', self.H, manager=self)
      for jpsi in range(ipsi):
        # Inner product with other states
        phi = self.psilist[jpsi]
        for site in range(self.N-1):
          self._registry[site,'l',jpsi,ipsi] = LeftTransferManaged(psi,
            site,'l', phi, manager=self)
          self._registry[site,'l',ipsi,jpsi] = LeftTransferManaged(phi,
            site,'l', psi, manager=self)
        for site in range(1,self.N):
          self._registry[site,'r',jpsi,ipsi] = RightTransferManaged(psi,
            site,'r', phi, manager=self)
          self._registry[site,'r',ipsi,jpsi] = RightTransferManaged(phi,
            site,'r', psi, manager=self)
      for jpass,phi in enumerate(self.passive_states):
        # Inner product with 'passive' states
        if jpass in self.passive_dependent[ipsi]:
          continue
        for site in range(self.N-1):
          self._registry[site,'l',ipsi,'p',jpass] = LeftTransferManaged(phi,
            site,'l',psi,manager=self)
        for site in range(1,self.N):
          self._registry[site,'r',ipsi,'p',jpass] = RightTransferManaged(phi,
            site,'r',psi,manager=self)
    super(GeneralManager,self)._initializeROs(statespec)

  def _validateROstate(self, spec):
    assert isinstance(spec,np.ndarray)
    assert spec.dtype == int
    assert spec.shape == (self.npsis,self.npsis+1,2)
    assert np.all(spec >= 0)
    assert np.all(spec < self.N)
    assert np.all(spec[:,-1,0] == spec[:,-1,1])
    assert np.all(spec[:,-1,:] < 2)

  def _validateRO(self, key):
    self.logger.debug('Validating RO %s',key)
    site,lr,*idx = key
    transf = self._registry[key]
    if lr == 'l':
      if not isinstance(transf, LeftTransferManaged):
        self.logger.warn('RO %s improper type',key)
        return False
    else:
      assert lr == 'r'
      if not isinstance(transf, RightTransferManaged):
        self.logger.warn('RO %s improper type',key)
        return False
    if site != transf.site:
      self.logger.warn('RO %s incorrect site %d',key,transf.site)
      return False
    if len(idx) == 1:
      if transf.depth != 1:
        self.logger.warn('RO %s wrong depth',key)
      elif transf.operators[0] != self.H:
        self.logger.warn('RO %s wrong operator',key)
      elif transf._psi2 is not None:
        self.logger.warn('RO %s wrong bra',key)
      elif transf.psi is not self.psilist[idx[0]]:
        self.logger.warn('RO %s wrong ket',key)
      return transf.depth == 1 and transf.operators[0] == self.H and transf._psi2 is None and transf.psi is self.psilist[idx[0]]
    elif len(idx) == 2:
      i,j = idx
      if transf.depth != 0:
        self.logger.warn('RO %s wrong depth',key)
      elif transf._psi2 is not self.psilist[i]:
        self.logger.warn('RO %s wrong bra',key)
      elif transf.psi is not self.psilist[j]:
        self.logger.warn('RO %s wrong ket',key)
      return transf.depth == 0 and transf._psi2 is self.psilist[i] and \
        transf.psi is self.psilist[j]
    else:
      i,s,j = idx
      assert s == 'p'
      if transf.depth != 0:
        self.logger.warn('RO %s wrong depth',key)
      elif transf._psi2 is not self.psilist[i]:
        self.logger.warn('RO %s wrong bra',key)
      elif transf.psi is not self.passive_states[j]:
        self.logger.warn('RO %s wrong ket',key)
      return transf.depth == 0 and transf._psi2 is self.psilist[i] and \
        transf.psi is self.passive_states[j]

  # TODO general version?
  def _printROspec(self, statespec):
    rv = ''
    for i in range(self.npsis):
      rv += f'\n{i: 4d}: 0'
      j0,(l0,r0) = 0,statespec[i,0]
      for j in range(1,self.npsis):
        l,r = statespec[i,j]
        if l != l0 or r != r0:
          # New pair
          if j-j0 > 1:
            rv += f'-{j-1}'
          rv += f'>{l0}L{r0}R,{j}'
          j0,l0,r0 = j,l,r
      if j0 < self.npsis-1:
        rv += f'-{self.npsis-1}'
      rv += f'>{l0}L{r0}R {"P" if statespec[i,-1,0] else "X"}'
    return rv
    
  def ROstatelist(self, statespec):
    states = []
    for i in range(self.npsis):
      for j in range(self.npsis):
        if i == j:
          states.extend((n,'l',i) for n in range(statespec[i,i,0]))
          states.extend((self.N-n-1,'r',i) for n in range(statespec[i,i,1]))
        else:
          states.extend((n,'l',i,j) for n in range(statespec[i,j,0]))
          states.extend((self.N-n-1,'r',i,j) for n in range(statespec[i,j,1]))
      for j in range(len(self.passive_states)):
        if j not in self.passive_dependent[i] and np.all(statespec[self.passive_dependency[j],-1,0]):
          states.extend((n,'l',i,'p',j) for n in range(statespec[i,i,0]))
          states.extend((self.N-n-1,'r',i,'p',j) for n in range(statespec[i,i,1]))
    return states
  
  def shiftleftupto(self, site):
    statespec = self._ROstatespec.copy()
    statespec[self.psiindex,:-1,0] = site
    self._updateROstate(statespec)
    if site:
      return self.fetchRO((site-1,'l',self.psiindex))

  def shiftrightupto(self, site):
    statespec = self._ROstatespec.copy()
    statespec[self.psiindex,:-1,1] = self.N-site-1
    self._updateROstate(statespec)
    if site != self.N-1:
      return self.fetchRO((site+1,'r',self.psiindex))
          
  def selectpsi(self, index, flush_to, log=True):
    if log:
      self.logger.info('Selecting state #%d',index)
    self.psiindex,oldindex = index,self.psiindex
    statespec = self._ROstatespec.copy()
    for j in range(self.npsis):
      if j != index:
        statespec[index,j,0] = max(statespec[j,index,0],statespec[index,j,0])
        statespec[index,j,1] = max(statespec[j,index,1],statespec[index,j,1])
        statespec[j,index,:] = 0
    # Collect dependent passive states for old index, delete for new
    statespec[oldindex,-1,:] = 1
    statespec[index,-1,:] = 0
    self._updateROstate(statespec,flip=True)
    if flush_to is not None:
      fleft,fright = flush_to

      self.shiftleftupto(fleft+1)
      self.shiftrightupto(fright-1)
    return self.npsis

  # TODO initialization for next states
  # def initstate(self):
  def shiftbenchmark(self):
    # Remove stored energies etc. for prior step, shift most recent back
    for i,E in enumerate(self.Es):
      if E is not None:
        # Otherwise has not changed
        self.E0s[i] = E
    self.Es = self.npsis*[None]
    self.dschmidts = self.npsis*[0]

  def benchmark1(self):
    # Get energy for current state
    self.getE(savelast=False,updaterel=False)
    config.streamlog.log(20, f'\t#%d %#-+12.4g E=%-0.10f',self.psiindex,self.E-self.E0,self.E)

  def benchmark2(self):
    # Get energy & schmidt-difference for current state
    #self.getE(savelast=False,updaterel=False)<--was this here for a reason?
    # TODO full RMS option?
    Ldiff = 0
    for n,l0 in enumerate(self.schmidt0):
      l1 = self.psi._schmidt[n]
      if isinstance(l0, dict):
        # Dictionary by charge sector
        V = self.psi.getbond(n)
        l1 = {k:l1[idx:idx+d*degen:d] for k,degen,d,idx in V.full_iter()}
        # Flip 'secondary' quaternionic irreps as necessary
        for k,nn in V:
          if V.group.indicate(k) == -2:
            l1[V.group.dual(k)] = l1.pop(k)
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
    self.dschmidts[self.psiindex] = Ldiff
    self.logger.log(15, '[#%d] schmidt diff %s', self.psiindex, Ldiff)
    self.getE(savelast=False,updaterel=False)
    try:
      self.logger.log(15, '[#%d]  energy diff %s', self.psiindex, self.E-self.E0)
    except TypeError:
      self.E0 = 0
      self.logger.log(15, '[#%d]  energy diff %s', self.psiindex, self.E-self.E0)
    config.streamlog.log(25,f'\t#%d %#-10.4g %#-+10.4g E=%0.10f',self.psiindex,Ldiff,self.E-self.E0,self.E)

  def get_schmidt_diff(self):
    # Return schmidt difference for current state -- assume benchmark has run
    return self.dschmidts[self.psiindex]

  def compare1(self, niter, which='all'):
    assert which == 'all'
    dEtot = np.mean([abs(self.Es[i]-self.E0s[i]) for i in range(self.npsis)])
    self.streamlog.log(30,f'[% 3d] %#-12.4g',niter,dEtot)
    return dEtot

  def compare2(self, niter, which='all'):
    assert which == 'all'
    try:
      dEavg = np.mean([abs(self.Es[i]-self.E0s[i]) for i in range(self.npsis)])
    except TypeError:
      psi0 = self.psiindex
      while any(E is None for E in self.Es):
        inone = self.Es.index(None)
        self.logger.critical('Energy of state #%d not recorded; recalculating',
          inone)
        self.selectpsi(inone,None)
        self.benchmark2()
      dEavg = np.mean([abs(self.Es[i]-self.E0s[i]) for i in range(self.npsis)])
      self.selectpsi(psi0,None)
        
    dEtot = np.sum(self.Es)-np.sum(self.E0s)
    Ldtot = np.mean(self.dschmidts)
    config.streamlog.log(30,f'[% 3d] %#-12.4g %#-12.4g : %#-+12.4g (%#0.10f)',niter,Ldtot,dEavg,dEtot,np.mean(self.Es))
    self._reldiff = dEavg
    if self.settings['eigtol_rel']:
      self.logger.log(10,'eigtol set to %s',self.eigtol)
    return Ldtot

  def initstate_from_fixed(self):
    # Initialize first state with 'ground state' at passive_states[0]
    site = self.N//2-1
    self.logger.info('Initializing first state from fixed with double update at site %d',site)
    psi = copy(self.passive_states[0])
    self.psilist[0] = psi
    self._initializeROs()
    self._guess_override = False
    self._reldiff = None
    self.doubleupdate(site,'r')
    self._guess_override = None
    self.logger.info('Completing by imposing canonical form')
    self.shiftleftupto(0)
    self.shiftrightupto(self.N-1)
    self.restorecanonical()
    self.shiftrightupto(2)
    self.getE()
    if self.settings['usereflection']:
      self.logger.info('Adding reflected state')
      self.add_passive_state(MPSReflected(psi),0)

  def newstate_center(self):
    # Initialize with previous
    # TODO move set-stage to wrapper
    config.streamlog.log(30,'[ Adding state #%d ]', self.npsis)
    self.setstage(self.chi, self.npsis+1)
    site = self.N//2-1
    self.logger.info('Initializing state #%d with double update at site %d',
      self.npsis,site)
    self.npsis += 1 
    psi = copy(self.psilist[-1])
    idx = self.npsis-1
    self.psilist.append(psi)
    Einit = self.Es[-1]
    if Einit is None:
      Einit = self.E0s[-1]
    self.Es.append(Einit)
    self.E0s.append(None)
    self.s0s.append(None)
    self.dschmidts.append(0)
    # New transfer vector containers
    newtransf = {}
    newtransf.update({(site,'l',idx):LeftTransferManaged(psi,site,'l',self.H,
      manager=self) for site in range(self.N-1)})
    newtransf.update({(site,'r',idx):RightTransferManaged(psi,site,'r',self.H,
      manager=self) for site in range(1,self.N)})
    for j in range(self.npsis):
      phi = self.psilist[j]
      newtransf.update({(site,'l',idx,j):LeftTransferManaged(phi,site,'l',psi,
        manager=self) for site in range(self.N-1)})
      newtransf.update({(site,'r',idx,j):RightTransferManaged(phi,site,'r',psi,
        manager=self) for site in range(1,self.N)})
      newtransf.update({(site,'l',j,idx):LeftTransferManaged(psi,site,'l',phi,
        manager=self) for site in range(self.N-1)})
      newtransf.update({(site,'r',j,idx):RightTransferManaged(psi,site,'r',phi,
        manager=self) for site in range(1,self.N)})
    for j in range(self.npassive):
      phi = self.passive_states[j]
      newtransf.update({(site,'l',idx,'p',j):LeftTransferManaged(phi,site,'l',psi,
        manager=self) for site in range(self.N-1)})
      newtransf.update({(site,'r',idx,'p',j):RightTransferManaged(phi,site,'r',psi,
        manager=self) for site in range(1,self.N)})
    self.passive_dependent[idx] = set()
    # Add new row & column
    statespec = np.pad(self._ROstatespec,[(0,1),(0,1),(0,0)])
    # Flip last two columns
    statespec[:,[-1,-2],:] = statespec[:,[-2,-1],:]
    self.expandROs(newtransf, statespec)
    self.selectpsi(self.npsis-1,(site-1,site+2))
    self._guess_override = False
    self.doubleupdate(site,'r')
    self._guess_override = None
    self.logger.info('Completing by imposing canonical form')
    self.restorecanonical()
    if self.settings['usereflection']:
      self.logger.info('Adding reflected state')
      self.add_passive_state(MPSReflected(psi),idx)
    self.getrighttransfers(2)
  
  def add_passive_state(self, psinew, dependency=None):
    idx = len(self.passive_states)
    self.passive_states.append(psinew)
    # For now assume single dependency
    # TODO compatibility with expandROs so that this is not necessary
    assert isinstance(dependency, int) and not self._ROstatespec[dependency,-1,0]
    self.passive_dependent[dependency].add(idx)
    self.passive_dependency[idx] = [dependency]
    newtransf = {}
    for iphi,phi in enumerate(self.psilist):
      if iphi != dependency:
        newtransf.update({(site,'l',iphi,'p',idx):LeftTransferManaged(psinew,site,
          'l',phi,manager=self) for site in range(self.N-1)})
        newtransf.update({(site,'r',iphi,'p',idx):RightTransferManaged(psinew,site,
          'r',phi,manager=self) for site in range(1,self.N)})
    self.expandROs(newtransf, self._ROstatespec)

  def collected_pairs(self):
    # All pairs of states represented in transfer matrices
    # TODO make dependent on callable
    regular = [(self.psiindex,i) for i in range(self.npsis) if i != self.psiindex]
    passive = [(self.psiindex,('p',i)) for i in range(self.npassive) if i not in self.passive_dependent[self.psiindex]]
    return regular+passive

  def getrighttransfers(self, n0):
    self.logger.log(20, 'Collecting right transfer matrices (to site %d)',n0)
    #TODO be more deliberate about when clearing database
    self.shiftleftupto(0)
    self.shiftrightupto(self.N-1)
    self.shiftrightupto(n0)

  def getlefttransfers(self, n0):
    # Only clear left here
    self.shiftleftupto(0)
    self.shiftleftupto(n0)

  #@property
  #def rtransflist(self):
  #  return [self.transfRs]+[self.dottransferR[idx] for idx in self.collected_pairs()]

  #@property
  #def ltransflist(self):
  #  return [self.transfLs]+[self.dottransferL[idx] for idx in self.collected_pairs()]

  def gettransfR(self, n, which=None):
    """Transfer matrix at site n; Hamiltonian if which is None,
    identified pair otherwise"""
    if which is None:
      return self.fetchRO((n,'r',self.psiindex))
    else:
      return self.fetchRO((n,'r')+which)

  def gettransfL(self, n, which=None):
    if which is None:
      return self.fetchRO((n,'l',self.psiindex))
    else:
      return self.fetchRO((n,'l')+which)

  def ortho_vectors(self, site, nsites):
    right = site+nsites == self.N
    left = site == 0
    for i,jdx in self.collected_pairs():
      if i != self.psiindex:
        continue
      if isinstance(jdx,tuple):
        assert jdx[0] == 'p'
        phi = self.passive_states[jdx[1]]
        self.logger.debug('protecting orthogonality of %d w/ passive %d: site(s) %s', i, jdx[1], tuple(range(site,site+nsites)))
      else:
        j,jdx = jdx,(jdx,)
        phi = self.psilist[j]
        self.logger.debug('protecting orthogonality of %d (current) w/ %d: site(s) %s', i, j, tuple(range(site,site+nsites)))
      T0 = phi.getTL(site)
      if not (nsites==1 and right):
        T0 = T0.diag_mult('r',phi.getschmidt(site))
      if nsites == 2:
        Tr = phi.getTR(site+1)
        T0 = T0.contract(Tr,'r-l;b>bl,~;b>br,~')
      T = T0#.conj()
      if not left:
        L = self.gettransfL(site-1,(i,)+jdx)
        self.logger.log(5,'left at %d',L.site)
        T = T.contract(L.T, 'l-t;~;b>l')#*')
      if not right:
        R = self.gettransfR(site+nsites,(i,)+jdx)
        self.logger.log(5,'right at %d',R.site)
        T = T.contract(R.T,'r-t;~;b>r')#*')
      yield T#.conj()
      
  def optimize_tensor(self, Heff, site, nsites, gauge=None, charge=None):
    # Project out vectors
    self.logger.info('Optimizing site %d, %d-site update',site,nsites)
    Hshift = Heff-self.Eshift
    if charge is not None:
      Hshift.charge(charge)
    # Project shifted operator away from effective complementary vectors
    Hshift.project_out(list(self.ortho_vectors(site,nsites)),
      tol=self.settings['ortho_tol'],tol_absolute=True,zero=True)
    self.logger.debug('Effective Hamiltonian projected down')
    # Prepare initial guess
    # TODO regauge after update? 
    if self.useguess:
      if gauge is None:
        self.logger.log(5,'No gauge')
      else:
        self.logger.log(5,'"%s" gauge present',gauge[0])
      if gauge is not None and gauge[0] == 'l':
        self.logger.log(5,'Gauging guess on right')
        M0 = self.psi.getTc(site).mat_mult('r-l',gauge[1])
        M0 = M0.diag_mult('l',self.psi.getschmidt(site-1))
      else:
        M0 = self.psi.getTL(site)
        if nsites==1 and gauge is not None and gauge[0] == 'r':
          self.logger.log(5,'Gauging guess on left')
          M0 = M0.mat_mult('l-r',gauge[1])
      if not (nsites==1 and site+1==self.N):
        M0 = M0.diag_mult('r',self.psi.getschmidt(site))
      if nsites == 2:
        M0 = M0.contract(self.psi.getTR(site+1),'r-l;b>bl,~;b>br,~')
    else:
      M0 = None
    etol = self.eigtol if self.eigtol is not None else 0
    w,v = Hshift.eigs(self.settings['keig'],which='LM',guess=M0, tol=etol)
    self.logger.log(16,'Effective energy computed as %0.10f',min(w)+self.Eshift)
    return v[np.argmin(w)]
    
  def doubleupdateleft(self):
    # TODO combine left/right/bulk?
    self.shiftleftupto(0)
    self.shiftrightupto(1)
    tR = self.gettransfR(2)
    self.logger.log(12, 'left edge double update')
    Heff = operators.NetworkOperator('(T);R;Ol;Or;R.t-T.r,Ol.r-Or.l,'
      'Or.r-R.c,Ol.t-T.bl,Or.t-T.br;R.b>r,Ol.b>bl,Or.b>br',
      tR.T,self.H.getT(0),self.H.getT(1))
    M = self.optimize_tensor(Heff, 0, 2)
    self.logger.debug('Dividing tensors')
    ML,s,MR = M.svd('bl|r,l|br,r',chi=self.chi,tolerance=self.settings['tol2'])
    self.psi.setTL(ML.renamed('bl-b'),0,unitary=True)
    self.psi.setTR(MR.renamed('br-b'),1,unitary=True)
    self.psi.setschmidt(s/np.linalg.norm(s),0)

  def doubleupdateright(self):
    self.shiftleftupto(self.N-2)
    self.shiftrightupto(self.N-1)
    tL = self.gettransfL(self.N-3)
    self.logger.log(12, 'right edge double update')
    Heff = operators.NetworkOperator('L;(T);Ol;Or;L.t-T.l,L.c-Ol.l,'
      'Ol.r-Or.l,Ol.t-T.bl,Or.t-T.br;L.b>l,Ol.b>bl,Or.b>br',
      tL.T,self.H.getT(self.N-2),self.H.getT(self.N-1))
    M = self.optimize_tensor(Heff, self.N-2, 2)
    self.logger.debug('Dividing tensors')
    ML,s,MR = M.svd('l,bl|r,l|br',chi=self.chi,tolerance=self.settings['tol2'])
    self.psi.setTL(ML.renamed('bl-b'),self.N-2,unitary=True)
    self.psi.setTR(MR.renamed('br-b'),self.N-1,unitary=True)
    self.psi.setschmidt(s/np.linalg.norm(s), self.N-2)

  def doubleupdate(self, n, direction):
    self.logger.log(15, 'site %s double update (%s)', n, direction)
    self.shiftleftupto(n)
    self.shiftrightupto(n+1)
    tR = self.gettransfR(n+2)
    tL = self.gettransfL(n-1)
    Heff = operators.NetworkOperator('L;(T);Ol;Or;R;L.t-T.l,R.t-T.r,'
      'L.c-Ol.l,Ol.r-Or.l,Or.r-R.c,Ol.t-T.bl,Or.t-T.br;'
      'L.b>l,R.b>r,Ol.b>bl,Or.b>br', tL.T,self.H.getT(n),self.H.getT(n+1),tR.T)
    if self.psi.charged_at(n) or self.psi.charged_at(n+1):
      ch = self.psi.irrep
    else:
      ch = None
    M = self.optimize_tensor(Heff, n, 2, charge=ch)
    self.logger.debug('Dividing tensor')
    if self.psi.charged_at(n+1):
      MR,s,ML = M.svd('r,br|l,r|bl,l',chi=self.chi,
        tolerance=self.settings['tol2'])
    else:
      ML,s,MR = M.svd('l,bl|r,l|br,r',chi=self.chi,
        tolerance=self.settings['tol2'])
    self.psi.setTL(ML.renamed('bl-b'),n,unitary=True)
    self.psi.setTR(MR.renamed('br-b'),n+1,unitary=True)
    self.psi.setschmidt(s/np.linalg.norm(s), n)

  def singleupdateleft(self):
    self.shiftleftupto(0)
    self.shiftrightupto(0)
    tR = self.gettransfR(1)
    self.logger.log(10, 'left edge single update')
    Heff = operators.NetworkOperator('(T);R;O;R.t-T.r,O.r-R.c,O.t-T.b;'
      'R.b>r,O.b>b', tR.T, self.H.getT(0))
    M = self.optimize_tensor(Heff, 0, 1)
    ML,s,MR = M.svd('b|r,l|r')
    self.psi.setTL(ML,0,unitary=True)
    schmidt = s/np.linalg.norm(s)
    self.psi.setschmidt(schmidt,0)
    return MR

  def singleupdateright(self, gauge):
    self.shiftleftupto(self.N-1)
    self.shiftrightupto(self.N-1)
    tL = self.gettransfL(self.N-2)
    self.logger.log(10, 'right edge single update')
    Heff = operators.NetworkOperator('(T);L;O;L.t-T.l,O.l-L.c,O.t-T.b;'
      'L.b>l,O.b>b', tL.T, self.H.getT(self.N-1))
    M = self.optimize_tensor(Heff, self.N-1,1,('r',gauge))
    ML,s,MR = M.svd('l|r,l|b')
    self.psi.setTR(MR,self.N-1,unitary=True)
    schmidt = s/np.linalg.norm(s)
    self.psi.setschmidt(schmidt,self.N-2)
    return ML

  def singleupdate(self, n, direction, gauge):
    self.logger.log(12, 'site %s single update (%s)', n, direction)
    self.shiftleftupto(n)
    self.shiftrightupto(n)
    right = direction == 'r'
    tR = self.gettransfR(n+1)
    tL = self.gettransfL(n-1)
    Heff = operators.NetworkOperator('L;(T);O;R;L.t-T.l,R.t-T.r,'
      'L.c-O.l,O.r-R.c,O.t-T.b;L.b>l,R.b>r,O.b>b', tL.T,self.H.getT(n),tR.T)
    if self.psi.charged_at(n):
      ch = self.psi.irrep
    else:
      ch = None
    M = self.optimize_tensor(Heff, n, 1, (direction, gauge), charge=ch)
    if right:
      ML,s,MR = M.svd('l,b|r,l|r',tolerance=self.settings['tol1'])
      self.psi.setTL(ML,n,unitary=True)
      self.psi.setschmidt(s/np.linalg.norm(s),n)
      return MR
    else:
      MR,s,ML = M.svd('r,b|l,r|l',tolerance=self.settings['tol1'])
      self.psi.setTR(MR,n,unitary=True)
      self.psi.setschmidt(s/np.linalg.norm(s),n-1)
      return ML

  def returnvalue(self):
    return list(self.psilist)

  def baseargs(self):
    return self.chis, self.N, self.npsi_tot, self.npsis, self.settings

def orthoBaseSupervisor(chis, N, npsitot, np0, settings, state=None):
  if state is None:
    yield 'init',()
    cidx = 0
    yield 'setchi',(chis[0],)
    npsis = np0
  else:
    chi,npsis,nupdate = state
    cidx = chis.index(chi)
    if nupdate == 1:
      # Complete bond-dimension cycle
      yield 'runsub',('single',N,npsis),(chi,npsis,1)
      cidx += 1
      if cidx == len(chis):
        yield 'complete',()
        return
      else:
        yield 'savestep', ()
        yield 'setchi',(chis[cidx],)
  # First add all states
  while npsis < npsitot:
    yield 'runsub',('addstate',N,npsis),(chis[cidx],npsis,0)
    npsis += 1
  while cidx < len(chis):
    yield 'runsub',('double',N,npsis),(chis[cidx],npsis,2)
    yield 'runsub',('single',N,npsis),(chis[cidx],npsis,1)
    cidx += 1
    if cidx < len(chis):
      yield 'savestep', ()
      yield 'setchi',(chis[cidx],)
  yield 'complete',()

# TODO bond dimension only for state initialization
def doubleOrthoSupervisor(N, npsis, settings, state=None):
  # Only perform double update--do not introduce new states
  yield 'announce', 'Double update'
  if state is None:
    psi0 = 0 # Starting index of state
    n0 = 0 # Current niter
    yield 'righttransf', (2,)
    yield 'saveschmidt', ()
  else:
    n0,psi0,npsis = state
  for niter in range(n0,settings['nsweep2']):
    for ipsi in range(psi0,npsis):
      E = yield 'runsub',('doublesweep',N),(niter,ipsi,npsis)
      yield 'benchmark2', ()
      if ipsi+1 != npsis:
        yield 'select', (ipsi+1,(-1,2))
      yield 'saveschmidt', ()
    psi0 = 0
    diff = yield 'compare2', (niter,)
    if diff < settings['schmidtdelta']:
      yield 'clearall', ()
      for ipsi in range(npsis):
        yield 'select',(ipsi,None)
        yield 'canonical', (True,)
      break
    if niter % settings['ncanon2'] == 0:
      yield 'clearall', ()
      for ipsi in range(npsis):
        yield 'select',(ipsi,None)
        yield 'canonical', (True,)
      for ipsi in range(npsis):
        yield 'select',(ipsi,None)
        yield 'righttransf', (2,) #TODO use gauge?
    yield 'select',(0,(-1,2))
    yield 'shiftbench', ()
    yield 'saveschmidt', ()
  yield 'shiftbench', ()
  yield 'select', (0,None)
  yield 'complete', ()

def mixedDoubleOrthoSupervisor(N, npsis, settings, state=None):
  # Perform single or double update depending on last step
  yield 'announce', 'Double update (mixed)'
  if state is None:
    psi0 = 0 # Starting index of state
    n0 = 0 # Current niter
    firstrun = settings['firstrun'] if 'firstrun' in settings else 'auto'
    if firstrun == 1:
      single_next = npsis*[True]
    elif firstrun == 2:
      single_next = npsis*[False]
    else:
      assert firstrun == 'auto'
      single_next = (npsis-1)*[True]+[False]
    # TODO rest of implementation
    yield 'righttransf', (2,)
    yield 'saveschmidt', ()
  else:
    n0,psi0,npsis,single_next = state
  if 'sd_switch' in settings:
    sd_switch = settings['sd_switch']
  else:
    sd_switch = .001
  for niter in range(n0,settings['nsweep2']):
    for ipsi in range(psi0,npsis):
      if single_next[ipsi]:
        E = yield 'runsub',('singlesweep',N),(niter,ipsi,npsis,single_next)
      else:
        E = yield 'runsub',('doublesweep',N),(niter,ipsi,npsis,single_next)
      yield 'benchmark2', ()
      diff = yield 'getdiff', ()
      single_next[ipsi] = (diff < sd_switch)
      if ipsi+1 != npsis:
        yield 'select', (ipsi+1,(-1,2-single_next[ipsi+1]))
      yield 'saveschmidt', ()
    psi0 = 0
    diff = yield 'compare2', (niter,)
    if diff < settings['schmidtdelta']:
      yield 'clearall', ()
      for ipsi in range(npsis):
        yield 'select',(ipsi,None)
        yield 'canonical', (True,)
      break
    if niter % settings['ncanon2'] == 0:
      yield 'clearall', ()
      for ipsi in range(npsis):
        yield 'select',(ipsi,None)
        yield 'canonical', (True,)
      for ipsi in range(npsis):
        yield 'select',(ipsi,None)
        yield 'righttransf', (2,) #TODO use gauge?
    yield 'select',(0,(-1,2-single_next[(ipsi+1)%npsis]))
    yield 'shiftbench', ()
    yield 'saveschmidt', ()
  yield 'shiftbench', ()
  yield 'select', (0,None)
  yield 'complete', ()


def addStateSupervisor(N, npsi, settings, state=None):
  # Supervisor for introducing one new state, baking in, & performing some
  # optimization
  if state is None:
    psi0 = 'bakein' # Will be starting by initializing new state
    niter = 0 # Iterations (after adding new state)
    yield 'newstate', ()
    yield 'shiftbench', ()
    npsi += 1
    yield 'righttransf', (2,)
    yield 'saveschmidt', ()
    yield 'resettol', ()
  else:
    niter,psi0,npsi = state
  if psi0 == 'bakein':
    psi0 = npsi-1
    for n in range(niter,settings['bakein']):
      yield 'announce', f'[{n}/{settings["bakein"]}]'
      yield 'runsub', ('doublesweep',N), (n,'bakein',npsi)
      yield 'benchmark2', ()
      yield 'shiftbench', ()
      yield 'saveschmidt', ()
      yield 'resettol', ()
    yield 'select', (0,(-1,2))
    yield 'saveschmidt', ()
    niter = 0
    psi0 = 0
  yield 'announce', 'Updating all'
  while niter < settings['nsweepadd']:
    if settings['addupdatedouble'] > 1:
      wdouble = (npsi%settings['addupdatedouble'])==0
    else:
      wdouble = settings['addupdatedouble']
    for ipsi in range(psi0,npsi):
      if wdouble:
        E = yield 'runsub',('doublesweep',N),(niter,ipsi,npsi)
      else:
        E = yield 'runsub',('singlesweep',N),(niter,ipsi,npsi)
      yield 'benchmark2', ()
      yield 'saveschmidt', ()
      if ipsi+1 != npsi:
        yield 'select', (ipsi+1,(-1,1+wdouble))
    psi0 = 0
    # Just put everything in canonical form
    yield 'clearall', ()
    for ipsi in range(npsi):
      yield 'select',(ipsi,None)
      yield 'canonical', (True,)
    for ipsi in range(npsi):
      yield 'select',(ipsi,None)
      yield 'righttransf', (2,) #TODO use gauge?
    diff = yield 'compare2', (niter,)
    yield 'select',(0,(-1,2))
    if diff < settings['newthresh']:
      break
    niter += 1
    yield 'shiftbench', ()
    yield 'saveschmidt', ()
  yield 'complete', ()

def doubleOrthoSupervisorCombined(N, settings, state=None):
  # Both introduce new states & perform ordinary double update
  # Depricated
  yield 'announce', 'Double update'
  if state is None:
    psi0 = 0 # Starting index of state
    niter = 0 # Iterations (after adding new state)
    yield 'righttransf', (2,)
    yield 'saveschmidt', ()
    npsi,newpsi = yield 'addstatecond', (0, None)
  else:
    niter,psi0,npsi = state
    if psi0 == 'bakein':
      psi0 = npsi-1
      for n in range(niter,settings['bakein']):
        yield 'announce', f'[{n}/{settings["bakein"]}]'
        yield 'runsub', ('doublesweep',N), (n,'bakein',npsi)
        yield 'benchmark2', ()
        yield 'shiftbench', ()
        yield 'saveschmidt', ()
      yield 'select', (0,(-1,2))
      niter = 0
      psi0 = 0
    newpsi = False
  while newpsi or (niter < settings['nsweep2']):
    if newpsi:
      if (niter-1)%settings['ncanon2'] != 0:
        yield 'select',(npsi-1,None)
        yield 'canonical', ()
        yield 'righttransf', (2,)
      yield 'newstate', ()
      yield 'shiftbench', ()
      npsi += 1
      yield 'righttransf', (2,)
      yield 'saveschmidt', ()
      for n in range(settings['bakein']):
        yield 'announce', f'[{n}/{settings["bakein"]}]'
        yield 'runsub', ('doublesweep',N), (n,'bakein',npsi)
        yield 'benchmark2', ()
        yield 'shiftbench', ()
        yield 'saveschmidt', ()
      yield 'select', (0,(-1,2))
      niter = 0
      psi0 = 0
    for ipsi in range(psi0,npsi):
      E = yield 'runsub',('doublesweep',N),(niter,ipsi,npsi)
      yield 'benchmark2', ()
      if ipsi+1 != npsi:
        yield 'select', (ipsi+1,(-1,2))
      yield 'saveschmidt', ()
    psi0 = 0
    if niter % settings['ncanon2'] == 0:
      yield 'clearall', ()
      for ipsi in range(npsi):
        yield 'select',(ipsi,None)
        yield 'canonical', (True,)
      for ipsi in range(npsi):
        yield 'select',(ipsi,None)
        yield 'righttransf', (2,) #TODO use gauge?
    diff = yield 'compare2', (niter,)
    yield 'select',(0,(-1,2))
    npsi,newpsi = yield 'addstatecond', (niter, diff)
    if not newpsi and diff < settings['schmidtdelta']:
      if niter % settings['ncanon2'] != 0:
        yield 'canonical', (True,)
      break
    niter += 1
    yield 'shiftbench', ()
    yield 'saveschmidt', ()
  if niter == settings['nsweep2']:
    niter -= 1
  if niter % settings['ncanon2'] != 0:
    yield 'clearall', ()
    for ipsi in range(npsi):
      yield 'select', (ipsi,None)
      yield 'canonical',(True,)
  yield 'shiftbench', ()
  yield 'select', (0,None)
  yield 'complete', ()
  
def singleOrthoSupervisor(N, npsis, settings, state=None):
  yield 'announce', 'Single update'
  if state is None:
    psi0 = 0
    i0 = 0
    resume = False
    yield 'righttransf', (1,)
    yield 'saveschmidt', ()
    npsis = yield 'select', (psi0,(-1,1))
  else:
    i0,psi0,npsis = state
    resume = True
  for niter in range(i0, settings['nsweep1']):
    for ipsi in range(psi0,npsis):
      if not resume:
        yield 'select', (ipsi,(-1,1))
        yield 'saveschmidt', ()
      E = yield 'runsub',('singlesweep',N),(niter,ipsi,npsis)
      resume = False
      yield 'benchmark2', ()
    psi0 = 0
    diff = yield 'compare2', (niter,)
    if diff < settings['schmidtdelta1']:
      break
    if niter % settings['ncanon1'] == 0:
      yield 'clearall', ()
      for ipsi in range(npsis):
        yield 'select',(ipsi,None)
        yield 'canonical', (True,)
      for ipsi in range(npsis):
        yield 'select',(ipsi,None)
        yield 'righttransf', (1,) #TODO use gauge?
      yield 'select', (0,None)
    yield 'shiftbench', ()
  if niter % settings['ncanon1'] != 0:
    yield 'clearall', ()
    for ipsi in range(npsis):
      yield 'select',(ipsi,None)
      yield 'canonical', (True,)
  yield 'shiftbench', ()
  yield 'select', (0,None)
  yield 'complete', ()
  
