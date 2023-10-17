from .mpsbasic import *
from .. import config
import os,os.path,shutil
import pickle,shelve,contextlib,itertools
import logging
import numpy as np
from copy import copy,deepcopy
from collections.abc import MutableMapping

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
      Mngr = pickle.load(open(filename,'rb'))
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
          Mngr.setchi(idx, chi)
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
            Mngr.setchi(idx, chi)
            Mngr.supervisors = 'paused'
            Mngr.supstatus = [(chi,2)]
            Mngr.suplabels = ['base']
            Mngr.E = sv
        return Mngr
      if override_chis:
        # Reset bond dimension list
        useas = override_chis if isinstance(override_chis,str) else None
        Mngr.resetchis(chis, use_as=useas)
      else:
        assert chis == Mngr.chis
      if Mngr.filename != filename:
        config.streamlog.warn('Updating save destination from %s to %s',
          Mngr.filename, filename)
        Mngr.filename = filename
        Mngr.saveprefix = file_prefix
      if override:
        # Additionally reset general settings
        dmrg_kw.pop('psi0',None)
        Mngr.resetsettings(dmrg_kw)
        if use_shelf:
          Mngr.setdbon()
        else:
          Mngr.setdboff(False)
      return Mngr
  return DMRGManager(Hamiltonian, chis, use_shelf=use_shelf, file_prefix=file_prefix, savefile=savefile, **dmrg_kw)


class DMRGManager:
  """Optimizer for DMRG"""
  def __init__(self, Hamiltonian, chis, psi0=None,
              file_prefix=None, savefile='auto', use_shelf=False,
              Edelta=1e-8, schmidtdelta=1e-6,ncanon1=10,ncanon2=10,
              nsweep1=1000,nsweep2=100,tol2=1e-12,tol1=None,tol0=1e-12,
              eigtol=None, eigtol_rel=None):
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
        (initial value tolerance is still eigtol)"""
    self.__version = '1.0'
    if isinstance(chis,int):
      self.chis = [chis]
    else:
      self.chis = chis
    self.H = Hamiltonian
    # Initialize state & its properties
    self.psi = psi0
    self.N = Hamiltonian.N
    self.chiindex = None
    self.chi = None
    self.schmidt0 = None
    self.E = None
    self.E0 = None
    self.eigtol = None

    self.saveprefix = file_prefix
    if savefile == 'auto' and isinstance(file_prefix,str):
      self.filename = (file_prefix+'.p') if savefile=='auto' else savefile
    else:
      self.filename = savefile

    if use_shelf:
      assert self.filename
      # Use autosave filename but with extension .db instead of .p
      self.dbpath = self.filename[:-1]+'d'
      self.database = PseudoShelf(self.dbpath)
    else:
      # Otherwise just use a dictionary instead
      self.dbpath = None
      self.database = {}
    # General (all-bond-dimension) settings
    self.settings_allchi = dict(Edelta=Edelta, schmidtdelta=schmidtdelta,
           ncanon1=ncanon1,ncanon2=ncanon2,nsweep1=nsweep1,nsweep2=nsweep2,
           tol0=tol0,tol1=tol1,tol2=tol2,eigtol=eigtol,eigtol_rel=eigtol_rel)
    self.settings = {}
    # Additional settings by bond dimension
    self.chirules = {}

    self._initfuncs()
    # Logging
    self._initlog()
                        
    self.supervisors = []
    self.supstatus = []
    self.suplabels = []
    self.savelevel = 2 # Max level (# supervisors) to save at

    # Registered objects--new in v1
    self._registry = {} # Replaces transfLs,transfRs
    # _ROstatespec replaces ntransfL,ntransfR
    # _activeRO, _inactiveRO identify occupied & empty containers 
    self._initializeROs()

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
                        'setchi':self.setchi,
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

  def _initlog(self):
    self.logger = config.logger.getChild('DMRG')
    self.logger.setLevel(10)

  def __getstate__(self):
    state = self.__dict__.copy()
    del state['chirules']
    del state['supfunctions']
    del state['supcommands']
    del state['supervisors']
    del state['logger']
    del state['callables']
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._initlog()
    # In all cases where "database"/"shelf" is in an old format,
    # load as dict & let conversion to new format occur in later processing
    if not config.loadonly:
      if '_DMRGManager__version' not in state:
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
      elif state['_DMRGManager__version'][0] == '0':
        # Registered-object form
        self._ROstatespec = [self.ntransfL,self.ntransfR]
        for n in range(self.ntransfL):
          self._registry[n,'l'] = self.transfLs[n]
          self._activeRO.add((n,'l'))
        for n in range(self.ntransfL,self.N-1):
          self._registry[n,'l'] = mpsbasic.LeftTransferManaged(self.psi,n,'l',
            self.H,manager=self)
          self._inactiveRO.add((n,'l'))
        for n in range(self.N-self.ntransfR,self.N):
          self._registry[n,'r'] = self.ntransfRs[n-self.N+self.ntransfR]
          self._activeRO.add((n,'r'))
        for n in range(1,self.N-self.ntransfR):
          self._registry[n,'r'] = mpsbasic.RightTransferManaged(self.psi,n,'r',
            self.H,manager=self)
          self._inactiveRO.add((n,'r'))
      # TODO recovery options if directory is expected but does not exist
    self.chirules = {}
    self._initfuncs()
    self.supervisors = 'paused'

  # "Registered objects" (here transfer vectors)--
  # all management should pass through these functions

  def fetchRO(self, key):
    if key not in self._activeRO:
      if key in self._inactiveRO:
        assert key in self._registry
        raise KeyError(f'Registered object requested {key} inactive')
      else:
        assert key not in self._registry
        raise KeyError(f'Registered object requested {key} not existent')
    return self._registry[key]

  def _produceRO(self, key):
    assert key in self._inactiveRO 
    assert key not in self._activeRO
    self._computeRO(key)
    self._activeRO.add(key)
    self._inactiveRO.remove(key)

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

  def refreshRO(self, key):
    assert key in self._activeRO
    self._computeRO(key)

  def _discardRO(self, key):
    self._registry[key].discard()
    self._activeRO.remove(key)
    self._inactiveRO.add(key)

  def _initializeROs(self,statespec=None):
    for site in range(self.N-1):
      self._registry[site,'l'] = LeftTransferManaged(self.psi,site,'l',self.H,
        manager=self)
    for site in range(1,self.N):
      self._registry[site,'r'] = RightTransferManaged(self.psi,site,'r',self.H,
        manager=self)
    self._ROstatespec = self._nullROstate
    self._inactiveRO = set(self._registry)
    self._activeRO = set()
    if statespec is not None:
      self._updateROstate(statespec)

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

  def _updateROstate(self, statespec):
    self._validateROstate(statespec)
    self.logger.debug('Updating registry to state %s',statespec)
    newactive = self.ROstatelist(statespec)
    # Add new *in order*
    for key in newactive:
      if key in self._inactiveRO:
        self._produceRO(key)
    self._ROstatespec = statespec
    # Discard extranneous
    for key in self._activeRO - set(newactive):
      self._discardRO(key)

  def _ROcheckdbentries(self, key):
    """Confirm existence of 'database' entry/entries"""
    transf = self._registry[key]
    if transf.id.hex not in self.database:
      # Re-compute
      site,lr = key
      self.logger.warn('%s transfer vector at site %d missing; restoring',
        'Right' if lr == 'r' else 'Left', site)
      if site == 0 or site == self.N-1:
        transf.compute(None)
      else:
        site0 = site + (1 if lr == 'r' else -1)
        transf.compute(self._registry[site0,lr].T)
    return {transf.id.hex}

  def clearall(self):
    self._updateROstate(self._nullROstate)
    assert not self._activeRO
    assert self._inactiveRO == set(self._registry)

  @property
  def _nullROstate(self):
    return (0,0)

  def ROstatelist(self, statespec):
    ntl,ntr = statespec
    return [(n,'l') for n in range(ntl)]+[(self.N-n-1,'r') for n in range(ntr)]

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

  def resetsettings(self,kw_args):
    assert set(kw_args.keys()).issubset(set(self.settings.keys()))
    self.settings_allchi.update(kw_args)
    # Update current as appropriate
    if self.chi:
      if self.chi in self.chirules:
        for k in set(kw_args) - set(self.chirules[self.chi]):
          self.settings[k] = kw_args[k]
      else:
        self.settings.update(kw_args)

  def resetchis(self, chis, use_as=None):
    """Reset list of bond dimensions; purge bond-dimension-dependent rules
    and pick chiindex"""
    self.chis = chis
    if self.supstatus == 'complete':
      if self.chi >= chis[-1]:
        # All completed
        return
      else:
        self.supstatus = []
    if self.chi not in chis:
      if not use_as:
        raise KeyError('Current bond dimension not in new list')
      nbelow = sum(c < self.chi for c in chis)
      if use_as == 'lower':
        assert nbelow > 0
        self.chiindex = nbelow-1
      else:
        assert use_as == 'higher' and nbelow < len(chis)
        self.chiindex = nbelow
      self.chi = chis[self.chiindex]
    else:
      self.chiindex = chis.index(self.chi)
    self.chirules = {}
    # Restart base supervisor
    if len(self.supstatus):
      self.supstatus[0] = (self.chi,)+self.supstatus[0][1:]
    if self.supervisors != 'paused' and len(self.supervisors):
      self.supervisors[0] = self.supfunctions['base'](self.chis,self.N,self.settings,self.supstatus[0])

  def settingbychi(self, chi, **kw_args):
    if isinstance(chi, int):
      self.logger.log(13, 'Adding rules for bond dimension %d',chi)
      if chi in self.chirules:
        self.chirules[chi].update(kw_args)
      else:
        self.chirules[chi] = kw_args
    else:
      for c in chi:
        self.settingbychi(c, **kw_args)
  
  def settingbyindex(self, idx, **kw_args):
    if isinstance(idx, int):
      self.settingbychi(self.chis[idx], **kw_args)
    else:
      for i in idx:
        self.settingbychi(self.chis[i], **kw_args)

  def set_command(self, name, func, bind=True):
    """Add additional supervisor-callable 'command' function
    if bind, make a quasi-bound method by adding self as argument"""
    if bind:
      import functools
      func = functools.partial(func,self)
    self.supcommands[name] = func

  def add_callable(self, name, func, bind=False):
    """Add additional directly-callable  function
    if bind, make a quasi-bound method by adding self as argument"""
    if bind:
      import functools
      func = functools.partial(func,self)
    self.callables[name] = func

  def useSupervisor(self,name,func):
    """Use function passed instead of standard"""
    self.supfunctions[name] = func

  def getE(self, savelast=True, updaterel=True):
    if savelast:
      self.E0 = self.E
    if self._ROstatespec[0] == 0 and self._ROstatespec[1] == 0:
      assert not self.E
      self.E = np.real(self.H.expv(self.psi))
      self.logger.log(25,'Initial energy %s',self.E)
      return self.E
    # TODO efficient outside of assumed case of no left transfers
    tR0 = self.gettransfR(self.N-self.nrighttransf)
    transf = tR0.moveby(self.N-self.nrighttransf-1,unstrict=True)
    self.E = np.real(transf.left(terminal=True))
    self.logger.log(20,'Energy calculated as %s',self.E) 
    if self.E0 is not None and self.settings['eigtol_rel'] and updaterel:
      dE = abs(self.E - self.E0)
      self.eigtol = self.settings['eigtol_rel']*dE
      self.logger.log(10,'eigtol_rel set to %s',self.eigtol)
    return self.E

  def save(self):
    if self.filename:
      try:
        pickle.dump(self,open(self.filename,'wb'))
      except MemoryError:
        self.logger.error('Encountered MemoryError while saving (consider adjusting)')
        import gc
        gc.collect()
        pickle.dump(self,open(self.filename,'wb'))

  def savecheckpoint(self, level):
    if config.haltsig:
      self.logger.log(40,'Saving and quitting')
      self.save() 
      return True
    if level <= self.savelevel:
      self.logger.log(15,'Saving at checkpoint')
      self.save()
    return False

  def savestep(self):
    """Save output after optimizing a single bond dimension"""
    self.logger.log(20,'Saving completed bond dimension %s',self.chi)
    pickle.dump((self.psi,self.E),open(f'{self.saveprefix}c{self.chi}.p','wb'))

  def resettol(self):
    """Reset eigtol_rel to eigtol"""
    if self.settings['eigtol'] is not None:
      self.logger.log(10,'Resetting eigtol_rel to %s',self.settings['eigtol'])
      self.eigtol = self.settings['eigtol']

  def initstate(self):
    # Options for passing MPS
    if isinstance(self.psi,str):
      self.logger.log(30, 'Loading initial state from %s',self.psi)
      rv = pickle.load(open(self.psi,'rb'))
      if isinstance(rv,tuple):
        rv, *sv = rv
      if isinstance(rv,DMRGManager):
        self.logger.log(15, 'Loaded from manager')
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
      self.psi = self.H.rand_MPS(bond=bond)
      self.restorecanonical()
    if 'process_psi0' in self.callables:
      self.psi = self.callables['process_psi0'](self.psi)
    self._initializeROs()
    if not self.psi.iscanon():
      self.restorecanonical()
    self.getE()

  def setchi(self, idx, chi):
    # Step data
    self.chiindex = idx
    self.chi = chi
    # Bond-dimension-dependent settings
    self.settings.update(self.settings_allchi)
    if chi in self.chirules:
      self.logger.debug('Found settings of %s at bond dimension %d',list(self.chirules[chi]),chi)
      self.settings.update(self.chirules[chi])
    # Eigenvalue tolerance
    self.eigtol = self.settings['eigtol']
    self.logger.log(10, 'eigtol_rel set to %s',self.eigtol)
    config.streamlog.log(30, 'chi = % 3d (#%s)', chi, idx)
  
  def getrighttransfers(self, n0):
    self.logger.log(10, 'Collecting right transfer matrices')
    # TODO gauge from canonical
    self.clearall()
    self._updateROstate((0,self.N-n0))

  def setdbon(self):
    dbpath = self.filename[:-1]+'d'
    if not self.dbpath:
      self.logger.warn('Transferring transfer tensors to shelf...')
      # Create database
      database = PseudoShelf(dbpath)
      database.update(self.database)
      del self.database
      self.database = database
      self.dbpath = dbpath
      self.logger.debug('Shelf created at %s',dbpath)
      self.save()
    elif dbpath != self.dbpath:
      self.logger.info('Copying shelf from %s to %s',dbpath,self.dbpath)
      if isinstance(self.database,PseudoShelf):
        self.dbpath = dbpath
        try:
          self.database.copyto(dbpath)
        except FileNotFoundError:
          self.logger.warn('Shelf directory not found, repopulating %d left '
            'and %d right transfer vectors', *self._ROstatespec)
          self.regenerate()
      else:
        self.database = PseudoShelf(dbpath)
        try:
          self.database.update(self.dbpath)
        except FileNotFoundError:
          self.logger.warn('Shelf directory not found, repopulating %d left and '
            '%d right transfer vectors', self.ntransfL,self.ntransfR)
          self.regenerate()
      # Move database
      self.save()
    else:
      self.logger.debug('Shelf already in place')
    try:
      self.checkdatabase()
    except AssertionError:
      self.logger.exception('Error checking database; recomputing transfers')
      self.database.clear()
      self.regenerate()

  def regenerate(self):
    statespec = self._ROstatespec
    self.clearall()
    self._initializeROs(statespec)

  def checkdatabase(self):
    self.logger.debug('Checking database and restoring entries as necessary')
    if self.dbpath:
      if not self.database:
        self.database = PseudoShelf(self.dbpath)
      dbkeys = self.database.checkentries()
    else:
      dbkeys = set(self.database)
    # Check converted: TODO depricate
    for key in self._registry:
      assert self._validateRO(self, key)
    activedbkeys = set()
    for key in self.ROstatelist(self._ROstatespec):
      activedbkeys.update(self._ROcheckdbentries(key))
    for k in dbkeys - activedbkeys:
      self.logger.info('Removing unidentified key %s',k)
      del self.database[k]
    assert len(self.database) == len(self.activeRO)
    # TODO change for case RO-database is not one-to-one

  def setdboff(self, remove=True):
    if self.dbpath:
      self.logger.warn('Converting shelf into local memory...')
      database = {}
      for k in self.database:
        database[k] = self.database[k]
      if remove:
        self.logger.warn('Deleting shelf database directory')
        shutil.rmtree(self.dbpath)
      self.database = database
      self.dbpath = None
      self.save()
    else:
      self.logger.debug('Transfer vectors already in local memory')
          
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
    config.streamlog.log(30,f'[% 3d] %+12.4g E=%0.10f',niter,Ediff,self.E)
    return abs(Ediff)

  def saveschmidt(self):
    self.schmidt0 = []
    for n,ll in enumerate(self.psi._schmidt):
      V = self.psi.getbond(n)
      if hasattr(V,'full_iter'):
        # Dictionary by charge sector
        l1 = {k:ll[idx:idx+d*degen:d] for k,degen,d,idx in V.full_iter()}
      else:
        l1 = copy(ll)
      # Update history
      self.schmidt0.append(l1)
    
  def compare2(self, niter):
    # Schmidt-coefficient difference
    if niter == 0:
      config.streamlog.log(30, 'Double update')
    # TODO full RMS option?
    Ldiff = 0
    for n,l0 in enumerate(self.schmidt0):
      l1 = self.psi._schmidt[n]
      if isinstance(l0, dict):
        # Dictionary by charge sector
        viter = self.psi.getbond(n).full_iter()
        l1 = {k:l1[idx:idx+d*degen:d] for k,degen,d,idx in viter}
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
    config.streamlog.log(30,f'[% 4d] %10.4g %+10.4g E=%0.10f',niter,Ldiff,self.E-self.E0,self.E)
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
    self.H.DMRG_opt_double(self.psi, n, self.chi,
        tL,tR,None,direction=='r', self.settings['tol2'], self.eigtol)

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

  def run(self):
    if self.supstatus == 'complete':
      print('Optimization already completed')
      return self.returnvalue()
    if not self.supervisors:
      # Initialize base
      self.logger.log(10, 'Initializing base supervisor')
      self.supervisors.append(self.supfunctions['base'](*self.baseargs()))
      self.suplabels.append('base')
      self.supstatus.append(None)
    elif self.supervisors == 'paused':
      self.logger.log(10, 'Resuming base supervisor')
      self.logger.log(5, 'Using state %s', self.supstatus[0])
      config.streamlog.log(30, 'Resuming: chi = % 3d (#%s)', self.chi, self.chiindex)
      if self.chi in self.chirules:
        self.logger.debug('Found settings of %s at bond dimension %d',list(self.chirules[self.chi]),self.chi)
        self.settings.update(self.chirules[self.chi])
      # Initialize saved supervisors
      sup = self.supfunctions['base'](*self.baseargs(), state=self.supstatus[0])
      self.supervisors = [sup]
      level = 0
      while len(self.supervisors) < len(self.supstatus):
        cmd,*argall = next(sup)
        while cmd == 'announce':
          config.streamlog.log(30, argall[0])
          cmd,*argall = next(sup)
        (label,*args),status = argall
        assert cmd == 'runsub' and label == self.suplabels[level+1]
        if self.supstatus[level] != status:
          self.logger.log(25, 'Supervisor %s status %s superceded as %s',
            label, self.supstatus[level], status)
          self.supstatus[level] = status
        level += 1
        self.logger.log(10, 'Resuming level-%s supervisor %s',level+1,label)
        self.logger.log(5, 'Using arguments %s', args)
        sup = self.supfunctions[label](*args, self.settings, state=self.supstatus[level])
        self.supervisors.append(sup)
    level = len(self.supervisors)
    sendval = None
    while level > 0:
      # Next step from current subroutine
      cmd,args,*status = self.supervisors[-1].send(sendval)
      self.logger.log(8, 'Received %s command from supervisor %s',
        cmd, self.suplabels[-1])
      sendval = None # Default to not sending anything
      if status:
        # Candidate save checkpoint--includes supervisor status
        self.logger.log(8, 'Setting status to %s', status)
        self.supstatus[-1], = status
        if self.filename:
          savequit = self.savecheckpoint(level)
          if savequit:
            import sys
            sys.exit()
      # Options for cmd
      if cmd == 'complete':
        # Subroutine completed execution
        self.supervisors.pop()
        self.supstatus.pop()
        label = self.suplabels.pop()
        level -= 1
        if args:
          # Pass arguments up the chain
          sendval = args
        self.logger.log(10, 'Supervisor %s complete',label)
      elif cmd == 'runsub':
        # New sub-supervisor 
        label,*supargs = args
        subsup = self.supfunctions[label](*supargs, self.settings)
        self.supervisors.append(subsup)
        self.supstatus.append(None)
        self.suplabels.append(label)
        level += 1
        self.logger.log(10, 'Starting level-%s supervisor %s',level,label)
      elif cmd == 'announce':
        # Announcement
        # TODO logging levels?
        config.streamlog.log(30, args)
      else:
        # Algorithm-specific options
        sendval = self.supcommands[cmd](*args)
    self.supstatus = 'complete'
    self.save()
    return self.returnvalue()

def baseSupervisor(chis, N, settings, state=None):
  if state is None:
    yield 'init',()
    cidx = 0
    yield 'setchi',(0,chis[0])
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
        yield 'setchi',(cidx,chis[cidx])
  while cidx < len(chis):
    yield 'runsub',('double',N),(chis[cidx],2)
    yield 'runsub',('single',N),(chis[cidx],1)
    cidx += 1
    if cidx < len(chis):
      yield 'savestep', ()
      yield 'setchi',(cidx,chis[cidx])
  yield 'complete',()

def doubleOptSupervisor(N, settings, state=None):
  if state is None:
    i0 = 0
    yield 'righttransf', (2,)
    yield 'saveschmidt', ()
  else:
    i0, = state
  for niter in range(i0,settings['nsweep2']):
    E = yield 'runsub',('doublesweep',N),(niter,)
    if niter % settings['ncanon2'] == 0:
      yield 'canonical', (True,)
      yield 'righttransf', (2,) #TODO use gauge?
    diff = yield 'compare2', (niter,)
    if diff < settings['schmidtdelta']:
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
  a1 = np.pad(v1,(0,numel-len(v1)))
  a2 = np.pad(v2,(0,numel-len(v2)))
  return np.linalg.norm(a1-a2)

class PseudoShelf(MutableMapping):
  # Dictionary-like class representing storage in a directory (rather than a
  # true database file)
  def __init__(self, dirname):
    dirname = os.path.abspath(dirname)
    if not os.path.isdir(dirname):
      os.mkdir(dirname)
    self.path = dirname

  def fname(self, key):
    return os.path.join(self.path,key+'.p')

  def __getitem__(self, key):
    if not os.path.isfile(self.fname(key)):
      raise KeyError(key)
    return pickle.load(open(self.fname(key),'rb'))

  def __setitem__(self, key, value):
    pickle.dump(value, open(self.fname(key),'wb'))

  def __delitem__(self, key):
    os.remove(self.fname(key))

  def __iter__(self):
    return (f[:-2] for f in os.listdir(self.path) if f[-2:] == '.p')
  
  def __len__(self):
    return len(os.listdir(self.path))

  def __contains__(self, key):
    return os.path.isfile(self.fname(key))

  def clear(self):
    for f in os.listdir(self.path):
      os.remove(os.path.join(self.path,f))

  def checkentries(self):
    """Load files to check. Removes invalid or corrupt files.
    Returns set of keys."""
    # TODO temporary destination for invalid/corrupt files?
    keys = set()
    for f in os.listdir(self.path):
      if f[-2:] != '.p':
        config.logger.error('Removing invalid file "%s" in database directory',f)
        os.remove(os.path.join(self.path,f))
      key = f[:-2]
      try:
        obj = pickle.load(open(os.path.join(self.path,f),'rb'))
        del obj
      except e:
        config.logger.exception('Entry with key %s corrupted; removing',key)
      keys.add(key)
    return keys

  def move(self, path2):
    """Change directory path (remove original)"""
    path2 = os.path.abspath(path2)
    if os.path.exists(path2):
      raise FileExistsError('New path already exists')
    config.logger.warn('Moving database directory from %s to %s',self.path,path2)
    shutil.move(self.path,path2)
    self.path = path2

  def copyto(self, path2):
    """Change directory path (keep original)"""
    path2 = os.path.abspath(path2)
    if os.path.exists(path2):
      raise FileExistsError('New path already exists')
    config.logger.warn('Copying database directory from %s to %s',self.path,path2)
    shutil.copytree(self.path,path2)
    self.path = path2

  def update(self, d):
    # If PseudoShelf or valid directory, copy files; otherwise proceed as normal
    if isinstance(d,PseudoShelf):
      d = d.path
    if isinstance(d,str):
      d = os.path.abspath(d)
      if not os.path.isdir(d):
        raise FileNotFoundError(d)
      for f in os.listdir(d):
        if f[:-2] == '.p':
          shutil.copy2(os.path.join(d,f),os.path.join(self.path,f))

def ortho_manager(Hamiltonian, chis, npsis, file_prefix=None, savefile='auto',
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
      Mngr = pickle.load(open(filename,'rb'))
      if override_chis:
        # Reset bond dimension list
        useas = override_chis if isinstance(override_chis,str) else None
        Mngr.resetchis(chis, use_as=useas)
        assert npsis >= Mngr.npsis
        Mngr.npsi_tot = npsis
      else:
        assert chis == Mngr.chis
        assert npsis == Mngr.npsi_tot
      if Mngr.filename != filename:
        config.streamlog.warn('Updating save destination from %s to %s',
          Mngr.filename, filename)
        Mngr.filename = filename
        Mngr.saveprefix = file_prefix
      if override:
        # Additionally reset general settings
        dmrg_kw.pop('psi0',None)
        dmrg_kw.pop('persiteshift',None)
        Mngr.resetsettings(dmrg_kw)
        if use_shelf:
          Mngr.setdbon()
        else:
          Mngr.setdboff(False)
      return Mngr
  return DMRGOrthoManager(Hamiltonian, chis, npsis, use_shelf=use_shelf, file_prefix=file_prefix, savefile=savefile, **dmrg_kw)


class DMRGOrthoManager(DMRGManager):
  """Set of low-lying orthogonal states"""
  def __init__(self, Hamiltonian, chis, npsis, ortho_tol=1e-7, keig=1,
      newthresh=1e-3,schmidtdelta1=2e-8,bakein=5,nsweepadd=10,persiteshift=0,
      **kw_args):
    self.psiindex = 0
    self.npsi_tot = npsis
    self.npsis = 1
    self.psilist = [None]
    self.dottransferR = {}
    self.dottransferL = {}
    self.transfcacheL = {}
    self.transfcacheR = {}
    self.Es = [None]
    self.s0s = [None]
    self.E0s = [None]
    self.dschmidts = [0]
    super().__init__(Hamiltonian, chis, **kw_args)
    if persiteshift <= 0:
      raise ValueError('persiteshift must be provided')
    self.Eshift = self.N*persiteshift
    self.__version = '0.2'
    self.settings_allchi.update(ortho_tol=ortho_tol,keig=keig,
      newthresh=newthresh,schmidtdelta1=schmidtdelta1,bakein=bakein,
      nsweepadd=nsweepadd,useguess=True)

  def __setstate__(self, state):
    super().__setstate__(state)
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
        self.supstatus[0] = (self.chi,self.npsis,0)
        self.suplabels[1] = 'addstate'
        if len(labels) > 1:
          if labels[1] == 'double':
            # Directly translates
            niter,psi0,npsi = state
            if psi0 == 'bakein':
              self.supstatus[1] = (niter,'bakein',npsi)
            else:
              niter1 = min(niter,self.settings['nsweepadd']-1)
              self.supstatus[1] = (niter1,psi0,npsi)
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
        self.supstatus[0] = self.supstatus[0]+(self.npsis,)
      self.settings['nsweepadd'] = 10
      self.__version = '0.2'
    if self.__version == '0.2':
      self.eigtol = abs(self.eigtol)

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
    self.supcommands['clearall'] = self.clearcache
    self.supcommands['shiftbench'] = self.shiftbenchmark
    self.supcommands['benchmark1'] = self.benchmark1
    self.supcommands['benchmark2'] = self.benchmark2

  def savestep(self):
    """Save output after completing bond-dimension optimization"""
    self.logger.log(20,'Saving states with bond dimension %d',self.chi)
    for i in range(self.npsis):
      pickle.dump((self.psilist[i],self.Es[i]),open(f'{self.saveprefix}c{self.chi}n{self.psiindex}.p','wb'))

  def check_add_state(self, niter, delta):
    if self.npsis == 1 and delta is None:
      # TODO setting for not-already-initialized state?
      return 1,True
    if self.npsis < self.npsi_tot and niter == self.settings['nsweep2']-1:
      return self.npsis,True
    return self.npsis, (self.npsi_tot > self.npsis and delta is not None and delta < self.settings['newthresh'])

  def flushtransferleft(self, site):
    """Remove or recalculate transfer vectors to get to correct site
    Unlike getlefttransfers, assumes any existing/cached are valid"""
    n = site+1
    self.logger.info('Flushing left transfer lists to site %d (length %d)',
      site,n)
    for itr,tL in enumerate(self.ltransflist):
      while len(tL) > n:
        # Longer than needed
        tL.pop().discard()
      if n and not len(tL):
        # Initialize
        if itr == 0:
          tL.append(self.H.getboundarytransfleft(self.psi, manager=self))
        else:
          i,j = self.collected_pairs()[itr-1]
          tL.append(LeftTransferManaged(self.psilist[j],0,'l',self.psilist[i],
            manager=self))
          tL[-1].compute(None)
      while len(tL) < n:
        tL.append(tL[-1].right())
    self.ntransfL = n

  def flushtransferright(self, site):
    n = self.N-site
    self.logger.info('Flushing right transfer lists to site %d (length %d)',
      site,n)
    for itr,tR in enumerate(self.rtransflist):
      while len(tR) > n:
        # Longer than needed
        tR.pop(0).discard()
      if n and not len(tR):
        # Initialize
        if itr == 0:
          tR.append(self.H.getboundarytransfright(self.psi,manager=self))
        else:
          i,j = self.collected_pairs()[itr-1]
          tR.append(RightTransferManaged(self.psilist[j],self.N-1,'r',
            self.psilist[i],manager=self))
          tR[0].compute(None)
      while len(tR) < n:
        tR.insert(0,tR[0].left())
    self.ntransfR = n

  def clearcache(self):
    for tll in (self.ltransflist,self.rtransflist,self.transfcacheL.values(),self.transfcacheR.values()):
      for tlist in tll:
        while len(tlist) > 0:
          tlist.pop().discard()
          
  def selectpsi(self, index, flush_to, log=True):
    if log:
      self.logger.info('Selecting state #%d',index)
    if index == self.psiindex:
      if flush_to is not None:
        # Check that correct transfers are there
        fleft,fright = flush_to
        self.flushtransferleft(fleft)
        self.flushtransferright(fright)
      return self.npsis
    self.transfcacheL[self.psiindex] = self.transfLs
    self.transfcacheR[self.psiindex] = self.transfRs
    if index in self.transfcacheL:
      self.transfLs = self.transfcacheL.pop(index)
    else:
      self.transfLs = []
    if index in self.transfcacheR:
      self.transfRs = self.transfcacheR.pop(index)
    else:
      self.transfRs = []
    if flush_to is not None:
      fleft,fright = flush_to
    else:
      fleft = len(self.transfLs)-1
      fright = self.N-len(self.transfRs)
    dottL = self.dottransferL
    dottR = self.dottransferR
    self.dottransferL = {}
    self.dottransferR = {}
    self.psiindex = index
    for i,j in self.collected_pairs():
      # check: already in `active' transfer list?
      tLs = []
      c = False
      if (i,j) in dottL:
        tLs = dottL.pop((i,j))
      elif (j,i) in dottL:
        tLs = dottL.pop((j,i))
        c = True
      elif (i,j) in self.transfcacheL:
        tLs = self.transfcacheL.pop((i,j))
      elif (j,i) in self.transfcacheR:
        tLs = self.transfcacheL.pop((j,i))
        c = True
      if c:
        for n in range(len(tLs)):
          T = tLs[n]
          tLs[n] = T.conj()
          T.discard()
      self.dottransferL[i,j] = tLs
      tRs = []
      c = False
      if (i,j) in dottR:
        tLs = dottR.pop((i,j))
      elif (j,i) in dottR:
        tRs = dottR.pop((j,i))
        c = True
      elif (i,j) in self.transfcacheR:
        tRs = self.transfcacheR.pop((i,j))
      elif (j,i) in self.transfcacheR:
        tRs = self.transfcacheR.pop((j,i))
        c = True
      if c:
        for n in range(len(tRs)):
          T = tRs[n]
          tRs[n] = T.conj()
          T.discard()
      self.dottransferR[i,j] = tRs
    self.flushtransferleft(fleft)
    self.flushtransferright(fright)
    self.transfcacheL.update(dottL)
    self.transfcacheR.update(dottR)
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
    config.streamlog.log(20, f'\t#%d %+12.4g E=%0.10f',self.psiindex,self.E-self.E0,self.E)

  def benchmark2(self):
    # Get energy & schmidt-difference for current state
    self.getE(savelast=False,updaterel=False)
    # TODO full RMS option?
    Ldiff = 0
    for n,l0 in enumerate(self.schmidt0):
      l1 = self.psi._schmidt[n]
      if isinstance(l0, dict):
        # Dictionary by charge sector
        viter = self.psi.getbond(n).full_iter()
        l1 = {k:l1[idx:idx+d*degen:d] for k,degen,d,idx in viter}
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
    self.logger.log(15, '[#%d]  energy diff %s', self.psiindex, self.E-self.E0)
    config.streamlog.log(25,f'\t#%d %#10.4g %#+10.4g E=%0.10f',self.psiindex,Ldiff,self.E-self.E0,self.E)

  def compare1(self, niter, which='all'):
    assert which == 'all'
    dEtot = np.mean([abs(self.Es[i]-self.E0s[i]) for i in range(self.npsis)])
    self.streamlog.log(30,f'[% 3d] %#12.4g',niter,dEtot)
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
    config.streamlog.log(30,f'[% 3d] %#12.4g %#12.4g : %#+12.4g',niter,Ldtot,dEavg,dEtot)
    if self.settings['eigtol_rel']:
      self.eigtol = self.settings['eigtol_rel']*dEavg
      self.logger.log(10,'eigtol_rel set to %s',self.eigtol)
    return Ldtot

  def newstate_center(self):
    # Initialize with previous
    config.streamlog.log(30,'[ Adding state #%d ]', self.npsis)
    site = self.N//2-1
    self.logger.info('Initializing state #%d with double update at site %d',
      self.npsis,site)
    self.npsis += 1 
    self.psilist.append(copy(self.psilist[-1]))
    Einit = self.Es[-1]
    if Einit is None:
      Einit = self.E0s[-1]
    self.Es.append(Einit)
    self.E0s.append(None)
    self.s0s.append(None)
    self.dschmidts.append(0)
    self.selectpsi(self.npsis-1,(site-1,site+2))
    #self.getrighttransfers(site+2)
    #self.getlefttransfers(site-1)
    ug,self.settings['useguess'] = self.settings['useguess'],False
    self.doubleupdate(site,'r')
    self.settings['useguess'] = True
    self.logger.info('Completing by imposing canonical form')
    self.restorecanonical()
    self.getrighttransfers(2)
  
  def collected_pairs(self):
    # All pairs of states represented in transfer matrices
    # TODO make dependent on callable
    return [(self.psiindex,i) for i in range(self.npsis) if i != self.psiindex]

  def getrighttransfers(self, n0):
    self.logger.log(20, 'Collecting right transfer matrices (to site %d)',n0)
    #TODO be more deliberate about when clearing database
    for n in range(self.ntransfR):
      self.transfRs[n].discard()
      for idx in self.dottransferR:
        self.dottransferR[idx][n].discard()
    for n in range(self.ntransfL):
      self.transfLs[n].discard()
      for idx in self.dottransferL:
        self.dottransferL[idx][n].discard()
    #self.database.clear()
    self.transfRs = self.H.right_transfer(self.psi, n0, collect=True, manager=self)
    self.transfLs = []
    self.ntransfR = len(self.transfRs)
    self.ntransfL = 0
    self.dottransferL = {}
    self.dottransferR = {}
    # TODO callable for options for which transfers to track
    #for i in range(self.psiindex):
    for i,j in self.collected_pairs():
      self.dottransferL[i,j] = []
      phi,psi = self.psilist[i],self.psilist[j]
      tR = RightTransferManaged(psi, self.N-1, 'r', phi, manager=self)
      tR.compute(None)
      self.dottransferR[i,j] = tR.moveby(self.N-1-n0,collect=True)[::-1]

  def getlefttransfers(self, n0):
    # Only clear left here
    for n in range(self.ntransfL):
      self.transfLs[n].discard()
      for idx in self.dottransferL:
        self.dottransferL[idx][n].discard()
    self.logger.log(20, 'Collecting left transfer matrices (to site %d)',n0)
    self.transfLs = self.H.left_transfer(self.psi, n0, collect=True, manager=self)
    self.ntransfL = len(self.transfLs)
    self.dottransferL = {}
    # TODO callable for options for which transfers to track
    #for i in range(self.psiindex):
    for i,j in self.collected_pairs():
      phi,psi = self.psilist[i],self.psilist[j]
      tL = LeftTransferManaged(psi, 0, 'l', phi, manager=self)
      tL.compute(None)
      self.dottransferL[i,j] = tL.moveby(n0-1,collect=True)

  def regenerate(self):
    if self.ntransfR:
      # Start with boundary 
      self.transfRs[-1].compute(None)
      for nr in range(self.ntransfR-2,-1,-1):
        self.transfRs[nr].compute(self.transfRs[nr+1].T)
      for dotR in self.dottransferR.values():
        dotR[-1].compute(None)
        for nr in range(self.ntransfR-2,-1,-1):
          dotR[nr].compute(dotR[nr+1].T)
    if self.ntransfL:
      self.transfLs[0].compute(None)
      for nl in range(1,self.ntransfL):
        self.transfLs[nl].compute(self.transfLs[nl-1].T)
      for dotL in self.dottransferL.values():
        dotL[0].compute(None)
        for nl in range(1,self.ntransfL):
          dotL[nl].compute(dotR[nl-1].T)

  def checkdatabase(self):
    self.logger.debug('Checking database and restoring entries as necessary')
    if self.dbpath:
      if not self.database:
        self.database = PseudoShelf(self.dbpath)
      keys = self.database.checkentries()
    else:
      keys = set(self.database)
    if self.ntransfR:
      for idx,tR in enumerate(self.rtransflist):
        if idx == 0:
          state = 'H'
        else:
          state = idx-1
        if tR[-1].id.hex not in keys:
          self.logger.warn('Right boundary transfer vector (%s) missing; restoring',state)
          tR[-1].compute(None)
        for n in range(1,self.ntransfR):
          if tR[-n-1].id.hex not in keys:
            if idx == self.psiindex:
              self.logger.warn('Right transfer vector at %s (%s) missing; '
                'restoring', self.N-n-1, state)
            tR[-n-1].compute(tR[-n].T)
        assert len(tR) == self.ntransfR
    if self.ntransfL:
      for idx,tL in enumerate(self.ltransflist):
        if idx == 0:
          state = 'H'
        else:
          state = idx-1
        if tL[0].id.hex not in keys:
          self.logger.warn('Left boundary transfer vector (%s) missing; restoring',state)
          tL[0].compute(None)
        for n in range(1,self.ntransfL):
          if tL[n].id.hex not in keys:
            self.logger.warn('Left transfer vector at %s (%s) missing; '
              'restoring',n,state)
          tL[n].compute(tL[n-1].T)
        assert len(tL) == self.ntransfL
    valid_keys = {t.id.hex for tlist in (self.rtransflist+self.ltransflist) for t in tlist}
    for k,tL in self.transfcacheL.items():
      n = 0
      while n < len(tL) and tL[n].id.hex in keys:
        valid_keys.add(tL[n].id.hex)
        n += 1
      # Delete additional
      if n < len(tL):
        self.logger.info('Cached left transfer chain %s broken at %d of %d,'
          'deleting remainder',k,n,len(tL))
        while len(tL) > n:
          T = tL.pop()
          if T.id.hex in keys:
            del self.database[T.id.hex]
    for k,tR in self.transfcacheR.items():
      nt = len(tR)
      n = nt-1
      while n >= 0 and tR[n].id.hex in keys:
        valid_keys.add(tR[n].id.hex)
        n -= 1
      # Delete additional
      if n >= 0:
        self.logger.info('Cached right transfer chain %s broken at %d of %d,'
          'deleting remainder',k,n,nt)
        while len(tR)+n >= nt:
          T = tR.pop(0)
          if T.id.hex in keys:
            del self.database[T.id.hex]
    for k in keys - valid_keys:
      self.logger.info('Removing unidentified key %s',k)
      del self.database[k]
    self.logger.log(8,'%d valid keys (%d), %d left transfers, %d right transfers, state #%d',len(valid_keys),len(self.database),self.ntransfL,self.ntransfR,self.psiindex)
    assert len(self.database) == (self.npsis)*(self.ntransfL + self.ntransfR) + sum(len(tL) for tL in self.transfcacheL.values()) + sum(len(tR) for tR in self.transfcacheR.values())
    # TODO change to match collection setting

  @property
  def rtransflist(self):
    return [self.transfRs]+[self.dottransferR[idx] for idx in self.collected_pairs()]

  @property
  def ltransflist(self):
    return [self.transfLs]+[self.dottransferL[idx] for idx in self.collected_pairs()]

  def gettransfR(self, n, which=None):
    """Transfer matrix at site n; Hamiltonian if which is None,
    identified pair otherwise"""
    # TODO combine options with proxy context manager?
    self.logger.debug('Fetching site %d right transfers (%s) (of %d presently)',n,which,self.ntransfR)
    if n == self.N-self.ntransfR-1:
      # Compute transfer
      self.logger.log(15,'Computing site %d right transfers',n)
      if self.ntransfR == 0:
        tR = self.H.getboundarytransfright(self.psi, manager=self)
        self.transfRs = [tR]
        #yield tR
        for i,j in self.collected_pairs():
          tR = RightTransferManaged(self.psilist[j],self.N-1,'r',
            self.psilist[i], manager=self)
          tR.compute(None)
          self.dottransferR[i,j] = [tR]
          #yield tR
      else:
        for tlist in self.rtransflist:
          tR = tlist[0].left()
          #yield tR
          tlist.insert(0,tR)
      self.ntransfR += 1
      #return
    nm = n-(self.N-self.ntransfR)
    assert nm >= 0
    if which is None:
      return self.transfRs[nm]
    else:
      return self.dottransferR[which][nm]

  def gettransfL(self, n, which=None):
    self.logger.debug('Fetching site %d left transfers (%s) (of %d presently)',n,which,self.ntransfL)
    if n == self.ntransfL:
      self.logger.log(15, 'Computing site %d left transfers',n)
      # Compute transfer
      if self.ntransfL == 0:
        tL = self.H.getboundarytransfleft(self.psi, manager=self)
        self.transfLs = [tL]
        #yield tL
        for i,j in self.collected_pairs():
          tL = LeftTransferManaged(self.psilist[j],0,'l',self.psilist[i],
            manager=self)
          tL.compute(None)
          self.dottransferL[i,j] = [tL]
        # yield tL
      else:
        for tlist in self.ltransflist:
          tL = tlist[-1].right()
          tlist.append(tL)
        #tL = self.transfLs[-1].right()
        #self.transfLs.append(tL)
      self.ntransfL += 1
      #eturn
    assert n <= self.ntransfL
    if which is None:
      return self.transfLs[n]
    else:
      return self.dottransferL[which][n]

  def discardleft(self):
    # Discard (rightmost) left transfer matrix
    self.logger.info('Discarding left transfer matrix (%d)',self.ntransfL-1)
    for tlist in self.ltransflist:
      tlist.pop().discard()
    self.ntransfL -= 1

  def discardright(self):
    # Discard (leftmost) right transfer matrix
    self.logger.info('Discarding right transfer matrix (%d)',self.N-self.ntransfR)
    for tlist in self.rtransflist:
      tlist.pop(0).discard()
    self.ntransfR -= 1

  def compare1(self, niter):
    # TODO energy comparison: individual or all together?
    if niter == 0:
      config.streamlog.log(30, 'Single update')
    self.getE()
    Ediff = self.E-self.E0
    self.logger.log(20, '[%s]  energy diff %s', niter, Ediff)
    config.streamlog.log(30,f'[% 3d] %+12.4g E=%0.10f',niter,Ediff,self.E)
    return abs(Ediff)

  def ortho_vectors(self, site, nsites):
    right = site+nsites == self.N
    left = site == 0
    for i,j in self.collected_pairs():
      phi = self.psilist[j]
      if i != self.psiindex:
        continue
      self.logger.debug('protecting orthogonality of %d (current) w/ %d: site(s) %s', i, j, tuple(range(site,site+nsites)))
      T0 = phi.getTL(site)
      if not (nsites==1 and right):
        T0 = T0.diag_mult('r',phi.getschmidt(site))
      if nsites == 2:
        Tr = phi.getTR(site+1)
        T0 = T0.contract(Tr,'r-l;b>bl,~;b>br,~')
      T = T0.conj()
      if not left:
        L = self.gettransfL(site-1,(i,j))
        self.logger.debug('left at %d',L.site)
        T = T.contract(L.T, 'l-t;~;b>l*')
      if not right:
        R = self.gettransfR(site+nsites,(i,j))
        self.logger.debug('right at %d',R.site)
        T = T.contract(R.T,'r-t;~;b>r*')
      yield T.conj()
      
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
    if self.settings['useguess']:
      if gauge is not None and gauge[0] == 'l':
        M0 = self.psi.getTc(site).mat_mult('r-l',gauge[1])
        M0 = M0.diag_mult('l',self.psi.getschmidt(site-1))
      else:
        M0 = self.psi.getTL(site)
      if not (nsites==1 and site+1==self.N):
        if nsites==1 and gauge is not None and gauge[0] == 'r':
          M0 = M0.mat_mult('l-r',gauge[1])
        M0 = M0.diag_mult('r',self.psi.getschmidt(site))
      if nsites == 2:
        M0 = M0.contract(self.psi.getTR(site+1),'r-l;b>bl,~;b>br,~')
    else:
      M0 = None
    w,v = Hshift.eigs(self.settings['keig'],which='LM',guess=M0, tol=self.eigtol)
    self.logger.log(16,'Effective energy computed as %0.10f',min(w)+self.Eshift)
    return v[np.argmin(w)]
    
  def doubleupdateleft(self):
    # TODO better way of handling transfer
    tR = self.gettransfR(2)
    self.logger.log(12, 'left edge double update')
    Heff = operators.NetworkOperator('(T);Ol;Or;R;R.t-T.r,Ol.r-Or.l,'
      'Or.r-R.c,Ol.t-T.bl,Or.t-T.br;R.b>r,Ol.b>bl,Or.b>br',
      self.H.getT(0),self.H.getT(1),tR.T)
    M = self.optimize_tensor(Heff, 0, 2)
    self.logger.debug('Dividing tensors')
    ML,s,MR = M.svd('bl|r,l|br,r',chi=self.chi,tolerance=self.settings['tol2'])
    self.psi.setTL(ML.renamed('bl-b'),0,unitary=True)
    self.psi.setTR(MR.renamed('br-b'),1,unitary=True)
    self.psi.setschmidt(s/np.linalg.norm(s),0)
    self.discardright()

  def doubleupdateright(self):
    # TODO better way of handling transfer
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
    self.discardleft()

  def doubleupdate(self, n, direction):
    self.logger.log(15, 'site %s double update (%s)', n, direction)
    if direction == 'r':
      while n+2 > self.N-self.ntransfR:
        self.discardright()
    else:
      while n < self.ntransfL:
        self.discardleft()
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
    if direction == 'r':
      self.discardright()
    else:
      self.discardleft()

  def singleupdateleft(self):
    tR = self.gettransfR(1)
    self.logger.log(10, 'left edge single update')
    Heff = operators.NetworkOperator('(T);O;R;R.t-T.r,O.r-R.c,O.t-T.b;'
      'R.b>r,O.b>b', self.H.getT(0), tR.T)
    M = self.optimize_tensor(Heff, 0, 1)
    ML,s,MR = M.svd('b|r,l|r')
    self.psi.setTL(ML,0,unitary=True)
    schmidt = s/np.linalg.norm(s)
    self.psi.setschmidt(schmidt,0)
    self.discardright()
    #return MR.diag_mult('l', schmidt)
    return MR

  def singleupdateright(self, gauge):
    tL = self.gettransfL(self.N-2)
    self.logger.log(10, 'right edge single update')
    Heff = operators.NetworkOperator('(T);O;L;L.t-T.l,O.l-L.c,O.t-T.b;'
      'L.b>l,O.b>b', self.H.getT(self.N-1), tL.T)
    M = self.optimize_tensor(Heff, self.N-1,1)
    ML,s,MR = M.svd('l|r,l|b')
    self.psi.setTR(MR,self.N-1,unitary=True)
    schmidt = s/np.linalg.norm(s)
    self.psi.setschmidt(schmidt,self.N-2)
    self.discardleft()
    return ML

  def singleupdate(self, n, direction, gauge):
    self.logger.log(12, 'site %s single update (%s)', n, direction)
    right = direction == 'r'
    if right:
      while n+1 > self.N-self.ntransfR:
        self.discardright()
    else:
      while n < self.ntransfL:
        self.discardleft()
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
      self.discardright()
      return MR
    else:
      MR,s,ML = M.svd('r,b|l,r|l',tolerance=self.settings['tol1'])
      self.psi.setTR(MR,n,unitary=True)
      self.psi.setschmidt(s/np.linalg.norm(s),n-1)
      self.discardleft()
      return ML

  def returnvalue(self):
    return list(self.psilist)

  def baseargs(self):
    return self.chis, self.N, self.npsi_tot, self.npsis, self.settings

def orthoBaseSupervisor(chis, N, npsitot, np0, settings, state=None):
  if state is None:
    yield 'init',()
    cidx = 0
    yield 'setchi',(0,chis[0])
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
        yield 'setchi',(cidx,chis[cidx])
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
      yield 'setchi',(cidx,chis[cidx])
  yield 'complete',()

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
  while niter < settings['nsweepadd']:
    for ipsi in range(psi0,npsi):
      E = yield 'runsub',('doublesweep',N),(niter,ipsi,npsi)
      yield 'benchmark2', ()
      yield 'saveschmidt', ()
      if ipsi+1 != npsi:
        yield 'select', (ipsi+1,(-1,2))
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
  
