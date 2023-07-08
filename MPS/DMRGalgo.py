from .mpsbasic import *
from .. import config
import os,os.path,shutil
import pickle,shelve,contextlib
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
    self.__version = '0.1'
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
    self.transfLs = []
    self.transfRs = []
    self.ntransfL = 0
    self.ntransfR = 0
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
                        'doubleupdateright':self.doubleupdateright}
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
    # TODO recovery options if directory is expected but does not exist
    self.chirules = {}
    self._initfuncs()
    self.supervisors = 'paused'

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
      self.supstatus[0] = (self.chi,self.supstatus[0][1])
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
      self.settingbychi(self.chis[idx])
    else:
      for i in idx:
        self.settingbychi(self.chis[i], **kw_args)

  def useSupervisor(self,name,func):
    """Use function passed instead of standard"""
    self.supfunctions[name] = func

  def getE(self):
    if self.ntransfR == 0:
      assert not self.E
      self.E = np.real(self.H.expv(self.psi))
      self.logger.log(25,'Initial energy %s',self.E)
      return self.E
    tR0 = self.gettransfR(self.N-self.ntransfR)
    transf = tR0.moveby(self.N-self.ntransfR-1,unstrict=True)
    self.E0,self.E = self.E,np.real(transf.left(terminal=True))
    self.logger.log(20,'Energy calculated as %s',self.E) 
    if self.E0 is not None and self.settings['eigtol_rel']:
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
    # TODO add_callable method
    if 'process_psi0' in self.callables:
      self.psi = self.callables[process_psi0](self.psi)
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
    self.database.clear()
    self.transfRs = self.H.right_transfer(self.psi, n0, collect=True, manager=self)
    self.transfLs = []
    self.ntransfR = len(self.transfRs)
    self.ntransfL = 0

  @contextlib.contextmanager
  def shelfcontext(self):
    raise NotImplementedError()
    # Open shelf context manager if using, otherwise imitate nullcontext
    if self.dbfile:
      try:
        if not self.shelfopen:
          self.database = shelve.open(self.dbfile)
        self.shelfopen += 1
        yield self.database
      finally:
        self.shelfopen -= 1
        assert self.shelfopen >= 0
        if not self.shelfopen:
          # No hanging dependencies; close
          self.database.close()
          self.database = None
    else:
      try:
        yield self.database
      finally:
        pass

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
          self.logger.warn('Shelf directory not found, repopulating %d left and '
            '%d right transfer vectors', self.ntransfL,self.ntransfR)
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
      if self.ntransfR:
        self.transfRs = self.H.right_transfer(self.psi,self.N-self.ntransfR,collect=True,manager=self)
      else:
        self.transfRs = []
      if self.ntransfL:
        self.transfLs = self.H.left_transfer(self.psi,self.ntransfL-1,collect=True,manager=self)
      else:
        self.transfLs = []

  def regenerate(self):
    if self.ntransfR:
      # Start with boundary 
      Ar = self.psi.getTc(self.N-1)
      TV = self.H.getT(self.N-1).contract(Ar,'t-b;l>c,b>b;l>t')
      TV = TV.contract(Ar,'b-b;~;l>b*')
      self.transfRs[-1].T = TV
      for nr in range(self.ntransfR-2,-1,-1):
        self.transfRs[nr].compute(self.transfRs[nr+1].T)
    if self.ntransfL:
      Al = self.psi.getTc(0)
      TV = self.H.getT(0).contract(Al, 't-b;r>c,b>b;r>t')
      TV = TV.contract(Al, 'b-b;~;r>b*')
      self.transfLs[0].T = TV
      for nl in range(1,self.ntransfL):
        self.transfLs[nl].compute(self.transfLs[nl-1].T)

  def checkdatabase(self):
    self.logger.debug('Checking database and restoring entries as necessary')
    if self.dbpath:
      if not self.database:
        self.database = PseudoShelf(self.dbpath)
      keys = self.database.checkentries()
    else:
      keys = set(self.database)
    # Check converted: TODO depricate
    sitecorrect = True
    for n,tL in enumerate(self.transfLs):
      if not isinstance(tL,LeftTransferManaged):
        self.logger.error('Left transfer at %d unconverted',tL.site)
        self.transfLs[n] = LeftTransferManaged(self.psi,tL.site,'l',
          self.H, manager=self)
        self.transfLs[n].T = tL.T
      self.logger.log(5,'Left transfer at %d vs %d',tL.site,n)
      sitecorrect = sitecorrect and (tL.site == n)
      assert sitecorrect
    for n,tR in enumerate(self.transfRs):
      if not isinstance(tR,RightTransferManaged):
        self.logger.error('Right transfer at %d unconverted',tR.site)
        self.transfRs[n] = RightTransferManaged(self.psi,tR.site,'r',
          self.H, manager=self)
        self.transfRs[n].T = tR.T
      self.logger.log(5,'Right transfer at %d vs %d',tR.site,self.N-self.ntransfR+n)
      sitecorrect = sitecorrect and (tR.site == self.N-self.ntransfR+n)
      assert sitecorrect
    assert sitecorrect
    if self.ntransfR:
      if self.transfRs[-1].id.hex not in keys:
        # TODO ideally can just call compute
        self.logger.warn('Right boundary transfer vector missing; restoring')
        Ar = self.psi.getTc(self.N-1)
        TV = self.H.getT(self.N-1).contract(Ar,'t-b;l>c,b>b;l>t')
        TV = TV.contract(Ar,'b-b;~;l>b*')
        self.transfRs[-1].T = TV
      for n in range(1,self.ntransfR):
        if self.transfRs[-n-1].id.hex not in keys:
          self.logger.warn('Right transfer vector at %s missing; restoring',
            self.N-n-1)
          self.transfRs[-n-1].compute(self.transfRs[-n].T)
      assert len(self.transfRs) == self.ntransfR
    if self.ntransfL:
      if self.transfLs[0].id.hex not in keys:
        self.logger.warn('Left boundary transfer vector missing; restoring')
        Al = self.psi.getTc(0)
        TV = self.H.getT(0).contract(Al, 't-b;r>c,b>b;r>t')
        TV = TV.contract(Al, 'b-b;~;r>b*')
        self.transfLs[0].T = TV
      for n in range(1,self.ntransfL):
        if self.transfLs[n].id.hex not in keys:
          self.logger.warn('Left transfer vector at %s missing; restoring',n)
          self.transfLs[n].compute(self.transfLs[n-1].T)
      assert len(self.transfLs) == self.ntransfL
    valid_keys = {t.id.hex for t in self.transfLs+self.transfRs}
    for k in keys - valid_keys:
      self.logger.info('Removing unidentified key %s',k)
      del self.database[k]
    self.logger.log(8,'%d valid keys (%d), %d left transfers (%d), %d right transfers (%d)',len(valid_keys),len(self.database),self.ntransfL,len(self.transfLs),self.ntransfR,len(self.transfRs))
    assert len(self.database) == self.ntransfL + self.ntransfR

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
          

  def gettransfR(self, n, pop=False):
    # TODO combine options with proxy context manager?
    self.logger.debug('Fetching right transfer %d',n)
    if self.ntransfR:
      self.logger.log(5,'(presently %d, or minimum %d)',
        self.ntransfR,self.transfRs[0].site)
    else:
      self.logger.log(5,'(presently none)')
    if n == self.N-self.ntransfR-1:
      # Compute transfer
      if self.ntransfR == 0:
        tR = self.H.getboundarytransfright(self.psi, manager=self)
      else:
        tR = self.transfRs[0].left()
      if pop:
        tR.discard = True
      else:
        self.transfRs.insert(0, tR)
        self.ntransfR += 1
      return tR
    else:
      assert n > self.N-self.ntransfR-1
      tR = self.transfRs[n-(self.N-self.ntransfR)]
      if pop:
        while n > self.N-self.ntransfR:
          self.logger.info('Discarding extranneous right transfer at %d')
          tR0 = self.transfRs.pop(0)
          tR0.discard = True
          self.ntransfR -= 1
        assert n == self.N-self.ntransfR
        tR.discard = True
        self.transfRs.pop(0)
        self.ntransfR -= 1
      return tR

  def gettransfL(self, n, pop=False):
    assert not pop # TODO depricate
    self.logger.debug('Fetching left transfer %d',n)
    if self.ntransfL:
      self.logger.log(5,'(presently %d, or maximum %d)',
        self.ntransfL,self.transfLs[-1].site)
    else:
      self.logger.log(5,'(presently none)')
    if n == self.ntransfL:
      # Compute transfer
      if self.ntransfL == 0:
        tL = self.H.getboundarytransfleft(self.psi, manager=self)
      else:
        tL = self.transfLs[-1].right()
      if pop:
        tL.discard = True
      else:
        self.transfLs.append(tL)
        self.ntransfL += 1
      return tL
    else:
      #assert n < self.ntransfL
      while n >= self.ntransfL:
        self.logger.info('Restoring excluded left transfer %d',self.ntransfL)
        if not self.ntransfL:
          tL = self.H.getboundarytransfleft(self.psi, manager=self)
        else:
          tL = self.transfLs[-1].right()
        self.transfLs.append(tL)
        self.ntransfL += 1
      tL = self.transfLs[n]
      if pop:
        assert n == self.ntransfL-1
        tL.discard = True
        self.transfLs.pop()
        self.ntransfL -= 1
      return tL

  def discardleft(self):
    # Discard (rightmost) left transfer matrix
    self.logger.info('Discarding left transfer matrix (%d)',self.ntransfL-1)
    tL = self.transfLs.pop()
    tL.discard()
    self.ntransfL -= 1

  def discardright(self):
    # Discard (leftmost) right transfer matrix
    self.logger.info('Discarding right transfer matrix (%d)',self.N-self.ntransfR)
    tR = self.transfRs.pop(0)
    tR.discard()
    self.ntransfR -= 1

  def restorecanonical(self, almost=False):
    self.logger.log(12,'Restoring to canonical form')
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
    # TODO better way of handling transfer
    tR = self.gettransfR(3).left()
    #tR.discard = True
    self.logger.log(12, 'left edge double update')
    self.H.DMRG_opt_double_left(self.psi, self.chi, tR, self.settings['tol2'])
    tR.discard()
    #self.gettransfL(0)

  def doubleupdateright(self):
    # TODO better way of handling transfer
    tL = self.gettransfL(self.N-4).right()
    #tL.discard = True
    self.logger.log(12, 'right edge double update')
    self.H.DMRG_opt_double_right(self.psi, self.chi, tL, self.settings['tol2'])
    tL.discard()
    #self.gettransfR(self.N-1)

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
    self.H.DMRG_opt_double(self.psi, n, self.chi,
        tL,tR,None,direction=='r', self.settings['tol2'], self.eigtol)
    if direction == 'r':
      self.discardright()
    else:
      self.discardleft()

  def singleupdateleft(self):
    tR = self.gettransfR(2).left()
    #tR.discard = True
    self.logger.log(10, 'left edge single update')
    mat = self.H.DMRG_opt_single_left(self.psi, tR, self.settings['tol1'])
    tR.discard()
    return mat.diag_mult('l', self.psi.getschmidt(0))

  def singleupdateright(self, gauge):
    tL = self.gettransfL(self.N-3).right()
    #tL.discard = True
    self.logger.log(10, 'right edge single update')
    mat = self.H.DMRG_opt_single_right(self.psi, tL, self.settings['tol2'])
    tL.discard()
    #self.transfRs = [self.H.getboundarytransfright(self.psi)]
    return mat

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
    gL,gR = (gauge, None) if right else (None, gauge)
    mat = self.H.DMRG_opt_single(self.psi, n, tL, tR,
        right, gL, gR, self.settings['tol1'], self.eigtol)
    if right:
      mat = mat.diag_mult('l',self.psi.getschmidt(n))
    if right:
      self.discardright()
    else:
      self.discardleft()
    #  self.transfRs.pop(0)
    #  self.transfLs.append(self.transfLs[-1].right())
    #else:
    #  self.transfLs.pop()
    #  self.transfRs.insert(0, self.transfRs[0].left())
    return mat

  def gauge_at(self, site, gauge, direction='r'):
    self.logger.log(10, 'gauging tensor at site %s', site)
    if direction == 'r':
      self.psi.rgauge(gauge,site)
    else:
      self.psi.lgauge(gauge,site)

  def run(self):
    if self.supstatus == 'complete':
      print('Optimization already completed')
      return self.psi,self.E
    if not self.supervisors:
      # Initialize base
      self.logger.log(10, 'Initializing base supervisor')
      self.supervisors.append(self.supfunctions['base'](self.chis, self.N, self.settings))
      self.suplabels.append('base')
      self.supstatus.append(None)
    elif self.supervisors == 'paused':
      self.logger.log(10, 'Resuming base supervisor')
      self.logger.log(5, 'Using state %s', self.supstatus[0])
      config.streamlog.log(30, 'Resuming: chi = % 3d (#%s)', self.chi, self.chiindex)
      # Initialize saved supervisors
      sup = self.supfunctions['base'](self.chis, self.N, self.settings, state=self.supstatus[0])
      self.supervisors = [sup]
      level = 0
      while len(self.supervisors) < len(self.supstatus):
        cmd,(label,*args),status = next(sup)
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
      else:
        # Algorithm-specific options
        sendval = self.supcommands[cmd](*args)
    self.supstatus = 'complete'
    self.save()
    return self.psi, self.E

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

  def __contains__(self):
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