from . import config
import os,os.path,shutil
import pickle,contextlib,itertools
from copy import copy,deepcopy
from collections.abc import MutableMapping
from abc import ABC,abstractmethod
from .settingsman import SettingsManager

class GeneralManager:
  """Base class for optimization managers"""
  _initializers = ()

  def __init__(self, savefile='auto', file_prefix=None,
      settings_from=[], **kw_args):
    # settings_from is list of (filename, key) pairs for configuration files
    #   * Earlier settings in list are given priority
    #   * Class defaults are loaded automatically with lowest priority
    #   * Settings provided directly as keywords are loaded with highest
    #       priority
    #   * If string is encountered instead of pair will use most recent
    #       filename
    #   * If "default" is encountered instead of filename will look in
    #       file containing defaults
    #   * If only using one config table may provide only that pair
    #       instead of the list
    # In initializing subclasses, define parameters expected for 'registered
    # object' configuration before calling & take actions dependent on settings
    # afterwards
    self.__version = '0.1'
    self.saveprefix = file_prefix
    if savefile == 'auto' and isinstance(file_prefix,str):
      self.filename = (file_prefix+'.p') if savefile=='auto' else savefile
    elif savefile == 'auto' or not savefile:
      self.filename = None
    else:
      self.filename = savefile

    self._configure_settings(settings_from, kw_args)
    self._configure_stage_settings()

    if self.use_shelf:
      assert self.filename
      # Use autosave filename but with extension .db instead of .p
      self.dbpath = self.filename[:-1]+'d'
      self.database = PseudoShelf(self.dbpath)
    else:
      # Otherwise just use a dictionary instead
      self.dbpath = None
      self.database = {}

    self._initfuncs()
    # Logging
    self._initlog()
                        
    self._initializers = ()
    self.supervisors = []
    self.supstatus = []
    self.suplabels = []
    self.savelevel = 2 # Max level (# supervisors) to save at

    self._registry = {}
    self._initializeROs()
    # TODO initialize stage?

  @abstractmethod
  def _initfuncs(self):
    pass

  def _configure_settings(self, settings_from, kw_args, reconfig=False):
    # TODO way of saying "this general setting does not override rules"?
    man_args = [kw_args]
    if bool(settings_from) and isinstance(settings_from[0],str):
      if len(settings_from) != 2:
        raise ValueError('Expected filename-key pair as first element of '
          'settings_from, got string "%s"'%settings_from[0])
      man_args.append(tuple(settings_from))
    else:
      fname = None
      for farg in settings_from:
        if isinstance(farg,str):
          man_args.append((fname,farg))
        else:
          fname,key = farg
          if fname == 'default':
            fname = self._default_file
          man_args.append((fname,key))
    settings_man = SettingsManager(self, *man_args)
    if reconfig:
      raise NotImplementedError() #TODO
    else:
      self.settings_man = settings_man
      self.settings = {}

  @abstractmethod
  def _configure_stage_settings(self):
    """Read off information about stages obtained from settings manager"""
    pass

  def set_setting(self, key, value, override=True, nocreate=False):
    """Specify a setting "key", set to value (for all stages)
    If override, wipes all conditional settings for key
    If nocreate, raises error if key does not exist
    Assumes that setting definitions will be re-processed (through setstage)
      before running"""
    # TODO option for re-processing at call time?
    self.settings_man.set_general(key, value,
      override=override,nocreate=nocreate)

  def set_settings(self, new_settings, **kw_args):
    """As with set_setting but provide key-value pairs as dict"""
    for key,value in new_settings.items():
      self.set_setting(key, value, **kw_args)

  def resetsettings(self,settings_from=[],stage_override=True,**kw_args):
    """Accepts settings arguments as in constructor"""
    # TODO option for merge instead of full override?
    self._configure_settings(settings_from, kw_args)
    self._configure_stage_settings(override=stage_override)

  def set_rule(self, key, value, condition='', priority=True, nocreate=False,
      **stage_args):
    """Specify a value for a conditional setting
    condition should be a string parseable by sympy
    If priority, will be given priority over existing rules; otherwise will
      be given last priority
    May additionally specify 'stage variables' as keywords
    If nocreate, raises error if key does not exist
    Assumes that setting definitions will be re-processed (through setstage)
      before running"""
    self.settings_man.set_conditional(key, condition, value, priority=priority,
      nocreate=nocreate, **stage_args)

  def set_rules(self, new_settings, *args, **kw_args):
    """As with set_rule but provide key-value pairs as dict"""
    for key,value in new_settings.items():
      self.set_rule(key, value, *args, **kw_args)

  #TODO should there be more (or less) going on with these properties?
  @property
  def use_shelf(self):
    return self.settings_man.global_settings['use_shelf']

  @use_shelf.setter
  def use_shelf(self, value):
    if value:
      self.setdbon()
    else:
      self.setdboff()

  @property
  def savesafe(self):
    return self.settings_man.global_settings['savesafe']

  @savesafe.setter
  def savesafe(self, value):
    self.settings_man.global_settings['savesafe'] = value

  @property
  def savelevel(self):
    return self.settings_man.global_settings['savelevel']

  @savelevel.setter
  def savelevel(self, value):
    self.settings_man.global_settings['savelevel'] = value

  def __getstate__(self):
    state = self.__dict__.copy()
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
      self._update_state_to_version(state)
    #self.chirules = {}
    #self._initfuncs()
    #self.supervisors = 'paused'

  @abstractmethod
  def _update_state_to_version(self, state):
    if '_GeneralManager__version' not in state:
      self.__version = '0.0'
      # TODO other updates?
      self.settings_allchi.update(dict(savelevel=state['savelevel'],use_shelf=isinstance(self.database,PseudoShelf)))
    if self.__version == '0.0':
      # Change 'chi' to 'stage'
      self.settings_man = SettingsManager(self,self.settings_allchi)
      del self.settings_allchi
      self.__version = '0.1'

  def setstage(self, *stage_data):
    # Child must update stage variables (e.g. bond dimension)   
    self._update_stage(*stage_data)
    self._evaluate_settings(*stage_data)

  def _evaluate_settings(self, *stage_data):
    # May pass without stage data in which case will use current stage
    if not stage_data:
      stage_data = self._get_stage_data()
    self.settings.clear()
    stage_dict = self.query_stage(*stage_data)
    self.settings.update(self.settings_man.eval_at_stage(stage_dict))
    self.logger.debug('Settings updated to:'+len(self.settings)*'\n\t%s = %s',
      *sum(self.settings.items(),()))

  @abstractmethod
  def query_stage(self, *stage_data):
    pass

  @abstractmethod
  def _update_stage(self, *stage_data):
    """"Set internal fields specifying 'stage'"""
    pass

  @abstractmethod
  def _get_stage_data(self):
    """Retrieve "stage data" to be passed to query_stage & _update_stage"""
    pass

  # "Registered objects" --
  # all management thereof should pass through these functions

  ### General methods 
  def fetchRO(self, key):
    if key not in self._activeRO:
      if key in self._inactiveRO:
        assert key in self._registry
        raise KeyError(f'Registered object requested {key} inactive')
      else:
        assert key not in self._registry
        raise KeyError(f'Registered object requested {key} not existent')
    return self._registry[key]

  def _produceRO(self, key, **kw_args):
    assert key in self._inactiveRO 
    assert key not in self._activeRO
    self._computeRO(key, **kw_args)
    self._activeRO.add(key)
    self._inactiveRO.remove(key)

  def refreshRO(self, key):
    assert key in self._activeRO
    self._computeRO(key)

  def _discardRO(self, key):
    self._registry[key].discard()
    self._activeRO.remove(key)
    self._inactiveRO.add(key)

  def _updateROstate(self, statespec, **kw_args):
    self._validateROstate(statespec)
    # TODO pretty-print (or at least flatten) statespec
    self.logger.debug('Updating registry to state %s',statespec)
    newactive = self.ROstatelist(statespec)
    # Add new *in order*
    for key in newactive:
      if key in self._inactiveRO:
        self._produceRO(key, **kw_args)
    self._ROstatespec = statespec
    # Discard extranneous
    for key in self._activeRO - set(newactive):
      self._discardRO(key)

  def clearall(self):
    # TODO use database.clear + context manager?
    self._updateROstate(self._nullROstate)
    self.database.clear()
    assert not self._activeRO
    assert self._inactiveRO == set(self._registry)

  def expandROs(self, extension, statespec):
    """Expand registered objects with dictionary of new *inactive* objects
    & new specifier
    Assume internal changes required to expand permissible registry have already
    been performed"""
    assert isinstance(extension,dict) and not (set(extension) & set(self._registry))
    assert set(self.ROstatelist(statespec)) == self._activeRO
    self._registry.update(extension)
    self._inactiveRO.update(extension)
    self._ROstatespec = statespec

  ### Specific methods
  def _computeRO(self, key):
    pass

  def _initializeROs(self, statespec=None):
    self._ROstatespec = self._nullROstate
    self._inactiveRO = set(self._registry)
    self._activeRO = set()
    if statespec is not None:
      self._updateROstate(statespec)

  @abstractmethod
  def _validateROstate(self, spec):
    pass

  def _validateRO(self, key):
    pass

  def _ROcheckdbentries(self, key):
    pass

  @abstractmethod
  def ROstatelist(self, statespec):
    pass

  def setdbon(self, dbpath=None):
    if not dbpath:
      dbpath = self.filename[:-1]+'d'
    if not self.dbpath:
      self.logger.warning('Transferring tensors to shelf...')
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
        self.dbpath,dbp0 = dbpath,self.dbpath
        try:
          self.database.copyto(dbpath)
        except FileNotFoundError:
          self.logger.warning('Shelf directory not found, repopulating %d left '
            'and %d right transfer vectors', *self._ROstatespec)
          self.regenerate()
        except FileExistsError:
          assert not os.path.isdir(dbp0)
          self.logger.warning('Assuming database has already been moved to %s',
            dbpath)
          # TODO it's natural to move paths, should have better checks
          self.database.path = os.path.abspath(dbpath)
      else:
        self.database = PseudoShelf(dbpath)
        try:
          self.database.update(self.dbpath)
        except FileNotFoundError:
          self.logger.warning('Shelf directory not found, repopulating')
          self.regenerate()
      # Move database
      # TODO should this save command be here?
      self.save()
    else:
      self.logger.debug('Shelf already in place')
      self.database.path = dbpath
    try:
      self.checkdatabase()
    except AssertionError:
      self.logger.exception('Error checking database; recomputing transfers')
      self.regenerate()
    self.settings_man.global_settings['use_shelf'] = True

  def regenerate(self):
    statespec = self._ROstatespec
    self.clearall()
    self._initializeROs(statespec)

  def checkdatabase(self):
    self.logger.debug('Checking database and restoring entries as necessary')
    if self.dbpath:
      if not isinstance(self.database,PseudoShelf):
        self.database = PseudoShelf(self.dbpath)
      elif not os.path.isdir(self.dbpath):
        self.logger.warning('Directory %s not found; creating',self.dbpath)
        os.mkdir(self.dbpath)
      dbkeys = self.database.checkentries()
    else:
      dbkeys = set(self.database)
    # Check converted: TODO depricate
    for key in self._registry:
      assert self._validateRO(key)
    activedbkeys = set()
    for key in self.ROstatelist(self._ROstatespec):
      activedbkeys.update(self._ROcheckdbentries(key))
    for k in dbkeys - activedbkeys:
      self.logger.info('Removing unidentified key %s',k)
      del self.database[k]
    assert len(self.database) == len(self._activeRO)
    # TODO change for case RO-database is not one-to-one

  def setdboff(self, remove=True, restore=True):
    if self.dbpath:
      self.logger.warning('Converting shelf into local memory...')
      database = {}
      for k in self.database:
        database[k] = self.database[k]
      if remove:
        self.logger.warning('Deleting shelf database directory')
        shutil.rmtree(self.dbpath)
      self.database = database
      self.dbpath = None
      # Confirm that correct entries are present 
      activedbkeys = set()
      for key in self.ROstatelist(self._ROstatespec):
        activedbkeys.update(self._ROcheckdbentries(key))
      for k in set(self.database) - activedbkeys:
        self.logger.info('Removing unidentified key %s',k)
        del self.database[k]
      self.save()
    else:
      self.logger.debug('Transfer vectors already in local memory')
    self.settings_man.global_settings['use_shelf'] = False

  @abstractmethod
  def query_system(self):
    return {}

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

  def save(self):
    if not self.filename:
      return
    if self.savesafe:
      oldfile = self.filename+'.old'
      if os.path.isfile(oldfile):
        os.rename(oldfile,oldfile+'.old')
      if os.path.isfile(self.filename):
        os.rename(self.filename,oldfile)
    safedump(self, self.filename)
    if self.savesafe and os.path.isfile(self.filename+'.old.old'):
      os.remove(self.filename+'.old.old')

  def savecheckpoint(self, level):
    if config.haltsig:
      self.logger.log(40,'Saving and quitting')
      self.save() 
      return True
    if level <= self.savelevel:
      self.logger.log(15,'Saving at checkpoint')
      self.save()
    return False

  @abstractmethod
  def returnvalue(self):
    """Define return value for completed run()"""
    pass

  @abstractmethod
  def baseargs(self):
    """Define arguments for base supervisor"""
    pass

  def resume_message(self):
    """Announce status on resumption"""
    pass

  def is_initializer(self, cmd, statcmp, *args):
    return cmd in self._initializers

  def _do_execution(self, send_value, rstrict=None, statcmp=None):
    """Obtain & execute commands from "supervisor" subroutines"""
    level = len(self.supervisors)
    label = self.suplabels[level-1]
    cmd,args,*stat_arg = self.supervisors[-1].send(send_value)
    self.logger.log(8, 'Received %s command from supervisor %s',
      cmd, self.suplabels[-1])
    sendstate = None
    if stat_arg:
      status, = stat_arg
      if rstrict is not None:
        # Perform checks, do not save
        # TODO was there a reason for not doing separate level=statcmp check?
        if level == len(statcmp):
          if status != statcmp[-1]:
            if rstrict < 3:
              self.logger.log(25, 'Supervisor %s status %s superceded as %s '
                'in %s command', label, statcmp[-1], status, cmd)
              self.supstatus[-1] = status
            else:
              raise ValueError('Supervisor %s status %s encountered in %s '
                'command, expected %s'%(label, status, cmd, statcmp[-1]))
          else:
            self.logger.debug('Resuming %s command with status %s',
              cmd, status)
            self.supstatus[-1] = status
        elif cmd != 'runsub':
          if not rstrict:
            self.logger.warning('Encountered save status in %s command of '
              '%s supervisor while resuming', cmd, label)
          else:
            raise ValueError('Encountered save status in %s command of '
              '%s supervisor while resuming'%(cmd, label))
        elif args[0] != self.suplabels[level]: #level<len(statcmp) and
          if not rstrict:
            self.logger.error('Expected supervisor "%s" at level %d, '
              'got "%s"', self.suplabels[level],level,args[0])
          else:
            raise ValueError('Expected supervisor "%s" at level %d, '
              'got "%s"'%(self.suplabels[level],level,label))
        else:
          if status != statcmp[level-1]:
            if rstrict < 3:
              self.logger.log(25, 'Supervisor %s status %s superceded as %s',
                label, statcmp[level-1], status, cmd)
              sendstate = statcmp[level]
            else:
              raise ValueError('Supervisor %s status %s encountered, '
                'expected %s'%(label, status, statcmp[-1]))
          self.supstatus[-1] = status
          sendstate = statcmp[level]
      else:
        # Candidate save checkpoint--includes supervisor status
        self.logger.log(8, 'Setting status to %s', status)
        self.supstatus[-1] = status
        if self.filename:
          savequit = self.savecheckpoint(level)
          if savequit:
            import sys
            sys.exit(config.SIGSAVE)
    # Options for cmd
    if cmd == 'complete':
      # Subroutine completed execution
      if rstrict:
        raise ValueError('Supervisor %s completed before full resumption' % \
          label)
      self.supervisors.pop()
      self.supstatus.pop()
      label = self.suplabels.pop()
      self.logger.log(15, 'Supervisor %s complete',label)
      if args:
        # Pass arguments up the chain
        return args
    elif cmd == 'runsub':
      # New sub-supervisor 
      label,*supargs = args
      subsup = self.supfunctions[label](*supargs, self.settings,
        state=sendstate)
      self.supervisors.append(subsup)
      self.supstatus.append(None)
      # TODO was there a reason for using rstrict checks?
      if level == len(self.suplabels):
        self.suplabels.append(label)
        self.logger.log(10, 'Starting level-%s supervisor %s',level+1,label)
      else:
        self.suplabels[level] = label
        self.logger.log(10, 'Resuming level-%s supervisor %s',level+1,label)
    elif cmd == 'announce':
      # Announcement
      # TODO logging levels?
      config.streamlog.log(30, args)
    else:
      # Algorithm-specific options
      if rstrict is not None and not self.is_initializer(cmd,statcmp,*args) \
          and not (level == len(statcmp) and stat_arg == [statcmp[-1]]):
        #print(cmd,statcmp,args,level,stat_arg,level==len(statcmp),stat_arg==[statcmp[-1]])
        if rstrict < 2:
          self.logger.info('Non-initializer command "%s" not expected before '
            'next sub-supervisor',cmd)
        else:
          raise ValueError('Non-initializer command "%s" not expected before '
            'next sub-supervisor'%cmd)
      return self.supcommands[cmd](*args)

  def run(self, resume_strictness=2):
    """Run manager
    resume_strictness specifies level of strictness used to ensure that a
      manager being resumed will return to the correct savepoint
      1: require that the same sequence of supervisors is called
      2: also check other commands with is_initializer
      3: also ensure correct "statuses" are encountered"""
    if self.supstatus == 'complete':
      print('Optimization already completed')
      return self.returnvalue()
    sval = None
    if not self.supervisors:
      # Initialize base
      self.logger.log(10, 'Initializing base supervisor')
      self.supervisors.append(self.supfunctions['base'](*self.baseargs()))
      self.suplabels.append('base')
      self.supstatus.append(None)
    elif self.supervisors == 'paused':
      self.logger.log(10, 'Resuming base supervisor')
      self.logger.log(5, 'Using state %s', self.supstatus[0])
      self.resume_message()
      self._evaluate_settings()
      # Initialize saved supervisors
      sup = self.supfunctions['base'](*self.baseargs(), state=self.supstatus[0])
      self.supervisors = [sup]
      level = 0
      old_status = list(self.supstatus)
      levold = len(old_status)
      self.supstatus = [None]
      while len(self.supervisors) <= levold and (len(self.supervisors)<levold \
          or (self.supstatus[-1] is None and old_status[-1] is not None)):
        sval = self._do_execution(sval, rstrict=resume_strictness,
          statcmp=old_status)
    while len(self.supervisors):
      # Next step from current subroutine
      sval = self._do_execution(sval)
    self.supstatus = 'complete'
    self.clearall()
    self.save()
    self.savestep() # TODO redundant to use both?
    return self.returnvalue()


class BondOptimizer(GeneralManager):
  """Optimization manager with bond-dimension dependence"""

  def _configure_stage_settings(self, override=False):
    if not override and hasattr(self, 'chis') and self.chis:
      return
    elif 'chis' in self.settings_man.global_settings:
      self.chis = list(self.settings_man.global_settings['chis'])
    elif 'chi' in self.settings_man.global_settings:
      self.chis = [self.settings_man.global_settings['chi']]

  def _update_stage(self, chi):
    self.chiindex = self.chis.index(chi)
    self.chi = chi
    config.streamlog.log(30, 'chi = % 3d (#%s)', chi, self.chiindex)

  def resetchis(self, chis, use_as=None):
    """Reset list of bond dimensions; purge bond-dimension-dependent rules
    and pick chiindex"""
    # TODO generalize, rectify with managed settings
    self.chis = chis
    inc = False
    if self.supstatus == 'complete':
      if self.chi >= chis[-1]:
        # All completed
        return
      else:
        self.supstatus = [None]
        self.suplabels = ['base']
        # TODO replace with option to call "next stage"?
        inc = True
    if self.chi not in chis:
      if inc:
        use_as = 'higher'
      if not use_as:
        raise KeyError('Current bond dimension not in new list')
      nbelow = sum(c < self.chi for c in chis)
      if use_as == 'lower':
        assert nbelow > 0
        chiindex = nbelow-1
      else:
        assert use_as == 'higher' and nbelow < len(chis)
        chiindex = nbelow
      self.setstage(chis[chiindex])
      #self.chi = chis[self.chiindex]
    else:
      self.chiindex = chis.index(self.chi)
      if inc:
        self.chiindex += 1
        self.chi = chis[self.chiindex]
    self.chirules = {}
    # Restart base supervisor
    if len(self.supstatus) and self.supstatus[0] is not None:
      self.supstatus[0] = (self.chi,)+self.supstatus[0][1:]
    if self.supervisors != 'paused' and len(self.supervisors):
      self.supervisors[0] = self.supfunctions['base'](self.chis,self.N,self.settings,self.supstatus[0])
    # TODO feels redundant
    self.set_setting('chis',chis)

  def settingbychi(self, chi, **kw_args):
    """Add rule for bond dimension"""
    if isinstance(chi, int):
      self.logger.log(13, 'Adding rules for bond dimension %d',chi)
      self.set_rules(kw_args, chi=chi)
    else:
      for c in chi:
        self.settingbychi(c, **kw_args)
  
  def settingbyindex(self, idx, **kw_args):
    """Add rule for bond dimension by index within sequence"""
    if isinstance(idx, int):
      self.logger.log(13, 'Adding rules for bond dimension with index %d',idx)
      if idx < 0:
        sarg = dict(back_index=idx)
      else:
        sarg = dict(index=idx)
      self.set_rules(kw_args, **sarg)
    else:
      for i in idx:
        self.settingbyindex(i, **kw_args)

  def query_stage(self, chi):
    idx = self.chis.index(chi)
    bidx = idx - len(self.chis)
    return dict(chi=chi, index=idx, back_index=bidx)

  def _get_stage_data(self):
    return (self.chi,)

  def resume_message(self):
    config.streamlog.log(30, 'Resuming: chi = % 3d (#%s)', self.chi, self.chiindex)
    

class PseudoShelf(MutableMapping):
  # Dictionary-like class representing storage in a directory (rather than a
  # true database file)
  def __init__(self, dirname):
    dirname = os.path.abspath(dirname)
    if not os.path.isdir(dirname):
      os.mkdir(dirname)
    self.path = dirname
    self._backup_dict = {}

  def __setstate__(self, state):
    # TODO include versions?
    if '_backup_dict' not in state:
      state['_backup_dict'] = {}
    self.__dict__.update(state)

  def fname(self, key):
    return os.path.join(self.path,key+'.p')

  def __getitem__(self, key):
    if key in self._backup_dict:
      return self._backup_dict[key]
    if not os.path.isfile(self.fname(key)):
      raise KeyError(key)
    try:
      return pickle.load(open(self.fname(key),'rb'))
    except (EOFError,IOError,OSError):
      config.logger.exception('Unable to access PseudoShelf file %s',self.fname(key))
      import time
      tic = time.time()
      for ntry in range(config.file_timeout):
        toc = time.time()
        try:
          return pickle.load(open(self.fname(key),'rb'))
        except (EOFError,IOError,OSError):
          config.logger.warning('Retry #%d failed',ntry)
        if toc-tic > config.file_timeout:
          break
      raise

  def __setitem__(self, key, value):
    if key in self._backup_dict:
      del self._backup_dict[key]
    try:
      pickle.dump(value, open(self.fname(key),'wb'))
    except:
      config.logger.exception('Unable to write to %s; using backup dictionary')
      self._backup_dict[key] = value

  def __delitem__(self, key):
    if key in self._backup_dict:
      del self._backup_dict[key]
    else:
      try:
        os.remove(self.fname(key))
      except FileNotFoundError:
        config.logger.error('Attempted to delete missing database item')

  def __iter__(self):
    dirgen = (f[:-2] for f in os.listdir(self.path) if f[-2:] == '.p')
    if self._backup_dict:
      return itertools.chain(self._backup_dict.keys(),dirgen)
    else:
      return dirgen
  
  def __len__(self):
    return len(os.listdir(self.path)) + len(self._backup_dict)

  def __contains__(self, key):
    return os.path.isfile(self.fname(key)) or key in self._backup_dict

  def clear(self):
    for f in os.listdir(self.path):
      os.remove(os.path.join(self.path,f))
    self._backup_dict.clear()

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
      except:
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
      
  
valid_paths=set()
# TODO check possibly redundant measures
def safedump(data, fname, newsave=False):
  savedir = os.path.abspath(fname)
  ftmp = os.path.join(os.path.dirname(savedir),'~'+os.path.basename(fname))
  if config.synclockfile is not None:
    import fcntl
    fd = open(config.synclockfile,'r')
    fcntl.lockf(fd,fcntl.LOCK_SH)
  else:
    fd = None
  try:
    try:
      pickle.dump(data, open(ftmp,'wb'))
    except MemoryError:
      config.logger.exception('Encountered MemoryError while saving (consider adjusting)')
      import gc
      gc.collect()
      pickle.dump(data, open(ftmp,'wb'))
  except Exception:
    if config.backupdir is None:
      config.logger.exception('File write failed; please correct')
    elif not newsave and savedir not in valid_paths:
      # Just break
      raise
    else:
      altname = os.path.join(config.backupdir,os.path.basename(fname))
      alttmp = os.path.join(config.backupdir,'~'+os.path.basename(fname))
      config.logger.exception('File write failed; writing to alternate location %s',altname)
      pickle.dump(data, open(alttmp,'wb'))
      os.rename(alttmp,altname)
  else:
    # Successful save
    os.rename(ftmp,fname)
    valid_paths.add(savedir)
  if fd is not None:
    import fcntl
    fcntl.lockf(fd,fcntl.LOCK_UN)
    fd.close()
