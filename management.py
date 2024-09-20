from . import config
import os,os.path,shutil
import pickle,contextlib,itertools
from copy import copy,deepcopy
from collections.abc import MutableMapping
from abc import ABC,abstractmethod

class GeneralManager:
  """Base class for optimization managers"""
  def __init__(self, savefile='auto', file_prefix=None, use_shelf=False):
    self.__version = '0.0'
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

    self._registry = {}
    self._initializeROs()

  @abstractmethod
  def _initfuncs(self):
    pass

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
      self._update_state_to_version(state)
    self.chirules = {}
    self._initfuncs()
    self.supervisors = 'paused'

  @abstractmethod
  def _update_state_to_version(self):
    pass

  # "Registered objects" --
  # all management should pass through these functions

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

  # TODO "stage" variable that is not specifically bond dimension? 
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

  @abstractmethod
  def returnvalue(self):
    """Define return value for completed run()"""
    pass

  @abstractmethod
  def baseargs(self):
    """Define arguments for base supervisor"""
    pass

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
    try:
      os.remove(self.fname(key))
    except FileNotFoundError:
      config.logger.error('Attempted to delete missing database item')

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

