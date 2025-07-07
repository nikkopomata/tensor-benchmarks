import tomlkit
import sympy
from sympy.parsing import sympy_parser
import os.path
from .config import settings_log as logger

# Defaults for "global settings" (having to do with saving, stages, etc.)
global_setting_defaults = dict(use_shelf=False,savesafe=False,savelevel=2)
val_transformations = sympy_parser.standard_transformations \
  + (sympy_parser.convert_xor,sympy_parser.implicit_multiplication,
      sympy_parser.factorial_notation)
cond_transformations = val_transformations + (sympy_parser.convert_equals_signs,)

class SettingsManager:
  """Class to handle specifications of settings (including from config files
  and runtime specifications)"""
  def __init__(self, owner, *args, use_defaults=True):
    # Following class of calling manager, arguments are either
    # (filename, key) pairs or dictionaries of settings values
    # Earlier arguments have higher precedence
    # Load configuration info
    self.__version = '0.0'
    fconf = os.path.join(os.path.dirname(__file__),owner._default_file)
    configs = ConfigLoader(fconf,owner.__class__.__name__)
    self.owner = owner
    
    stagevar = list(configs.stage_variables)
    self._var_stage = dict(zip(stagevar, sympy.symbols(stagevar)))
    sysvar = list(configs.system_variables)
    self._var_sys = dict(zip(sysvar, sympy.symbols(sysvar)))
    self.stringopts = configs.stringopts
    fallbacks = {}
    rules = {}
    self.global_settings = {}

    for arg in args:
      if isinstance(arg,dict):
        # TODO allow for pre-initialization runtime definition of rules?
        for key in arg:
          if key in configs.global_keys:
            if key not in self.global_settings:
              logger.debug('Found global setting for %s in dict',key)
              self.global_settings[key] = arg[key]
          elif key not in fallbacks:
            logger.debug('Found fallback setting for %s in dict',key)
            fallbacks[key] = arg[key]
      else:
        loader = SettingsLoader(*arg, configs)
        # Incorporate rules first to maintain precedence of fallbacks
        for key in loader.rules:
          if key not in fallbacks:
            logger.debug('Adding new rule for %s',key)
            if key not in rules:
              rules[key] = []
            rules[key].extend(loader.rules[key])
          else:
            logger.debug('Rule for %s omitted (fallback found)')
        logger.debug('Found fallback settings for %s',
          set(loader.fallbacks)-set(fallbacks))
        fallbacks = loader.fallbacks | fallbacks
        self.global_settings = loader.global_settings | self.global_settings
    if use_defaults:
      loader = SettingsLoader(configs.filename,configs.clsname,configs,True)
      for key in loader.rules:
        if key not in fallbacks:
          if key not in rules:
            rules[key] = []
          rules[key].extend(loader.rules[key])
          logger.debug('Found default rule for %s',key)
          if key not in rules:
            rules[key] = []
        else:
          logger.debug('Default rule for %s omitted (fallback found)')
      logger.debug('Found default fallback settings for %s',
        set(loader.fallbacks)-set(fallbacks))
      fallbacks = loader.fallbacks | fallbacks
    # Now have all (initial) settings names
    setvar = list(set(fallbacks) | set(rules))
    self._var_setting = dict(zip(setvar, sympy.symbols(setvar))) 
    # Finally sympify expressions
    self.general_settings = {}
    for key,value in fallbacks.items():
      logger.log(15,'Setting fallback %s to %s',key,value)
      self.general_settings[key] = self.parse_value(key,value)
    for key,rulelist in rules.items():
      logger.log(15,'Setting rules for %s',key)
      for i,(conds,val) in enumerate(rulelist):
        logger.debug('Adding condition %s => %s',conds,val)
        rulelist[i] = (self.parse_condition(conds),
          self.parse_value(key,val))
    self.conditional_settings = rules
    # Expand global settings in case any are missing
    global_missing = configs.global_keys - set(self.global_settings)
    self.global_settings.update({key:None for key in global_missing})

  @property
  def prior_vars(self):
    return self._var_stage | self._var_sys

  @property
  def all_vars(self):
    return self.prior_vars | self._var_setting

  def parse_condition(self, conditions):
    parsed = []
    for cond in conditions:
      if isinstance(cond, str):
        # Preprocess to not have to worry about = or ==
        parsed.append(sympy_parser.parse_expr(cond.replace('==','='),
          self.prior_vars,transformations=cond_transformations))
      else:
        key,value = cond
        parsed.append(sympy.Eq(self._var_stage[key], value))
    return sympy.And(*parsed)

  def parse_value(self, key, value):
    if isinstance(value,str) and key not in self.stringopts:
      if value == 'none':
        return None
      else:
        return sympy_parser.parse_expr(value, self.all_vars,
          transformations=val_transformations)
    else:
      return value

  def set_general(self, key, value, override=False, nocreate=False):
    """Specify a value for a general (i.e. fallback) setting
    May specify an expression as a string parseable by sympy
    Unless override is True, conditional settings will still take precedence
    If nocreate is True, raises error if key does not exist"""
    # May be otherwise unrecognized
    if key in self.global_settings:
      logger.debug('Updating global setting %s to %s',key,value)
      self.global_settings[key] = value
      return
    if key not in self._var_setting:
      if nocreate:
        raise KeyError(key)
      else:
        logger.debug('Adding setting %s',key)
        self._var_setting[key] = sympy.symbols(key)
    logger.debug('Updating setting %s to %s',key,value)
    self.general_settings[key] = self.parse_value(key,value)
    if override:
      logger.debug('Wiping conditional settings for key %s',key)
      self.conditional_settings[key] = []

  def set_conditional(self, key, condition, value, priority=True,
      nocreate=False, **stage_args):
    """Specify a value for a conditional setting
    condition should be a string parseable by sympy
    If nocreate, raises error if key does not exist
    If priority, will be given priority over existing rules; otherwise will
      be given last priority
    May additionally specify 'stage variables' as keywords"""
    conditions = list(stage_args.items())
    if condition:
      conditions.append(condition)
    logger.debug('Updating setting %s with value %s subject to conditions %s',
      key,value,condition)
    expr = self.parse_condition(conditions)
    if key not in self._var_setting:
      if nocreate:
        raise KeyError(key)
      else:
        self._var_setting[key] = sympy.symbols(key)
    val = self.parse_value(key,value)
    if key not in self.conditional_settings:
      self.conditional_settings[key] = [(expr,val)]
    elif priority:
      self.conditional_settings[key].insert(0,(expr,val))
    else:
      self.conditional_settings[key].append((expr,val))

  def eval_at_stage(self, stage_dict):
    """Evaluate settings given values for stage variables in stage_dict"""
    sys_dict = self.owner.query_system()
    prior_subs = [(symbol,stage_dict[key]) \
         for key,symbol in self._var_stage.items()] \
     + [(symbol,sys_dict[key]) for key,symbol in self._var_sys.items()]
    all_subs = dict(prior_subs)
    # Find which rules to follow
    evaluated = {}
    unevaluated = {}
    for key in self._var_setting:
      value = self.setting_at_stage(key, prior_subs)
      if isinstance(value,sympy.Basic):
        unevaluated[key] = value
      else:
        evaluated[key] = value
        all_subs[self._var_setting[key]] = value
    # Evaluate expressions for settings
    while unevaluated:
      n_uneval = len(unevaluated)
      for key,expr in list(unevaluated.items()):
        if expr.free_symbols.issubset(set(all_subs)):
          # Can be evaluated
          value = expr.subs(all_subs)
          if not isinstance(value,sympy.core.numbers.Number):
            # TODO may need to ensure integer conversion in some cases
            value = value.evalf()
          evaluated[key] = value
          all_subs[self._var_setting[key]] = value
          unevaluated.pop(key)
      if n_uneval == len(unevaluated):
        # unevaluated has not changed, implying a cyclic dependency
        raise ValueError('Cyclic dependency encountered among settings %r'%\
          set(unevaluated))
    return evaluated

  def setting_at_stage(self, key, substitutions):
    """Get applicable setting given values of stage & system variables"""
    # Test conditionals
    if key in self.conditional_settings:
      for cond,val in self.conditional_settings[key]:
        query = cond.subs(substitutions)
        if query:
          return val
    # All conditionals have failed, return fallback if present
    return self.general_settings.get(key)


class SettingsLoader:
  """Load settings from file"""
  def __init__(self, filename, key, configs, defaults=False):
    self.configs = configs
    # TODO paths
    logger.info('Loading settings from table %s of %s',key,filename)
    with open(filename,'r') as f:
      confs = tomlkit.parse(f.read())
    table = confs[key]
    if defaults:
      table = table['defaults']
    if not defaults and 'parent' in table:
      fparent = table.pop('parent-file',filename)
      self._parent = SettingsLoader(fparent, table.pop('parent'), configs)
    elif defaults and configs.parent is not None:
      self._parent = SettingsLoader(configs.parent.filename,
        configs.parent.clsname, configs.parent, True)
    else:
      self._parent = None
    self.fallbacks = {}
    self.global_settings = {}
    self.rules = {}
    # Populate fallbacks from table
    rules = []
    for key in table:
      if key in configs.global_keys:
        self.global_settings[key] = table[key]
      elif key == 'rules':
        rules = table[key]
      else:
        self.fallbacks[key] = table[key]
    # Populate everything from parent
    if self._parent is not None:
      # TODO way of determining which stage settings should preempt which
      self.global_settings = self._parent.global_settings | self.global_settings
      # TODO fallbacks that do not fully preempt inherited rules?
      for key in self._parent.rules:
        if key not in self.fallbacks:
          self.rules[key] = self._parent.rules[key]
      # Now that rules have been populated, add fallbacks
      self.fallbacks = self._parent.fallbacks | self.fallbacks
    else:
      self.global_settings = global_setting_defaults | self.global_settings
      self.rules = {}
    # Populate rules
    # Precedence is essentially FIFO:
    # earliest rules get highest precedence, child rules override parent
    # TODO explicit precedence system?
    # self.rules values will be list of (conditions, value) pairs
    for rule in rules[::-1]:
      # Conditions are specified by the key 'condition' or a stage variable
      # (used as a key)
      # All other keys are assumed to name settings
      conditions = []
      settings = {}
      for key in rule:
        if key == 'condition':
          conditions.append(rule[key])
        elif key in configs.stage_variables:
          conditions.append((key,rule[key]))
        else:
          settings[key] = rule[key]
      logger.debug('Rule set found mapping %s -> %s',conditions,settings)
      for key in settings:
        if key not in self.rules:
          self.rules[key] = []
        self.rules[key].insert(0,(conditions,settings[key]))
      

class ConfigLoader:
  """Load config settings from file (table with default)"""
  def __init__(self, filename, clsname):
    # TODO paths
    self.stage_variables = set()
    self.system_variables = set()
    self.stringopts = set()
    self.filename = filename
    self.clsname = clsname
    logger.info('Loading configurations for class %s from file %s',
      self.clsname, self.filename)
    with open(filename,'r') as f:
      confs = tomlkit.parse(f.read())
    table = confs[clsname]
    if 'inherit-from' in table:
      if 'inherit-from-file' in table:
        fparent = table['inherit-from-file']
      else:
        fparent = filename
      parent = ConfigLoader(fparent, table['inherit-from'])
      # Inherit where not overridden
      if not table.get('stage-vars-override',default=False):
        self.stage_variables.update(parent.stage_variables)
      if not table.get('system-vars-override',default=False):
        self.system_variables.update(parent.system_variables)
      if not table.get('str-opts-override',default=False):
        self.stringopts.update(parent.stringopts)
      self.global_keys = set(parent.global_keys)
      self.parent = parent
    else:
      self.global_keys = set(global_setting_defaults)
      self.parent = None
    if 'stage-vars' in table:
      self.stage_variables.update(table['stage-vars'])
      #self.stage_variables.update(dict(zip(stagevars,sympy.symbols(stagevars))))
    if 'system-vars' in table:
      self.system_variables.update(table['system-vars'])
    if 'str-opts' in table:
      self.stringopts.update(table['str-opts'])
    if 'globals' in table:
      self.global_keys.update(table['globals'])
