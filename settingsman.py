# Notes on TOML files for GeneralManager subclasses:
# Config file contains table with same name as class
# Keys
# - stage-vars (list) indicates names of variables used to specify stage
#   (keys of dictionary returned by query_stage)
# - system-vars (list) names ``global'', permanent variables (e.g. system size)
# - str-opts (list) names settings options which have string values
#   (i.e. shuold not attempt to parse)
# - globals (list) indicates global options (e.g. saving behavior,
#   specifications for stages like bond dimensions)
# - submans (table) gives names of sub-managers, mapped to class names
#   (may be (cls, filename) tuple if not in same file)
# - sub-keys (table) identifies how settings of sub-managers are to be
#   specified
#     For any sub-manager may give table mapping
#     key in sub = key in super or formula 
#     (TODO formula dependent on stages in sub)
# - inherit-from (str) names class of manager that configs are inherited from
#   (does not have to be true superclass)
#   - If table does not belong to same file, inherit-from-file should give
#     filename
#   - globals adds to names given in parent 
#   - stage-vars, system-vars, str-opts adds to names given in parent unless
#     stage-vars-override, system-vars-override, and str-opts-override,
#     respectively, are given
#   - similarly for sub-managers, but may be overriden with sub-override
#     OR sub-exclude may be provided as list
#   - TODO all default settings are inherited, i.e. no way to exclude settings
# - defaults subtable gives table of default values, parsed like other
#   settings tables.

# Rules for settings tables:
# - A string value for a settings key not in str-opts, unless 'none', will 
#   be parsed as a formula using sympy rules (including the sympy parser's
#   standard_transformations, convert_xor, implicit_multiplication,
#   factorial_notation). Allowable variable names will be other settings
#   keys, stage variables (as returned by query_stage), and "system" variables.
#   Formulae may be dependent on other stage-dependent variables, but
#   circular dependencies are forbidden.
# - 'rules' may be given as a list of tables, each containing one or more
#   conditions and settings
#   Conditions may be provided either by assigning a value to a key
#   corresponding to a stage variable (e.g. back_index = -1)
#   or setting the 'condition' key to a conditional formula (parsed using
#   the same sympy rules as before, with the additional convert_equals_sign
#   transformation). Conditional formualae may only be dependent on 'stage'
#   variables.
#   The settings imposed under this condition are given in the same kind of
#   key-value pair as the other ("fallback") settings.
#   Conditional settings are prioritized according to the order in the
#   rules list: the first rule whose condition evaluates to true which contains
#   the key will set it; if no rule sets the key, it will revert to the
#   'fallback' value
# - 'parent' gives the name of a table to inherit rules from
#   (an additional key "parent-file" may specify the file to load this table
#   from, otherwise it will be sought in the same file)
#   Settings in the current table will always be given priority over settings
#   in the 'parent' table; in particular a "fallback" setting for a given
#   key in the current table will override any conditional settings in the
#   parent table.
# - "sub-manager" tables: if there are any sub-managers, they may be given a
#   subtable (key is the name of the sub).
#   Any expressions in the subtable will be parsed as referencing
#   the super's variables; to reference the sub's variables use a period:
#   e.g. evolver.delta = "gen_index * evolver.timestep" sets the key delta of
#   the sub-manager evolver to be the super's stage variable gen_index times the
#   sub's stage variable timestep.
#   - 'parent' and 'parent-file' may be given as in ordinary tables (should
#     not indicate other subtables)
#   Altogether there are 5 ways for the user to specify settings for a sub-
#   manager:
#   * set keys of the super-manager, as mapped via sub-keys
#   * supply settings at runtime with names of form <sub>.<key>
#     (or sub-dictionary of keyword dictionary)
#   * supply <sub>.settings-from=(name,filename) at runtime
#     (or <sub>=(name,filename), or <sub>={'settings-from':(name,filename)})
#   * set subtable of settings table with name <sub>
#   * inherit settings into subtable, i.e. set <sub>.parent
#   Within sources of the same origin (as supplied to the manager) priority is
#   in that order; in particular keys with sub-keys mapping are given
#   precedence (TODO it may be desirable to override this) including over
#   other means of setting the same key from prior sources (as this
#   correspondence will be evaluated at runtime, after loading sources)
#    TODO this precedence will currently be unpredictably implemented
# TODO explicit priority system:
# * sources get "baseline" priority as a multiple of 10, 0 for keys assigned
#   at runtime, 10 for first file, etc., unless otherwise specified
#   (last tuple element)
# * "relative" priorities (typically 0-10) may be assigned within sources
#   but would typically relate to various forms of "inheritance"
import tomlkit
import sympy
from sympy.parsing import sympy_parser
import os.path
import numpy as np
from .config import settings_log as logger
from tokenize import (NAME, OP)
from collections import defaultdict

# Defaults for "global settings" (having to do with saving, stages, etc.)
global_setting_defaults = dict(use_shelf=False,savesafe=False,savelevel=2)
def unsplit_names0(tokens, local_dict, global_dict):
  """Use symbol names with . (when already in identified namespace"""
  dotnames = [k for k in local_dict if '.' in k] + [k for k in global_dict if '.' in k]
  if not dotnames:
    return tokens
  for name in list(dotnames):
    seq = name.split('.')
    while len(seq)>2 and '.'.join(seq[:-1]) not in dotnames:
      seq.pop()
      dotnames.append('.'.join(seq))
  result = []
  itok = 0
  print(dotnames)
  while itok < len(tokens):
    token = tokens[itok]
    if itok > 0 and itok < len(tokens)-1 and token == (OP, '.'):
      print(tokens[itok-1:itok+2])
      if tokens[itok-1][0] == NAME and tokens[itok+1][0] == NAME:
        candidate = '%s.%s'%(tokens[itok-1][1],tokens[itok+1][1])
        if candidate in dotnames:
          token = (NAME, candidate)
          print(token)
          result.pop()
          itok += 1
        else:
          print(candidate)
    result.append(token)
    itok += 1
  return result
      
class SympyNamespace:
  def __init__(self, name, odict):
    self.__dict__.update(odict)
    self.__name__ = name

  @property
  def name(self):
    return self.__name__

def unsplit_names(tokens, local_dict, global_dict):
  """Use symbol names with . (when parent object is in identified namespace)"""
  namespaces = {k:v for k,v in (global_dict|local_dict).items() if isinstance(v,SympyNamespace)}
  if not namespaces:
    return tokens
  result = []
  itok = 0
  while itok < len(tokens):
    token = tokens[itok]
    if itok > 0 and itok < len(tokens)-1 and token == (OP, '.'):
      if tokens[itok-1][0] == NAME and tokens[itok+1][0] == NAME:
        if result[-1][1] in namespaces:
          combo = '%s.%s'%(result[-1][1],tokens[itok+1][1])
          if combo in local_dict or combo in global_dict:
            token = (NAME,combo)
            result.pop()
            itok += 1
    result.append(token)
    itok += 1
  return result


# Cut auto_symbol from standard_transformations
val_transformations = (unsplit_names,
                       sympy_parser.lambda_notation,
                       sympy_parser.repeated_decimals,
                       sympy_parser.auto_number,
                       sympy_parser.factorial_notation,
                       sympy_parser.convert_xor,
                       sympy_parser.implicit_multiplication,
                       sympy_parser.factorial_notation)
cond_transformations = (*val_transformations, sympy_parser.convert_equals_signs)

class SettingsManager:
  """Class to handle specifications of settings (including from config files
  and runtime specifications)"""
  def __init__(self, owner, *args, use_defaults=True):
    # Following class of calling manager, arguments are either
    # (filename, key) pairs or dictionaries of settings values
    # Earlier arguments have higher precedence
    # Load configuration info
    self.__version = '0.1'
    fconf = os.path.join(os.path.dirname(__file__),owner._default_file)
    configs = ConfigLoader(fconf,owner._configs_id)
    self._configs = configs
    self.owner = owner
    
    stagevar = list(configs.stage_variables)
    self._var_stage = dict(zip(stagevar, sympy.symbols(stagevar)))
    sysvar = list(configs.system_variables)
    self._var_sys = dict(zip(sysvar, sympy.symbols(sysvar)))
    self.stringopts = configs.stringopts
    self._fallbacks = {}
    self._rules = defaultdict(list)
    self.global_settings = {}
    self.submans = {}

    for arg in args:
      if isinstance(arg,dict):
        # TODO allow for pre-initialization runtime definition of rules?
        for key in arg:
          if key in configs.global_keys:
            if key not in self.global_settings:
              logger.debug('Found global setting for %s in dict',key)
              self.global_settings[key] = arg[key]
          elif key in configs.submans:
            self._set_to_subman(key,arg[key])
          elif '.' in key:
            self._subman_key(key,arg[key])
          elif key not in self._fallbacks:
            logger.debug('Found fallback setting for %s in dict',key)
            self._fallbacks[key] = arg[key]
      else:
        loader = SettingsLoader(*arg, configs)
        self._load_loader(loader)
    if use_defaults:
      logger.debug('Loading default settings')
      loader = SettingsLoader(configs.filename,configs.clsname,configs,True)
      self._load_loader(loader)
    # Now have all (initial) settings names
    setvar = list(set(self._fallbacks) | set(self._rules))
    self._var_setting = dict(zip(setvar, sympy.symbols(setvar))) 
    self._complete()

  @property
  def prior_vars(self):
    return self._var_stage | self._var_sys

  @property
  def all_vars(self):
    return self.prior_vars | self._var_setting

  def prior_at_level(self, lvl):
    assert lvl == 0
    return self.prior_vars

  def all_at_level(self, lvl):
    assert lvl == 0
    return self.all_vars

  def parse_condition(self, conditions):
    parsed = []
    for cond in conditions:
      if isinstance(cond, str):
        # Preprocess to not have to worry about = or ==
        try:
          expr = sympy_parser.parse_expr(cond.replace('==','='),
            self.prior_vars,transformations=cond_transformations)
        except NameError as e:
          raise KeyError('Unrecognized symbol "%s" in condition "%s"' \
            % (e.name,cond))
        parsed.append(expr)
      else:
        key,value = cond
        assert not isinstance(value,str) # TODO maybe revisit
        parsed.append(sympy.Eq(self._var_stage[key], value))
    return sympy.And(*parsed)

  def parse_value(self, key, value):
    if isinstance(value,str) and key not in self.stringopts:
      if value == 'none':
        return None
      else:
        try:
          expr = sympy_parser.parse_expr(value, self.all_vars,
            transformations=val_transformations)
        except NameError as e:
          raise KeyError('Unrecognized symbol "%s" in expression "%s" for '
            'setting %s' % (e.name,value,key))
        return expr
    else:
      return value

  def set_general(self, key, value, override=False, nocreate=False):
    """Specify a value for a general (i.e. fallback) setting
    May specify an expression as a string parseable by sympy
    Unless override is True, conditional settings will still take precedence
    If nocreate is True, raises error if key does not exist"""
    # May be otherwise unrecognized
    if '.' in key:
      sub,subkey = key.split('.',1)
      if isinstance(value,tuple):
        subval = (sub,)+value
      else:
        subval = (sub,value)
      self.submans[sub].set_general(subkey,subval,override,nocreate)
      return
    if key in self.global_settings:
      logger.debug('Updating global setting %s to %s',key,value)
      self.global_settings[key] = value
      return
    if key not in self._var_setting:
      # TODO should this be avoided for string options?
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
    self._set_conditional(key, conditions, value, priority, nocreate)

  def _set_conditional(self, key, conditions, value, priority, nocreate,
      names=()):
    if '.' in key:
      sub,subkey = key.split('.',1)
      self.submans[sub]._set_conditional(subkey,conditions,value,priority,
        nocreate,names+(sub,))
      return
    elif names:
      # Prepare sub-manager-compatible
      lvl = len(names)
      if isinstance(value,str) and key not in self.stringopts:
        value = names + (value,)
      newcond = []
      if isinstance(conditions[-1],str):
        newcond.append(names + (conditions.pop(),))
      for skey,sval in conditions[::-1]:
        *snames,skey = skey.split('.')
        slvl = lvl-len(snames)
        assert names[:-slvl] == snames # May actually be important in this case
        # TODO this shit is as good an argument as any for just indicating level
        # in heirarchy with an integer
        newcond.append((names[-slvl:]+(skey,),sval))
      conditions = newcond[::-1]
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
      elif value is not None:
        evaluated[key] = value
        all_subs[self._var_setting[key]] = value
    # Evaluate expressions for settings
    while unevaluated:
      n_uneval = len(unevaluated)
      for key,expr in list(unevaluated.items()):
        if expr.free_symbols.issubset(set(all_subs)):
          # Can be evaluated
          value = expr.subs(all_subs)
          if value.is_integer:
            value = int(value)
          elif value.is_infinite:
            # Don't expect various infinities to be handleable
            value = np.nan
          else:
            value = float(value)
          #if not isinstance(value,sympy.core.numbers.Number):
          #  # TODO may need to ensure integer conversion in some cases
          #  value = value.evalf()
          evaluated[key] = value
          all_subs[self._var_setting[key]] = value
          unevaluated.pop(key)
      if n_uneval == len(unevaluated):
        # unevaluated has not changed, implying a cyclic dependency
        raise ValueError('Cyclic dependency encountered among settings %r'%\
          set(unevaluated))
    # TODO possibility for "hypothetical" unstored settings?
    self._evaluated_settings = evaluated | stage_dict
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

  def __setstate__(self, state):
    # TODO how well/consistently do sympy objects pickle?
    self.__dict__.update(state)
    if self.__version == '0.0':
      self.submans = {}
    self.__version = '0.1'

  def _get_subman(self, name):
    # Get sub-manager, creating if necessary
    if name not in self.submans:
      self.submans[name] = SettingsSubManager(name, self,
        *self._configs.submans[name])
    return self.submans[name]

  def _set_to_subman(self, name, value):
    # Argument from settings dictionary
    sub = self._get_subman(name)
    if isinstance(value,dict):
      sub._load_from_dict(value)
    else:
      sub._load_from_file(*value)

  def _subman_key(self, key, value):
    name,*subkeys = key.split('.')
    self._get_subman(name)._set_key(subkeys,value)

  def _load_loader(self, loader):
    # Incorporate rules first to maintain precedence of fallbacks
    self._load_rules(loader)
    self._load_fallbacks(loader)
    self._load_globals(loader)
    for sub in loader.submans:
      self._get_subman(sub)._load_loader(loader.submans[sub])

  def _load_rules(self, loader):
    for key in loader.rules:
      if key not in self._fallbacks:
        logger.debug('Adding new rule for %s',key)
        if key not in self._rules:
          self._rules[key] = []
        self._rules[key].extend(loader.rules[key])
      else:
        logger.debug('Rule for %s omitted (fallback found)')

  def _load_fallbacks(self, loader):
    logger.debug('Found fallback settings for %s',
      set(loader.fallbacks)-set(self._fallbacks))
    self._fallbacks = loader.fallbacks | self._fallbacks

  def _load_globals(self, loader):
    logger.debug('Found global settings for %s',
      set(loader.global_settings)-set(self.global_settings))
    self.global_settings = loader.global_settings | self.global_settings

  def _complete(self):
    # Finish initialization
    # Finally sympify expressions
    self.general_settings = {}
    for key,value in self._fallbacks.items():
      logger.log(15,'Setting fallback %s to %s',key,value)
      self.general_settings[key] = self.parse_value(key,value)
    for key,rulelist in self._rules.items():
      logger.log(15,'Setting rules for %s',key)
      for i,(conds,val) in enumerate(rulelist):
        logger.debug('Adding condition %s => %s',conds,val)
        rulelist[i] = (self.parse_condition(conds),
          self.parse_value(key,val))
    self.conditional_settings = dict(self._rules)
    # Expand global settings in case any are missing
    global_missing = self._configs.global_keys - set(self.global_settings)
    self.global_settings.update({key:None for key in global_missing})
    for sub in self.submans:
      self.submans[sub]._complete_sub()
    del self._configs
    del self._rules
    del self._fallbacks

  def promote_subman(self, subname):
    """Get final form out of sub-settings-manager"""
    # TODO incorporate save_level and use_shelf settings
    # --maybe those shouldn't be settings-manager-controlled at all?
    subs_sup = getattr(self, '_sup_substitutions', {})
    subs_current = {self.all_vars[key]:value for key,value in \
      self._evaluated_settings.items()}
    newman = PromotedSettingsManager(self.submans[subname],
      subs_sup | subs_current)
    return newman


class PromotedSettingsManager(SettingsManager):
  # Take info from SettingsSubManager & act as full SettingsManager
  def __init__(self, original, substitutions):
    self.__version = '0.0'
    self.owner = None #TODO
    self._var_stage = dict(original._var_stage)
    self._var_sys = dict(original._var_sys)
    self._var_setting = dict(original._var_setting)
    self.stringopts = original.stringopts
    self.global_settings = dict(original.global_settings)
    self.general_settings = {}
    self.conditional_settings = {}
    # First: collect (unsubstituted) fallbacks
    self.general_settings = dict(original.general_settings)
    # Then apply to conditions, discarding or promoting if evaluatable
    for key in original.conditional_settings:
      self.conditional_settings[key] = []
      for cond,val in original.conditional_settings[key]:
        query = cond.subs(substitutions)
        if query == True:
          logger.debug('Promoting conditional setting %s=%s', key,val)
          self.general_settings[key] = val
          self.conditional_settings.pop(key)
          break
        elif query == False:
          logger.debug('Discarding conditional setting %s=%s',key,val)
          continue
        else:
          if isinstance(val,sympy.Basic):
            val = val.subs(substitutions)
          self.conditional_settings[key].append((query,val))
    # Now apply key mappings from super-manager
    # Only where all dependencies != None (TODO may be worth reconsidering?)
    non_null = set(v for v in substitutions if substitutions[v] is not None)
    for key,expr in original._supkeys.items():
      if isinstance(expr,str):
        # TODO should it be required that either both or neither are
        # global settings?
        if key in self.stringopts:
          self.general_settings[key] = original._supman.general_settings[expr]
        else:
          self.global_settings[key] = original._supman.global_settings[expr]
        continue
      dependencies = expr.free_symbols
      if dependencies.issubset(non_null):
        logger.debug('Applying mapped value %s=%s',key,expr)
        self.general_settings[key] = expr
      else:
        logger.debug('Not applying mapped value %s=%s '
          '(null/unset dependencies)', key, expr)
    for key,value in self.general_settings.items():
      if isinstance(value,sympy.Basic):
        self.general_settings[key] = value.subs(substitutions)
    self._sup_substitutions = substitutions
    self.submans = original.submans


class SettingsSubManager(SettingsManager):
  # Contain information needed to produce settings for currently inactive
  # sub-managers
  def __init__(self, name, supman, clsname, configs, supkeys):
    self.name = name
    *supnames,subname = name.split('.')
    self.subname = subname
    self._supman = supman
    self._supkeys = supkeys
    self._configs = configs
    self.submans = {}
    self._var_stage = {v:self._symbol(v) for v in configs.stage_variables}
    self._var_sys = {v:self._symbol(v) for v in configs.system_variables}
    self.stage_namespace = SympyNamespace(name,self._var_stage | self._var_sys)
    self.stringopts = configs.stringopts
    self._fallbacks = {}
    self._rules = {}
    #self._configs = configs
    self.global_settings = {}
    self.general_settings = {}
    self.conditional_settings = {}

  def _symbol(self, varname):
    return sympy.symbols('%s.%s'%(self.name,varname))

  def _get_subman(self, name):
    # Get sub-manager, creating if necessary
    if name not in self.submans:
      self.submans[name] = SettingsSubManager('.'.join((self.name,name)), self,
        *self._configs.submans[name])
    return self.submans[name]

  def _set_key(self, keys, value, supkeys=()):
    key,*subkeys = keys
    if subkeys:
      self._get_subman(key)._set_key(subkeys,value,
        supkeys=supkeys+(self.subname,))
    elif key in self._configs.global_keys:
      if key not in self.global_settings:
        logger.debug('Found global setting for %s.%s',self.name,key)
        self.global_settings[key] = value
    elif key in self._configs.submans:
      self._set_to_subman(key, value)
    else:
      # TODO can we be sure that no dotted keys will appear here?
      assert '.' not in key
      if key not in fallbacks:
        logger.debug('Found fallback setting for %s.%s',self.name,key)
        if isinstance(value,str) and key not in self.stringopts:
          self._fallbacks[key] = supkeys + (self.subname,value)
        else:
          self._fallbacks[key] = value

  def _load_from_dict(self, d):
    for k in d:
      keys = k.split('.')
      self._set_key(keys, d[k])

  #def _load_rules(self, loader):
  #  logger.debug('Adding rules for sub-manager %s',self.name)
  #  if isinstance(loader, SubSettingsLoader):
  #    # Rules are already prepared as appropriate
  #    super()._load_rules(loader)
  #  else:
  #    for key in loader.rules:
  #      if key not in self._fallbacks:
  #        logger.debug('Adding new rule for setting %s of sub-manager %s',
  #          key,self.name)
  #        if key not in self._rules:
  #          self._rules[key] = []
  #        for oldcond,value in loader.rules[key]:
  #          conditions = []
  #          for cond in oldcond:
  #            if isinstance(cond,str):
  #              conditions.append((cond,))
  #            else:
  #              stagekey,stageval = cond
  #              conditions.append(((stagekey,),stageval))

  #def _load_fallbacks(self, loader):
  #  if 
  #  logger.debug('Found fallback settings for %s',
  #    set(loader.fallbacks)-set(self._fallbacks))
  #  self._fallbacks = loader.fallbacks | self._fallbacks

  def _load_from_file(self, filename, name):
    logger.debug('Loading from %s:%s into sub-manager %s',
      filename,name,self.name)
    # To use extant machinery, load as "parent" into blank sub-loader
    loader = SettingsLoader(filename, name, self._configs)
    subloader = SubSettingsLoader({}, self.name, None, self._configs, False)
    subloader._parent = loader
    subloader._parentpopulate()
    self._load_rules(loader)
    self._load_fallbacks(loader)
    self._load_globals(loader)

  def _complete_sub(self):
    # Finish initialization
    self._var_setting = {}
    for key in list(set(self._fallbacks)|set(self._rules)|set(self._supkeys)):
      self._var_setting[key] = sympy.symbols('%s.%s'%(self.name,key))
    self.namespace = SympyNamespace(self.name, self.all_vars)
    if isinstance(self._supman,SettingsSubManager):
      setattr(self._supman.namespace,self.subname,self.namespace)
    logger.log(15, 'Finalizing settings for sub-manager %s',self.name)
    # Parse keys from super-manager
    supvars = self._supman.all_vars
    for key,value in list(self._supkeys.items()):
      if value in self._supman.global_settings or key in self._configs.stringopts:
        expr = value
      elif value in supvars:
        expr = supvars[value]
      else:
        expr = sympy_parser.parse_expr(value, supvars,
          transformations=val_transformations)
      self._supkeys[key] = expr
    self._complete()

  def get_stagevar(self, names, key):
    if names:
      *supnames,name = names
      assert name == self.subname
      return self._supman.get_stagevar(supnames, key)
    else:
      return self._var_stage[key]

  def prior_at_level(self, lvl):
    if lvl == 0:
      return self.prior_vars
    else:
      names = self.name.split('.')[-lvl:]
      return self._supman.prior_at_level(lvl-1) | {'.'.join(names):self.namespace}
      # TODO 'prior'-only namespace?
      nmsp = {'.'.join(names):self.namespace}
      priors = {'.'.join(names+[key]):symb for key,symb in self.prior_vars.items()}
      return priors | nmsp | self.supman.prior_at_level(lvl-1)

  def all_at_level(self, lvl):
    if lvl == 0:
      return self.all_vars
    else:
      names = self.name.split('.')[-lvl:]
      nmsp = {'.'.join(names):self.namespace}
      return self._supman.all_at_level(lvl-1) | nmsp
      #vardict = {'.'.join(names+[key]):symb for key,symb in self.all_vars.items()}
      #return vardict | nmsp | self._supman.all_at_level(lvl-1)

  def parse_condition(self, conditions):
    parsed = []
    for cond in conditions:
      if isinstance(cond[0], str):
        *names, unparsed = cond
        # Preprocess to not have to worry about = or ==
        try:
          expr = sympy_parser.parse_expr(unparsed.replace('==','='),
            self.prior_at_level(len(names)),transformations=cond_transformations)
        except NameError as e:
          msg = 'Unrecognized symbol "%s" in condition "%s" in ' \
            'sub-manager %s' % (e.name, unparsed, self.name)
          if not names:
            msg += ' (minimal context)'
          else:
            msg += ' (context %s)'%('.'.join(names))
          raise KeyError(msg)
        parsed.append(expr)
      else:
        (*names,key),value = cond
        assert not isinstance(value,str)
        parsed.append(sympy.Eq(self.get_stagevar(names,key), value))
    return sympy.And(*parsed)

  def parse_value(self, key, value):
    if isinstance(value,tuple):
      # TODO may need extra steps to ensure tuple is not accidentally provided
      # e.g. via dict
      *names, value = value
      if value == 'none':
        return None
      if key in self.stringopts or not isinstance(value,str):
        return value
      ldict = self.all_at_level(len(names))
      try:
        expr = sympy_parser.parse_expr(value, ldict,
          transformations=val_transformations)
      except NameError as e:
        msg = 'Unrecognized symbol "%s" in expression "%s" for ' \
          'sub-manager setting %s.%s' % (e.name, value, self.name, key)
        if not names:
          msg += ' (minimal context)'
        else:
          msg += ' (context %s)'%('.'.join(names))
        raise KeyError(msg)
      return expr
    elif value == 'none':
      return None
    else:
      return value

def _flattentable(table):
  # Flatten dict or TOML table, such that sub-tables are instead represented
  # using dotted keys
  if hasattr(table,'unwrap'):
    table = table.unwrap()
  result = {}
  for key,value in table.items():
    if isinstance(value,dict):
      for subkeys,subvalue in _subflatten((key,),value):
        result['.'.join(subkeys)] = subvalue
    else:
      result[key] = value
  return result
        
def _subflatten(superkeys,subtable):
  result = {}
  for subkey,value in subtable.items():
    keys = superkeys+(subkey,)
    if isinstance(value,dict):
      yield from _subflatten(keys,value)
    else:
      yield keys,value

class SettingsLoader:
  """Load settings from file"""
  def __init__(self, filename, key, configs, defaults=False):
    self.configs = configs
    self._defaults = defaults
    self.filename = filename
    # TODO paths
    logger.info('Loading settings from table %s of %s',key,filename)
    with open(filename,'r') as f:
      confs = tomlkit.parse(f.read())
    table = confs[key]
    if self._defaults:
      table = table['defaults']
    if not self._defaults and 'parent' in table:
      fparent = table.pop('parent-file',filename)
      self._parent = SettingsLoader(fparent, table.pop('parent'), configs)
    elif self._defaults and configs.parent is not None:
      self._parent = SettingsLoader(configs.parent.filename,
        configs.parent.clsname, configs.parent, True)
    else:
      self._parent = None
    # Populate fallbacks from table
    rules = self._populate(table)
    # Populate everything from parent
    self._parentpopulate()
    # Populate rules
    # Precedence is essentially FIFO:
    # earliest rules get highest precedence, "child" rules override parent
    # TODO explicit precedence system?
    # self.rules values will be list of (conditions, value) pairs
    for rule in rules[::-1]:
      # Conditions are specified by the key 'condition' or a stage variable
      # (used as a key)
      # All other keys are assumed to name settings
      conditions, settings = self._transform_rule(rule)
      logger.debug('Rule set found mapping %s -> %s',conditions,settings)
      for key in settings:
        if '.' in key:
          self._addsubrule(key,conditions,settings[key])
        else:
          if key not in self.rules:
            self.rules[key] = []
          self.rules[key].insert(0,(conditions,settings[key]))
    for sub in self.submans:
      self.submans[sub]._complete_init()

  def _populate(self, table):
    rules = []
    self.fallbacks = {}
    self.global_settings = {}
    self.submans = {}
    self.rules = {}
    for key in table:
      if key in self.configs.global_keys:
        self.global_settings[key] = table[key]
      elif key == 'rules':
        rules = table[key]
      elif key in self.configs.submans:
        self._initsub(key, table[key])
      else:
        self.fallbacks[key] = table[key]
    if self._defaults:
      # Defaults for any sub-managers that have been missed
      for sub in self.configs.submans:
        self._initsub(sub, {})
    return rules

  def _parentpopulate(self):
    if self._parent is None:
      self.global_settings = global_setting_defaults | self.global_settings
      # TODO different way to handle defaults for sub-manager?
      return
    self.global_settings = self._parent.global_settings | self.global_settings
    for key in self._parent.rules:
      if key not in self.fallbacks:
        if key not in self.rules:
          self.rules[key] = []
        for rule in self._parent.rules[key]:
          self._addparentrule(key,rule)
    for key in self._parent.fallbacks:
      if key not in self.fallbacks:
        self._addparentfallback(key,self._parent.fallbacks[key])
    for sub in set(self._parent.submans) & set(self.configs.submans):
      # Exclude as appropriate--TODO improve
      if self.configs.parent is None or self._parent.configs.clsname != self.configs.parent.clsname or sub not in self.configs.xsub:
        if sub not in self.submans:
          self._initsub(sub, {})
        self.submans[sub]._superparentsub(self._parent.submans[sub])

  def _addparentrule(self, key, rule):
    self.rules[key].append(rule)

  def _addparentfallback(self, key, value):
    self.fallbacks[key] = value

  def _initsub(self, subname, subtable):
    self.submans[subname] = SubSettingsLoader(subtable, subname, self,
      self.configs.submans[subname][1], self._defaults)
      
  def _transform_rule(self, rule):
    conditions = []
    settings = {}
    frule = _flattentable(rule)
    for key in frule:
      if key == 'condition':
        conditions.append(frule[key])
      elif key in self.configs.stage_variables:
        conditions.append((key,frule[key]))
      else:
        settings[key] = frule[key]
    return conditions, settings

  def _addsubrule(self, key, conditions, value):
    allnames = key.split('.')
    names = allnames[:-1] # Hierarchy of sub-managers
    subname,*subkeys = allnames
    assert subname in self.configs.submans
    condnew = []
    # Wrap all values here & unwrap as necessary at final level
    value = (*names, value)
    for cond in conditions:
      # All conditions are in super-manager naming context
      if isinstance(cond,str):
        condnew.append((*names,cond))
      else:
        # For now at least keys specified in super-manager rules must be
        # super-manager variables
        stagekey,val = cond
        condnew.append(((*names,stagekey),val))
    if subname not in self.submans:
      self._initsub(subname,{})
    self.submans[subname]._addrule(subkeys, condnew, value)


class SubSettingsLoader(SettingsLoader):
  def __init__(self, subtable, sub_name, supman, configs, defaults):
    self.configs = configs
    self._defaults = defaults
    self.name = sub_name
    self.filename = supman.filename
    logger.debug('Loading "sub-manager" settings for %s',sub_name)
    if 'parent' in subtable:
      fparent = subtable.pop('parent-file',supman.filename)
      self._parent = SettingsLoader(fparent, subtable.pop('parent'), configs)
    elif self._defaults:
      self._parent = SettingsLoader(configs.filename,configs.clsname,
        configs,True)
    else:
      self._parent = None
    self._rulelist = self._populate(subtable)
    # Will need this to be persistent to wait for rules from 'superior'

    names = self.name.split('.')
    for key in self.fallbacks:
      if isinstance(self.fallbacks[key],str) and key not in configs.stringopts:
        self.fallbacks[key] = (*names,self.fallbacks[key])
    # Populate from 'direct' parent first (this will NOT be a sub loader)
    self._parentpopulate()

  def _initsub(self, subname, subtable):
    # Need to acknowledge hierarchy
    subsubname = '.'.join((self.name,subname))
    self.submans[subname] = SubSettingsLoader(subtable, subsubname, self,
      self.configs.submans[subname][1], self._defaults)
    #super()._initsub('.'.join((self.name,sub_name)),subtable)

  def _addparentrule(self, key, rule):
    condold,value = rule
    if isinstance(value,str) and key not in self.configs.stringopts:
      value = (value,)
    conditions = []
    for cond in condold:
      if isinstance(cond,str):
        conditions.append((cond,))
      else:
        stagekey,val = cond
        conditions.append(((stagekey,),cond))
    super()._addparentrule(key, (conditions,value))

  def _addparentfallback(self, key, value):
    if isinstance(value,str) and key not in self.configs.stringopts:
      value = (value,)
    super()._addparentfallback(key, value)
    
  def _superparentsub(self, subparent):
    # Inherit from submanager of supermanager's parent
    assert self.name.endswith(subparent.name)
    self.global_settings = subparent.global_settings | self.global_settings
    for key in subparent.rules:
      if key not in self.fallbacks:
        if key not in self.rules:
          self.rules[key] = []
        self.rules[key].extend(subparent.rules[key])
    self.fallbacks = subparent.fallbacks | self.fallbacks
    for sub in set(subparent.submans) & set(self.configs.submans):
      # TODO is there an appropriate way to figure out which of these should
      # be excluded?
      if sub not in self.submans:
        self._initsub(sub, {})
      self.submans[sub]._superparentsub(subparent.submans[sub])

  def _addrule(self, allnames, conditions, value):
    # Add rule that may belong to self or sub-manager further down the chain
    if len(allnames) == 1:
      key, = allnames
      if key in self.configs.stringopts or not isinstance(value[-1],str):
        value = value[-1]
      if key not in self.rules:
        self.rules[key] = []
      self.rules[key].insert(0,(conditions,value))
    else:
      subname,*subkeys = allnames
      assert subname in self.configs.submans
      if subname not in self.submans:
        self._initsub(subname,{})
      self.submans[subname]._addrule(subkeys, conditions, value)

  def _complete_init(self):
    # Finish initialization from instruction by super
    names = self.name.split('.')
    for rule in self._rulelist[::-1]:
      conditions, settings = self._transform_rule(rule)
      logger.log(7,'Rule set found mapping %s -> %s',conditions,settings)
      for key in settings:
        *subnames,subkey = key.split('.')
        condnew = []
        for cond in conditions:
          if isinstance(cond,str):
            condnew.append((*names,*subnames,cond))
          else:
            stagekey,val = cond
            condnew.append(((*subnames,stagekey),val))
        # Wrap value: originated from self, undotted means own setting (TODO is this the correct wrapping?)
        value = (*names,*subnames,settings[key])
        self._addrule((*subnames,subkey),condnew,value)
    for sub in self.submans:
      self.submans[sub]._complete_init()


class ConfigLoader:
  """Load config settings from file (table with default)"""
  def __init__(self, filename, clsname):
    # TODO paths
    self.stage_variables = set()
    self.system_variables = set()
    self.stringopts = set()
    self.submans = {}
    self.xsub = set()
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
      if not table.get('sub-override',default=False):
        self.xsub = set(table.get('sub-exclude',default=[]))
        self.submans.update({k:(cls,conf,dict(d)) \
          for k,(cls,conf,d) in parent.submans.items() if k not in self.xsub})
      else:
        self.xsub = set(parent.submans)
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
    if 'sub-mans' in table:
      for sub in table['sub-mans']:
        if isinstance(table['sub-mans'][sub], str):
          sub_cls = table['sub-mans'][sub]
          subfile = self.filename
        else:
          sub_cls,subfile = table['sub-mans'][sub]
        if sub in self.submans:
          # Inherited, should check consistency & allow sub-keys to inherit
          assert sub_cls == self.submans[sub][0]
          assert subfile == self.submans[sub][1].filename
        else:
          self.submans[sub] = (sub_cls,ConfigLoader(subfile,sub_cls),{})
    if 'sub-keys' in table:
      for sub in table['sub-keys']:
        if sub not in self.submans:
          raise KeyError('Key mapping provided for previously unspecified '
            'sub-manager "%s"'%sub)
        self.submans[sub][2].update(table['sub-keys'][sub])
