import re
import pickle
import time
import functools
import itertools
import collections
from .npstack import np,linalg,safesvd
from .tensors import Tensor,_findrep1,dictproperty
from copy import copy,deepcopy

from . import links, config


class Network:
  # Internal fields:
  # _tlist is list of distinct tensor objects
  # _tdict is dictionary of tensor names to index in _tlist
  # _tset is inverse of _tdict (list of sets corresponding to tensor object)
  # _conj is dictionary of tensor names to conjugation (relative to _tlist)
  # _tbonds is a dictionary of tensor names to
  #   dict mapping from index to (name,index) bonded
  # _tout is the same, to output names
  # _tchained is a chained combination of the two
  # _bonded maps from tensor name to all adjacent tensors
  # _out maps from (tensor, index) to index name in output tensor
  # _cache determines whether or not a cached result is kept, and contains
  #   said result if applicable
  # _tree assigns the contraction order

  def __init__(self, tensors, tensordict, conjdict, cached=True):
    self._tlist = list(tensors)
    self._tdict = dict(tensordict)
    self._tset = [set() for t in tensors]
    self._conj = dict(conjdict)
    self._tbonds = {t:{} for t in tensordict}
    self._tout = {t:{} for t in tensordict}
    self._tchained = {}
    for t,i in tensordict.items():
      self._tset[i].add(t)
      self._tchained[t] = collections.ChainMap(self._tbonds[t],self._tout[t])
    self._bonded = {t:set() for t in tensordict}
    self._out = {}
    self._tree = None
    self._cache = cached

  @classmethod
  def network(cls, parsestr, *tensors, compat=True, cached=False):
    """Initialize tensor network, with structure given by parsestr
      & list of tensors
    parsestr will be semicolon-separated list of comma-separated groups
      of names for nodes within network corresponding to that tensor 
      (with * when conjugated)
    optional two final clauses of parsestr:
      list of bonds, of the form A.a-B.b
        may have `inner-product style' subclause of the form
        (A,B)[a,b,c] (indicates A.a-B.a,A.b-B.b,A.c-B.c)
      list of index names in contracted tensor, of the form
        A.l>a,B.l>b
        terminated with ~ for autocompletion
          (all indices not specified should have the same name in output tensor)
        or # for strict completion (no free indices should be left unnamed)
    compat indicates whether to test for compatibility during initialization
      (flag as False if tensors provided may contain indices that cannot
      be contracted)"""
    substrs = parsestr.split(';')
    N = len(tensors)
    if len(substrs) < N:
      raise ValueError('Tensors not all identified')
    elif len(substrs) > N+2:
      raise ValueError('%d too many substrings provided'%(len(substrs)-N-2))
    tdict = {}
    tobjdict = {}
    conj = {}
    for i in range(N):
      for ss in substrs[i].split(','):
        m = re.fullmatch(r'(\w+)(\*?)',ss)
        if not m:
          raise ValueError('Invalid tensor-identification substring %s'%ss)
        t,c = m.groups()
        if t in tdict:
          raise ValueError('Tensor %s repeated'%t)
        tdict[t] = i
        tobjdict[t] = tensors[i]
        conj[t] = bool(c)
    Net = cls(tensors, tdict, conj, cached=bool(cached))
    if len(substrs) > N:
      pstr = substrs[N]
      if not re.fullmatch(r'(?!,)((^|,)(\w+\.\w+-\w+\.\w+|\(\w+,\w+\)'
          '\[\w+(,\w+)*\]))*', pstr):
        raise ValueError('Invalid bond substring \'%s\''%pstr)
      for t0,l0,t1,l1 in re.findall(r'\b(\w+)\.(\w+)-(\w+)\.(\w+)\b',pstr):
        try:
          T0 = tobjdict[t0]
          T1 = tobjdict[t1]
          c = conj[t0]^conj[t1]
          assert l0 not in Net._tbonds[t0] and l1 not in Net._tbonds[t1]
        except KeyError:
          if t0 not in tdict:
            raise KeyError('Tensor %s absent'%t0)
          if t1 not in tdict:
            raise KeyError('Tensor %s absent'%t1)
        except AssertionError:
          if l0 in Net._tbonds[t0]:
            raise ValueError('Index %s.%s repeated'%(t0,l0))
          if l1 in Net._tbonds[t1]:
            raise ValueError('Index %s.%s repeated'%(t1,l1))
        if l0 not in T0:
          raise KeyError('Index %s.%s absent'%(t0,l0))
        if l1 not in T1:
          raise KeyError('Index %s.%s absent'%(t1,l1))
        if compat and not T0._dspace[l0].cmp(T1._dspace[l1], not c):
          raise ValueError('Indices %s.%s, %s.%s cannot be contracted' \
              % (t0,l0,t1,l1))
        Net._tbonds[t0][l0] = (t1,l1)
        Net._tbonds[t1][l1] = (t0,l0)
        Net._bonded[t0].add(t1)
        Net._bonded[t1].add(t0)
      for t0,t1,ls in re.findall(r'\((\w+),(\w+)\)\[([\w,]+)\]',pstr):
        ls = ls.split(',')
        for t in (t0,t1):
          if t not in tdict:
            raise KeyError('Tensor %s absent'%t)
        c = conj[t0]^conj[t1]
        T0 = tobjdict[t0]
        T1 = tobjdict[t1]
        for ll in ls:
          for t in (t0,t1):
            if ll not in tobjdict[t]:
              raise KeyError('Index %s.%s absent'%(t,ll))
            if ll in Net._tbonds[t]:
              raise ValueError('Index %s.%s repeated'%(t,ll))
          if compat and not T0._dspace[ll].cmp(T1._dspace[ll], not c):
            raise ValueError('Index %s of %s and %s cannot be contracted' \
              % (ll,t0,t1))
          Net._tbonds[t0][ll] = (t1,ll)
          Net._tbonds[t1][ll] = (t0,ll)
        Net._bonded[t0].add(t1)
        Net._bonded[t1].add(t0)
    if len(substrs) == N+2:
      pstr = substrs[-1]
      if not re.fullmatch(r'(?!,)((,|^)\w+\.\w+\>\w+)*(,?(\~|\#))?',pstr):
        raise ValueError('Invalid output substring \'%s\''%pstr)
      if pstr and pstr[-1] == '~':
        auto,strict = True,False
      elif pstr and pstr[-1] == '#':
        auto,strict = False,True
      else:
        auto = strict = False
      for t,l0,l1 in re.findall(r'\b(\w+)\.(\w+)\>(\w+)\b',pstr):
        if t not in tdict:
          raise KeyError('Tensor %s absent'%t)
        if l0 not in tobjdict[t]:
          raise KeyError('Tensor %s does not have index %s'%(t,l0))
        if l0 in Net._tbonds[t]:
          raise ValueError('Bonded index %s.%s assigned output name'%(t,l0))
        if l0 in Net._tout[t]:
          raise ValueError('Index %s.%s assigned multiple output names'%(t,l0))
        if l1 in Net._out.values():
          raise ValueError('Output index name %s already assigned'%l1)
        Net._tout[t][l0] = Net._out[t,l0] = l1
      if auto:
        for t,l in Net.freeindices(True):
          if l in Net._out.values():
            raise ValueError('Auto-completed index %s already assigned'%l)
          Net._out[t,l] = Net._tout[t][l] = l
      elif strict:
        for t,l in Net.freeindices(True):
          raise ValueError('Index %s.%s missing with strict assignment'%(t,l))
    return Net

  def freeindices(self, unsetonly=False):
    """Return generator listing unbonded (tensor, index) pairs
    If unsetonly is True, exclude indices set for output"""
    if unsetonly:
      for t,ii in self._tdict.items():
        for l in self._tlist[ii].idxset - set(self._tchained[t].keys()):
          yield t,l
    else:
      for t,ii in self._tdict.items():
        for l in self._tlist[ii].idxset - set(self._tbonds[t].keys()):
          yield t,l

  @property
  def tensors(self):
    """Return list of tensors"""
    return list(self._tdict.keys())

  def __contains__(self, key):
    """tensor belongs to network"""
    return key in self._tdict

  def __getitem__(self, key):
    """tensor corresponding to 't' (returns copy if conjugated)"""
    T = self._tlist[self._tdict[key]]
    if self._conj[key]:
      return T.conj()
    return T

  def __setitem__(self, key, t):
    """set tensor corresponding to key
    if Tensor object is provided, add new tensor and set as tensor
    if string is provided (existing tensor with optional *) set to be the
      same tensor as specified"""
    if isinstance(t, Tensor):
      self.replace(key, t)
    else:
      self.updateto(key, t)

  def _set_bonds(self, bdict):
    # Update bonds with dictionary as in self._tbonds
    for t in bdict:
      self._tbonds[t].update(bdict[t])
      for t1,l1 in bdict[t].values():
        self._bonded[t].add(t1)

  def _set_output(self, odict):
    # Update output index names with dictionary as in self._tout
    for t in odict:
      self._tout[t].update(odict[t])
      for l0,l1 in odict[t].items():
        self._out[t,l0] = l1

  def setorder(self, parsestr):
    """Set tree determining contraction order
    Use matched brackets ( (),[],{} ) to denote branches and commas to
    separated branches or leaves"""
    if not re.fullmatch(r'[()\[\]{}\w,]+', parsestr):
      raise ValueError('Invalid argument \'%s\'' % parsestr)
    elements = re.findall(r'[\(\)\[\]{},]|\w+', parsestr)
    if ''.join(elements) != parsestr:
      raise ValueError('Unrecognized element in argument of \'%s\''%parsestr)
    nesting = []
    stack = [nesting]
    current = nesting
    stacksymb = []
    for el in elements:
      if el in '([{':
        current = []
        # Add new branch
        stack[-1].append(current)
        stack.append(current)
        stacksymb.append(el)
      elif el in ')]}':
        # Check that brackets are paired correctly
        if not stacksymb or stacksymb.pop()+el not in {'()','[]','{}'}:
          raise ValueError('Unpaired bracket in \'%s\''%parsestr)
        # Remove branch
        stack.pop()
        current = stack[-1]
      elif el == ',':
        pass
      else:
        # Add element to most recent branch
        current.append(el)
    if stacksymb:
      raise ValueError('Unpaired bracket in \'%s\''%parsestr)
    self._tree = NetworkTree.setfromlist(nesting)

  @property
  def tree(self):
    return self._tree

  @property
  def expense(self):
    return self.contraction_expense(0)

  def uncache(self):
    """Delete cache, if applicable"""
    if isinstance(self._cache, Tensor):
      self._cache = True

  def updateto(self, t0, t1, c=False, rel=False):
    """Update tensor t0 by identifying as t1
    conjugation data may optionally be provided by argument c, or with * in t1
      rel is relative to t0; automatically relative to t1
    """
    self.uncache()
    if not isinstance(t1,str):
      raise ValueError('t1 must identify tensor')
    if not re.fullmatch(r'\w+\*?', t1):
      raise ValueError('\'%s\' invalid assigment for tensor in network'%t1)
    if not isinstance(t0,str):
      raise ValueError('t0 must identify tensors')
    if t0 not in self._tdict:
      if not re.fullmatch(r'\w+',t0) and \
          re.fullmatch(r'(\w+\*?(,|$))*(?<!,)',t0):
        for t,c0 in re.findall(r'(\w+)(\*?)',t0):
          #print('\n',t,t1,self._conj[t],self._conj[t1],c0,bool(c0)^c)
          self.updateto(t,t1,c=bool(c0)^c,rel=rel)
        return
      else:
        raise KeyError('tensor \'%s\' does not belong to network'%t0)
    if t1[-1] == '*':
      c = not c
      t1 = t1[:-1]
    if rel:
      c ^= self._conj[t0]
    if t1 not in self._tdict:
      raise ValueError('Assigned tensor \'%s\' absent'%t1)
    i0 = self._tdict[t0]
    i1 = self._tdict[t1]
    if i0 == i1:
      if c^self._conj[t0]^self._conj[t1]:
        self._tlist[i0].matches(self._tlist[i0],True)
        self._conj[t0] = not self._conj[t0]
      return
    self._tlist[i0].matches(self._tlist[i1],c^self._conj[t0]^self._conj[t1])
    self._updateto(t0, i1, c^self._conj[t1])

  def _updateto(self, t, idx, c):
    """Change tensor identified by t to tensor at idx,
    with conjugation data c"""
    i0 = self._tdict[t]
    if len(self._tset[i0]) == 1:
      if idx > i0:
        idx -= 1
      for s in self._tset[i0+1:]:
        for ta in s:
          self._tdict[ta] -= 1
      self._tset.pop(i0)
      self._tlist.pop(i0)
    else:
      self._tset[i0].remove(t)
    self._tdict[t] = idx
    self._tset[idx].add(t)
    self._conj[t] = c

  def replace(self, t, T, c=False, rel=False):
    """Change tensor(s) indicated by t to T,
      optional conjugacy c (otherwise not conjugated)
      if rel, c is relative to current conjugation (False by default)"""
    self.uncache()
    if not t in self._tdict:
      if re.fullmatch(r'(\w+\*?(,|$))+(?<!,)',t):
        ts = re.findall(r'(\w+)(\*)?',t)
        t0,c00 = ts[0]
        c0 = bool(c00)
        # Previously failed to catch nonexistent index
        if t0 not in self._tdict:
          raise KeyError(f'index {t0} referenced not found in network')
        self.replace(t0, T, c^c0, rel)
        for t1,c1 in ts[1:]:
          raise KeyError(f'index {t1} referenced not found in network')
          try:
            self.updateto(t1,t0,c=c^c0^bool(c1),rel=rel)
          except:
            raise ValueError('Cannot update %s%s to %s%s'%(t1,c1,t0,c00))
        return
      raise KeyError('Tensor %s missing'%t)
    i0 = self._tdict[t]
    if rel:
      c ^= self._conj[t]
    self._tlist[i0].matches(T, c ^ self._conj[t], strict=False)
    self._conj[t] = c
    if len(self._tset[i0]) == 1:
      # Full replacement
      self._tlist[i0] = T
    else:
      # New, distinct tensor
      i1 = len(self._tlist)
      self._tlist.append(T)
      self._tdict[t] = i1
      self._tset.append({t})
      self._tset[i0].remove(t)

  def updateas(self, t0, T, c=False, rel=False):
    """Update tensor t0 by identifying as tensor T,
    adding if T is *not* already represented in self"""
    self.uncache()
    if t0 not in self._tdict:
      raise KeyError('Tensor %s missing'%t0)
    if rel:
      c ^= self._conj[t]
    T.matches(self._tlist[self._tdict], c^self._conj[t0])
    for ii in range(len(self._tlist)):
      if self._tlist[ii] is T:
        if ii == self._tdict[t0]:
          if c is not None and c != self._conj[t0]:
            self._conj[t0] = c
          return
        self._updateto(t, ii, c)
        return
    self.replace(t, T, c)

  def replaceall(self, parsestr, *tensors, strict=False):
    """Replace tensor indicated by parsestr with arguments, for every node
      that points to that tensor
    if strict is True, compares exactly with vector spaces of current tensor;
      if False, only raise error if a conflict is caused"""
    self.uncache()
    if not re.fullmatch(r'(?!,)((,|^)\w+\*?)*', parsestr):
      raise ValueError('Invalid tensor list \'%s\''%parsestr)
    ts = parsestr.split(',')
    nten = len(tensors)
    if len(ts) != nten:
      raise ValueError('%d tensors provided versus %d substrings' \
        % (len(ts),nten))
    idxold = [] # Index in original list
    conjs = [] # Conjugation relative to said list
    for ii in range(nten):
      T = tensors[ii]
      if ts[ii][-1] == '*':
        t0 = ts[ii][:-1]
        conj = True
      else:
        t0 = ts[ii]
        conj = False
      if t0 not in self._tdict:
        raise KeyError('name %s not a tensor in network'%t0)
      idx = self._tdict[t0]
      if idx in idxold:
        ifirst = idxold.index(idx)
        raise ValueError('Tensor identified by %s already referenced as %s' \
          % (t0, ts[ifirst]))
      idxold.append(idx)
      conj ^= self._conj[t0]
      if strict:
        # Just check against existing tensor
        self._tlist[idx].matches(T, conj)
      else:
        # Start by checking that all previously-contracted indices are still
        # there
        contracted = set()
        for t1 in self._tset[idx]:
          contracted |= self._tbonds[t1].keys()
        if contracted - T.idxset:
          for t1 in self._tset[idx]:
            ls = set(self._tbonds[t1].keys()) - T.idxset
            if ls:
              l0 = ls.pop()
              tb,lb = self._tbonds[t1][l0]
              raise KeyError(f'Tensor, identified as {t0}, lacking index '
                f'{t1}.{l0} contracted with {tb}.{lb}')
          assert False
      conjs.append(conj)
    if not strict:
      # Test bonds individually
      for ii in range(nten):
        T = tensors[ii]
        ti = idxold[ii]
        for l in T._idxs:
          matchold = False # Contracts with unchanged tensor so must be
                           # compatible with self
          matchnew = set() # (new idx, contracted index name, conjugation)
                           # for indices contracted with tensors being changed
          for t0 in self._tset[ti]:
            if l in self._tbonds[t0]:
              t1,l1 = self._tbonds[t0][l]
              i1 = self._tdict[t1]
              if i1 in idxold:
                if i1 < ti or (i1 == ti and l == l1 and conjs[ii]):
                  # Will have already been checked
                  continue
                c = self._conj[t0]^self._conj[t1]^conjs[idxold.index(i1)]
                matchnew.add((i1, l1, c))
              else:
                matchold = True
          if matchold:
            if not T._dspace[l].cmp(self._tlist[ti]._dspace[l], conjs[ii]):
              raise ValueError('Index %s.%s changed improperly'%(ts[ti],l))
          for i1,l1,c in matchnew:
            tinew = idxold.index(i1)
            if not T._dspace[l].cmp(tensors[tinew]._dspace[l1], not c):
              # Retrieve t1
              t1 = self._tbonds[t0][l][0]
              raise ValueError('Bond %s.%s-%s.%s changed incompatibly' \
                % (t0,l,t1,l1))
    for ii in range(nten):
      # Change the network
      ti = idxold[ii]
      self._tlist[ti] = tensors[ii]
      if conjs[ii]:
        for t in self._tset[ti]:
          self._conj[t] = not self._conj[t]

  def derive(self, parsestr, *tensorsadd):
    """Derive a new network from self
    Clauses of parsestr (semicolon-separated) may include:
      |T0.l0,T1.l1,^T2|a,b - mirroring a network
        subclause T0.l0 indicates contraction of index
          with mirrored version of itself
        subclause ^T2 indicates that T2 is not mirrored
        alternatively |~| indicates all free indices are paired with each other
        final clause a,b indicates original, new nodes will add
          'a' and 'b' to their names respectively
          (only applies to nodes that are actually mirrored)
        end with >~ to update output index name T0.l0>l to T0a.l0>la,T0b.l0>lb
          (requiring that all mirrored free indices are assigned as output)
      +T1,T2*,T3 - add next tensor argument with node names as indicated
      -T1,T2,T3 - remove nodes indicated
      T0.l0-T1.l1,T2.l3>l, etc - add bonds and output index names
      conclude with # (strict) or ~ (autocomplete) as in Network.network()
    Returns new network"""
    if not isinstance(parsestr,str):
      raise ValueError('First argument must be string')
    if not all(isinstance(T, Tensor) for T in tensorsadd):
      raise ValueError('All subsequent arguments must be tensors')
    auto = strict = False
    m = re.search(r';?((?<!\>)\~|\#)$', parsestr)
    if m:
      endclause = m.group(0)
      parsestr = parsestr[:-len(endclause)]
      if endclause[-1] == '~':
        auto = True
      else:
        strict = True
    tlist = list(self._tlist)
    tensors = dict(self._tdict)
    tidx = 0
    conj = dict(self._conj)
    out = deepcopy(self._tout)
    bonds = deepcopy(self._tbonds)

    for clause in parsestr.split(';'):
      if re.fullmatch(r'\|(?!,)(((,|(?<=\|))\w+\.\w+|\^\w+)*|\~)\|\w*,\w*(\>\~)?', clause):
        # Mirror network
        if clause[:3] == '|~|':
          # Automatically double
          exclude = set()
          pairs = list(self.freeindices())
        else:
          pairs = re.findall(r'\b(\w+)\.(\w+)\b', clause)
          exclude = set(re.findall(r'\^(\w+)\b', clause))
        tnl,tnr = re.search(r'\|(\w*),(\w*)\b', clause).groups()
        if tnl == tnr:
          raise ValueError('Left and right name modifications must be distinct')
        autocomplete = (clause[-2:] == '>~')
        # Find which tensors to mirror
        mirrored = set(tensors.keys()) - exclude
        if len(mirrored) + len(exclude) > len(tensors):
          raise KeyError('Excluded tensor %s already absent' \
            % list(exclude-set(tensors.keys()))[0])
        if tnl in tnr:
          nr = len(tnr)-len(tnl)
          if tnr[nr:] == tnl:
            diff = tnr[:nr]
            for t in mirrored:
              if t[-nr:] == diff and t[:-nr] in mirrored:
                raise ValueError('Overlapping names of mirrored nodes '
                  '%s and %s' % (t,t[:-nr]))
        elif tnr in tnl:
          nl = len(tnl)-len(tnr)
          if tnl[nl:] == tnl:
            diff = tnl[:nl]
            for t in mirrored:
              if t[-nl:] == diff and t[:-nl] in mirrored:
                raise ValueError('Overlapping names of mirrored nodes '
                  '%s and %s' % (t,t[:-nl]))
        newtens = {}
        newconj = {}
        newbonds = {}
        newout = {}
        for t in exclude:
          newtens[t] = tensors[t]
          newconj[t] = conj[t]
          newbonds[t] = bonds[t]
          if autocomplete:
            newout[t] = out[t]
          else:
            newout[t] = {}
          # Correct bonds where necessary
          for l in bonds[t]:
            t1,l1 = bonds[t][l]
            if t1 in mirrored:
              newbonds[t][l] = (t1+tnl,l1)
        paired = collections.defaultdict(set)
        for t,l in pairs:
          if t not in mirrored:
            if t in exclude:
              raise ValueError('Self-paired index %s.%s belongs to tensor'
                ' excluded from mirroring'%(t,l))
            else:
              raise KeyError('Tensor %s missing'%t)
          if l in bonds[t]:
            raise ValueError('Self-paired index %s.%s already bonded'%(t,l))
          paired[t].add(l)
        for t in mirrored:
          tl = t+tnl
          tr = t+tnr
          if tl in exclude:
            raise ValueError('Mirrored node %s overlaps with name %s'%(tl,t))
          if tr in exclude:
            raise ValueError('Mirrored node %s overlaps with name %s'%(tr,t))
          newtens[tl] = newtens[tr] = tensors[t]
          newconj[tl] = conj[t]
          newconj[tr] = not conj[t]
          newbonds[tl] = {}
          newbonds[tr] = {}
          newout[tl] = {}
          newout[tr] = {}
          for l in bonds[t]:
            t1,l1 = bonds[t][l]
            if t1 is not None:
              if t1 in exclude:
                newbonds[tl][l] = (t1,l1)
              else:
                newbonds[tl][l] = (t1+tnl,l1)
                newbonds[tr][l] = (t1+tnr,l1)
            elif autocomplete:
              newout[tl,l] = l1+tnl
              newout[tr,l] = l1+tnr
          for l in paired[t]:
            newbonds[tl][l] = (tr,l)
            newbonds[tr][l] = (tl,l)
        tensors = newtens
        conj = newconj
        bonds = newbonds
        out = newout
      elif re.fullmatch(r'\+(?!,)((,|(?<=\+))\w+\*?)+(\:\w+)?',clause):
        # Add node(s)
        m = re.search('\:(\w+)$',clause)
        if m:
          # From existing node
          t0 = m.group(1)
          if t0 not in tensors:
            raise ValueError('Reference node %s does not exist'%t0)
          idx = tensors[t0]
          c0 = conj[t0]
        else:
          # Tensor from argument list
          if len(tensorsadd) == tidx:
            raise ValueError('%d tensors given as argument, expected %d' \
              % (tidx, len(re.findall('\\+', parsestr))))
          c0 = False
          idx = len(tlist)
          tlist.append(tensorsadd[tidx])
          tidx += 1
        for t,c in re.findall(r'(?<=,|\+)(\w+)\b(\*)?',clause):
          if t in tensors:
            raise ValueError('Tensor name %s duplicated'%t)
          tensors[t] = idx
          conj[t] = c0 ^ bool(c)
          bonds[t] = {}
          out[t] = {}
      elif re.fullmatch(r'-\w+(,\w+)*',clause):
        # Remove node(s)
        for t in re.findall(r'\b\w+\b',clause):
          if t not in tensors:
            raise ValueError('Removed node %s does not exist'%t)
          it = tensors[t]
          del tensors[t]
          del conj[t]
          bs = bonds.pop(t)
          for t1,l1 in bs.values():
            del bonds[t1][l1]
          del out[t]
      elif re.fullmatch(r'(?!,)((,|^)\w+\.\w+(-\w+\.\w+|\>\w+))*',clause):
        # Add bonds & name outputs
        for t0,l0,t1,l1 in re.findall(r'\b(\w+)\.(\w+)-(\w+)\.(\w+)\b',clause):
          for t,l in [(t0,l0),(t1,l1)]:
            if t not in tensors:
              raise KeyError('Bonded tensor %s missing'%t)
            if l not in tlist[tensors[t]]:
              raise KeyError('Bonded index %s.%s missing'%(t,l))
            if l in bonds[t]:
              raise ValueError('Index %s.%s already bonded'%(t,l))
            out[t].pop(l,None) # Remove if in output
          bonds[t0][l0] = (t1,l1)
          bonds[t1][l1] = (t0,l0)
        for t,l0,l1 in re.findall(r'\b(\w+)\.(\w+)\>(\w+)\b', clause):
          if t not in tensors:
            raise KeyError('Tensor %s missing'%t)
          if l0 not in tlist[tensors[t]]:
            raise KeyError('Index %s.%s missing'%(t,l0))
          if l0 in bonds[t]:
            raise ValueError('Index %s.%s already bonded'%(t,l0))
          out[t][l0] = l1
      else:
        raise ValueError('Invalid clause \'%s\''%clause)
    # Verify that output names are not duplicated
    # (should be the only issue possible at this point)
    # Initialize Network object
    # Check that all tensors listed remain
    shifted = list(range(len(tlist)))
    vset = set(tensors.values())
    if len(vset) < len(tlist):
      tmap = sorted(vset)
      rmap = {tmap[i]:i for i in range(len(tmap))}
      for t in tensors:
        tensors[t] = rmap[tensors[t]]
      tlist = [tlist[tmap[i]] for i in range(len(tmap))]
    net = Network(tlist, tensors, conj)
    net._set_bonds(bonds)
    net._set_output(out)
    if auto:
      for t,ll in net.freeindices(True):
        net._tout[t][ll] = ll
        net._out[t,ll] = ll
    if strict:
      try:
        tl = next(net.freeindices(True))
        raise ValueError('Free index %s.%s unaccounted for'%tl)
      except StopIteration:
        pass
    if len(set(net._out.values())) < len(net._out):
      raise ValueError('Output index name %s repeated'%_findrep1(net._out.values()))
    return net

  def subnetwork(self, minus, out=None, cached=False):
    """Returns subnetwork with tensors listed (iterable or comma-separated)
    removed
    May specify additional output index names by:
      using a second clause of the form t0.l0->l,..., (or
        by setting out to be a dictionary from (node, index) to new name)
      and, alone or in combination with a second clause,
      out='reuse': original output names are used
      out='env': environment mode - name of contracted index
      out='auto' (or ~ flag): index name is kept as-is"""
    # TODO additional out argument for environment bonds?
    if isinstance(minus, str):
      if not re.fullmatch(r'\w+(,\w+)*(;(\w+\.\w+\>\w+(,|$|\~$))+\~?(?<!,))?'
          ,minus):
        raise ValueError('Invalid argument \'%s\''%minus)
      if minus[-1] == '~':
        if minus[-2] == ',':
          minus = minus[:-2]
        else:
          minus = minus[:-1]
        mode = 'auto'
        if not isinstance(out,dict):
          out = {}
      elif out == 'auto':
        mode = 'auto'
        out = {}
      elif out in ['env','reuse']:
        mode = out
        out = {}
      else:
        mode = None
        if not out:
          out = {}
        elif not isinstance(out,dict):
          raise ValueError('out must be dictionary or valid mode')
        auto = False
      mo = minus.split(';')
      if len(mo) == 2:
        minus,outstr = mo
        out = {}
        for t0,l0,l1 in re.findall('(?:^|,)(\w+)\.(\w+)\>(\w+)(?=,|$)',outstr):
          out[t0,l0] = l1
      else:
        minus = mo[0]
      minus = minus.split(',')
    else:
      mode = None
    for t in minus:
      if t not in self._tdict:
        raise KeyError('Node %s missing'%t)
    sminus = set(minus)
    if len(sminus) < len(minus):
      raise ValueError('Excluded node %s repeated'%_findrep1(minus))
    for t,l in out:
      if t not in self._tdict:
        raise KeyError('Node %s missing'%t)
      if l not in self._tlist[self._tdict[t]]:
        raise KeyError('Index %s.%s missing'%(t,l))
    if mode == 'env' or mode == 'reuse':
      # First employ existing output names
      for t,l in self._out:
        if t not in sminus and l not in self._tbonds[t]:
          out[t,l] = self._out[t,l]
    if mode == 'env':
      # Now add names corresponding to removed tensors
      for t in sminus:
        for l in self._tbonds[t]:
          t1,l1 = self._tbonds[t][l]
          if t1 not in sminus:
            out[t1,l1] = l
    if mode == 'auto':
      for t in set(self._tdict.keys())-sminus:
        tb = self._tbonds[t]
        for l in self._tlist[self._tdict[t]].idxset:
          if (l not in tb or tb[l][0] in sminus) and (t,l) not in out:
            out[t,l] = l
    if len(set(out.values())) > len(out):
      raise ValueError('Output index name %s duplicated'%_findrep1(out))
    net = SubnetworkView(self, sminus, out, bool(cached))
    if self._tree is not None and not any(self.freeindices()):
      net._tree = self._tree.subnettree(sminus)
    return net

  def contract(self, setout={}, fulloverride=False, auto=False):
    """Contract tensors
    Optional setout indicates index names of output
      (either dictionary (t,l0)->l or string 't.l0>l,...')
    fulloverride indicates that all free indices must be renamed
      (also indicated by '#' at end of string setout)
    auto indicates that any index t.l not specified in setout retains name
      (also indicated by '~' at end of setout)"""
    if not setout and isinstance(self._cache, Tensor):
      return self._cache
    if isinstance(setout,str):
      if not re.fullmatch(r'(\w+\.\w+\>\w+(,|$))(\~|\#)?(?!<,)',setout):
        raise ValueError('Invalid argument \'%s\''%setout)
      fulloverride = ('#' in setout)
      auto = ('~' in setout)
      setout = {(t,l0):l for t,l0,l in \
        re.findall(r'\b(\w+)\.(\w+)\>(\w+)\b',setout)}
    if len(self._tdict) > 1:
      if self._tree is None:
        self.optimize()
      T,leaves,contract,c = self._tree_contract(self._tree)
      assert len(contract) == 0
      assert leaves == self._tdict.keys()
    else:
      # TODO do more explicitly
      t = next(iter(self._tdict))
      T = self._tlist[self._tdict[t]]
      T = T.renamed({l:t+'.'+l for l in T.idxset})
      c = self._conj[t]
    if c:
      T = T.conj()
    if not isinstance(T,Tensor):
      return T
    idxmap = {}
    if setout:
      iset = T.idxset
      for t,l0 in setout:
        tl = '%s.%s'%(t,l0)
        if tl not in iset:
          raise KeyError('Tensor.index %s does not represent a free index')
        idxmap[tl] = setout[t,l0]
        iset.remove(tl)
      if iset:
        if fulloverride:
          raise ValueError('Free index %s not accounted for'%tl)
        elif auto:
          for tl in iset:
            idxmap[tl] = tl.split('.')[1]
    else:
      for tl in T._idxs:
        t,l = tl.split('.')
        if (t,l) in self._out:
          idxmap[tl] = self._out[t,l]
        elif auto:
          idxmap[tl] = l
        else:
          raise ValueError('Free index %s not accounted for'%tl)
    if len(set(idxmap.values())) < len(idxmap):
      raise ValueError('output index %s repeated'%_findrep1(idxmap))
    T._rename_dynamic(idxmap)
    if self._cache:
      self._cache = T
    return T

  def _tree_contract(self, branch):
    # Contract branch of tree
    # Return contracted tensor; leaves included; indices to be contracted;
    #   conjugated or not
    lbranch = branch.left
    rbranch = branch.right
    if isinstance(lbranch, str):
      # Leaf
      lleaves = {lbranch}
      cl = self._conj[lbranch]
      Tl,lcontract = self._tensor_prep(lbranch)
    else:
      Tl,lleaves,lcontract,cl = self._tree_contract(lbranch)
    if isinstance(rbranch, str):
      # Leaf
      rleaves = {rbranch}
      cr = self._conj[rbranch]
      Tr,rcontract = self._tensor_prep(rbranch)
    else:
      Tr,rleaves,rcontract,cr = self._tree_contract(rbranch)
    outl = set(Tl._idxs)
    outr = set(Tr._idxs)
    # Find indices to be contracted
    contracted = []
    for tr in rleaves:
      if tr in lcontract:
        for ll,lr in lcontract.pop(tr):
          lr1 = '%s.%s'%(tr,lr)
          contracted.append((ll,lr1))
          outl.remove(ll)
          outr.remove(lr1)
    outl = {l:l for l in outl}
    outr = {l:l for l in outr}
    #print(Tl._idxs,Tl._T.shape,Tr._idxs,Tr._T.shape,contracted)
    Tc = Tl._do_contract(Tr, contracted, outl, outr, cl^cr)
    del Tl,Tr
    for tt in rcontract:
      if tt not in lleaves:
        lcontract[tt] |= rcontract[tt]
    return Tc, lleaves|rleaves, lcontract, cl

  def _tensor_prep(self, t): 
    contract = collections.defaultdict(set)
    bonds = self._tbonds[t]
    traceidx = set()
    tracel = []
    T = self._tlist[self._tdict[t]]
    for l0 in bonds:
      t1,l1 = bonds[l0]
      if t1 == t:
        if l0 not in traceidx:
          tracel.append((l0,l1))
          if not (T._dspace[l0] ^ T._dspace[l1]):
            raise ValueError('Cannot trace indices %s.(%s,%s) within network'\
              %(t,l0,l1))
        traceidx |= {l0,l1}
      else:
        contract[t1].add(('%s.%s'%(t,l0),l1))
    if traceidx:
      outxs = set(T._idxs)-traceidx
      Tx = T._do_trace(tracel, {l0:'%s.%s'%(t,l0) for l0 in outxs})
    else:
      Tx = T.renamed({l0:'%s.%s'%(t,l0) for l0 in T._idxs})
    return Tx,contract

  def __copy__(self):
    cpy = self.__class__(self._tlist, self._tdict, self._conj)
    for t in self._tdict:
      cpy._tout[t].update(self._tout[t])
      cpy._tbonds[t].update(self._tbonds[t])
    cpy._out = dict(self._out)
    cpy._bonded = deepcopy(self._bonded)
    cpy._tree = copy(self._tree)
    return cpy

  def __deepcopy__(self):
    cpy = self.__copy__()
    for i in range(len(self._tlist)):
      cpy._tlist[i] =  copy(self._tlist[i])

  def conj(self):
    cpy = self.__copy__()
    for t,c in self._conj.items():
      cpy._conj[t] = not c
    return cpy

  def optimize(self):
    """Optimize contraction order of network.
    Currently uses simple (which is to say non-optimized) algorithm"""
    bonddim = collections.defaultdict(lambda: 1)
    for t,ti in self._tdict.items():
      T = self._tlist[ti]
      bdict = self._tbonds[t]
      for l in T._idxs:
        if l in bdict:
          bonddim[t,bdict[l][0]] *= T.shape[l]
        else:
          bonddim[t] *= T.shape[l]
    tensors = set(self._tdict.keys())
    if len(tensors) == 1:
      return
    tenl = tuple(sorted(tensors))
    if config.memcap:
      memcap = int(config.memcap//config.memratio)
    else:
      memcap = None
    stepexp = []
    resdict = {}
    for i in range(1,len(self._tdict)//2+1):
      half = (2*i == len(self._tdict))
      for sub in itertools.combinations(tenl,i):
        ssub = set(sub)
        scomp = tensors - ssub
        comp = tuple(sorted(scomp))
        assert isinstance(sub,tuple) and isinstance(comp,tuple)
        if half and comp < sub:
          continue
        lfree = 1
        rfree = 1
        contracted = 1
        for t0 in ssub:
          lfree *= bonddim[t0]
          for t1 in self._bonded[t0] & scomp:
            contracted *= bonddim[t0,t1]
        for t0 in scomp:
          rfree *= bonddim[t0]
        #print(np.log(contracted)/np.log(2),np.log(lfree)/np.log(2),np.log(rfree)/np.log(2))
        if memcap:
          if contracted > memcap:
            continue
          elif lfree*contracted > memcap:
            if rfree*contracted <= memcap:
              resdict[comp] = None
            continue
          elif rfree*contracted > memcap:
            resdict[sub] = None
            continue
        stepexp.append((sub,comp,lfree*rfree*contracted))
        resdict[comp] = None
        resdict[sub] = None
    expsort = sorted(stepexp,key=lambda it:it[2])
    try:
      exp, tree = self.__optimize_child_simple(tenl,bonddim,resdict,expsort)
    except OptimizationOverflow:
      raise ValueError('Network optimization could not be found within memory'
        ' limit %0.2fGiB'%(np.dtype(config.FIELD).itemsize*memcap/2**30))
    mem = self.memexpense(tree,reorder=True)
    self._tree = tree
    if config.opt_verbose >= 1:
      # TODO work into logging
      print(f'Optimized with {exp:.0e} steps, memory',end=' ')
      memk = mem/2**14
      if memk < 1000:
        print(f'{round(memk):d}KB')
      else:
        memm = memk/2**10
        if memm < 1000:
          print(f'{memm:0.1f}MB')
        elif memm < 2**20:
          print(f'{memm/2**10:0.1f}GB')
        else:
          print(f'{memm/2**20:0.1f}TB (!!!)')
      tree.pprint()
    
  def __optimize_child_simple(self, tlist, bonddim, resdict, expmin):
    assert isinstance(tlist,tuple)
    tensors = set(tlist)
    tall = set(self._tdict.keys())
    if len(tlist) == 2:
      # Quick calculation of contraction expense
      x = bonddim[tlist[0],tlist[1]]
      for t0 in tensors:
        x *= bonddim[t0]
        for t1 in self._bonded[t0] - tensors:
          x *= bonddim[t0,t1]
      return x, NetworkTree(tlist[0],tlist[1])
    elif len(tlist) == 1:
      return 0, tlist[0]
    if isinstance(expmin, list):
      # Parent - sorted expenses provided
      expsort = expmin
      expmin = None
    else:
      # Check if this subnetwork has already been optimized
      r0 = resdict[tlist]
      if r0:
        if isinstance(r0, tuple):
          if expmin and r0[0] > expmin:
            # Max expense exceeded
            raise OptimizationOverflow(expmin)
          return r0
        else:
          # Greatest expense tested
          if expmin is not None and r0 >= expmin:
            raise OptimizationOverflow(expmin)
      # Generate expenses for viable subnetworks sorted
      rank = len(tlist)
      fullcomp = tall - tensors
      stepexp = []
      for i in range(1,rank//2+1):
        half = (2*i == rank)
        for sub in itertools.combinations(tlist,i):
          ssub = set(sub)
          scomp = tensors - ssub
          comp = tuple(sorted(scomp))
          sc2 = tall - ssub
          if half and comp < sub:
            continue
          if sub not in resdict or comp not in resdict:
            continue
          x = 1
          for t0 in ssub:
            x *= bonddim[t0]
            for t1 in self._bonded[t0] & sc2:
              x *= bonddim[t0,t1]
          for t0 in scomp:
            x *= bonddim[t0]
            for t1 in self._bonded[t0] & fullcomp:
              x *= bonddim[t0,t1]
          stepexp.append((sub,comp,x))
      expsort = sorted(stepexp,key=lambda it:it[2])
    tree = None
    for lsub,rsub,exp in expsort:
      assert isinstance(lsub,tuple) and isinstance(rsub,tuple)
      if expmin and exp > expmin:
        break
      try:
        lexp, ltree = self.__optimize_child_simple(lsub, bonddim, resdict,expmin)
        rexp, rtree = self.__optimize_child_simple(rsub, bonddim, resdict,expmin)
      except OptimizationOverflow:
        continue
      exp1 = exp + lexp + rexp
      if expmin is None or exp1 < expmin:
        tree = NetworkTree(ltree, rtree, verify=False)
        expmin = exp1
    if tree is None:
      resdict[tlist] = expmin
      raise OptimizationOverflow(expmin)
    resdict[tlist] = (expmin, tree)
    return expmin, tree

  def contraction_expense(self, tree, root=True):
    """Determine time complexity of contraction order given by NetworkTree"""
    if not tree:
      tree = self.tree
    if isinstance(tree.left, NetworkTree):
      ltens = tree.left.leaves
      lexp,lfree,lcont = self.contraction_expense(tree.left,False)
    else:
      t = tree.left
      T = self._tlist[self._tdict[t]]
      lfree = 1
      lcont = {t1:1 for t1 in self._bonded[t]}
      bonds = self._tbonds[t]
      for l in T._idxs:
        d = T.shape[l]
        if l in bonds:
          lcont[bonds[l][0]] *= d
        else:
          lfree *= d
      ltens = {t}
      lexp = 0
    if isinstance(tree.right, NetworkTree):
      rtens = tree.right.leaves
      rexp,rfree,rcont = self.contraction_expense(tree.right,False)
    else:
      t = tree.right
      T = self._tlist[self._tdict[t]]
      rfree = 1
      rcont = {t1:1 for t1 in self._bonded[t]}
      bonds = self._tbonds[t]
      for l in T._idxs:
        d = T.shape[l]
        if l in bonds:
          rcont[bonds[l][0]] *= d
        else:
          rfree *= d
      rtens = {t}
      rexp = 0
    exp1 = lfree*rfree
    if lcont:
      exp1 *= functools.reduce(int.__mul__, lcont.values())
    for t1 in tuple(rcont.keys()):
      if t1 in ltens:
        rcont.pop(t1)
      else:
        exp1 *= rcont[t1]
    exp = exp1 + lexp + rexp
    if root:
      return exp
    for t1 in lcont.keys() - rtens:
      if t1 in rcont:
        rcont[t1] *= lcont[t1]
    return exp, lfree*rfree, rcont

  def memexpense(self, tree, root=True, reorder=False):
    """Determine spatial complexity of contraction order given by NetworkTree"""
    if not tree:
      tree = self.tree
    if isinstance(tree.left, NetworkTree):
      ltens = tree.left.leaves
      lmem,lfree,lcont = self.memexpense(tree.left,False,reorder)
    else:
      t = tree.left
      T = self._tlist[self._tdict[t]]
      lfree = 1
      lcont = {t1:1 for t1 in self._bonded[t]}
      bonds = self._tbonds[t]
      for l in T._idxs:
        d = int(T.shape[l])
        if l in bonds:
          lcont[bonds[l][0]] *= d
        else:
          lfree *= d
      ltens = {t}
      lmem = 0
    lx = lfree*functools.reduce(int.__mul__, lcont.values())
    if isinstance(tree.right, NetworkTree):
      rtens = tree.right.leaves
      rmem,rfree,rcont = self.memexpense(tree.right,False,reorder)
    else:
      t = tree.right
      T = self._tlist[self._tdict[t]]
      rfree = 1
      rcont = {t1:1 for t1 in self._bonded[t]}
      bonds = self._tbonds[t]
      for l in T._idxs:
        d = int(T.shape[l])
        if l in bonds:
          rcont[bonds[l][0]] *= d
        else:
          rfree *= d
      rtens = {t}
      rmem = 0
    rx = rfree*functools.reduce(int.__mul__, rcont.values())
    if reorder and rmem+lx > lmem+rx:
      tree.left,tree.right = tree.right,tree.left
      tree._ldepth,tree._rdepth = tree._rdepth,tree._ldepth
      mem = lmem+rx
    else:
      mem = rmem+lx
    if root:
      return mem
    cont = {}
    for t1 in (set(lcont.keys()) | set(rcont.keys())) - ltens - rtens:
      cont[t1] = 1
      if t1 in lcont:
        cont[t1] *= lcont[t1]
      if t1 in rcont:
        cont[t1] *= rcont[t1]
    return mem, lfree*rfree, cont


class SubnetworkView(Network):
  """View of a Network based on an existing network, representing a subnetwork
  thereof"""

  def __init__(self, net, sminus, out, cached=False):
    """Create a view of net based on a subnetwork excluding sminus,
    given (additional) output names out"""
    self._parent = net
    self._tensors = set(net._tdict.keys()) - sminus
    self._out = dict(out)
    self._tout = {t:{} for t in self._tensors}
    for t,l in out:
      self._tout[t][l] = out[t,l]
    self._tchained = {}
    self._tbonds = {}
    for t in self._tensors:
      for l in net._tbonds[t]:
        if net._tbonds[t][l][0] in sminus and (t,l) not in out:
          self._tout[t][l] = None
      self._tchained[t] = collections.ChainMap(self._tout[t],net._tbonds[t])
      self._tbonds[t] = dictproperty(self.__bondget,None,self.__hasbondidx,
        self.__bondcopy, t)
    self._tree = None
    self._tlist = net._tlist
    self._tdict = dictproperty(self.__tdictget,self.__tdictset,
      self._tensors, self.__tdictcopy)
    self._conj = dictproperty(self.__conjget, self.__conjset,
      self._tensors, self.__conjcopy)
    self._bonded = dictproperty(self.__bondedget,None,self._tensors,
      self.__bondedcopy)
    self._tset = dictproperty(self.__tsetget,None,self.__tidxs,self.__tsetcopy)
    self._cache = cached

  # When necessary provide internal fields of Network
  def __istensor(self, key):
    return key in self._tensors

  def __tdictcopy(self):
    return {t: self._parent._tdict[t] for t in self._tensors}

  def __tdictget(self, key):
    return self._parent._tdict[key]

  def __tdictset(self, key, value):
    self._parent._tdict[key] = value

  def __conjcopy(self):
    return {t: self._parent._conj[t] for t in self._tensors}

  def __conjget(self, key):
    return self._parent._conj[key]

  def __conjset(self, key, value):
    self._parent._conj[key] = value

  def __hasbondidx(self, l, t):
    return l in self._parent._tbonds[t] \
      and self._parent._tbonds[t][l][0] in self._tensors

  def __bondcopy(self, t):
    td = self._parent._tbonds[t]
    return {l: td[l] for l in td if td[l][0] in self._tensors}

  def __bondget(self, key, t):
    return self._parent._tbonds[t][key]
 
  def __bondedcopy(self):
    return {t: self._parent._bonded[t]&self._tensors for t in self._tensors}

  def __bondedget(self, key):
    return self._parent._bonded[key] & self._tensors

  def __tidxs(self, key):
    return isinstance(key, int) and key >= 0 and key < len(self._parent._tlist)

  def __tsetcopy(self):
    return [s & self._tensors for s in self._parent._tset]

  def __tsetget(self, key):
    return self._parent._tset[key] & self._tensors

  def clone(self):
    """Reproduce as regular Network"""
    tdclone = copy(self._tdict)
    tlclone = list(self._tlist)
    idiff = 0
    for i0 in range(len(self._tlist)):
      s = self._tset[i0]
      if not s:
        tlclone.pop(i0-idiff)
        idiff += 1
      elif idiff:
        for t in s:
          tdclone[t] -= idiff
    net = Network(tlclone, tdclone, copy(self._tconj))
    bonds = {t: copy(self._tbonds[t]) for t in self._tensors}
    net._set_bonds(bonds)
    net._set_output(self._tout)
    return net


class Tensor_NetworkView(Tensor):
  """View of a Network as the tensor resulting from its contraction"""

  def __init__(self, net):
    self.__net = net
    self._dspace = {}
    for t, ll in self.network.freeindices():
      try:
        l1 = net._out[t,ll]
      except:
        raise KeyError('Output index %s.%s not set'%(t,ll))
      V = net._tlist[net._tdict[t]]._dspace[ll]
      if net._conj[t]:
        self._dspace[l1] = V.dual()
      else:
        self._dspace[l1] = V
    self._spaces = tuple(self._dspace.values())
    self._idxs = tuple(self._dspace.keys())
    self.shape = dictproperty(self._Tensor__shape_get, None,
      self._Tensor__shape_has, self._Tensor__shape_copy)

  @property
  def _T(self):
    return self.__net.contract().permuted(self._idxs)

  def __copy__(self):
    if isinstance(self.__net._cache, Tensor):
      return self.__net._cache.copy()
    else:
      return self.__net.contract()


class NetworkTree:
  """Tree class for encoding contraction order
  Has left and right pointers to sub-trees"""

  def __init__(self, left, right, verify=True):
    self.left = left
    self.right = right
    if isinstance(left, NetworkTree):
      self._ldepth = left.depth
      lleaves = left.leaves
    else:
      self._ldepth = 0
      lleaves = {left}
    if isinstance(right, NetworkTree):
      self._rdepth = right.depth
      rleaves = right.leaves
    else:
      self._rdepth = 0
      rleaves = {right}
    if verify and lleaves & rleaves:
      raise ValueError('Overlap within leaves, including %s' \
        % list(left&right)[0])
    self._leafset = lleaves | rleaves
  
  def __copy__(self):
    return NetworkTree(copy(self.left), copy(self.right), verify=False)

  @classmethod
  def setfromlist(cls, nestedlist):
    """Initialize tree based on nested lists"""
    if isinstance(nestedlist,str):
      return nestedlist
    elif not isinstance(nestedlist,list) and not isinstance(nestedlist,tuple):
      raise ValueError('Argument not nested list')
    if len(nestedlist) == 0:
      return None
    if len(nestedlist) == 1:
      return cls.setfromlist(nestedlist[0])
    else:
      left = nestedlist[:-1]
      right = nestedlist[-1]
      return NetworkTree(cls.setfromlist(left),cls.setfromlist(right))

  @property
  def depth(self):
    """Total depth of tree"""
    return max(self._ldepth, self._rdepth)+1

  @property
  def leaves(self):
    return set(self._leafset)

  @property
  def leftleaves(self):
    if isinstance(self.left, NetworkTree):
      return self.left.leaves
    elif self.left is None:
      return set()
    else:
      return {self.left}

  @property
  def rightleaves(self):
    if isinstance(self.right, NetworkTree):
      return self.right.leaves
    elif self.right is None:
      return set()
    else:
      return {self.right}

  def subnettree(self, minus, side=None):
    """Tree based on subnetwork from subtracting given nodes
    Produced by restructuring to put nodes on top
    Only works when minus is a subtree; returns None otherwise"""
    lleaves = self.leftleaves
    if minus & lleaves:
      if not minus.issubset(lleaves):
        # Not proper subtree
        return None
      if minus == lleaves:
        # Bring up right subtree
        rearrange = self.right
      else:
        # Join right with what remains of left
        rearrange = self.left.subnettree(minus, self.right)
    elif minus == self.rightleaves:
      rearrange = self.left
    else:
      rearrange = self.right.subnettree(minus, self.left)
    if side is None:
      return rearrange
    else:
      # Join with other side of parent tree
      return NetworkTree(rearrange, side)

  def pprint(self, pattern=(), isleft=False, use_ascii=False):
    if use_ascii:
      assert not pattern and not isleft
      self.pprint_ascii()
      return
    # First print root (already aligned at $depth columns)
    depth = len(pattern)
    kw = dict(end='',flush=config.flush)
    if not depth:
      # Root node is |-
      print('\u251C',**kw)
    else:
      # Root node is tee
      print('\u252C',**kw)
    # Print 'left' node
    if isinstance(self.left,NetworkTree):
      self.left.pprint(pattern+(1,),True)
    else:
      print(self.left)
    # Before printing right: align to $depth
    for b in pattern:
      if b:
        print('\u2502',**kw)
      else:
        print(' ',**kw)
    print('\u2514',**kw)
    if isinstance(self.right,NetworkTree):
      self.right.pprint(pattern+(0,),False)
    else:
      print(self.right,flush=config.flush)

  def pprint_ascii(self, pattern=(), isleft=False):
    # First print root (already aligned at $depth columns)
    depth = len(pattern)
    kw = dict(end='',flush=config.flush)
    if not depth:
      # Root node is |-
      print('L_',**kw)
    else:
      # Root node is tee
      print('__',**kw)
    # Print 'left' node
    if isinstance(self.left,NetworkTree):
      self.left.pprint_ascii(pattern+(1,),True)
    else:
      print(self.left)
    # Before printing right: align to $depth
    for b in pattern:
      if b:
        print('| ',**kw)
      else:
        print('  ',**kw)
    print('L_',**kw)
    if isinstance(self.right,NetworkTree):
      self.right.pprint_ascii(pattern+(0,),False)
    else:
      print(self.right,flush=config.flush)


class OptimizationOverflow(MemoryError):
  """Error thrown when network or subnetwork cannot be optimized relative to
  memory or time constraints
  Used primarily to message upstream"""
  def __init__(self, expense):
    msg = 'Unable to find valid optimization order'
    if config.memcap:
      memcap = config.memcap//config.memratio
      self.memory_constraint = memcap
      self.time_constraint = expense
      msg += ' with maximum %d elements'%memcap
    if expense:
      msg += ' in under %d FPOs'%expense
    self.message = msg
