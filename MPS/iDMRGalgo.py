from .DMRGalgo import *
from .impsbasic import *
from scipy import linalg

def open_manager(Hamiltonian, chis, file_prefix=None, savefile='auto',
    override=True, override_chis=True, resume_from_old=False,
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
    consisting of (psi, transfers, (chi, single_or_double, niter, E, E0))
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
        psi,(lt,rt),sv = Mngr
        if isinstance(sv,tuple):
          chi,ds,niter,E,E0 = sv
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
            Mngr.suplabels = ['base','single']
          else:
            Mngr.saveschmidt()
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
      else:
        assert isinstance(Mngr, iDMRGManager)
      if override_chis:
        # Reset bond dimension list
        useas = override_chis if isinstance(override_chis,str) else None
        if chis is None:
          chi_from_settings = True
        else:
          Mngr.resetchis(chis, use_as=useas)
          chi_from_settings = False
      else:
        assert chis is None or chis == Mngr.chis
        chi_from_settings = False
      if Mngr.filename != filename:
        config.streamlog.warn('Updating save destination from %s to %s',
          Mngr.filename, filename)
        Mngr.filename = filename
        Mngr.saveprefix = file_prefix
      if override:
        # Additionally reset general settings
        dmrg_kw.pop('psi0',None)
        Mngr.resetsettings(stage_override=chi_from_settings,**dmrg_kw)
      return Mngr
  return iDMRGManager(Hamiltonian, chis, file_prefix=file_prefix, savefile=savefile, **dmrg_kw)

class iDMRGManager(DMRGManager):
  """Optimizer for iDMRG"""
  def __init__(self, Hamiltonian, chis, N=None, **dmrg_kw):
    """MPO Hamiltonian; list of bond dimensions (or single bond dimension)
    length of unit cell (if not same as H)
    transf_init maximum number of steps for initializing transfer matrices"""
    self.__version = '1.0'
    if N is not None:
      assert N%Hamiltonian.N == 0
      self.__N = N
    else:
      self.__N = Hamiltonian.N
    self.right_transfer = None
    self.left_transfer = None
    # Number of sites transfers have moved since last reduction
    self.dx_right = None
    self.dx_left = None
    # Total energy according to transfer matrices & # sites covered at last record
    self.Eblock = None
    self.nblock = None
    self.ocenter = None
    #assert 'use_shelf' not in dmrg_kw
    super().__init__(Hamiltonian, chis, **dmrg_kw)
    assert not self.use_shelf
    # TODO is this completely depricated?
    self._usenormalizers = False

  @property
  def N(self):
    return self.__N

  @N.setter
  def N(self, value):
    # Do nothing
    pass

  def _initfuncs(self):
    # Labels for supervisors
    self.supfunctions = {'base':baseSupervisor,
                         'double':iDoubleOptSupervisor,
                         'single':iSingleOptSupervisor,
                         'doublesweep':iDoubleSweepSupervisor,
                         'singlesweep':iSingleSweepSupervisor}
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
                        'singleupdate':self.singleupdate,
                        'doubleupdate':self.doubleupdate,
                        'resettol':self.resettol,
                        'shift':self.shiftto}
    self.callables = {}
  
  # No registered objects (for now)

  @property
  def rtransf_site(self):
    return self.right_transfer.site

  @property
  def ltransf_site(self):
    return self.left_transfer.site

  def shift_rtransf(self, n):
    # Shift right transfer matrix n sites
    self.psi.logcanon()
    if n == 0:
      return
    self.right_transfer = self.right_transfer.moveby(n)
    if self._usenormalizers:
      self._rnormalizer = self._rnormalizer.moveby(n)
    self.dx_right += n

  def shift_ltransf(self, n):
    # Shift left transfer matrix n sites
    self.psi.logcanon()
    if n == 0:
      return
    self.left_transfer = self.left_transfer.moveby(n)
    if self._usenormalizers:
      self._lnormalizer = self._lnormalizer.moveby(n)
    self.dx_left += n

  def reduce_transfers(self):
    # Renormalize transfer vectors: reset terminal boundary condition
    lEb = self.left_transfer.Ereduce()
    rEb = self.right_transfer.Ereduce()
    if self.Eblock is not None:
      self.Eblock -= lEb + rEb
      self.nblock -= self.dx_left + self.dx_right
    lE = lEb/self.dx_left
    rE = rEb/self.dx_right
    self.dx_left = 0
    self.dx_right = 0
    return lE,rE

  def shiftto(self, nl,nr,direction):
    # Shift transfer matrices in (direction), to sites nl & nr
    # move orthogonality center (to nl+ if left, to nr- if right)
    nocross = False
    if direction==0 and self.ocenter != 'canon':
      # Infer correct direction
      rcross = (self.rtransf_site-nr)%self.N > (self.rtransf_site-self.ocenter)%self.N
      lcross = (nl - self.ltransf_site)%self.N > (self.ocenter-1 - self.ltransf_site)%self.N
      if rcross and lcross:
        raise ValueError('Both left (%d->%d) and right (%d<-%d) transfers must cross orthogonality center at %d'%(self.ltransf_site%self.N,nl%self.N,nr%self.N,self.rtransf_site%self.N,self.ocenter%self.N))
      if rcross or lcross:
        direction = -1 + 2*rcross
      else:
        #direction = -1 + 2*((self.rtransf_site-nr)%self.N > (nl - self.ltransf_site)%self.N)
        nocross = True
    if self.ocenter == 'canon' or nocross or (self.N == 2 and (nl-nr)%2):
      # Fully orthogonal - shift instead of doing anything with orthogonality
      # center
      self.logger.debug('Moving left transfer %d->%d, right transfer %d->%d',
        self.ltransf_site,nl,self.rtransf_site,nr)
      self.shift_rtransf((self.rtransf_site-nr)%self.N)
      self.shift_ltransf((nl-self.ltransf_site)%self.N)
    elif direction == -1:
      # Leftward
      # Move right transfer 
      self.logger.debug('Moving left transfer %d->%d, right transfer %d->%d,'
        ' orthogonality center %d->%d',
        self.ltransf_site,nl,self.rtransf_site,nr,self.ocenter,nl+1)
      self.shift_rtransf((self.rtransf_site-nr)%self.N)
      assert (self.ocenter-nr)%self.N == 0
      Ul = self.psi.rectifyleft(self.ocenter,nl+1)
      self.ocenter = nl+1 #TODO is this the right adjustment?
      # TODO triple-check this is how gauging should work
      if Ul is not None:
        self.gauge_at(nl+1,Ul,'l')
      # Move left transfer (almost) one unit cell right
      self.shift_ltransf((nl-self.ltransf_site)%self.N)
    else:
      # Rightward
      assert direction == 1
      self.logger.debug('Moving left transfer %d->%d, right transfer %d->%d,'
        ' orthogonality center %d->%d',
        self.ltransf_site,nl,self.rtransf_site,nr,self.ocenter,nr)
      self.shift_ltransf((nl-self.ltransf_site)%self.N)
      assert (self.ocenter-nl-1)%self.N == 0
      Ur = self.psi.rectifyright(self.ocenter,nr)
      self.ocenter = nr
      if Ur is not None:
        self.gauge_at(nr-1,Ur,'r')
      self.shift_rtransf((self.rtransf_site-nr)%self.N)

  def init_transfers(self):
    self.left_transfer = self.H.getboundarytransfleft(self.psi,0)
    self.right_transfer = self.H.getboundarytransfright(self.psi,-1)
    self.dx_left = 0
    self.dx_right = 0
    self.Eblock = None
    self.logger.log(5,'Initializer: %0.6e %0.6e',
      np.real(self.left_transfer.initializer()),
      np.real(self.right_transfer.initializer()))
    if not self.settings['transf_init']:
      self.logger.info('Initializing with boundary transfer matrices')
      return
    config.streamlog.info('Initializing with %d applications of full transfer matrix', self.settings['transf_init'])
    ncheck = self.settings['transf_check']
    delta = self.settings['transf_delta']
    if delta is None:
      delta = self.settings['Edelta']
    persweep = self.N * ncheck
    lE0,rE0 = None,None
    lT0,rT0 = self.left_transfer.T,self.right_transfer.T
    self._initnormalizers()
    for n in range(0,self.settings['transf_init'],ncheck):
      self.shift_ltransf(persweep)
      self.shift_rtransf(persweep)
      lE,rE = self.reduce_transfers()
      diff = abs(self.left_transfer.T-lT0)+abs(self.right_transfer.T-rT0)
      config.streamlog.info('[% 4d]\tE=%+0.6f/%+0.6f\t%0.2e',
        n+ncheck,np.real(lE),np.real(rE),diff)
      if n != 0:
        dEl = lE0-lE
        dEr = rE0-rE
        if diff < delta and abs(dEl)+abs(dEr) < delta and abs(lE-rE) < delta:
          break
      lE0,rE0 = lE,rE
      lT0,rT0 = self.left_transfer.T,self.right_transfer.T
    # TODO remove references to normalization transfers
    return np.real(lE+rE)/2

  def _initnormalizers(self):
    if not self._usenormalizers or self.left_transfer is None:
      return
    self._lnormalizer = LeftTransfer(self.psi,-1,'l')
    self._lnormalizer.setstrict()
    self._lnormalizer.setvalue(self.left_transfer.T.id_fromT('t:t-b'))
    self._rnormalizer = RightTransfer(self.psi,0,'r')
    self._rnormalizer.setstrict()
    self._rnormalizer.setvalue(self.right_transfer.T.id_fromT('t:t-b'))

  def updateEblock(self):
    # Measure E through transfer matrix contraction
    tL = self.left_transfer
    dx = (self.rtransf_site - self.ltransf_site - 1)%self.N
    nL = np.square(self.psi.getschmidt(tL.site))
    nL = tL.T.init_fromT(np.diag(nL),'t|t,b|b')
    if dx:
      tL = tL.moveby(dx, unstrict=True)
      ntL = LeftTransfer(self.psi,self.ltransf_site,'r')
      ntL.setvalue(nL)
      nL = ntL.moveby(dx).T
    norm = nL.trace('t-b')
    s0 = self.psi.getschmidt(self.rtransf_site-1)
    T = tL.T.diag_mult('t',s0).diag_mult('b',s0)
    Eblock = T.contract(self.right_transfer.T,'t-t,c-c,b-b')/norm
    self.Eblock = np.real(Eblock)
    self.nblock = self.dx_left+self.dx_right+dx
    self.logger.log(5, 'Energy transfers contracted to %0.10f %+0.4ei '
      '(%d sites right, %d+%d left',self.Eblock,np.imag(Eblock),
      self.dx_right,self.dx_left,dx)

  def getE(self, savelast=True, updaterel=True):
    if savelast:
      self.E0 = self.E
    if not self.right_transfer or not self.left_transfer:
      assert not self.E
      self.E = self.init_transfers()
    Eblock0 = self.Eblock
    nblock0 = self.nblock
    self.updateEblock()
    if Eblock0 is not None and self.nblock != nblock0:
      self.E = (self.Eblock-Eblock0)/(self.nblock-nblock0)
    return self.E

  def reset_transfer(self, lr, mat, ncheck=10):
    delta = self.settings['tol0']
    degen_tol = 1e-8
    nmax = self.settings['transf_restore']
    bdy = (self.H.Rboundary.renamed('l-c'),self.H.Lboundary.renamed('r-c'))
    term = (self.H.Rterm.renamed('l-c'),self.H.Lterm.renamed('r-c'))
    if lr == 'l':
      transf = self.left_transfer
      if transf is None:
        return
      dx = (-1-transf.site)%self.N
    else:
      transf = self.right_transfer
      if transf is None:
        return
      dx = transf.site%self.N
    if isinstance(mat,list):
      assert len(mat) == self.N
      mat = mat[transf.site%self.N]
    transf.gauge(mat)
    if dx != 0:
      # Move to origin
      transf = transf.moveby(dx)
    # Remove accumulator
    transf.Ereduce()
    # Normalize by initializer
    inorm = transf.initializer()
    self.logger.debug('Initializer is %0.4g + %0.4gi',
      np.real(inorm),np.imag(inorm))
    transf.T /= inorm
    self.logger.info('Performing Arnoldi iteration to converge %s transfer',
      'left' if lr=='l' else 'right')
    transfer_norm = abs(transf.T)
    transf.T /= transfer_norm
    krylov = []
    Hessenberg = np.zeros((0,0),dtype=complex)
    hk = 0
    Es = []
    for k in range(1,nmax):
      Hessenberg = np.pad(Hessenberg,(0,1))
      Hessenberg[k-1,k-2] = hk
      krylov.append(transf.T)
      # Apply transfer matrix
      transf = transf.moveby(self.N)
      # Project out accumulator
      Ek = transf.Ereduce()
      Es.append(Ek)
      for j,T in enumerate(krylov):
        h = transf.T.contract(T,'t-t,b-b,c-c*')
        transf.T -= h*T
        Hessenberg[j,k-1] = h
      hk = abs(transf.T)
      self.logger.log(8,'Arnoldi iteration %d has norm %0.4g', k, hk)
      if hk < delta:
        self.logger.debug('Exiting Arnoldi iteration (Krylov convergence)')
        break
      transf.T /= hk
      if k%ncheck == 0:
        w,v = linalg.eig(Hessenberg)
        imax = np.argmax(np.abs(w))
        dv = linalg.norm(v[-ncheck:,imax])
        self.logger.debug('Max eigenvector contribution %0.4g from Arnoldi '
          'iterations %d-%d', dv, k-ncheck+1,k)
        if dv < delta:
          self.logger.debug('Exiting Arnoldi iteration (Ritz eigenvector '
            'convergence)')
          break
        # TODO Hermiticity/other stabilization?
    # Ritz eigenvalues/vectors
    w,v = linalg.eig(Hessenberg)
    nk = len(w)
    imax = np.argmax(np.abs(w))
    self.logger.info('Max eigenvalue |lambda - 1| = %0.4g',
      abs(w[imax]-1))
    # If degeneracy exists optimize by initializer
    # TODO and/or accumulator?
    degen = np.abs(w/max(np.abs(w)) - 1) < degen_tol
    ndegen = sum(degen)
    if ndegen > self.psi._transfer_degen:
      self.logger.critical('Degeneracy %d of energy transfer matrix '
        '> degeneracy %d of norm transfer matrix',
        ndegen,self.psi._transfer_degen)
    Tinits = [v.contract(term[lr=='r'],'c-c;~') for v in krylov]
    chi = self.psi.getchi(-1)
    traces = [v1.trace('t-b')/chi for v1 in Tinits]
    if ndegen > 1:
      self.logger.info('Selecting from %d-fold transfer matrix degeneracy',
        ndegen)
      gram = np.zeros((nk,nk),dtype=complex)
      traces = np.zeros(nk,dtype=complex)
      for i,v1 in enumerate(Tinits):
        for j,v2 in enumerate(Tinits[:i+1]):
          gram[j,i] = v1.contract(v2,'t-t,b-b*')
          gram[i,j] = np.conj(gram[j,i])
      # Project onto approximate eigenspace
      # TODO check
      gram_degen = v[:,degen].conj().dot(gram).dot(v[:,degen])
      gw,gv = linalg.eigh(gram_degen)
      # Pseudo-inverse
      gpos = gw > degen_tol
      gvp = gv[:,gpos]
      init_basis = gvp.dot(np.diag(np.power(gw[gpos],-.5)))
      coeffs = gvp.dot(np.diag(np.power(gw[gpos],-1))).dot(gvp.T.conj()).dot(np.conj(traces))
    else:
      coeffs = v[:,imax]
      coeffs /= coeffs.dot(traces)
    T = coeffs[0] * krylov[0]
    for i in range(1,nk):
      T += coeffs[i] * krylov[i]
    transf.T = T
    Etransf = coeffs.dot(Es)/self.N
    self.logger.log(15, 'Energy from transfer matrix %0.10f %+0.2gi (%s)',
      np.real(Etransf),np.imag(Etransf),lr)
    # Check normalization to initializer
    s = self.psi.getschmidt(0)
    init_0 = T.contract(term[lr=='r'],'c-c;~')
    init_s = init_0.diag_mult('t',s).diag_mult('b',s)
    inorm = abs(init_s - init_s.id_like('t-b').diag_mult('t',np.square(s)))
    self.logger.info('Difference between initializer & identity: '
      '%0.4g weighted, %0.4g unweighted',inorm,abs(init_0-init_s.id_like('t-b')))
    if inorm > degen_tol:
      raise ValueError('Initializer excessively divergent from identity')
    #inorm = transf.initializer()
    #transf.T /= inorm
    if lr == 'l':
      self.left_transfer = transf
      self.dx_left = 0
    else:
      self.right_transfer = transf
      self.dx_right = 0
    return np.real(Etransf)

  def restorecanonical(self, almost=False):
    self.logger.info('Restoring to canonical form')
    mLs,mRs = self.psi.restore_canonical(gauge=True,almost_canon=almost,
      tol=self.settings['tol1'])
    El = self.reset_transfer('l',mRs)
    Er = self.reset_transfer('r',mLs)
    if El is not None:
      config.streamlog.debug('\t%+0.10f / %+0.10f',El,Er)
    self._initnormalizers()
    if self.right_transfer is not None and self.left_transfer is not None:
      self.updateEblock()
    #if self.left_transfer is not None:
    #  self.left_transfer = self.H.reset_ltransfer(self.left_transfer,mRs[self.ltransf_site%self.N],
    #    self.settings['tol0'],self.settings['transf_restore'])
    #  self.dx_left = 0
    #if self.right_transfer is not None:
    #  self.right_transfer = self.H.reset_rtransfer(self.right_transfer,mLs[self.rtransf_site%self.N],
    #    self.settings['tol0'],self.settings['transf_restore'])
    #  self.dx_right = 0
    #  if self.left_transfer is not None:
    #    self.updateEblock()
    self.ocenter = 'canon'

  def initstate(self):
    # Options for passing MPS
    if isinstance(self.psi,str):
      self.logger.log(30, 'Loading initial state from %s',self.psi)
      rv = pickle.load(open(self.psi,'rb'))
      if isinstance(rv,tuple):
        rv, transfs, *sv = rv
        self.left_transfer,self.right_transfer = transfs
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

  def doubleupdate(self, n, direction):
    self.logger.log(15, 'site %s double update (%s)', n, direction)
    assert self.ocenter=='canon' or 0<=(self.ocenter-n)%self.N<=2
    self.H.DMRG_opt_double(self.psi, n, self.chi,
      self.left_transfer,self.right_transfer,None,direction=='r',
      self.settings['tol2'], self.eigtol)
    self.ocenter = n+1

  def singleupdate(self, n, direction): #, gauge):
    self.logger.log(12, 'site %s single update (%s)', n, direction)
    right = (direction+1)//2
    assert self.ocenter=='canon' or (self.ocenter-n)%self.N in [0,1]
    # TODO gauge within this step
    #gL,gR = (gauge, None) if right else (None, gauge)
    gL,gR = None,None
    mat = self.H.DMRG_opt_single(self.psi, n, self.left_transfer,
      self.right_transfer, right, gL, gR, self.settings['tol1'], self.eigtol)
    if right:
      self.right_transfer.gauge(mat)
      if self._usenormalizers:
        self._rnormalizer.gauge(mat)
      self.psi.lgauge(mat,n+1)
    else:
      self.left_transfer.gauge(mat)
      if self._usenormalizers:
        self._lnormalizer.gauge(mat)
      self.psi.rgauge(mat,n-1)
    self.ocenter = n+right
    self.shiftto(n-1+direction,n+1+direction,direction)
    return mat

# TODO mixed double/single?
def iDoubleOptSupervisor(N, settings, state=None):
  if state is None:
    i0 = 0
    yield 'saveschmidt', ()
  else:
    i0, = state
  for niter in range(i0,settings['nsweep2']):
    E = yield 'runsub',('doublesweep',N, None, None),(niter,)
    diff = yield 'compare2', (niter,)
    if (niter+1) % settings['ncanon2'] == 0:
      yield 'canonical', (True,)
    if diff < settings['schmidtdelta']:
      if (niter+1) % settings['ncanon2'] != 0:
        yield 'canonical', (True,)
      break
  yield 'complete', ()

def iSingleOptSupervisor(N, settings, state=None):
  if state is None:
    i0 = 0
  else:
    i0, = state
  for niter in range(i0,settings['nsweep1']):
    E = yield 'runsub',('singlesweep',N, None, None),(niter,)
    diff = yield 'compare1', (niter,)
    if (niter+1) % settings['ncanon1'] == 0:
      yield 'canonical', (True,)
    if diff < settings['Edelta']:
      if (niter+1) % settings['ncanon1'] != 0:
        yield 'canonical', (True,)
      break
  yield 'complete', ()

def iSingleSweepSupervisor(N, site0, sequence, settings, state=None):
  # TODO vary starting place/number in each direction/etc?
  # sequence is (after initial site site0) for each step +/- for right/left
  #   (including for move after last step)
  # default: site 0 followed by N right, N+1 left
  if site0 is None:
    site0 = 0
  if sequence is None:
    sequence = N*(1,)+(N+1)*(-1,)+(1,)
  nstep = len(sequence)
  if state is None:
    direction = sequence[0]
    state = (0, site0)
    yield 'shift', (site0-1,site0+1,0)
  step, site = state
  assert site == site0+sum(sequence[:step])
  while step < nstep:
    direction = sequence[step]
    yield 'singleupdate', (site,direction), (step,site)
    # New orthogonality center is site if left-moving, site+1 if right-moving
    step += 1
    site += direction
  yield 'complete', ()

def iDoubleSweepSupervisor(N, site0, sequence, settings, state=None):
  # TODO vary starting place/number in each direction/etc?
  # sequence is (after initial site site0) for each step +/- for right/left
  #   (including for move after last step)
  # default: site 0 followed by N-1 right, N left
  if site0 is None:
    site0 = 0
  if sequence is None:
    sequence = (N-1)*(1,)+N*(-1,)+(1,1)
  nstep = len(sequence)-1
  if state is None:
    direction = sequence[0]
    state = (0,site0+direction)
    #if direction == -1:
    #  # TODO is this the correct starting move?
    #  yield 'shift', (site0,site0-3,site0,direction)
    #else:
    yield 'shift', (site0-1+direction,site0+2+direction,direction)
  step, site = state
  assert site == site0+sum(sequence[:step+1])
  direction = sequence[step+1]
  while step < nstep:
    direction = sequence[step+1]
    yield 'doubleupdate', (site,direction), (step,site)
    step += 1
    site += direction
    if step < nstep:
      yield 'shift', (site-1,site+2,direction)
  yield 'complete', ()
