# TODO: Unify with METTS/mpsbasic
import quantrada
from quantrada import tensors, networks, operators, links, config
from copy import copy
import numpy as np
from numbers import Number
from itertools import compress
from collections import defaultdict
import os.path, pickle,sys
import traceback
from .mpsabstract import *

config.verbose = 1
config.VDEBUG = 2
keig = 5

def randMPS(indphys, indvirt, N=None):
  if not N:
    N = len(indphys)
  elif not isinstance(indphys, list):
    indphys = N*[indphys]
  elif N != len(indphys):
    indphys = (N//len(indphys))*indphys
    indphys = indphys[:N]
  if isinstance(indvirt, int):
    chi0 = indvirt
    indvirt = [links.VSpace(chi0) for n in range(N)]
  elif isinstance(indvirt, links.VSAbstract):
    indvirt = N*[indvirt]
  else:
    for n,V in enumerate(indvirt):
      if isinstance(V,int):
        indvirt[n] = links.VSpace(V)
    if N != len(indvirt):
      assert N%len(indvirt) == 0
      indvirt = (N//len(indvirt))*indvirt
  Ms = [ tensors.Tensor(None,('b','l','r'),(indphys[n],indvirt[n-1].dual(),indvirt[n])) for n in range(N)]
  return iMPS([M.rand_like() for M in Ms])


class iMPS(MPSgeneric):
  # TODO: Compare various possible forms for stability
  # Note: Index Schmidt such that it is to the right of matrix with same
  #   index -- matches finite MPS but conflicts with old version
  def __init__(self,Ms,schmidt=None):
    self._matrices = list(Ms)
    N = len(Ms)
    self._Nsites = N
    self._leftcanon = N*[False]
    self._rightcanon = N*[False]
    if schmidt:
      assert len(schmidt) == N
      self._schmidt = list(schmidt)
    else:
      self._schmidt = []
      for n in range(self._Nsites):
        if not (Ms[n]._dspace['r'] ^ Ms[(n+1)%N]._dspace['l']):
          raise ValueError(f'Failure in bond matching at sites {n}-{n+1}')
        d = Ms[n].dshape['r']
        self._schmidt.append(d*[1/d])
      self.restore_canonical()

  def __copy__(self):
    psi = iMPS(list(self._matrices),list(self._schmidt))
    psi._leftcanon = list(self._leftcanon)
    psi._rightcanon = list(self._rightcanon)
    return psi

  def issite(self, n):
    return True

  def setTc(self, M, n):
    MPSgeneric.setTc(self, M, n%self._Nsites)

  def setTL(self, M, n, schmidt=None, unitary=False):
    MPSgeneric.setTL(self, M, n%self._Nsites, schmidt, unitary)

  def setTR(self, M, n, schmidt=None, unitary=False):
    MPSgeneric.setTR(self, M, n%self._Nsites, schmidt, unitary)

  def setschmidt(self, s, n, strict=True):
    MPSgeneric.setschmidt(self, s, n%self._Nsites, strict)

  def resetcanon(self, n, bondright=False):
    if bondright and (n+1)%self._Nsites == 0:
      # Wraparound
      self._leftcanon[0] = False
      self._rightcanon[-1] = False
    else:
      MPSgeneric.resetcanon(self, n%self._Nsites, bondright)

  def lgauge(self, U, n, unitary=True):
    MPSgeneric.lgauge(self, U, n%self._Nsites, unitary)

  def rgauge(self, U, n, unitary=True):
    MPSgeneric.rgauge(self, U, n%self._Nsites, unitary)

  def regauge_left(self, n, U, s, sameside=True, unitary=True):
    # Apply gauge transformation to the left of the Schmidt matrix
    # sameside=True:
    # T[n-1]--old_s--T[n] => T[-]*U^d -- s -- s^-1*U*old_s*T[+]
    # sameside=False:
    # T[n-1]--old_s--T[n] => T[-]*U^d*s^-1 -- s -- U*old_s*T[+]
    # TODO general names for U indices?
    if unitary:
      Ud = U.conj().renamed('l-r,r-l')
    else:
      Ud = U.inv('l-r')
    old_schmidt = self.getschmidt(n-1)
    if sameside:
      Ul = Ud
      self.setTc(self.getTc(n-1).mat_mult('r-l',Ud), n-1)
      self.setschmidt(s, n-1)
      Ur = U.diag_mult('r',old_schmidt).diag_mult('l',np.power(s,-1))
    else:
      # T[-]*U^d defines "right-canonical" form at n-1
      Ul = Ud.diag_mult('r',np.power(s,-1))
      self.setTR(self.getTc(n-1).mat_mult('r-l',Ud), n-1, s)
      Ur = U.diag_mult('r',old_schmidt)
    self.setTc(self.getTc(n).mat_mult('l-r',Ur), n)
    # TODO check these are the right matrices
    return Ul,Ur

  def regauge_right(self, n, U, s, sameside=True, unitary=True):
    # Apply gauge transformation to the right of the Schmidt matrix
    # sameside=True:
    # T[n]--old_s--T[n+1] => T[-]*old_s*U*s^-1 -- s -- U^d*T[+]
    # sameside=False
    # T[n]--old_s--T[n+1] => T[-]*old_s*U -- s -- s^-1*U^d*T[+]
    # TODO general names for U indices?
    # TODO this function is now untested
    if unitary:
      Ud = U.conj().renamed('l-r,r-l')
    else:
      Ud = U.inv('l-r')
    old_schmidt = self.getschmidt(n)
    if sameside:
      Ur = Ud
      self.setTc(self.getTc(n+1).mat_mult('r-l',Ud), n+1)
      self.setschmidt(s, n)
      Ul = U.diag_mult('l',old_schmidt).diag_mult('r',np.power(s,-1))
    else:
      Ur = Ud.diag_mult('l',np.power(s,-1))
      self.setTL(self.getTc(n+1).mat_mult('l-r',Ud), n+1, s)
      Ul = U.diag_mult('l',old_schmidt)
    self.setTc(self.getTc(n).mat_mult('r-l',Ul), n)
    # TODO check these are the right matrices
    return Ul,Ur

  def restore_canonical_stabilized(self, transf_degen_tol=1e-8, tol=None):
    # "gentler" version for when already "almost" canonical
    # TODO look into whether transf_degen_tol should be lower/higher
    # TODO provide tolerance for deviation at which to use fallback
    N = self._Nsites
    oldschmidts = list(self._schmidt)
    matL = N*[None]
    matR = N*[None]
    mats = (matL,matR)
    # U & d from left-canonical part
    Ul = N*[None]
    dl = N*[None]
    dlinv = N*[None]
    # U & d from right-canonical part
    Ur = N*[None]
    dr = N*[None]
    # Unit-cell transfer matrix
    transfnet = ';'.join(f'T{n},B{n}*' for n in range(N)) + ';(L);' + \
      ','.join(f'T{n-1}.r-T{n}.l,B{n-1}.r-B{n}.l,T{n}.b-B{n}.b' for n in range(1,N)) + \
      f',T0.b-B0.b,L.t-T0.l,L.b-B0.l;T{N-1}.r>t,B{N-1}.r>b'
    # TODO Automatic optimization?
    ords = [f'T{n},B{n}' for n in range(N)]
    transfnet = operators.NetworkOperator(transfnet,
      *[self.getTL(n) for n in range(N)],
      order='L,'+','.join(ords), adjoint_order='L,'+','.join(ords[::-1]))
    guess = self.getTc(0).id_from('l:b-t',self.getTc(0))
    w,v = transfnet.eigs(1,which='LM',guess=guess,herm=False)
    lam = np.real(w[0]) # Target eigenvalue
    assert abs(lam/abs(lam) - 1) < transf_degen_tol # Check phase of lam
    # Normalize/phase-correct to trace-1
    Ptot = v[0]/v[0].trace('t-b')
    # Check result is positive-semidefinite
    wcheck = Ptot.eig('t-b',vecs=False)
    if abs(Ptot - Ptot.ctranspose('t-b')) > transf_degen_tol or \
        min(wcheck) < -transf_degen_tol:
      raise ValueError('Positive-semidefinite transfer-matrix eigenvector not found') 
    if min(wcheck) < -transf_degen_tol**2:
      print('Failure to obtain positive-semidefinite transfer-matrix'
            ' eigenvector at "non-critical" threshold')
    # Move max eigenvector through the unit cell
    Ptots = self.lefttransfers(Ptot, 0, 'l')
    # Re-normalization to compensate for Schmidt-vector normalization
    probs = [M.T.trace('t-b') for M in Ptots]
    renorms = [probs[(i+1)%N]/probs[i] for i in range(N)]
    renorms[-1] *= abs(lam)
    nmlz_l = np.sqrt(np.abs(renorms))
    if config.verbose >= config.VDEBUG: # DEBUG
      print('Deviation from normalization via left transfer eigenvalue:',
        abs(lam-1))
      print('Total deviation from overall normalization:',sum(abs(nm-1) for nm in renorms))
    # Intermediate matrices
    matrices = [self.getTL(n) for n in range(N)]
    # Use max eigenvectors to obtain left-canonical form
    for n in range(N):
      if config.verbose >= config.VDEBUG: #DEBUG
        print(f'site {n}, left: compare identity with {abs(Ptots[n].T - Ptots[n].T.id_like("t-b")/self.getchi(n-1)):0.5g}')
      w,U = Ptots[n].T.eig('t-b',mat=True,zero_tol=tol,reverse=True)
      Ul[n] = U.renamed('c-l,t-r')
      assert min(w) > transf_degen_tol # DEBUG?
      dl[n] = np.sqrt(w/sum(w))
      dlinv[n] = np.power(dl[n],-1)
      s_old = self.getschmidt(n-1)
      # -s-T- <- -U^d-1/d- -d- -U-s_old-T-
      M = matrices[n-1].contract(Ul[n],'r-r;~;l>r*')
      matrices[n-1] = M.diag_mult('r',dlinv[n])
      #matrices[n-1] = M
      matrices[n] = matrices[n].mat_mult('l-r',Ul[n])/nmlz_l[n]
    self._leftcanon = N*[True]
    # Right transfer eigenvectors
    dmats = [Ptot.init_from(np.diag(np.power(self.getschmidt(n),2)),
      'b|r,t|r*',matrices[n]) for n in range(N)]
    netL = transfnet.network
    netL.replaceall('L*,'+','.join(f'T{n}' for n in range(N)),
      dmats[-1], *[matrices[n].diag_mult('l',dl[n]) for n in range(N)])
    #for n in range(N): DEBUG
    #  TL = matrices[n].diag_mult('l',dl[n])
    #  print(abs(TL.contract(TL,'b-b,l-l;r>t;r>b*')-TL.id_from('r:t-b',TL)))
    netL = netL.derive(f'-L;+R;R.t-T{N-1}.r,R.b-B{N-1}.r,T0.l>t,B0.l>b',
      dmats[-1]) # Just use transpose network?
    netL.setorder('R,'+','.join(ords[::-1]))
    transfnet = operators.NetworkOperator(netL, 'R', adjoint_order='R,'+','.join(ords))
    w,v = transfnet.eigs(1, which='LM',guess=dmats[-1],herm=False)
    lam2 = np.real(w[0])
    assert abs(lam2/abs(lam2) - 1) < transf_degen_tol # Check phase of lam
    # Normalize/phase-correct to trace-1
    Ptot = v[0]/v[0].trace('t-b')
    # Check result is positive-semidefinite
    wcheck = Ptot.eig('t-b',vecs=False)
    if abs(Ptot - Ptot.ctranspose('t-b')) > transf_degen_tol or \
        min(wcheck) < -transf_degen_tol:
      raise ValueError('Positive-semidefinite transfer-matrix eigenvector not found') 
    if min(wcheck) < -transf_degen_tol**2:
      print('Failure to obtain positive-semidefinite transfer-matrix'
            ' eigenvector at "non-critical" threshold')
    # Move max eigenvector through the unit cell
    ds = [m.shape['r'] for m in matrices]
    psitemp = iMPS(matrices,dl[1:]+[dl[0]])
    #for n in range(N): #DEBUG
    #  TL = matrices[n].diag_mult('l',dl[n])
    #  #print(abs(TL.contract(TL,'b-b,l-l;r>t;r>b*')-TL.id_from('r:t-b',TL)))
    #  print(abs(psitemp.getTL(n)-TL))
    Ptots = psitemp.righttransfers(Ptot, 0, 'l')
    # Check normalization
    probs = [M.T.trace('t-b') for M in Ptots]
    renorms = [probs[(i+1)%N]/probs[i] for i in range(N)]
    renorms[-1] *= lam2
    nmlz_r = np.sqrt(np.abs(renorms))
    if config.verbose >= config.VDEBUG: # DEBUG
      print('Deviation from normalization via right transfer eigenvalue:',
        abs(lam2-1))
      print('Total deviation from overall normalization:',sum(abs(nm-1) for nm in renorms))
    # Final matrices
    matfin = [self.getTc(n) for n in range(N)]
    # Use max eigenvectors to obtain right-canonical form
    for k in range(N):
      n = (-k)%N
      w,U = Ptots[k].T.eig('b-t',mat=True,zero_tol=tol,reverse=True)
      if config.verbose >= config.VDEBUG: #DEBUG
        print(f'site {n}, right: compare Schmidt with {abs(Ptots[k].T/probs[k] - dmats[n-1]):0.5g}, {np.linalg.norm(np.array(sorted(w))-np.array(sorted(self.getschmidt(n-1)))**2):0.5g}')
      Ur[n] = U.renamed('c-l,b-r')
      s_old = self.getschmidt(n-1)
      assert min(w) > min(s_old)**2 * transf_degen_tol # DEBUG?
      dr[n] = np.sqrt(w/sum(w))
      # s <- d2
      # -s-T- <- -U1^d-1/d1-U2- -d2- -1/d2-U2^d-d1-U1-s_old-T-
      # "Safer" way to apply 1/d2
      mL0 = Ur[n].diag_mult('r',dl[n]).mat_mult('r-l',Ul[n])
      mLT = mL0.permuted(('l','r'))
      dim = len(s_old)
      assert mLT.shape == (dim,dim)
      for i in range(dim):
        for j in range(dim):
          mLT[i,j] *= (s_old[j]/dr[n][i])
      mL = mL0.init_like(mLT,'l,r')
      mR = Ul[n].conj().renamed('r-l,l-r').diag_mult('r',dlinv[n]).contract(Ur[n],'r-r;~;l>r*')
      matfin[n-1] = matfin[n-1].mat_mult('r-l',mR)
      matfin[n] = matfin[n].mat_mult('l-r',mL)/nmlz_r[k]/nmlz_l[n]
      matR[n-1] = mR
      matL[n] = mL/nmlz_r[k]/nmlz_l[n]
    # Set
    for n in range(N):
      self.setTc(matfin[n],n)
      self.setschmidt(dr[n],n-1)
    self._rightcanon = N*[True]
    self._leftcanon = N*[True]
    # TODO check gauge matrices fill required purpose
    #if config.verbose >= config.VDEBUG:
    if config.verbose >= config.VDEBUG:
      #DEBUG
      for n in range(N):
        TL = self.getTL(n-1)
        TTL = TL.contract(TL,'l-l,b-b;r>b;r>t*')
        TR = self.getTR(n)
        TTR = TR.contract(TR,'r-r,b-b;l>t;l>b*')
        I = TR.id_from('l:t-b',TR)
        d = TR.shape['l']
        sold = matL[n].id_from('r:r-l',matL[n]).diag_mult('l',oldschmidts[n-1])/nmlz_r[-n]/nmlz_l[n]
        print(f'check at site {n-1}-{n}:',abs(TTL-I)/d,abs(TTR-I)/d)
        print('\t',abs(TTL/TTL.trace('t-b')-I/d), abs(TTR/TTR.trace('t-b')-I/d),
          abs(matR[n-1].diag_mult('r',dr[n]).contract(matL[n],'r-l;~;~')-sold))
    return mats

  def restore_canonical(self, transf_degen_tol=1e-8, tol=None, gauge=False,
      almost_canon=False):
    # TODO look into whether transf_degen_tol should be lower/higher
    # TODO "gentler" version for when already "almost" canonical?
    N = self._Nsites
    if almost_canon:
      try:
        return self.restore_canonical_stabilized(transf_degen_tol=transf_degen_tol, tol=tol)
      except Exception as e:
        if config.verbose:
          print('Stabilized restore_canonical failed; falling back')
          traceback.print_exc()
          # DEBUG
          if config.verbose >= config.VDEBUG:
            s = input('Continue? y/n ')
            if s != 'y':
              sys.exit()
    if gauge:
      matL = [T.id_from('l:l-r',T) for T in self._matrices]
      matR = [T.id_from('r:r-l',T) for T in self._matrices]
      mats = (matL,matR)
    else:
      mats = None
    # Unit-cell transfer matrix
    transfnet = ';'.join(f'T{n},B{n}*' for n in range(N)) + ';(L);' + \
      ','.join(f'T{n-1}.r-T{n}.l,B{n-1}.r-B{n}.l,T{n}.b-B{n}.b' for n in range(1,N)) + \
      f',T0.b-B0.b,L.t-T0.l,L.b-B0.l;T{N-1}.r>t,B{N-1}.r>b'
    # TODO Automatic optimization?
    ords = [f'T{n},B{n}' for n in range(N)]
    # TODO maybe should be using form w/ schmidt on right instead?
    transfnet = operators.NetworkOperator(transfnet,
      *[self.getTL(n) for n in range(N)],
      order='L,'+','.join(ords), adjoint_order='L,'+','.join(ords[::-1]))
    guess = self.getTc(0).id_from('l:b-t',self.getTc(0))
    w,v = transfnet.eigs(keig,which='LM',guess=guess,herm=False)
    # TODO increase keig as necessary
    lidx = np.argmax(np.real(w))
    lam = w[lidx] # Target eigenvalue
    assert abs(lam/abs(lam) - 1) < transf_degen_tol # Check phase of lam
    # Degeneracy to handle:
    degen = np.abs(np.array(w)/abs(lam) - 1) < transf_degen_tol
    ndegen = sum(degen)
    wmask,vmask = list(compress(w,degen)),list(compress(v,degen))
    if ndegen == 1:
      # Effectively nondegenerate
      # Normalize/phase-correct to trace-1
      Ptot = vmask[0]/vmask[0].trace('t-b')
    else:
      if config.verbose:
        print('transfer degeneracy (left)', ndegen)
      Ptot = positive_contribution(vmask, transf_degen_tol)
    # Check result is positive-semidefinite
    wcheck = Ptot.eig('t-b',vecs=False)
    if abs(Ptot - Ptot.ctranspose('t-b')) > transf_degen_tol or \
        min(wcheck) < -transf_degen_tol:
      print('Failure in restore_canonical:')
      print('  positive-semidefinite transfer-matrix eigenvector not found') 
      return mats
    if min(wcheck) < -transf_degen_tol**2:
      print('Failure to obtain positive-semidefinite transfer-matrix'
            ' eigenvector at "non-critical" threshold')
    # Move max eigenvector through the unit cell
    Ptots = self.lefttransfers(Ptot, 0, 'l')
    # Re-normalization to compensate for Schmidt-vector normalization
    probs = [M.T.trace('t-b') for M in Ptots]
    renorms = [probs[(i+1)%N]/probs[i] for i in range(N)]
    renorms[-1] *= abs(lam)
    # Use max eigenvectors to obtain left-canonical form
    for n in range(N):
      # Normalization factor
      nmlz = np.sqrt(abs(renorms[n]))
      w,U = Ptots[n].T.eig('t-b',mat=True,zero_tol=tol,reverse=True)
      U = U.renamed('c-l,t-r')
      schmidt = np.sqrt(np.abs(w))
      schmidt /= np.linalg.norm(schmidt)
      s_old = self.getschmidt(n-1)
      Ul,Ur = self.regauge_left(n, U, schmidt, sameside=False)
      self._matrices[n] /= nmlz
      if gauge:
        matR[n-1] = Ul
        matL[n] = Ur/nmlz
    for n in range(N):
      TL = self.getTL(n)
      X = TL.contract(TL,'b-b,l-l;r>t;r>b*')
      p = X.trace('t-b')
      d = X.dshape['t']
    self._leftcanon = N*[True]
    # Right transfer eigenvectors
    # TODO this is messy
    transfnet.network.replaceall('L,'+','.join(f'T{n}' for n in range(N)), self.getTc(0).id_from('l:b-t',self.getTc(0)),*[self.getTL(n) for n in range(N)])
    netL = transfnet.network.derive(f'-L;+R;R.t-T{N-1}.r,R.b-B{N-1}.r,T0.l>t,B0.l>b',self.getTc(-1).id_from('r:b-t',self.getTc(-1))) # Just use transpose network?
    #netL.replaceall(','.join(f'T{n}' for n in range(N)), *[self.getTR(n) for n in range(N)])
    netL.setorder('R,'+','.join(ords[::-1]))
    transfnet = operators.NetworkOperator(netL, 'R', adjoint_order='R,'+','.join(ords))
    guess = self.getTc(0).init_from(np.diag(np.power(self.getschmidt(-1),2)),
      't|l,b|l*',self.getTc(0))
    w,v = transfnet.eigs(keig, which='LM',guess=guess,herm=False)
    lam2 = max(np.real(w))
    degen = np.abs(np.array(w)/abs(lam2) - 1) < transf_degen_tol
    ndegen2 = sum(degen)
    wmask,vmask = list(compress(w,degen)),list(compress(v,degen))
    if ndegen2 != ndegen and config.verbose:
      print(f'Degeneracy changed {ndegen}->{ndegen2} for right transfer, ratio {abs(w[-1]/lam2-1):0.1g} (compare {transf_degen_tol:0.1g}') 
    if ndegen2 == 1:
      # Effectively nondegenerate: normalize/phase-correct
      Ptot = vmask[0]/vmask[0].trace('t-b')
    else:
      Ptot = positive_contribution(vmask, transf_degen_tol)
    # Check result is positive-semidefinite
    wcheck = Ptot.eig('t-b',vecs=False)
    if abs(Ptot - Ptot.ctranspose('t-b')) > transf_degen_tol or \
        min(wcheck) < -transf_degen_tol:
      print('Failure in restore_canonical:')
      print('  positive-semidefinite transfer-matrix eigenvector not found') 
      return mats
    if min(wcheck) < -transf_degen_tol**2:
      print('Failure to obtain positive-semidefinite transfer-matrix'
            ' eigenvector at "non-critical" threshold')
    Ptots = self.righttransfers(Ptot, 0, 'l')
    probs = [M.T.trace('t-b') for M in Ptots]
    renorms = [probs[(i+1)%N]/probs[i] for i in range(N)]
    renorms[-1] *= lam2
    # Use max eigenvectors to obtain right-canonical form
    for k in range(N):
      n = -k-1
      # Normalization factor
      nmlz = np.sqrt(abs(renorms[k]))
      w,U = Ptots[k].T.eig('b-t',mat=True,zero_tol=tol,reverse=True)
      U = U.renamed('c-l,b-r')
      schmidt = np.sqrt(np.abs(w))
      schmidt /= np.linalg.norm(schmidt)
      s_old = self.getschmidt(n)
      Ul,Ur = self.regauge_left(n+1, U, schmidt)
      self._matrices[n+1] /= nmlz
      if gauge:
        matR[N-k-1] = matR[N-k-1].mat_mult('r-l',Ul)
        matL[-k] = matL[-k].mat_mult('l-r',Ur)/nmlz
    self._rightcanon = N*[True]
    self._leftcanon = N*[True]
    # TODO check gauge matrices fill required purpose
    if config.verbose >= config.VDEBUG:
      #DEBUG
      for n in range(N):
        TL = self.getTL(n)
        TR = self.getTR(n)
        IL = TL.id_from('r:t-b',TL)
        IR = TR.id_from('l:t-b',TR)
        print(abs(TL.contract(TL,'l-l,b-b;r>t;r>b*')-IL),abs(TR.contract(TR,'r-r,b-b;l>t;l>b*')-IR))
    return mats

  def correlationlength(self, tolerance=1e-10):
    # Unit-cell transfer matrix
    N = self.N
    transfnet = ';'.join(f'T{n},B{n}*' for n in range(N)) + ';(L);' + \
      ','.join(f'T{n-1}.r-T{n}.l,B{n-1}.r-B{n}.l,T{n}.b-B{n}.b' for n in range(1,N)) + \
      f',T0.b-B0.b,L.t-T0.l,L.b-B0.l;T{N-1}.r>t,B{N-1}.r>b'
    # TODO Automatic optimization?
    ords = [f'T{n},B{n}' for n in range(N)]
    transfnet = operators.NetworkOperator(transfnet,
      *[self.getTL(n) for n in range(N)],
      order='L,'+','.join(ords), adjoint_order='L,'+','.join(ords[::-1]))
    guess = self.getTc(0).init_from(np.diag(np.power(self.getschmidt(-1),2)),
      'b|l,t|l*',self.getTc(0))
    wa = [1]
    nv = keig
    while (1 - min(wa)/max(wa)) < tolerance:
      try:
        w,v = transfnet.eigs(nv,which='LM',guess=guess,herm=False)
      except operators.ArpackError:
        pass
      else:
        wa = np.abs(w)
      nv += 5
    w0 = max(wa)
    w1 = max(wa[(w0-wa)/w0 > tolerance])
    return N/np.log(w0/w1)

  def printcanon(self,compare=False):
    # Ends up displaying arrows for directions that transfer matrices can be
    # "pushed"
    for n in range(self.N):
      print('<' if self._rightcanon[n] else '-',end='')
      print('>' if self._leftcanon[n] else '-',end='')
      print(' ' if self.getbond(n,'r')^self.getbond(n,'l') else '|',end='')
    print()
    if compare:
      for n in range(self.N):
        TL = self.getTL(n)
        TR = self.getTR(n)
        IL = TL.id_from('r:t-b',TL)
        IR = TR.id_from('l:t-b',TR)
        lc = abs(TL.contract(TL,'l-l,b-b;r>t;r>b*')-IL) < 1e-8
        rc = abs(TR.contract(TR,'r-r,b-b;l>t;l>b*')-IR) < 1e-8
        print('<' if rc else '-',end='')
        print('>' if lc else '-',end='')
        print(' ' if self.getbond(n,'r')^self.getbond(n,'l') else '|',end='')
      print()

  def rectifyleft(self, n0, nf, tol=None):
    # At most 1 unit cell:
    N = self.N
    dn = (nf-n0-1)%N
    nf = n0+dn+1
    return MPSgeneric.rectifyleft(self, n0, nf, tol)

  def rectifyright(self, n0, nf, tol=None):
    # At most 1 unit cell:
    N = self.N
    dn = (n0-nf-1)%N
    nf = n0-dn-1
    return MPSgeneric.rectifyright(self, n0, nf, tol)

  # TODO fidelity density

  def tebd_sweep(self, Us, chi):
    # TODO
    pass

  def tebd_sweeps(self, Us, chi, niter=1000, ncheck=10, delta=1e-5):
    # TODO
    pass

  def do_tebd(self, H, chis, taus, niters, deltas):
    # TODO
    pass

class iMPO(MPOgeneric):
  def __init__(self,Os,Lbdy,Rbdy,Lterm=None,Rterm=None):
    self._matrices = tuple(Os)
    self._Nsites = len(Os)
    self.Lboundary = Lbdy
    self.Rboundary = Rbdy
    if Lterm:
      self.Lterm = Lterm
    elif Rbdy:
      self.Lterm = Rbdy.conj().renamed('l-r')
    if Rterm:
      self.Rterm = Rterm
    elif Lbdy:
      self.Rterm = Lbdy.conj().renamed('r-l')

  def __getitem__(self,n):
    return self._matrices[n%self._Nsites]

  def expandunitcell(self, M):
    return self.__class__(M*self._matrices, self.Lboundary, self.Rboundary, self.Lterm, self.Rterm)

  @classmethod
  def _construct_FSM(cls, Ms, initial, final, transition, idx_state,
      state_lists, statedim):
    d0 = Ms[0].shape['l']
    # Boundary condition matrices
    BCL = np.zeros(d0)
    BCR = np.zeros(d0)
    BCL[idx_state[0][initial]] = 1
    BCR[idx_state[0][final]] = 1
    BCL = Ms[-1].init_fromT(BCL,'r|r')
    BCR = Ms[0].init_fromT(BCR,'l|l')
    return iMPO(Ms,BCL,BCR), state_lists


  def rand_MPS(self, bonds=None, bond=None):
    if bond:
      assert bonds is None
      bonds = self.N*[bond]
    else:
      assert isinstance(bonds,list) and len(bonds) == self.N
    T = self._matrices[0]
    Ms = [T.rand_from('b;l*;r',self._matrices[0],bonds[-1],bonds[0])]
    for n in range(1,self.N-1):
      Ms.append(T.rand_from('b;r:l*;r',self._matrices[n],Ms[-1],bonds[n]))
    Ms.append(T.rand_from('b;r:l*;l:r*',self._matrices[-1],Ms[-1],Ms[0]))
    return iMPS(Ms)

  def expv(self, psi, nmax=1000, ncheck=10, delta=1e-10):
    # Expectation value of finite-range MPO
    ltransf = self.getboundarytransfleft(psi,0)
    rtransf = self.getboundarytransfright(psi,-1)
    persweep = self._Nsites*ncheck
    lE0,rE0 = None,None
    lT0,rT0 = ltransf.T,rtransf.T
    for n in range(0,nmax,ncheck):
      ltransf = ltransf.moveby(persweep)
      lE = ltransf.Ereduce()
      rtransf = rtransf.moveby(persweep)
      rE = rtransf.Ereduce()
      diff = abs(ltransf.T-lT0) + abs(rtransf.T-rT0)
      if config.verbose:
        print(f'[{n+ncheck}]\tE={np.real(lE/ncheck):+0.6f}/{np.real(rE/ncheck):+0.6f}\t{diff:0.2e}')
      if n:
        dEl = lE0-lE
        dEr = rE0-rE
        if diff < delta and abs(dEl)/persweep+abs(dEr)/persweep < delta and abs(lE-rE)/persweep < delta:
          break
      lE0,rE0 = lE,rE
      lT0,rT0 = ltransf.T,rtransf.T
    E0 = (lE+rE)/(2*persweep)
    return E0

  def getT(self, n):
    return self._matrices[n%self._Nsites]

  def getboundarytransfleft(self, psi, n, lr='l'):
    assert n == 0 # TODO push through BC
    T = LeftTransfer(psi, n-1, lr, self)
    T.setstrict()
    T0 = psi.getTc(n)
    bid = T0.id_from('l:b-t',T0)
    if lr == 'r':
      bid = bid.diag_mult('t',np.power(self.getschmidt(n-1),2))
    T.setvalue(bid.contract(self.Lboundary,';~;r>c'))
    return T
  
  def getboundarytransfright(self, psi, n, lr='r'):
    assert (n+1)%self.N == 0 # TODO push through BC
    T = RightTransfer(psi, n+1, lr, self)
    T.setstrict()
    T0 = psi.getTc(n)
    bid = T0.id_from('r:b-t',T0)
    if lr == 'l':
      bid = bid.diag_mult('t',np.power(self.getschmidt(n),2))
    T.setvalue(bid.contract(self.Rboundary,';~;l>c'))
    return T

  def reset_ltransfer(self, ltransf, U, delta, nmax=10000):
    # Gauge by U
    ltransf.gauge(U)
    # Remove accumulator
    ltransf.Ereduce()
    # Normalize by initializer
    inorm = ltransf.initializer()
    ltransf.T /= inorm
    # Subtract out projection onto boundary vector
    Mbdy = ltransf.T.contract(self.Lboundary,'c-r;~*').contract(self.Lboundary,';~;r>c')
    ltransf.T -= Mbdy
    # Add proper boundary vector back in
    tbdy = self.getboundarytransfleft(ltransf.psi,0)
    ltransf.T += tbdy.T
    # Obtain convergence
    for n in range(nmax):
      T0 = ltransf.T
      ltransf = ltransf.moveby(self._Nsites)
      El = ltransf.Ereduce()
      diff = abs(ltransf.T-T0)
      if config.verbose >= config.VDEBUG:
        print('left transfer convergence',diff,np.real(El/self.N))
      if diff < delta:
        break
    return ltransf

  def reset_rtransfer(self, rtransf, U, delta, nmax=10000):
    # Gauge by U
    rtransf.gauge(U)
    # Remove accumulator
    rtransf.Ereduce()
    # Normalize by initializer
    inorm = rtransf.initializer()
    rtransf.T /= inorm
    # Subtract out projection onto boundary vector
    Mbdy = rtransf.T.contract(self.Rboundary,'c-l;~*').contract(self.Rboundary,';~;l>c')
    rtransf.T -= Mbdy
    # Add proper boundary vector back in
    tbdy = self.getboundarytransfright(rtransf.psi,-1)
    rtransf.T += tbdy.T
    # Obtain convergence
    for n in range(nmax):
      T0 = rtransf.T
      rtransf = rtransf.moveby(self._Nsites)
      Er = rtransf.Ereduce()
      diff = abs(rtransf.T-T0)
      if config.verbose >= config.VDEBUG:
        print('right transfer convergence',diff,np.real(Er)/self.N)
      if diff < delta:
        break
    return rtransf

  def DMRGsweep_single(self, psi, ltransf, rtransf, tol=None):
    N = self.N
    # Start on left side: transfers at 0
    Ur = psi.rectifyright(0,1)
    Ul = None
    rtransf = rtransf.moveby(N-1)
    # T -- => T -- s(new) -- gauge unitary -- s(old)^-1 --
    # -s- Tr -s- <= -- s(new) -- gauge -- s(old)^-1 -- s(old) -- Tr -s-
    # Sweep right
    for n in range(0,N):
      if config.verbose >= config.VDEBUG:
        print('single-update at site',n,'->') #DEBUG
      Ul = self.DMRG_opt_single(psi, n, ltransf, rtransf, True, Ul, Ur, tol)
      # -- T -- => -- T -- s(new) -- gauge unitary -- s(old)^-1 --
      ltransf = ltransf.right()
      rtransf.gauge(Ul)
      Ur = psi.rectifyright(n+1,n+2)
      rtransf = rtransf.moveby(N-1)
    # Sweep left
    if config.verbose >= config.VDEBUG: #DEBUG
      print('single-update at site',N,'<-')
    Ur = self.DMRG_opt_single(psi, N, ltransf, rtransf, False, Ul, Ur, tol)
    # -- T => -- s(old)^-1 -- gauge unitary -- s(new) -- T
    # Sweep left
    for n in range(self.N-1,-2,-1):
      # -- T -- => -- s(old)^-1 -- gauge unitary -- s(new) -- T --
      rtransf = rtransf.left()
      ltransf.gauge(Ur)
      Ul = psi.rectifyleft(n+1,n)
      ltransf = ltransf.moveby(N-1)
      if config.verbose >= config.VDEBUG:
        print('single-update at site',n,'<-') #DEBUG
      Ur = self.DMRG_opt_single(psi, n, ltransf, rtransf, n==-1, Ul, Ur, tol)
      # In last step reverse by performing at -1
    Ul = Ur # U is mislabeled due to direction reversal
    # right transfer matrix is already at 0
    # Correct with gauge unitary; make consistent orthogonality center at 0
    # ltransf is already at 0
    if config.verbose >= config.VDEBUG:
      psi.printcanon() #DEBUG
    rtransf.gauge(Ul)
    ltransf = ltransf.right()
    psi.lgauge(Ul,0)
    return ltransf,rtransf

  def DMRGsweep_double(self, psi, ltransf, rtransf, chi, tol=None):
    # Should start out as canonical (?)
    N = self.N
    # Sweep rightward
    # Assume: start with transfers at 0
    # Move right transfer to +2
    if N > 2:
      Ur = psi.rectifyright(0,2)
    else:
      Ur = None
    rtransf = rtransf.moveby(N-2)
    for n in range(0,N):
      if config.verbose >= config.VDEBUG:
        #DEBUG
        print('double-update at site',n,'->')
      self.DMRG_opt_double(psi, n, chi, ltransf, rtransf, Ur, True, tol)
      # Add unit cell to move transfers +1
      ltransf = ltransf.right()
      if N > 2:
        Ur = psi.rectifyright(n+1,n+3)
      else:
        Ur = None
      rtransf = rtransf.moveby(N-1)
    # Left sweep
    if config.verbose >= config.VDEBUG:
      #DEBUG
      print('double-update at site',N,'<-')
    self.DMRG_opt_double(psi, N, chi, ltransf, rtransf, Ur, True, tol)
    for n in range(N-1,-2,-1):
      rtransf = rtransf.left()
      if N > 2:
        Ul = psi.rectifyleft(n+2,n)
      else:
        Ul = None
      ltransf = ltransf.moveby(N-1)
      if config.verbose >= config.VDEBUG:
        #DEBUG
        print('double-update at site',n,'<-')
      self.DMRG_opt_double(psi, n, chi, ltransf, rtransf, Ul, False, tol)
    # Send back to 0
    ltransf = ltransf.right()
    rtransf = rtransf.left()
    return ltransf,rtransf

  def do_dmrg(self, psi0, chi, Edelta=1e-10, delta2=1e-8,ncanon=10,ncanon2=10,
              nsweep=1000,nsweep2=100,tol=1e-12,tol1=None,tol0=1e-12,
              transfinit=100,savefile=None,saveprefix=None,cont=False):
    # Instead of alternating single + double update perform double update
    # until reaching threshold delta2 (or nsweep2 iterations) then
    # perform single update until reaching energy-difference threshold
    # Use tol for error threshold in chi^2 truncations (two-site update),
    # tol1 for error threshold in chi^1 svd (one-site update + canonical form)
    # tol0 is threshold for initialization
    # Save psi, (LT,RT), E when complete
    #   At intermediate stage save psi, (LT,RT),(chi,#site-per-update,sweep) 
    if isinstance(chi,list):
      chis,chi = chi,chi[0]
    else:
      chis = [chi]
    ic0 = 0
    sd = 2
    sweep = 0
    ltransf = None
    if savefile and cont and os.path.isfile(savefile):
      print('reloading from',savefile)
      psi, (ltransf,rtransf), sv = pickle.load(open(savefile,'rb'))
      if isinstance(sv, tuple):
        chi, sd, sweep = sv
        ic0 = chis.index(chi)
      else:
        ic0 = -1
        # Check if result represents now-intermediate bond dimension
        if saveprefix and os.path.isfile(f'{saveprefix}c{chis[0]}.p'):
          # Find first uncomputed bond dimension
          ic0 = 1
          while os.path.isfile(f'{saveprefix}c{chis[ic0]}.p') and ic0<len(chis):
            print('Found state with chi =',chis[ic0])
            ic0 += 1
        elif len(chis) > 1:
          # Get bond dimension from state
          chi = min(psi.getchi(n) for n in range(psi.N))
          ic0 = 0
          while chis[ic0] <= chi and ic0 < len(chis):
            ic0 += 1
          print('Last bond dimension presumed equal to',chis[ic0])
        if ic0 == -1 or ic0 == len(chis):
          print('Already completed: E =',sv)
          return psi,sv
        else:
          chi = chis[ic0]
          if ic0:
            print(f'Completed with chi={chis[ic0-1]} (E={sv:0.8f})')
      psi0 = copy(psi)
    else:
      if isinstance(psi0,str):
        print('loading from',psi0)
        psi0, (ltransf,rtransf),sv = pickle.load(open(psi0,'rb'))
      elif not isinstance(psi0,iMPS):
        psi0 = self.rand_MPS(bond=chi)
        psi0.restore_canonical(tol=tol0)
      psi = copy(psi0)
    if ltransf is None:
      # Initialize transfer matrices
      ltransf = self.getboundarytransfleft(psi,0)
      rtransf = self.getboundarytransfright(psi,-1)
      if config.verbose > config.VDEBUG:
        print('Initializer:',np.real(ltransf.initializer()),np.real(rtransf.initializer()))
      if transfinit:
        transfinit = int(transfinit)
        print('Initializing with %d applications of full transfer matrix'%transfinit)
        ncheck = 10
        persweep = self._Nsites*ncheck
        lE0,rE0 = None,None
        lT0,rT0 = ltransf.T,rtransf.T
        for n in range(0,transfinit,ncheck):
          ltransf = ltransf.moveby(persweep)
          lE = ltransf.Ereduce()
          rtransf = rtransf.moveby(persweep)
          rE = rtransf.Ereduce()
          diff = abs(ltransf.T-lT0) + abs(rtransf.T-rT0)
          print(f'[{n+ncheck}]\tE={np.real(lE/ncheck):+0.6f}/{np.real(rE/ncheck):+0.6f}\t{diff:0.2e}')
          if n:
            dEl = lE0-lE
            dEr = rE0-rE
            if diff < Edelta and abs(dEl)/persweep+abs(dEr)/persweep < Edelta and abs(lE-rE)/persweep < Edelta:
              break
          lE0,rE0 = lE,rE
          lT0,rT0 = ltransf.T,rtransf.T
        E0 = (lE+rE)/(2*ncheck)
        dE0 = (dEl+dEr)/(2*ncheck)
    for ic,chi in enumerate(chis):
      if ic0 > ic:
        continue
      print(f'{chi=:>3} (#{ic})')
      persweep = (psi.N+1)*psi.N
      # right: N-2 + N*(N-1) + N+2 #rtransf
      # left : N + (N+1)*(N-1) + 1 #ltransf
      for niter in range(nsweep2):
        if config.verbose > config.VDEBUG: #DEBUG
          print('Initializer:',np.real(ltransf.initializer()),np.real(rtransf.initializer()))
        # Perform two-site DMRG
        # Energy before sweep
        s0 = psi.getschmidt(-1)
        T = ltransf.T.diag_mult('t',s0).diag_mult('b',s0)
        E0 = T.contract(rtransf.T,'t-t,c-c,b-b')
        ltransf,rtransf = self.DMRGsweep_double(psi,ltransf,rtransf,chi,tol=tol)
        if config.verbose >= config.VDEBUG:
          psi.printcanon() #DEBUG
        s0 = psi.getschmidt(-1)
        T = ltransf.T.diag_mult('t',s0).diag_mult('b',s0)
        E = T.contract(rtransf.T,'t-t,c-c,b-b')
        if config.verbose or (niter%ncanon2 == 0):
          print(f'[{niter}]\t{np.real(E-E0)/(2*persweep)}')
        if (niter+1)%ncanon2 == 0:
          # Reset transfer matrices
          El = ltransf.Ereduce()/persweep
          Er = rtransf.Ereduce()/persweep
          mLs,mRs = psi.restore_canonical(tol=tol1,gauge=True,almost_canon=True)
          # Re-gauge transfers
          ltransf = self.reset_ltransfer(ltransf,mRs[-1],tol0)
          rtransf = self.reset_rtransfer(rtransf,mLs[0],tol0)
          # Difference in Schmidt indices
          Ldiff = 0
          for n in range(self.N-1):
            la = np.array(psi0._schmidt[n])
            lb = np.array(psi._schmidt[n])
            lmin = min(len(la),len(lb))
            Ld2 = np.linalg.norm(la[:lmin]-lb[:lmin])**2
            # Corrections for extra indices may not be necessary?
            if len(la) > lmin:
              Ld2 += np.linalg.norm(la[lmin:])**2
            elif len(lb) > lmin:
              Ld2 += np.linalg.norm(lb[lmin:])**2
            Ldiff += np.sqrt(Ld2)
          Ldiff /= ncanon2*self.N # Normalize to approximate # of updates
          print(f'\t{Ldiff:0.6g}\t{np.real(El/ncanon2):0.6f}/{np.real(Er/ncanon2):0.6f}')
          if Ldiff < delta2:
            break
          psi0 = copy(psi)
        if savefile:
          pickle.dump((psi,(ltransf,rtransf),(chi,2,niter)),open(savefile,'wb'))
      Es0 = np.real(E-E0)
      persweep = (psi.N+1)*psi.N
      # right: (N+1)*(N-1) + N+1
      # left : N + (N+1)*(N-1) + 1
      for niter in range(nsweep):
        if config.verbose > config.VDEBUG: #DEBUG
          print('Initializer:',np.real(ltransf.initializer()),np.real(rtransf.initializer()))
        E0 = E
        if niter % ncanon == 0:
          mLs,mRs = psi.restore_canonical(gauge=True,tol=tol1,almost_canon=True)
          if config.verbose >= config.VDEBUG:
            psi.printcanon(compare=True)
          ltransf = self.reset_ltransfer(ltransf,mRs[-1],tol0)
          rtransf = self.reset_rtransfer(rtransf,mLs[0],tol0)
          if config.verbose >= config.VDEBUG: #DEBUG
            print('after canon:',np.real(ltransf.initializer()),np.real(rtransf.initializer()))
          s0 = psi.getschmidt(-1)
          T = ltransf.T.diag_mult('t',s0).diag_mult('b',s0)
          E = T.contract(rtransf.T,'t-t,c-c,b-b')
          ltransf = ltransf.moveby(psi.N)
          rtransf = rtransf.moveby(psi.N)
          s0 = psi.getschmidt(-1)
          T = ltransf.T.diag_mult('t',s0).diag_mult('b',s0)
          E0 = T.contract(rtransf.T,'t-t,c-c,b-b')
          if config.verbose:
            print(f'\t {np.real(E0-E)/(2*psi.N):0.8f}')
          Esweep = np.real(E0-E)/(2*psi.N)*persweep
        ltransf,rtransf = self.DMRGsweep_single(psi, ltransf,rtransf, tol=tol1)
        # Compute energy
        s0 = psi.getschmidt(-1)
        T = ltransf.T.diag_mult('t',s0).diag_mult('b',s0)
        E = T.contract(rtransf.T,'t-t,c-c,b-b')
        Esweep = np.real(E-E0)
        if config.verbose or (niter%ncanon == 0):
          print(f'[{niter}] E={Esweep/(2*persweep):0.8f} ({(Es0-Esweep):+0.4g})')
        if abs(Es0-Esweep)<Edelta:
          break
        Es0 = Esweep
        if savefile:
          pickle.dump((psi,(ltransf,rtransf),(chi,1,niter)),open(savefile,'wb'))
      Efin = Esweep/(2*persweep)
      # Restore canonical form
      mLs,mRs = psi.restore_canonical(gauge=True,tol=tol1,almost_canon=True)
      if config.verbose >= config.VDEBUG:
        psi.printcanon(compare=True)
      ltransf = self.reset_ltransfer(ltransf,mRs[-1],tol0)
      rtransf = self.reset_rtransfer(rtransf,mLs[0],tol0)
      if saveprefix:
        pickle.dump((psi,(ltransf,rtransf),Efin),
          open(f'{saveprefix}c{chi}.p','wb'))
    if savefile:
      pickle.dump((psi,(ltransf,rtransf),Efin),open(savefile,'wb'))
    return psi,Efin

def redecompose_mpo(O, tol=1e-15):
  # Debugging function to find component terms given an MPO
  # First: verify boundary properties
  N = O.N
  dphys = [O[n].shape['b'] for n in range(N)]
  dvirt = [O[n].shape['l'] for n in range(N)]
  termL = [O.Lterm] + (N-1)*[None]
  termR = [O.Rterm] + (N-1)*[None]
  BCL = O.Lboundary
  BCR = O.Rboundary
  idxf = N*[None]
  idxi = N*[None]
  # Propagate termL forwards
  for n in range(1,N):
    termL[n] = O[n-1].contract(termL[n-1],'l-r;~').trace('t-b')/dphys[n-1]
  # Propagate termR backwards
  tR = termR[0]
  for n in range(N-1,0,-1):
    tR = O[n].contract(tR,'r-l;~').trace('t-b')/dphys[n]
    termR[n] = tR
  # Check properties of terminal vectors, extract initial/final indices
  for n in range(N):
    tL = termL[n]._T
    tR = termR[n]._T
    idxf[n] = np.argmax(np.abs(tL))
    tchk = tL.copy()
    tchk[idxf[n]] -= 1
    assert np.linalg.norm(tchk) < tol
    idxi[n] = np.argmax(np.abs(tR))
    assert idxi[n] != idxf[n]
    tchk = tR.copy()
    tchk[idxi[n]] -= 1
    assert np.linalg.norm(tchk) < tol
    # Correct deviation
    for i in range(dphys[n]):
      tL[i] = (i == idxf[n])
      tR[i] = (i == idxi[n])
  assert abs(termL[0].contract(BCR,'r-l')-1) < tol
  assert abs(termR[0].contract(BCL,'l-r')-1) < tol
  assert abs(abs(BCR)-1) < tol and abs(abs(BCL)-1) < tol
  # Check term -> term x I
  for n in range(N):
    OL = O[n].contract(termL[n],'l-r;~')
    assert abs(OL - OL.id_from('t-b',OL).contract(termL[(n+1)%N],'~')) < tol
    OR = O[n].contract(termR[(n+1)%N],'r-l;~')
    assert abs(OR - OR.id_from('t-b',OR).contract(termR[n],'~')) < tol
  # Next: Generate "rules" of FSM from virtual-only perspective
  fsm_adj = [np.linalg.norm(O[n].permuted(('r','l','t','b')),axis=(2,3)) > tol for n in range(N)]
  # Check that FSM covers all states & terminates in finite steps
  for n in range(N):
    # If all states at one site are reachable, all states at the next are
    assert np.all(fsm_adj[n].dot(np.ones(dvirt[n],dtype=int)))
  # Assume max steps is N*chi^2
  nmax = N*max(dvirt)**2
  # All are reachable
  v = np.zeros(dvirt[0],dtype=int)
  v[idxi[0]] = 1
  niter = 0
  while not np.all(v):
    v = fsm_adj[niter%N].dot(v)
    assert niter < nmax
    niter += 1
  # All terminate: Set initial state to 0
  v[idxi[niter%N]] = 0
  while np.any(v):
    v = fsm_adj[niter%N].dot(v)
    niter += 1
    v[idxf[niter%N]] = 0 # Zero final state
    assert niter < nmax
  # Now collect generated sequences 
  seqs = []
  id_adj = [] # Sub-adjacency-matrix for connections that are just the identity
  Ms = [O[n].permuted(('l','r','b','t')) for n in range(N)]
  #return fsm_adj
  for n in range(N):
    id_adj.append(fsm_adj[n].copy())
    for j,i in np.argwhere(fsm_adj[n]):
      id_adj[n][j,i] = (np.linalg.norm(Ms[n][i,j,:,:]-np.identity(dphys[n])) < tol)
    seqs.append([])
    for i in np.argwhere(fsm_adj[n][:,idxi[n]]).flatten():
      if i != idxi[(n+1)%N]:
        seqs[n].extend(fsm_seq_gen(fsm_adj, idxf, n+1, N, nmax, (idxi[n],i)))
  # Now generate final outcome
  for n0 in range(N):
    for s in seqs[n0]:
      out = []
      for n in range(n0,n0+len(s)-1):
        i0,i1 = s[n-n0:n-n0+2]
        if id_adj[n%N][i1,i0]:
          # No need to record
          continue
        mat = Ms[n%N][i0,i1,:,:]
        md = np.diag(mat)
        if np.linalg.norm(mat - np.diag(md)) < tol:
          out.append((n,md))
        else:
          out.append((n,mat))
      yield out

def fsm_seq_gen(adj, idxf, n0, N, nmax, s0):
  # Recursive method to generate admissible sequences of virtual states
  idx = s0[-1]
  if idx == idxf[n0%N]:
    return [s0]
  for n in range(n0, nmax):
    idxs = adj[n%N][:,idx].nonzero()[0]
    if len(idxs) == 1:
      # Only one continuation -- keep going unless it is idxf
      idx = idxs[0]
      s0 = s0+(idx,)
      if idx == idxf[(n+1)%N]:
        return [s0]
    else:
      # Fork
      rv = []
      for idx in idxs:
        rv.extend(fsm_seq_gen(adj, idxf, n+1, N, nmax, s0+(idx,)))
      return rv
  raise ValueError('Reached maximum range')
