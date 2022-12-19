from . import mpsbasic
from .. import tensors
import numpy as np
import numpy.random as rand
import os, pickle

def collapse_1site(rho, basis):
  d = len(basis)
  ps = [e.dot(rho).dot(e.conj()) for e in basis]
  assert abs(sum(ps) - 1) < 1e-5
  ps = np.real(ps)
  x = rand.rand()
  ptot = 0
  for i in range(d-1):
    ptot += ps[i]
    if x < ptot:
      return i, ps[i], basis[i]
  return d-1, ps[-1], basis[-1]

def collapse(psi, basis_gen):
  T = psi[0]
  M1d = []
  Ptot = 1
  for n in range(psi.N-1):
    T = T.diag_mult('r',psi._schmidt[n])
    rho = T.contract(T,'r-r;b>t;b>b*').permuted(('b','t'))
    basis = basis_gen(n)
    i, p, e = collapse_1site(rho, basis)
    Ptot *= p
    et = tensors.Tensor.init_from(e, 'b|b', T)
    M1d.append(et)
    bc = T.contract(et,'b-b;~*')/np.sqrt(p)
    T = psi[n+1].contract(bc,'l-r;~')
  rho = T.contract(T, ';b>t;b>b*').permuted(('b','t'))
  basis = basis_gen(psi.N-1)
  i, p, e = collapse_1site(rho, basis)
  Ptot *= p
  et = tensors.Tensor.init_from(e, 'b|b', T)
  M1d.append(et)
  # Extend to 1d MPS
  vr = et.init_from([1],'r')
  Ms = [M1d[0].contract(vr,'~')]
  vl = vr.conj().renamed('r-l')
  for e in M1d[1:-1]:
    Ms.append(e.contract(vr,'~').contract(vl,'~'))
  Ms.append(M1d[-1].contract(vl,'~'))
  return mpsbasic.MPS(Ms), Ptot

def collapse_1site_tens(rho, basis):
  d = len(basis)
  ps = [rho.contract(e,'t-b*;~').contract(e,'b-b') for e in basis]
  assert abs(sum(ps) - 1) < 1e-5
  ps = np.real(ps)
  x = rand.rand()
  ptot = 0
  for i in range(d-1):
    ptot += ps[i]
    if x < ptot:
      return i, ps[i], basis[i]
  return d-1, ps[-1], basis[-1]

def collapse_tens(psi, basis_gen):
  T = psi[0]
  M1d = []
  Ptot = 1
  for n in range(psi.N-1):
    T = T.diag_mult('r',psi._schmidt[n])
    rho = T.contract(T,'r-r;b>t;b>b*')
    basis = basis_gen(n)
    i, p, e = collapse_1site_tens(rho, basis)
    Ptot *= p
    M1d.append(e)
    bc = T.contract(e,'b-b;~*')/np.sqrt(p)
    T = psi[n+1].contract(bc,'l-r;~')
  rho = T.contract(T, ';b>t;b>b*')
  basis = basis_gen(psi.N-1)
  i, p, e = collapse_1site_tens(rho, basis)
  Ptot *= p
  M1d.append(e)
  # Extend to 1d MPS
  vr = e.init_from([1],'r')
  Ms = [M1d[0].contract(vr,'~')]
  vl = vr.conj().renamed('r-l')
  for e in M1d[1:-1]:
    Ms.append(e.contract(vr,'~').contract(vl,'~'))
  Ms.append(M1d[-1].contract(vl,'~'))
  return mpsbasic.MPS(Ms), Ptot

def evenodd_sweep(psi, Us, chi):
  N = psi.N
  # Even sites
  norm = psi.tebd_left(Us[0],chi)
  for n in range(2, N-2, 2):
    norm *= psi.tebd_bulk(Us[n], chi, n)
  if N%2 == 0:
    norm *= psi.tebd_right(Us[-1], chi)
  norm *= psi.restore_canonical()
  # Odd sites
  for n in range(1,N-2,2):
    norm *= psi.tebd_bulk(Us[n], chi, n)
  if N%2 == 1:
    norm *= psi.tebd_right(Us[-1], chi)
  return norm*psi.restore_canonical()


allmethods = {'order2','evenodd'}
def do_metts(Hs, beta, chi, ntau, ngen, basis_gen, savepath,
    method='order2', nrect=10, resume=True):
  assert method in allmethods
  resume = resume and os.path.exists(savepath)
  nfile = os.path.join(savepath, 'norm.p')
  if not resume:
    os.mkdir(savepath)
  tau = beta/ntau/2
  if method == 'order2':
    tau /= 2
  Us = [H.exp('tl-bl,tr-br', -tau) for H in Hs]
  indphys = [H._dspace['bl'] for H in Hs]
  indphys.append(Hs[-1]._dspace['br'])
  psi = mpsbasic.randMPS(indphys, 1)
  if method == 'order2':
    psi.tebd_sweep(Us, chi)
  elif method == 'evenodd':
    evenodd_sweep(psi, Us, chi)
  if resume and os.path.exists(nfile):
    norms = pickle.load(open(nfile,'rb'))
  else:
    norms = []
  for step in range(ngen):
    savefile = os.path.join(savepath, f'{step}.p')
    if os.path.exists(savefile):
      psi = savefile
      continue
    if isinstance(psi, str):
      psi = pickle.load(open(psi,'rb'))
    print(f'METTS {step:>4}/{ngen}')
    psi, P = collapse(psi, basis_gen)
    norm = 1
    for t in range(ntau):
      if method == 'order2':
        if t and t%nrect == 0:
          norm *= psi.restore_canonical()
        norm *= psi.tebd_sweep(Us, chi)
      elif method == 'evenodd':
        norm *= evenodd_sweep(psi, Us, chi)
    if method != 'evenodd':
      norm *= psi.restore_canonical()
    pickle.dump(psi, open(savefile,'wb'))
    if len(norms) > step:
      print('overwriting norm')
      norms[step] = norm
    else:
      norms.append(norm)
    pickle.dump(norms, open(nfile,'wb'))
  assert len(norms) == ngen
  return norms

def psi_at(path, sample):
  return pickle.load(open(os.path.join(path,f'{sample}.p'),'rb'))

def rdm(psi, ell, n):
  T = psi[n].renamed('b-b0')
  if n != 0:
    T = T.diag_mult('l',psi._schmidt[n-1])
  for m in range(1,ell):
    T1 = psi[n+m].diag_mult('l',psi._schmidt[n+m-1]) 
    T = T.contract(T1,f'r-l;~;b>b{m},~')
  if n+ell != psi.N:
    T = T.diag_mult('r',psi._schmidt[n+ell-1])
    if n:
      parse = 'l-l,r-r;'
    else:
      parse = 'r-r;'
  else:
    parse = 'l-l;'
  parse += ','.join(f'b{m}>b{m}' for m in range(ell)) + ';'
  parse += ','.join(f'b{m}>t{m}' for m in range(ell)) + '*'
  return T.contract(T, parse)

def getfid(rho1,sqrtrho2,ell):
  endo = ','.join(f't{m}-b{m}' for m in range(ell))
  M = sqrtrho2.contract(rho1,endo+'~').contract(sqrtrho2,endo+'~')
  w = M.eig(endo,vecs=False)
  return sum(np.sqrt(w))
