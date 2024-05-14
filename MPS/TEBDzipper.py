from quantrada.MPS import mpsabstract,mpsbasic
from quantrada import config,networks
import numpy as np

logger = config.logger.getChild('TEBD')
stdlog = config.streamlog
ztol = 1e-14
gcollect = False

# This method was probably completed on home machine
def adjustH(H, lH, dt, E0=0):
  # Convert H to 1 - dt*(H-E0) (dt may be real or imaginary)
  N = H.N
  Ml = H[0].permuted(('r','t','b'))
  ir = lH[0].index('i')
  fr = lH[0].index('f')
  d0 = Ml.shape[1]
  # Adjust by dt
  Ml[:ir] *= -dt
  Ml[ir+1:] *= -dt
  # Add transition into unit
  Ml[fr] = np.identity(d0)*(1-E0*dt)
  Ol = H[0].init_like(Ml,'r,t,b')
  Ms = [Ol]
  for site in range(1,N-1):
    M = H[site].permuted(('l','r','t','b'))
    il,ir = ir,lH[site].index('i')
    # Adjustment by dt
    M[il,:ir] *= -dt
    M[il,ir+1:] *= -dt
    Ms.append(H[site].init_like(M,('l,r,t,b')))
  Mr = H[N-1].permuted(('l','t','b'))
  Mr[ir] *= -dt
  Ms.append(H[N-1].init_like(Mr,'l,t,b'))
  return mpsbasic.MPO(Ms)

def zip_site(TL,TR,chi0):
  M = TL.T.contract(TR.T,'b-b,c1-c1;t>tl,c0>bl;t>tr,c0>br')
  logger.info('Site %d/%d norm-squared %0.10f',TL.site,TR.site,
    np.real(M.trace('tl-tr,bl-br')))
  if gcollect:
    import gc
    gc.collect()
  w,vR,vL = M.eig('tl-tr,bl-br',herm=False,selection=chi0,left=True,mat=True,
    zero_tol=ztol)
  return vR,vL

def do_zipper(psi,O,chi,chi0):
  N = psi.N
  TL = mpsabstract.LeftTransfer(psi,0,'l',O,O.dagger())
  TL.compute(None)
  TLs = TL.moveby(N-2,collect=True)
  TR = mpsabstract.RightTransfer(psi,N-1,'l',O,O.dagger())
  TR.compute(None)
  mats = []
  schmidt = psi.getschmidt(N-2)
  for nr in range(N-1,0,-1):
    TL = TLs.pop()
    wR,wL1 = zip_site(TL,TR,chi0)
    if nr == N-1:
      net = networks.Network.network('A;O;L;A.l-L.tl,A.b-O.t,O.l-L.bl;'
        'L.c>l,O.b>b',psi.getTL(nr),O[nr],wR)
    else:
      net = networks.Network.network('A;O;L;R;A.l-L.tl,A.r-R.tr,A.b-O.t,'
        'O.l-L.bl,O.r-R.br;L.c>l,R.c>r,O.b>b',psi.getTL(nr),O[nr],wR,wL0)
    mats.insert(0,net.contract())
    projnet = networks.Network.network('T;BL*,TL;BR*,TR;T.t-TL.tl,T.c0-TL.bl,T.c1-BL.bl,T.b-BL.tl,TL.c-TR.c,BL.c-BR.c;TR.tr>t,TR.br>c0,BR.br>c1,BR.tr>b',
      TR.T,wR,wL1)
    TR.setvalue(projnet.contract())
    TR = TR.left(terminal=(nr==1))
    wL0 = wL1
  logger.debug('Final norm %0.10f',np.real(TR))
  net = networks.Network.network('A;O;R;A.r-R.tr,A.b-O.t,O.r-R.br;R.c>r,O.b>b',
    psi.getTL(0),O[0],wL0)
  mats.insert(0,net.contract())
  if hasattr(psi,'irrep'):
    psi1 = mpsbasic.MPSirrep(mats,psi.irrep,psi.charge_site,tol=None)
  else:
    psi1 = mpsbasic.MPS(mats,tol=None)
  logger.debug('State has norm-squared %0.10f',np.real(psi1.normsq()))
  logger.info('Enforcing canonical form on chi-%d state',chi0)
  norm1 = psi1.restore_canonical()
  logger.debug('Norm %0.10f',norm1)
  logger.info('Truncating to chi-%d',chi)
  psi1.truncate_tochi(chi)
  norm2 = psi1.restore_canonical()
  logger.debug('Norm %0.10f',norm2)
  return psi1

def do_imag_tebd_first(H,lH,psi0,chi,chi0,T,dt,delta=0,E0=0):
  O = adjustH(H,lH,dt,E0)
  for ni in range(int(np.round(T/dt))):
    psi1 = do_zipper(psi0,O,chi,chi0)
    diff = psi1.schmidtdiff(psi0)
    stdlog.info('[% 4d]\t%0.6g',ni,diff)
    if diff<delta:
      break
    psi0 = psi1
  return psi1

def do_imag_tebd_second(H,lH,psi0,chi,chi0,T,dt,delta=0,E0=0):
  Oa = adjustH(H,lH,dt/(1+1j),E0)
  Ob = adjustH(H,lH,dt/(1-1j),E0)
  for ni in range(int(T//dt)):
    psia = do_zipper(psi0,Oa,chi,chi0)
    psi1 = do_zipper(psia,Ob,chi,chi0)
    diff = psi1.schmidtdiff(psi0)
    stdlog.info('[% 4d]\t%0.6g',ni,diff)
    if diff<delta:
      break
    psi0 = psi1
  return psi1

def do_real_tebd_first(H,lH,psi0,chi,chi0,T,dt,delta=0,E0=0):
  O = adjustH(H,lH,1j*dt,E0)
  for ni in range(int(np.round(T/dt))):
    psi1 = do_zipper(psi0,O,chi,chi0)
    diff = psi1.schmidtdiff(psi0)
    stdlog.info('[% 4d]\t%0.6g',ni,diff)
    if diff<delta:
      break
    psi0 = psi1
  return psi1

def do_real_tebd_second(H,lH,psi0,chi,chi0,T,dt,delta=0,E0=0):
  Oa = adjustH(H,lH,dt*(1j+1)/2,E0)
  Ob = adjustH(H,lH,dt*(1j-1)/2,E0)
  for ni in range(int(T//dt)):
    psia = do_zipper(psi0,Oa,chi,chi0)
    psi1 = do_zipper(psia,Ob,chi,chi0)
    diff = psi1.schmidtdiff(psi0)
    stdlog.info('[% 4d]\t%0.6g',ni,diff)
    if diff<delta:
      break
    psi0 = psi1
  return psi1
