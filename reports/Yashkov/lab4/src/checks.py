"""Численные и структурные тесты корректности вычислений, NumPy без ML-библиотек."""
from pathlib import Path
import json
import numpy as np
from forecast import *

def run_checks():
 r=generate_series();lo,hi=train_minmax(r);xn=normalize(r,lo,hi)
 assert r.shape==(200,) and np.allclose(r,denormalize(xn,lo,hi))
 assert abs(lo-np.min(r[:140]))<1e-13 and abs(hi-np.max(r[:140]))<1e-13
 for p in (2,4,8,16):
  (X,y,ki),(Xt,yt,kt)=make_windows(xn,p)
  assert len(y)==140-p and len(yt)==60
  assert np.array_equal(ki,np.arange(p,140))
  assert np.array_equal(kt,np.arange(140,200))
  for idx in [0,len(kt)-1]:assert np.allclose(Xt[idx],xn[kt[idx]-p:kt[idx]])
 # Numerical derivative checking at fixed parameter and input.
 m=MLP(8,5,10,lr=.12);x=xn[30:38];y=xn[38]
 h=sigmoid(x@m.w1+m.b1);pred=float(m.predict(x))
 deriv=(pred-y)*h[0]
 old=float(m.w2[0]);eps=1e-6
 m.w2[0]=old+eps;plus=.5*(float(m.predict(x))-y)**2
 m.w2[0]=old-eps;minus=.5*(float(m.predict(x))-y)**2
 m.w2[0]=old
 assert abs((plus-minus)/(2*eps)-deriv)<1e-8
 # Confirm pure NumPy online training matches cached first seed.
 Xtr,Ytr,_=make_windows(xn,8)[0]
 m.fit(Xtr,Ytr,10,max_epochs=117,ee=0)
 cached=MLP.load(Path(__file__).parent/'results'/'model_p8_N5_seed10.npz')
 assert np.max(np.abs(m.w1-cached.w1))<1e-9
 assert np.max(np.abs(m.w2-cached.w2))<1e-9
 saved=json.loads((Path(__file__).parent/'results'/'metrics.json').read_text(encoding='utf8'))
 assert len(saved['runs'])==35 and len(saved['noise'])==5
 print('PASS: формула, разбивка, отсутствие утечки min/max, скользящее окно, градиент, воспроизводимость, 35 запусков.')

if __name__=='__main__':run_checks()
