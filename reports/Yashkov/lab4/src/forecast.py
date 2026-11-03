"""МРЗвИС, лабораторная №4, вариант 10. Только NumPy и стандартная библиотека.

Ряд: x(k)=2.2 cos(0.27 k)+1.1 sin(0.62 k), k=0,...,199.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

VARIANT=10
A,B,C,D=2.2,0.27,1.1,0.62
TRAIN_END=140
EE=1e-4
MAX_EPOCHS=20_000
DEFAULT_LR=0.12
SEEDS=(10,20,30,40,50)


def generate_series():
    k=np.arange(200,dtype=np.float64)
    return A*np.cos(B*k)+C*np.sin(D*k)


def train_minmax(raw):
    x=np.asarray(raw)
    low, high=float(np.min(x[:TRAIN_END])),float(np.max(x[:TRAIN_END]))
    if high<=low:raise ValueError('Нулевой диапазон ряда')
    return low,high


def normalize(x,low,high):
    return (np.asarray(x,dtype=np.float64)-low)/(high-low)


def denormalize(x,low,high):
    return low+(high-low)*np.asarray(x,dtype=np.float64)


def make_windows(normalized,p):
    """Цель индекс k. Окно x[k-p:k], 0<=k<=199. Тест может использовать историю train."""
    normalized=np.asarray(normalized,dtype=np.float64)
    if not(1<=p<TRAIN_END):raise ValueError('Недопустимый p')
    xx=np.asarray([normalized[k-p:k] for k in range(p,len(normalized))])
    yy=np.asarray([normalized[k] for k in range(p,len(normalized))])
    kk=np.arange(p,len(normalized),dtype=int)
    tr=kk<TRAIN_END
    return (xx[tr],yy[tr],kk[tr]),(xx[~tr],yy[~tr],kk[~tr])


def sigmoid(x):
    x=np.clip(x,-50.,50.)
    return 1.0/(1.0+np.exp(-x))


class MLP:
    """MLP p-N-1. Hidden sigmoid, linear output, SGD online (one example/update)."""
    def __init__(self,p,n,seed,lr=DEFAULT_LR):
        self.p,self.n,self.lr=int(p),int(n),float(lr)
        r=np.random.RandomState(int(seed))
        self.w1=r.uniform(-.1,.1,(p,n)).astype(np.float64)
        self.b1=r.uniform(-.1,.1,n).astype(np.float64)
        self.w2=r.uniform(-.1,.1,n).astype(np.float64)
        self.b2=float(r.uniform(-.1,.1))

    def predict(self,x):
        x=np.asarray(x,dtype=np.float64)
        h=sigmoid(x@self.w1+self.b1)
        return h@self.w2+self.b2

    def train_step(self,x,target):
        """Градиент L=1/2(ŷ-y)^2; веса обновляются после КАЖДОГО примера."""
        h=sigmoid(x@self.w1+self.b1)
        out=float(h@self.w2+self.b2)
        delta=out-float(target)
        # Производная скрытого слоя: δ_h=δ_out*w2*h*(1-h).
        delta_h=delta*self.w2*h*(1-h)
        self.w2-=self.lr*delta*h
        self.b2-=self.lr*delta
        self.w1-=self.lr*np.outer(x,delta_h)
        self.b1-=self.lr*delta_h

    def fit(self,x,y,seed,max_epochs=MAX_EPOCHS,ee=EE,keep_history=True):
        rng=np.random.RandomState(1000+int(seed))
        order=np.arange(len(y));history=[]
        for epoch in range(1,max_epochs+1):
            rng.shuffle(order)
            for j in order:self.train_step(x[j],y[j])
            # Es = mean_i (ŷ_i-y_i)^2, в нормализованной шкале.
            es=float(np.mean((self.predict(x)-y)**2))
            if keep_history:history.append(es)
            if es<=ee:break
        return {'epochs':epoch,'Es':es,'converged':bool(es<=ee),'history':np.asarray(history)}

    def save(self,path):
        np.savez(path,w1=self.w1,b1=self.b1,w2=self.w2,b2=self.b2,p=self.p,n=self.n,lr=self.lr)

    @classmethod
    def load(cls,path):
        z=np.load(path);obj=cls(int(z['p']),int(z['n']),seed=10,lr=float(z['lr']))
        obj.w1=z['w1'];obj.b1=z['b1'];obj.w2=z['w2'];obj.b2=float(z['b2'])
        return obj


def metrics(y,yhat):
    y=np.asarray(y,dtype=np.float64);yhat=np.asarray(yhat,dtype=np.float64)
    err=yhat-y
    mae=float(np.mean(np.abs(err)))
    rmse=float(np.sqrt(np.mean(err**2)))
    denom=np.abs(y)+np.abs(yhat)
    # sMAPE безопасен вблизи переходов ряда через ноль.
    smape=float(100*np.mean(np.divide(2*np.abs(err),denom,out=np.zeros_like(err),where=denom>1e-12)))
    return {'MAE':mae,'RMSE':rmse,'sMAPE':smape}


def forecast_recursive(model,norm,start=140,steps=30):
    history=list(np.asarray(norm[:start],dtype=float));pred=[]
    for _ in range(steps):
        v=float(model.predict(np.asarray(history[-model.p:])))
        pred.append(v);history.append(v)
    return np.asarray(pred)


def infer(model,original,low,high,k=None,values=None):
    if (k is None)==(values is None):raise ValueError('Нужно выбрать k ИЛИ значения')
    if k is not None:
        if not(model.p<=k<len(original)):raise ValueError('k вне диапазона')
        raw=np.asarray(original[k-model.p:k],dtype=float)
        true=float(original[k])
    else:
        raw=np.asarray(values,dtype=float)
        if raw.shape!=(model.p,):raise ValueError(f'Требуется ровно {model.p} чисел')
        true=None
    xn=normalize(raw,low,high)
    pn=float(model.predict(xn))
    pr=float(denormalize(pn,low,high))
    return {'k':k,'inputs':raw.tolist(),'norm_inputs':xn.tolist(),
            'forecast_norm':pn,'forecast':pr,'true':true,
            'abs_error':None if true is None else abs(pr-true)}
