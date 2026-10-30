"""Time series MLP p-N-1 with online backpropagation (NumPy / Matplotlib)."""
import numpy as np

def series():
    k=np.arange(200,dtype=float)
    return 1.2*np.cos(.28*k)+.6*np.sin(.65*k)

def fit_scale(values):
    lo=float(values[:140].min());hi=float(values[:140].max());return lo,hi

def normalize(v,lo,hi):return (np.asarray(v)-lo)/(hi-lo)
def denormalize(v,lo,hi):return lo+(hi-lo)*np.asarray(v)

def windows(x,p):
    # For testing, prior observations from training are available, but targets never leak.
    xx=np.asarray([x[k-p:k] for k in range(p,len(x))],dtype=float)
    yy=x[p:].copy()
    inds=np.arange(p,len(x))
    return (xx[inds<140],yy[inds<140],inds[inds<140],
            xx[inds>=140],yy[inds>=140],inds[inds>=140])

def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-60,60)))

class ForecastMLP:
    def __init__(self,p=6,n=4,seed=6,lr=.08):
        r=np.random.RandomState(seed)
        self.w1=r.uniform(-.1,.1,(p,n));self.b1=r.uniform(-.1,.1,n)
        self.w2=r.uniform(-.1,.1,n);self.b2=float(r.uniform(-.1,.1))
        self.p,self.n,self.lr=p,n,lr
    def predict(self,x):
        h=sigmoid(np.asarray(x)@self.w1+self.b1)
        return h@self.w2+self.b2
    def step(self,x,y):
        h=sigmoid(x@self.w1+self.b1)
        prediction=float(h@self.w2+self.b2)
        delta=prediction-y
        dh=delta*self.w2*h*(1-h)
        self.w2-=self.lr*delta*h
        self.b2-=self.lr*delta
        self.w1-=self.lr*np.outer(x,dh)
        self.b1-=self.lr*dh
    def fit(self,x,y,ee=1e-4,max_epochs=20000,seed=6):
        rng=np.random.RandomState(54321+seed)
        order=np.arange(len(y));history=[]
        for epoch in range(1,max_epochs+1):
            rng.shuffle(order)
            for j in order:self.step(x[j],y[j])
            pred=self.predict(x)
            es=float(.5*np.mean((pred-y)**2))
            history.append(es)
            if es<=ee:break
        return {'epochs':epoch,'Es':es,'converged':es<=ee,'history':np.array(history)}

def metrics(y,pred):
    err=np.asarray(pred)-np.asarray(y)
    mae=np.mean(abs(err));rmse=np.sqrt(np.mean(err**2))
    # Series crosses 0, so symmetric MAPE is used, with zero/zero treated as 0.
    den=np.abs(np.asarray(y))+np.abs(np.asarray(pred))
    smape=100*np.mean(np.where(den>1e-12,2*abs(err)/np.maximum(den,1e-12),0.))
    return dict(MAE=float(mae),RMSE=float(rmse),sMAPE=float(smape))

def recursive(model,observed_norm,start=140,steps=30):
    history=list(observed_norm[:start])
    preds=[]
    for _ in range(steps):
        v=float(model.predict(np.asarray(history[-model.p:])))
        preds.append(v);history.append(v)
    return np.asarray(preds)

def inference(model,source,lo,hi,k=None,custom=None):
    if custom is None:
        if k is None or not(model.p<=k<200):raise ValueError('Bad k')
        original=source[k-model.p:k]
        actual=source[k]
    else:
        original=np.asarray(custom,dtype=float)
        if len(original)!=model.p:raise ValueError('Provide exactly p values')
        actual=None
    xn=normalize(original,lo,hi)
    pred_norm=float(model.predict(xn))
    pred=float(denormalize(pred_norm,lo,hi))
    return {'k':k,'inputs':original.tolist(),'inputs_norm':xn.tolist(),'pred_norm':pred_norm,
            'pred':pred,'true':None if actual is None else float(actual),
            'AE':None if actual is None else abs(pred-actual)}
