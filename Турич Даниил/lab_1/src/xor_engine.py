"""MLP 2-2-1 and single layer perceptron. NumPy is the only dependency.
All losses are totals over the four training examples, evaluated after an epoch.
"""
import numpy as np

C0, C1 = -9.0, 6.0
X_RAW = np.array([[-9.,-9.],[-9.,6.],[6.,-9.],[6.,6.]])
Y_RAW = np.array([-9.,6.,6.,-9.])
X = (X_RAW - C0) / (C1-C0)
Y = (Y_RAW - C0) / (C1-C0)

def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1 / (1+np.exp(-z))

def relu(z):
    return np.maximum(z, 0.)

class MLP:
    def __init__(self, seed=7, activation='sigmoid', loss='bce', lr=1.0):
        rng=np.random.RandomState(seed)
        self.w1=rng.uniform(-.1,.1,(2,2))
        self.b1=rng.uniform(-.1,.1,2)
        self.w2=rng.uniform(-.1,.1,2)
        self.b2=float(rng.uniform(-.1,.1))
        self.activation,self.loss,self.lr=activation,loss,lr

    def forward(self, x):
        z1=x@self.w1+self.b1
        h=sigmoid(z1) if self.activation=='sigmoid' else relu(z1)
        p=sigmoid(h@self.w2+self.b2)
        return z1,h,p

    def predict(self, x):
        return np.asarray(self.forward(np.asarray(x, dtype=float))[2])

    def train_step(self,x,t):
        z,h,p=self.forward(x)
        # BCE + sigmoid: dL/dz = p-t; MSE=1/2*(p-t)^2.
        delta=(p-t) if self.loss=='bce' else (p-t)*p*(1-p)
        dh=(self.w2*delta)*(h*(1-h) if self.activation=='sigmoid' else (z>0))
        old_w2=self.w2.copy()
        self.w2-=self.lr*delta*h
        self.b2-=self.lr*delta
        self.w1-=self.lr*np.outer(x,dh)
        self.b1-=self.lr*dh

    def total_loss(self, xx, yy):
        p=np.clip(self.predict(xx),1e-12,1-1e-12)
        if self.loss=='bce': return float(-np.sum(yy*np.log(p)+(1-yy)*np.log1p(-p)))
        return float(.5*np.sum((p-yy)**2))

    def fit(self,xx,yy,max_epochs=20000,ee=.01,shuffle=True):
        rng=np.random.RandomState(42)
        history=[]
        for epoch in range(1,max_epochs+1):
            order=rng.permutation(len(xx)) if shuffle else range(len(xx))
            for i in order:self.train_step(xx[i],yy[i])
            error=self.total_loss(xx,yy)
            history.append(error)
            if error<=ee:break
        return dict(epochs=epoch,Es=error,converged=error<=ee,history=np.asarray(history))

class SinglePerceptron:
    def __init__(self,seed=7,lr=1.):
        rng=np.random.RandomState(seed)
        self.w=rng.uniform(-.1,.1,2)
        self.b=float(rng.uniform(-.1,.1))
        self.lr=lr
    def predict(self,xx):return sigmoid(np.asarray(xx)@self.w+self.b)
    def fit(self,xx,yy,max_epochs=20000,ee=.01):
        rng=np.random.RandomState(42);h=[]
        for ep in range(1,max_epochs+1):
            for i in rng.permutation(len(xx)):
                x,t=xx[i],yy[i]
                p=float(self.predict(x)); delta=(p-t)*p*(1-p)
                self.w-=self.lr*delta*x;self.b-=self.lr*delta
            p=self.predict(xx);es=float(.5*np.sum((p-yy)**2));h.append(es)
            if es<=ee:break
        return dict(epochs=ep,Es=es,converged=es<=ee,history=np.asarray(h))

def stats(model, raw_output=False):
    out=model.predict(X)
    if raw_output:
        yr=out;cls=np.where(abs(yr-C0)<=abs(yr-C1),C0,C1)
    else:
        yr=C0+(C1-C0)*out;cls=np.where(out<.5,C0,C1)
    return {'accuracy':float(np.mean(cls==Y_RAW)), 'mae':float(np.mean(abs(yr-Y_RAW))),
            'pred_raw':np.asarray(yr).tolist(),'pred_norm':np.asarray(out).tolist(),
            'pred_class':np.asarray(cls).tolist()}

def inference(model,a,b,raw_output=False):
    if not (-10<=a<=10 and -10<=b<=10):raise ValueError('A and B must be in [-10,10]')
    pn=float(model.predict((np.array([a,b])-C0)/(C1-C0)))
    val=pn if raw_output else C0+15*pn
    cls=C0 if abs(val-C0)<=abs(val-C1) else C1
    return {'A':a,'B':b,'output_norm':pn,'output_raw':val,'class':cls}

TEST_PAIRS=[(-9,-9),(-9,6),(6,-9),(6,6),(-6,4),(0,0),(6,0),(-10,10)]
