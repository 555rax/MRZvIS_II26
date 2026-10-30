"""Optional experiment: input noise on the training set, never on test.
The Gaussian noise has sigma=0.03 in normalized input units.
Only NumPy is required; this may take a few minutes without acceleration.
"""
import json
from pathlib import Path
import numpy as np
from ts_engine import (series,fit_scale,normalize,denormalize,windows,
                       ForecastMLP,metrics)

def main():
    src=series();lo,hi=fit_scale(src)
    tr,targets,train_ids,test,_,test_ids=windows(normalize(src,lo,hi),6)
    records=[]
    for seed in (6,7,11,14,18):
        rng=np.random.RandomState(1000+seed)
        noisy=tr+rng.normal(loc=0.0,scale=0.03,size=tr.shape)
        network=ForecastMLP(6,4,seed=seed,lr=.08)
        result=network.fit(noisy,targets,ee=1e-4,max_epochs=20000,seed=seed)
        preds=denormalize(network.predict(test),lo,hi)
        records.append({'seed':seed,'epochs':result['epochs'],
              'Es':result['Es'],'converged':result['converged'],
              **metrics(src[test_ids],preds)})
        print(records[-1])
    with open(Path(__file__).parent/'results'/'noise_run.json','w',encoding='utf8') as f:
        json.dump(records,f,ensure_ascii=False,indent=2)

if __name__=='__main__':main()
