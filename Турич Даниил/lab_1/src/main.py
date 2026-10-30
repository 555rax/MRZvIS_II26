"""Lab 1 (variant 6). Strict specification and a separately labeled scaled diagnostic."""
import json, argparse
from pathlib import Path
from xor_engine import MLP,SinglePerceptron,X,Y,Y_RAW,stats,inference,TEST_PAIRS

def main(max_epochs=20000, interactive=False):
    models=[('MLP raw targets',MLP(seed=6,loss='mse',lr=.5),Y_RAW,True),
            ('Single raw targets',SinglePerceptron(seed=6,lr=.5),Y_RAW,True),
            ('MLP scaled diagnostic',MLP(seed=6,loss='mse',lr=.5),Y,False),
            ('Single scaled diagnostic',SinglePerceptron(seed=6,lr=.5),Y,False)]
    outputs={}
    for name,model,t,raw in models:
        fit=model.fit(X,t,max_epochs=max_epochs,ee=.01)
        final=stats(model,raw_output=raw)
        outputs[name]={**{k:v for k,v in fit.items() if k!='history'},**final,
            'tests':[inference(model,a,b,raw) for a,b in TEST_PAIRS]}
        print(name,'epochs=',fit['epochs'],'Es=',round(fit['Es'],6),
              'accuracy=',final['accuracy'],'converged=',fit['converged'])
    (Path(__file__).parent/'results').mkdir(exist_ok=True)
    with open(Path(__file__).parent/'results'/'latest_run.json','w',encoding='utf8') as f:
        json.dump(outputs,f,ensure_ascii=False,indent=2)
    if interactive:
        print('Enter A B, or q to quit')
        while True:
            line=input('A B > ').strip()
            if line.lower()=='q':break
            try:
                a,b=map(float,line.split())
                print(inference(models[2][1],a,b))
            except (ValueError,IndexError) as e:print('Input error:',e)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true')
    ap.add_argument('--interactive',action='store_true');args=ap.parse_args()
    main(1000 if args.quick else 20000,args.interactive)
