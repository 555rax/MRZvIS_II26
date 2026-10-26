"""Строит все графики лабораторной №4 из сохранённых метрик и моделей."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from forecast import *

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':120,
     'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.22,
     'savefig.bbox':'tight'})


def save(fig,name):
    fig.savefig(OUT/name,dpi=175,facecolor='white')
    plt.close(fig)


def build():
 setup()
 data=json.loads((OUT/'metrics.json').read_text(encoding='utf8'))
 raw=generate_series();lo,hi=train_minmax(raw);sc=normalize(raw,lo,hi)
 base=MLP.load(OUT/'model_p7_N6_seed10.npz')
 (X,y,k),(Xt,yt,kt)=make_windows(sc,7)
 train=denormalize(base.predict(X),lo,hi);test=denormalize(base.predict(Xt),lo,hi)
 # 1. Source series and train/test one step overlay.
 fig,ax=plt.subplots(figsize=(10,4.8))
 ax.plot(np.arange(200),raw,'#a5b1c2',lw=1.5,label='Исходный ряд x(k)')
 ax.plot(k,train,'#2878b5',lw=1.15,alpha=.95,label='Одношаговый прогноз: train')
 ax.plot(kt,test,'#d15f3c',lw=1.45,label='Одношаговый прогноз: test')
 ax.axvline(139.5,color='#2b2b2b',ls='--',lw=1.1,label='Граница train/test')
 ax.set(xlabel='k',ylabel='x(k)',title='Вариант 5: исходный ряд и одношаговый прогноз (p=7, N=6, seed=10)')
 ax.legend(loc='lower left',ncol=2,fontsize=8);save(fig,'01_series_one_step.png')
 # 2. Loss (base and 2 alternative windows) log Y
 fig,ax=plt.subplots(figsize=(9.6,4.8))
 for p,c in [(4,'#bd632b'),(7,'#2878b5'),(14,'#40906f')]:
  h=np.loadtxt(OUT/f'convergence_p{p}_N6.csv',skiprows=1,delimiter=',')
  ax.semilogy(h[:,0],h[:,1],color=c,lw=1.8,label=f'p={p}, N=6')
 ax.axhline(EE,color='#464646',ls='--',label=f'Ee={EE:g}')
 ax.set(xlabel='Эпоха',ylabel='Es = средняя квадратичная ошибка (log scale)',title='Сходимость: базовая модель и два альтернативных окна')
 ax.legend();save(fig,'02_convergence_log.png')
 # 3. test scatter and diagonal.
 fig,ax=plt.subplots(figsize=(6.2,5.4))
 ax.scatter(raw[kt],test,c=kt,cmap='viridis',s=27,alpha=.85,label='60 тестовых точек')
 mi=min(np.min(raw[kt]),np.min(test));ma=max(np.max(raw[kt]),np.max(test));ax.plot([mi,ma],[mi,ma],'k--',lw=1,label='Идеально: y=x')
 ax.set(xlabel='Истинное значение x(k)',ylabel='Предсказание (исходная шкала)',title='Одношаговый прогноз: тестовый участок')
 ax.axis('equal');ax.legend();save(fig,'03_scatter_test.png')
 # 4. Recursive forecast and absolute error.
 rep=data['selected_run'];model=MLP.load(OUT/f"model_p{rep['p']}_N{rep['N']}_seed{rep['seed']}.npz")
 rec=denormalize(forecast_recursive(model,sc,140,30),lo,hi)
 idx=np.arange(140,170);error=np.abs(rec-raw[140:170])
 fig,ax=plt.subplots(figsize=(9.4,4.7))
 ax.plot(idx,raw[140:170],marker='.',lw=1.6,color='#334155',label='Эталон')
 ax.plot(idx,rec,marker='o',markersize=3,lw=1.35,color='#b6543a',label='Рекурсивный прогноз')
 ax.set(xlabel='k (140–169)',ylabel='x(k)',title=f"30-шаговый рекурсивный прогноз: p={rep['p']}, N={rep['N']}, seed={rep['seed']}")
 ax.legend();save(fig,'04_recursive_forecast.png')
 fig,ax=plt.subplots(figsize=(9.4,4.2))
 ax.plot(np.arange(1,31),error,'-o',lw=1.5,markersize=3,color='#286eaf',label='Абсолютная ошибка')
 ax.axhline(data['recursive']['threshold_5pct'],color='#ba4343',ls='--',label='5% полного размаха ряда')
 ax.set(xlabel='Шаг рекурсивного прогноза (1–30)',ylabel='|прогноз − эталон|',title='Абсолютная ошибка 30-шагового прогноза')
 ax.legend();save(fig,'05_recursive_error.png')
 # 5. Window study MAE + epochs.
 fig,(ax,ax2)=plt.subplots(1,2,figsize=(10.5,4.6))
 wins=[a for a in data['aggregates'] if a['N']==6]
 xs=np.arange(len(wins));ax.bar(xs,[d['MAE_test_mean'] for d in wins],yerr=[d['MAE_test_sd'] for d in wins],capsize=5,color='#5582b4')
 ax.set_xticks(xs,[str(d['p']) for d in wins]);ax.set(xlabel='Размер окна p',ylabel='Средняя MAE ± SD',title='Тестовая точность')
 ax2.bar(xs,[d['epochs_mean'] for d in wins],color='#73a59a')
 ax2.set_xticks(xs,[str(d['p']) for d in wins]);ax2.set(xlabel='Размер окна p',ylabel='Число эпох (среднее)',title='Скорость и сходимость')
 ax2.text(0,20000*.86,'5/5: не достигли Ee',ha='center',fontsize=8,color='#7d2f2f')
 fig.suptitle('Исследование длины входного окна (N=6, 5 запусков на конфигурацию)',fontsize=12)
 save(fig,'06_window_study.png')
 # 6. Neuron study.
 fig,(ax,ax2)=plt.subplots(1,2,figsize=(10.5,4.6))
 widths=sorted([a for a in data['aggregates'] if a['p']==7],key=lambda x:x['N'])
 xs=np.arange(len(widths));ax.bar(xs,[d['MAE_test_mean'] for d in widths],yerr=[d['MAE_test_sd'] for d in widths],capsize=5,color='#7385ba')
 ax.set_xticks(xs,[str(d['N']) for d in widths]);ax.set(xlabel='Число нейронов N',ylabel='Средняя MAE ± SD',title='Тестовая точность')
 ax2.bar(xs,[d['epochs_mean'] for d in widths],color='#8db1a3')
 ax2.set_xticks(xs,[str(d['N']) for d in widths]);ax2.set(xlabel='Число нейронов N',ylabel='Число эпох (среднее)',title='Сходимость')
 fig.suptitle('Влияние числа скрытых нейронов (p=7, 5 запусков)',fontsize=12)
 save(fig,'07_hidden_study.png')
 # 7. Noise with matched seeds.
 fig,ax=plt.subplots(figsize=(7.9,4.3))
 noise=data['noise'];ix=np.arange(len(noise));bw=.35
 ax.bar(ix-bw/2,[r['test_mae_baseline'] for r in noise],width=bw,color='#3c83ab',label='Без шума')
 ax.bar(ix+bw/2,[r['test_mae_clean'] for r in noise],width=bw,color='#d2855c',label='Шум на входах train, σ=0.02')
 ax.set_xticks(ix,[str(v['seed']) for v in noise]);ax.set(xlabel='Seed',ylabel='MAE на чистом тесте',title='Контролируемый эксперимент с шумом (p=7, N=6)')
 ax.legend();save(fig,'08_noise_comparison.png')
 # 8. Schematic pipeline: exact sequence from the methodical Appendix 1.
 from matplotlib.patches import FancyBboxPatch
 fig,ax=plt.subplots(figsize=(12.0,6.9));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
 steps=[
   ('1. Генерация ряда','k = 0…199'),('2. Train/Test split','140 train / 60 test'),
   ('3. Min–Max train','только k = 0…139'),('4. Sliding window','p значений → цель'),
   ('5. MLP p–N–1','sigmoid / linear'),('6. Shuffle each epoch','перемешать train-окна'),
   ('7. Online backprop','обновлять после примера'),('8. Stop: Es ≤ 10⁻⁴?','да / нет; максимум эпох'),
   ('9. Denormalization','возврат в исходную шкалу'),('10. MAE, RMSE, sMAPE','метрики train / test'),
   ('11. Recursive forecast','30 шагов с k=140')]
 xs=[.135,.378,.621,.864];ys=[.82,.515,.21]
 centers=([ (xs[j],ys[0]) for j in range(4)]
   +[(xs[3-j],ys[1]) for j in range(4)]
   +[(xs[j],ys[2]) for j in range(3)])
 for i,((title,sub),(cx,cy)) in enumerate(zip(steps,centers)):
  rect=FancyBboxPatch((cx-.105,cy-.081),.21,.162,
    boxstyle='round,pad=0.008,rounding_size=0.017',
    fc='#edf2f9' if i<4 else ('#ebf7f1' if i<8 else '#fff1e5'),
    ec='#5a6b80',lw=1.1,zorder=3)
  ax.add_patch(rect)
  ax.text(cx,cy+.025,title,fontsize=9.2,weight='bold',ha='center',va='center',zorder=5)
  ax.text(cx,cy-.035,sub,fontsize=8.3,ha='center',va='center',zorder=5)
 for i in range(10):
  start=centers[i];end=centers[i+1]
  if abs(start[1]-end[1])<.01:
   direction=1 if end[0]>start[0] else -1
   x1=start[0]+direction*.114;x2=end[0]-direction*.114
   y1=y2=start[1]
  else:
   x1=x2=start[0];y1=start[1]-.089;y2=end[1]+.09
  ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
    arrowprops=dict(arrowstyle='-|>',color='#52647b',lw=1.6),zorder=2)
 # The stopping test 'No' returns to shuffling, next epoch.
 ax.plot([centers[7][0],centers[7][0]],[centers[7][1]+.091,.681],color='#ad6549',lw=1.4)
 ax.plot([centers[7][0],centers[5][0]],[.681,.681],color='#ad6549',lw=1.4)
 ax.annotate('',xy=(centers[5][0],centers[5][1]+.091),xytext=(centers[5][0],.681),
   arrowprops=dict(arrowstyle='-|>',color='#ad6549',lw=1.4),zorder=1)
 ax.text(.38,.702,'Нет — следующая эпоха',color='#ac5438',fontsize=9,ha='center')
 ax.text(.115,.37,'Да ↓',color='#2a805b',fontsize=9)
 ax.text(.02,.045,'Не использовать: Batch update • Normalize using whole dataset • Sigmoid output • MAPE для ряда с нулевыми пересечениями',fontsize=8.5,color='#9e3a32')
 ax.set_title('Корректный пайплайн варианта 5 (образец для рукописной блок-схемы)',fontsize=13,pad=12)
 save(fig,'09_pipeline_reference.png')
 print('Created all charts in',OUT)

if __name__=='__main__':build()
