"""Быстрая демонстрация работы уже обученной модели; переобучение не нужно."""
from pathlib import Path
from forecast import MLP,generate_series,train_minmax,infer

if __name__=='__main__':
 here=Path(__file__).resolve().parent
 model=MLP.load(here/'results'/'model_p8_N5_seed10.npz')
 raw=generate_series();lo,hi=train_minmax(raw)
 print('МРЗвИС, ЛР №4, вариант 10: базовая модель p=8, N=5, seed=10')
 print('Введите k (8..199), значения x через пробел или q для выхода')
 while True:
  s=input('> ').strip()
  if s.lower() in ('q','quit','exit'):break
  try:
   if ' ' in s or ',' in s:
    vals=[float(x.replace(',','.')) for x in s.split()]
    row=infer(model,raw,lo,hi,values=vals)
   else:
    row=infer(model,raw,lo,hi,k=int(s))
   print('Нормализованные входы:',*[f'{v:.4f}' for v in row['norm_inputs']])
   print(f"Прогноз в нормализованной шкале: {row['forecast_norm']:.4f}")
   print(f"Прогноз в исходной шкале: {row['forecast']:.6f}")
   if row['true'] is not None:print(f"Эталон: {row['true']:.6f}, абсолютная ошибка: {row['abs_error']:.6f}")
  except (ValueError,TypeError) as err:print('Ошибка ввода:',err)
