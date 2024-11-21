# Установка:
1. Скачать папки assets, model и scene по ссылке https://disk.yandex.com/d/HlCCxiESonoKzQ в репозиторий.
2. Установить необходимые модули:
   ```
   ./python.sh -m pip install ftfy regex tqdmip
   ./python.sh -m pip install git+https://github.com/openai/CLIP.git
   ./python.sh -m pip install ultralytics
   ```
3. Изменить общий путь до проекта в переменной general_path расположенной в файле configs/main_config.py
# Работа с пайплайном:
```
alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
```
в файле configs/main_cnfig.py в переменной general_path оределить путь до директории проекта

в файле train.py/eval.py в переменной в функции gymnasium.make вставить версию (пример: tasks:rlmodel-v0):

Для пункта 4.2 (Обучение с картой знаний)

rlmodel-v0 - обучение на графе знаний

rlmodel-v1 - обучение без графа знаний

необходимо получать loss следующим образом:

в isaac-sim-4.1.0/kit/python/lib/python3.10/site-packages/stable_baselines3/

sac/sac.py на строчке 277 добавить:
```
import torch
    torch.save(actor_loss, 'вставить пуь до проекта/loss.pt')
```
## обучение:
1. В переменной расположенной eval в файле configs/main_config.py установить значение False
```
PYTHON_PATH train.py
```
## инференс:
В в файле configs/main_config.py
1. выбрать в переменной load_policy модель, которую необходимо протестировать;
2. eval = True
3. задать радиус и угол начального отклонения eval_radius, eval_angle
```
PYTHON_PATH train.py
```
