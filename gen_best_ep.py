import os
from glob import glob
import numpy as np
from config import Config


config = Config()

eval_txts = sorted(glob('e_results/*VD_eval.txt'))
print('eval_txts:', [_.split(os.sep)[-1] for _ in eval_txts])
score_panel = {}
sep = '&'
metric = ['sm', 'wfm', 'hce'][2]
for idx_et, eval_txt in enumerate(eval_txts):
    with open(eval_txt) as f:
        lines = [l for l in f.readlines()[3:] if '.' in l]
    for idx_line, line in enumerate(lines):
        properties = line.strip().strip(sep).split(sep)
        dataset = properties[0].strip()
        ckpt = properties[1].strip()
        if int(ckpt.split('--ep')[-1].strip()) < 0:
            continue
        targe_idx = {
            'sm': [5, 2, 2],
            'wfm': [3, 3, 8],
            'hce': [7, -1, -1]
        }[metric][['DIS5K', 'COD', 'SOD'].index(config.dataset)]
        if metric != 'hce':
            score_sm = float(properties[targe_idx].strip())
        else:
            score_sm = int(properties[targe_idx].strip().strip('.'))
        if idx_et == 0:
            score_panel[ckpt] = []
        score_panel[ckpt].append(score_sm)

metrics_min = ['hce', 'mae']
max_or_min = min if metric in metrics_min else max
score_max = max_or_min(score_panel.values(), key=lambda x: np.sum(x))

good_models = []
for k, v in score_panel.items():
    if (np.sum(v) <= np.sum(score_max)) if metric in metrics_min else (np.sum(v) >= np.sum(score_max)):
        print(k, v)
        good_models.append(k)

# Write
with open(eval_txt) as f:
    lines = f.readlines()
info4good_models = lines[:3]
for good_model in good_models:
    for idx_et, eval_txt in enumerate(eval_txts):
        with open(eval_txt) as f:
            lines = f.readlines()
        for line in lines:
            if set([good_model]) & set([_.strip() for _ in line.split(sep)]):
                info4good_models.append(line)
info4good_models.append(lines[-1])
print(''.join(info4good_models))
