import os
from glob import glob
import numpy as np


eval_txts = sorted(glob('e_results/*_eval.txt'))
print('eval_txts:', [_.split(os.sep)[-1] for _ in eval_txts])
score_panel = {}
for idx_et, eval_txt in enumerate(eval_txts):
    with open(eval_txt) as f:
        lines = [l for l in f.readlines()[3:] if '0.' in l]
    for idx_line, line in enumerate(lines):
        properties = line.split('|')
        dataset = properties[1].strip()
        ckpt = properties[2].strip()
        score_sm = float(properties[3].strip())
        if idx_et == 0:
            score_panel[ckpt] = []
        score_panel[ckpt].append(score_sm)

score_max = max(score_panel.values(), key=lambda x: np.sum(x))

good_models = []
for k, v in score_panel.items():
    if np.sum(v) >= np.sum(score_max):
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
            if set([good_model]) & set([_.strip() for _ in line.split('|')]):
                info4good_models.append(line)
info4good_models.append(lines[-1])
print(''.join(info4good_models))
