import os
from glob import glob
import numpy as np
from config import Config


config = Config()

eval_txts = sorted(glob('e_results/*_eval.txt'))
print('eval_txts:', [_.split(os.sep)[-1] for _ in eval_txts])
score_panel = {}
sep = '&'
metrics = ['sm', 'wfm', 'hce']    # we used HCE for DIS and wFm for others.
if 'DIS5K' not in config.task:
    metrics.remove('hce')

for metric in metrics:
    print('Metric:', metric)
    current_line_nums = []
    for idx_et, eval_txt in enumerate(eval_txts):
        with open(eval_txt, 'r') as f:
            lines = [l for l in f.readlines()[3:] if '.' in l]
        current_line_nums.append(len(lines))
    for idx_et, eval_txt in enumerate(eval_txts):
        with open(eval_txt, 'r') as f:
            lines = [l for l in f.readlines()[3:] if '.' in l]
        for idx_line, line in enumerate(lines[:min(current_line_nums)]):    # Consist line numbers by the minimal result file.
            properties = line.strip().strip(sep).split(sep)
            dataset = properties[0].strip()
            ckpt = properties[1].strip()
            if int(ckpt.split('--epoch_')[-1].strip()) < 0:
                continue
            targe_idx = {
                'sm': [5, 2, 2, 5, 5, 2],
                'wfm': [3, 3, 8, 3, 3, 8],
                'hce': [7, -1, -1, 7, 7, -1]
            }[metric][['DIS5K', 'COD', 'HRSOD', 'General', 'General-2K', 'Matting'].index(config.task)]
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
    with open(eval_txt, 'r') as f:
        lines = f.readlines()
    info4good_models = lines[:3]
    metric_names = [m.strip() for m in lines[1].strip().strip('&').split('&')[2:]]
    testset_mean_values = {metric_name: [] for metric_name in metric_names}
    for good_model in good_models:
        for idx_et, eval_txt in enumerate(eval_txts):
            with open(eval_txt, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if set([good_model]) & set([_.strip() for _ in line.split(sep)]):
                    info4good_models.append(line)
                    metric_scores = [float(m.strip()) for m in line.strip().strip('&').split('&')[2:]]
                    for idx_score, metric_score in enumerate(metric_scores):
                        testset_mean_values[metric_names[idx_score]].append(metric_score)

    if 'DIS5K' in config.task:
        testset_mean_values_lst = ['{:<4}'.format(int(np.mean(v_lst[:-1]).round())) if name == 'HCE' else '{:.3f}'.format(np.mean(v_lst[:-1])).lstrip('0') for name, v_lst in testset_mean_values.items()]  # [:-1] to remove DIS-VD
        sample_line_for_placing_mean_values = info4good_models[-2]
        numbers_placed_well = sample_line_for_placing_mean_values.replace(sample_line_for_placing_mean_values.split('&')[1].strip(), 'DIS-TEs').strip().split('&')[3:]
        for idx_number, (number_placed_well, testset_mean_value) in enumerate(zip(numbers_placed_well, testset_mean_values_lst)):
            numbers_placed_well[idx_number] = number_placed_well.replace(number_placed_well.strip(), testset_mean_value)
        testset_mean_line = '&'.join(sample_line_for_placing_mean_values.replace(sample_line_for_placing_mean_values.split('&')[1].strip(), 'DIS-TEs').split('&')[:3] + numbers_placed_well) + '\n'
        info4good_models.append(testset_mean_line)
    info4good_models.append(lines[-1])
    info = ''.join(info4good_models)
    print(info)
    with open(os.path.join('e_results', 'eval-{}_best_on_{}.txt'.format(config.task, metric)), 'w') as f:
        f.write(info + '\n')
