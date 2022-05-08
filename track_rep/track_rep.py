import multiprocessing
import joblib
import pathlib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from utils_track_rep import read_dataset, extract_feat, get_apoz
from compare_rep import measure_change
import sys
sys.path.append('../')
from compare_2OI import prune_apoz


def get_embedding(dataset, model_path, layer='fc2'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, testloader, idx_label = read_dataset(dataset=dataset)
    net = torch.load(model_path, map_location=torch.device(device))
    feats_unpruned, labels = extract_feat(net=net,
                                          layer=layer,
                                          dataloader=testloader,
                                          device=device)
    feats_unpruned = feats_unpruned * (feats_unpruned > 0)
    # print(feats_unpruned.shape)
    return feats_unpruned, labels, idx_label


def compare_rep(feats_mat, labels, idx_label):
    def _parallel_one_prune_mat(n_dim):
        if n_dim == feats_mat.shape[1]:
            pruned_mat = feats_mat  # error for unpruned
        else:
            pruned_mat = prune_apoz(emb_dataset=feats_mat,
                                    apoz_stat=apoz_stat,
                                    apoz_rate=n_dim,
                                    percent=False)
        scores = [measure_change(original_mat=feats_mat,
                                 changed_mat=pruned_mat,
                                 method=m,
                                 labels=labels,
                                 idx_label=idx_label)
                  for m in ['rsa']]#['procrustes', 'rsa', 'plsca', 'cca']]
        return scores

    apoz_stat = get_apoz(feats_mat)

    with joblib.Parallel(n_jobs=MAX_THREAD, prefer='threads') as parallel:
        delayed_funcs = [joblib.delayed(_parallel_one_prune_mat)(int(feats_mat.shape[1]*p))
                         for p in np.arange(1.0, 0, -0.1)]
        score_list = parallel(delayed_funcs)

    return score_list


if __name__ == '__main__':
    MAX_THREAD = int(multiprocessing.cpu_count() * 0.7)

    dataset = 'mnist'
    pathlib.Path(f'./res/lenet5/{dataset}').mkdir(parents=True, exist_ok=True)

    for model_idx in range(0, 50):
        model_path = f'../../data/models/lenet5/{dataset}/{model_idx}_model.pth.tar'

        feats, labels, idx_label = get_embedding(dataset=dataset, model_path=model_path)
        score_list = compare_rep(feats_mat=feats, labels=labels, idx_label=idx_label)
        score_list = np.hstack([np.array([[p, int(feats.shape[1]*p)] for p in np.arange(1.0, 0, -0.1)]),
                                np.array(score_list)])
        score_df = pd.DataFrame(data=score_list, columns=['p_dim', 'n_dim', 'rsa'])#['p_dim', 'n_dim', 'procrustes', 'rsa', 'plsca', 'cca'])
        score_df['dataset'] = dataset
        score_df['model_idx'] = model_idx
        print(score_df)
        score_df.to_csv(f'./res/lenet5/{dataset}/{model_idx}.csv', index=False)
