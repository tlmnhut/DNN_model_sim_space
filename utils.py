import pickle
import glob
import numpy as np


def get_quantile_apoz():
    cate_list = np.load('../data/priya/hsim_peterson.npz').files
    apoz_stats = []
    for cate in cate_list:
        with open(f'apoz_stats/vgg_peterson_{cate}_apoz_fc.pkl', 'rb') as f:
            apoz_stats.append(pickle.load(f)[-1])
    apoz_stats = np.array(apoz_stats)
    # print(apoz_stats.shape)

    quantile_list = []
    for i in range(apoz_stats.shape[0]):
        quantile_cate = []
        for j in range(4096-20, 0, -20):
            keep_idx = np.argpartition(apoz_stats[i], j)[:j]
            quantile_cate.append(np.max(apoz_stats[i][keep_idx]))
        quantile_list.append(quantile_cate)
    # print(quantile_list)

    for i in range(apoz_stats.shape[0]):
        for quantile in np.arange(0.9, 0.4, -0.1):
            min_apoz = min(quantile_list[i], key=lambda x:abs(x-quantile))
            print(min_apoz, quantile_list[i].index(min_apoz))


def test_sign():
    cate_list = np.load('../data/priya/hsim_peterson.npz').files
    score_cate_list = []
    for i in range(len(cate_list)):
        score_cate = np.load(f'apoz_stats/2OI_vgg16_{cate_list[i]}_pearson_numnode_4096_16_20.npy',
                             allow_pickle=True).astype(float)
        score_cate_list.append(score_cate[i])
    score_cate_list = np.array(score_cate_list)
    print(score_cate_list.shape)

    rand_list = []
    for path in glob.glob('apoz_stats/2OI_vgg16_random/*'):
        rand_list.append(np.load(path, allow_pickle=True).astype(float))
    rand_list = np.array(rand_list)

