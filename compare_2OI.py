import pickle
import pandas as pd
import numpy as np
import joblib
import multiprocessing
import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import seaborn as sns
from vgg import vgg16
from itertools import combinations
from scipy.stats import pearsonr
import matplotlib.pylab as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score



class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def extract_feat(net, layer, dataloader, device, prune_type):
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    net._modules['classifier'][layer].register_forward_hook(get_features('feat'))

    feats, labels = [], []
    # loop through batches
    with torch.no_grad():
        for images, label, path in dataloader:
            # print(path)
            outputs = net(images.to(device))
            feats.append(features['feat'].detach().cpu().numpy())
            labels.append(label.numpy())

    feats = np.concatenate(feats)
    labels = np.concatenate(labels)

    if prune_type == 'node':
        weight = net._modules[layer].weight.detach().cpu().numpy()
        node_mask = np.sum(np.abs(weight), axis=1) != 0
        feats = feats[:, node_mask]

    return feats.reshape(len(dataloader.dataset), -1), labels


def get_emb():
    model = vgg16(pretrained=True).to(device)
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths('../data/peterson/datasets', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=4, shuffle=False,
        num_workers=1, pin_memory=True)

    # print(model)
    feats, labels = extract_feat(net=model, layer=3, dataloader=val_loader,
                                 device=device, prune_type=None)
    # print(feats)
    # print(feats.shape)
    # np.save('./resnet_peterson', feats)
    np.savez('../data/priya/F_embeddings_peterson_after_relu.npz',
             animals=feats[120*0: 120*1],
             automobiles=feats[120*1: 120*2],
             fruits=feats[120*2: 120*3],
             furniture=feats[120*3: 120*4],
             various=feats[120*4: 120*5],
             vegetables=feats[120*5: 120*6])


def compute_sim_mat(feat_mat):
    n_sample = feat_mat.shape[0]
    sim_mat = np.ones([n_sample, n_sample])
    for comb in combinations(range(n_sample), 2):
        sim_mat[comb[0], comb[1]] = pearsonr(feat_mat[comb[0]], feat_mat[comb[1]])[0] ** 2
        # sim_mat[comb[0], comb[1]] = spearmanr(avg_features[comb[0]], avg_features[comb[1]])[0]
        # sim_mat[comb[0], comb[1]] = cosine_similarity(feat_mat[comb[0]].reshape(1, -1),
        #                                               feat_mat[comb[1]].reshape(1, -1))
        sim_mat[comb[1], comb[0]] = sim_mat[comb[0], comb[1]]
    return sim_mat


def upper_tri(r):
    # Extract off-diagonal elements of each Matrix
    ioffdiag = np.triu_indices(r.shape[0], k=1)  # indices of off-diagonal elements
    r_offdiag = r[ioffdiag]
    return r_offdiag


def compute_2OI(emb_dataset, apoz_file, save_file):
    hsim = np.load('../data/priya/hsim_peterson.npz')
    with open(apoz_file, 'rb') as f:
        apoz_stat = pickle.load(f)[-1]

    score_cate = []
    cate_list = []
    for category in hsim.files:
        print(category)
        score_list = []
        hsim_cate = hsim[category]
        emb = emb_dataset[category]
        csim_cate = compute_sim_mat(emb)
        score = pearsonr(upper_tri(hsim_cate), upper_tri(csim_cate))[0] ** 2
        score_list.append(score)

        for i in np.arange(4096-20, 0, -20):
            if not (4096 - i) % 100: print(i)
            #score = pearsonr(hsim_cate.flatten(), csim_cate.flatten())[0]
            #emb_pruned = prune_apoz(emb_dataset=emb, apoz_stat=apoz_stat, apoz_rate=i, percent=False)
            emb_pruned = prune_random(emb_dataset=emb, keep_amount=i, percent=False)
            csim_cate_pruned = compute_sim_mat(emb_pruned)
            try:
                score_pruned = pearsonr(upper_tri(hsim_cate), upper_tri(csim_cate_pruned))[0] ** 2
            except ValueError:
                score_pruned = None
                print(i, None)
            score_list.append(score_pruned)

        score_cate.append(score_list)
        cate_list.append(category)

    np.save(save_file, np.array(score_cate))


def prune_apoz(emb_dataset, apoz_stat, apoz_rate, percent=True):
    if percent:
        keep_idx = [i for i in range(len(apoz_stat)) if apoz_stat[i] <= apoz_rate]
    else:
        # keep_idx = np.argsort(-apoz_stat)[:apoz_rate]
        keep_idx = np.argpartition(apoz_stat, apoz_rate)[:apoz_rate]
    return emb_dataset[:, keep_idx]


def prune_random(emb_dataset, keep_amount, percent=False):
    if percent:
        prune_amount = 100 - keep_amount
        prune_idx = np.random.choice(emb_dataset.shape[1],
                                     int(emb_dataset.shape[1] * prune_amount), replace=False)
    else:
        prune_amount = emb_dataset.shape[1] - keep_amount
        prune_idx = np.random.choice(emb_dataset.shape[1],
                                     prune_amount, replace=False)
    return np.delete(emb_dataset, prune_idx, axis=1)


def plot_res(score_cate, save_file):
    plt.figure(figsize=(20, 15))

    # with open(apoz_file, 'rb') as f:
    #     apoz_stat = pickle.load(f)[-1]
    cate_list = np.load('../data/priya/hsim_peterson.npz').files
    # num_alive = ['4096, 1.000']
    # for r in np.arange(4096-20, 0, -20):
    #     # keep_idx = [i for i in range(len(apoz_stat)) if apoz_stat[i] <= r]
    #     # num_alive.append(str(len(keep_idx)))
    #     idx = np.argpartition(apoz_stat, r)[:r][-1]
    #     num_alive.append(str(r) + ', ' + '{:.3f}'.format(apoz_stat[idx]))
    num_alive = []
    for r in np.arange(4096, 0, -20):
        num_alive.append('{:.0f}'.format(r / 4096 * 100))
    # x_ax = num_alive[0: 180: 20] + num_alive[180:-1]


    plot_df = pd.DataFrame(data=score_cate.T, index=num_alive, columns=cate_list)
    # plot_df = plot_df[plot_df.index.isin(x_ax)]
    print(plot_df)
    fig = sns.lineplot(data=plot_df, lw=5,
                       dashes=False, sort=False)

    plt.legend(fontsize=38)
    #fig.set(xscale='log')
    fig.set_ylim([0, 0.65])
    fig.set_yticklabels([round(i, 1) for i in fig.get_yticks()], size=40)
    plt.xticks(
        # x_ax[::2],  # Odd rows only
        num_alive[::12],
        rotation=90,
        fontsize=40
        # horizontalalignment='right',
        # fontweight='light'
    )
    fig.set_xlabel("% of dim remain", fontsize=40)
    fig.set_ylabel("Pearson $R^2$", fontsize=40)
    # fig.set_title('2OI, randomly pruned VGG-16 vs. human similarity judgement', fontsize=30)

    # # max_scores = np.array([[num_alive[np.argmax(cate)], cate[np.argmax(cate)]] for cate in score_cate])
    # idx_max, max_scores = [], []
    # for cate in score_cate:
    #     idx_max.append(num_alive[np.nanargmax(cate)])
    #     max_scores.append(cate[np.nanargmax(cate)])
    # # max_scores = pd.DataFrame(max_scores, columns=['idx', 'value'])
    # # fig = sns.scatterplot(data=max_scores, x='idx', y='value', marker="+")
    # plt.scatter(idx_max, max_scores, marker="x", s=200, c='black', linewidths=4.0)

    fig.get_figure().savefig(save_file)


# def plot_all_scores():
#     cate_list = np.load('../data/priya/hsim_peterson.npz').files
#     num_alive = []
#     for r in np.arange(4096, 0, -20):
#         num_alive.append('{:.0f}'.format(r / 4096 * 100))
#
#     fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(20, 15))
#
#     score_cate = np.load('./apoz_stats/2OI_vgg16_furniture_pearson_numnode_4096_16_20.npy',
#                          allow_pickle=True).astype(float)
#     plot_df = pd.DataFrame(data=score_cate.T, index=num_alive, columns=cate_list)
#     sns.lineplot(data=plot_df, lw=4, dashes=False, sort=False,
#                  ax=axs[1, 0])
#     idx_max, max_scores = [], []
#     for cate in score_cate:
#         idx_max.append(num_alive[np.nanargmax(cate)])
#         max_scores.append(cate[np.nanargmax(cate)])
#     # max_scores = pd.DataFrame(max_scores, columns=['idx', 'value'])
#     # fig = sns.scatterplot(data=max_scores, x='idx', y='value', marker="+")
#     axs[1, 0].scatter(idx_max, max_scores, marker="x", s=200, c='black', linewidths=4.0)
#
#
#
#     sns.lineplot(data=plot_df, lw=4, dashes=False, sort=False,
#                  ax=axs[1, 1])
#     sns.lineplot(data=plot_df, lw=4, dashes=False, sort=False,
#                  ax=axs[1, 2])
#
#     # axs[2, 0] = sns.lineplot(data=plot_df, lw=5, dashes=False, sort=False)
#     # axs[2, 1] = sns.lineplot(data=plot_df, lw=5, dashes=False, sort=False)
#     # axs[2, 2] = sns.lineplot(data=plot_df, lw=5, dashes=False, sort=False)
#
#     handles, labels = axs[1, 0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center')
#
#     fig.savefig('./figures/plot.png')


def plot_all():
    cate_list = np.load('../data/priya/hsim_peterson.npz').files
    num_alive = []
    for r in np.arange(4096, 0, -20):
        num_alive.append('{:.0f}'.format(r / 4096 * 100))

    score_cate_list = []
    for i in range(len(cate_list)):
        score_cate = np.load(f'apoz_stats/2OI_vgg16_{cate_list[i]}_pearson_numnode_4096_16_20.npy',
                             allow_pickle=True).astype(float)
        score_cate_list.append(score_cate[i])
    score_cate_list = np.array(score_cate_list)

    plot_df = pd.DataFrame(data=score_cate_list.T, index=num_alive, columns=cate_list)
    # print(plot_df)

    plt.figure(figsize=(20, 15))
    fig = sns.lineplot(data=plot_df, lw=5,
                       dashes=False, sort=False)
    plt.legend(fontsize=30)
    fig.set_ylim([0, 0.65])
    fig.set_yticklabels([round(i, 1) for i in fig.get_yticks()], size=30)
    plt.xticks(
        num_alive[::12],  # Odd rows only
        rotation=90,
        fontsize=30
        # horizontalalignment='right',
        # fontweight='light'
    )
    fig.set_xlabel("% node remain", fontsize=30)
    fig.set_ylabel("Pearson R^2", fontsize=30)
    # fig.set_title('Compare repr. between pruned VGG-16 and human similarity judgement', fontsize=30)

    idx_max, max_scores = [], []
    for cate in score_cate_list:
        idx_max.append(num_alive[np.nanargmax(cate)])
        max_scores.append(cate[np.nanargmax(cate)])
    plt.scatter(idx_max, max_scores, marker="x", s=200, c='black', linewidths=4.0)



    fig.get_figure().savefig('./figures/plot_all.png')


def plot_random():
    score_random = []
    for i in range(48):
        score_cat = np.load(f'apoz_stats/2OI_vgg16_random/2OI_vgg16_random{i}_pearson_numnode_4096_16_20.npy',
                            allow_pickle=True).astype(float)
        score_random.append(score_cat)
    score_random = np.array(score_random).astype(float)
    plot_res(np.mean(score_random, axis=0), save_file='./figures/plot_random_all_1.png')


def compute_jaccard():
    def _jaccard(set_a, set_b):
        return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

    cate_list = np.load('../data/priya/hsim_peterson.npz').files
    imp_idx_list = []
    for cate in cate_list:
        with open(f'apoz_stats/vgg_peterson_{cate}_apoz_fc.pkl', 'rb') as f:
            apoz_stat = pickle.load(f)[-1]
        # imp_idx_list.append(np.argpartition(apoz_stat, 410)[: 410])
        imp_idx_list.append(np.argpartition(apoz_stat, 410)[410:])

    jaccard_score_arr = np.ones([len(cate_list), len(cate_list)])
    for comb in combinations(range(len(cate_list)), 2):
        jaccard_score_arr[comb[0], comb[1]] = _jaccard(set(imp_idx_list[comb[0]]), set(imp_idx_list[comb[1]]))
        # jaccard_score_arr[comb[0], comb[1]] = jaccard_score(imp_idx_list[comb[0]], imp_idx_list[comb[1]], average=None)
        jaccard_score_arr[comb[1], comb[0]] = jaccard_score_arr[comb[0], comb[1]]

    df_jaccard_score = pd.DataFrame(data=jaccard_score_arr, index=cate_list, columns=cate_list)
    # sns.set_theme(style="white")
    mask = np.triu(np.ones_like(jaccard_score_arr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 9))
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)
    ax = sns.heatmap(df_jaccard_score, mask=mask, annot=True,
                     linewidths=.5,
                     cmap='rocket',
                     # vmin=0.05,
                     # vmax=0.40,
                     )
    plt.yticks(va='center')
    ax.tick_params(labelsize=12)
    ax.get_figure().savefig('./figures/jaccard_score_90.png')


def cross_validate(emb_dataset, apoz_file, category, n_folds):
    def _one_step_cv(trial):
        val_idx = list(range(int(trial / n_folds * train_mat_all.shape[0]),
                             int((trial + 1) / n_folds * train_mat_all.shape[0])))
        val_mat = train_mat_all[val_idx]
        train_idx = list(set(list(range(train_mat_all.shape[0]))) - set(val_idx))
        train_mat = train_mat_all[train_idx]
        # print(train_idx, val_idx)

        score_list = []
        csim_cate = compute_sim_mat(train_mat)
        score = pearsonr(upper_tri(hsim_cate), upper_tri(csim_cate))[0] ** 2
        score_list.append(score)
        for i in np.arange(4096 - 1, 0, -1):
            if not (4096 - i) % 100: print(i)
            emb_pruned = prune_apoz(emb_dataset=train_mat, apoz_stat=apoz_stat, apoz_rate=i, percent=False)
            csim_cate_pruned = compute_sim_mat(emb_pruned)
            try:
                score_pruned = pearsonr(upper_tri(hsim_cate), upper_tri(csim_cate_pruned))[0] ** 2
            except ValueError:
                score_pruned = None
                print(i, None)
            score_list.append(score_pruned)
        np.save(f'./cv/{trial}', np.array(score_list))

        max_idx = np.argmax(score_list)
        csim_cate = compute_sim_mat(val_mat)
        score_unpruned_val = pearsonr(upper_tri(hsim_cate), upper_tri(csim_cate))[0] ** 2
        emb_pruned_val = prune_apoz(emb_dataset=val_mat, apoz_stat=apoz_stat, apoz_rate=4096 - max_idx, percent=False)
        score_pruned_val = compute_sim_mat(emb_pruned_val)
        print('trial', trial, 'unpruned score train', score_list[0],
              'pruned score train', max(score_list), 'unpruned score val', score_unpruned_val,
              'pruned score val', score_pruned_val)

    hsim = np.load('../data/priya/hsim_peterson.npz')
    with open(apoz_file, 'rb') as f:
        apoz_stat = pickle.load(f)[-1]
    hsim_cate = hsim[category]
    emb = emb_dataset[category]

    n_total = emb.shape[0]
    train_mat_all = emb[0:int(n_total*0.8)]
    test_mat = emb[int(n_total*0.8):]

    max_threads = int(multiprocessing.cpu_count() * 0.7)
    joblib.Parallel(n_jobs=max_threads)(joblib.delayed(_one_step_cv)(trial) for trial in range(n_folds))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get_emb()

    # feats = np.load('../data/priya/F_embeddings_peterson_after_relu.npz')
    # compute_2OI(feats, apoz_file='./apoz_stats/vgg_peterson_furniture_apoz_fc.pkl',
    #             save_file='./apoz_stats/2OI_vgg16_furniture_pearson_numnode_4096_16_20.npy')

    # score_cate = np.load('./apoz_stats/2OI_vgg16_random4_pearson_numnode_4096_16_20.npy',
    #                      allow_pickle=True).astype(float)
    # plot_res(score_cate, save_file='./figures/plot_random4.png')
    # with open('./vgg_imagenet_apoz_fc.pkl', 'rb') as f:
    #     apoz_stat = pickle.load(f)[-1]
    #
    # fig = sns.histplot(data=apoz_stat, binwidth=0.02)
    # fig.get_figure().savefig("hist_apoz_vgg16_imagenet.png")

    # feats = np.load('../data/priya/F_embeddings_peterson_after_relu.npz', allow_pickle=True)
    # for i in range(60, 70, 1):
    #     print(i)
    #     compute_2OI(feats, apoz_file='./apoz_stats/vgg_peterson_furniture_apoz_fc.pkl',
    #                 save_file=f'./apoz_stats/2OI_vgg16_random{i}_pearson_numnode_4096_16_20.npy')

    plot_all()
    # plot_random()

    # compute_jaccard()

