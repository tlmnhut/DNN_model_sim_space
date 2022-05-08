import pandas as pd
import numpy as np
import seaborn as sns
import glob
import pathlib
import matplotlib.pyplot as plt

from track_rep import get_embedding
from utils_track_rep import get_apoz


def plot_track_rep(method):
    res_list = []
    for filename in list(glob.glob('./res/lenet5/*/*')):
        res_list.append(pd.read_csv(filename))
    res_df = pd.concat(res_list)

    fig = sns.lineplot(data=res_df, x="p_dim", y=method, hue="dataset")
    fig.set_xticks(np.arange(1.0, 0, -0.1))
    fig.set_xticklabels([np.round(i, 2) for i in np.arange(1.0, 0, -0.1)])
    fig.invert_xaxis()
    fig.set_ylabel("Pearson $R^2$")
    fig.set_xlabel("% of dim remain")
    plt.grid(linestyle='--')
    handles, labels = fig.get_legend_handles_labels()
    fig.legend(handles=handles[1:], labels=['MNIST', 'CIFAR-10'])

    fig.get_figure().savefig(f"./figs/{method}.png")


def avg_hist(dataset):
    hist_list = []
    for model_idx in range(0, 50):
        model_path = f'../../data/models/lenet5/{dataset}/{model_idx}_model.pth.tar'
        feats, labels, idx_label = get_embedding(dataset=dataset, model_path=model_path)
        apoz_stat = get_apoz(feats)
        hist, bin_edges = np.histogram(apoz_stat, bins=np.arange(0, 1.1, 0.1), density=False)
        hist_list.append(hist)
    hist_list = np.array(hist_list) / hist_list[0].sum()
    mean_hist = np.mean(hist_list, axis=0)
    std_hist = np.std(hist_list, axis=0)
    return mean_hist, std_hist


def get_highest_apoz_at_percentage_keep(dataset, load=True):
    if not load:
        apoz_stat_list = []
        for model_idx in range(0, 50):
            model_path = f'../../data/models/lenet5/{dataset}/{model_idx}_model.pth.tar'
            feats, labels, idx_label = get_embedding(dataset=dataset, model_path=model_path)
            apoz_stat = get_apoz(feats)
            apoz_stat_list.append(apoz_stat)
        apoz_stat_list = np.array(apoz_stat_list)
        np.save(f'./res/stats/apoz_{dataset}.npy', apoz_stat_list)
    else:
        apoz_stat_list = np.load(f'./res/stats/apoz_{dataset}.npy', allow_pickle=True)

    # get highest apoz at x percentage of kept dimensions
    # for percent in np.arange(1.0, 0, -0.1):
    #     n_keep_dim = int(apoz_stat_list.shape[1] * percent)
    #     idx = np.argpartition(apoz_stat_list, kth=n_keep_dim, axis=0)
    max_apoz_arr = []
    for i in range(apoz_stat_list.shape[0]):
        max_apoz_list = []
        for percent in np.arange(1.0, 0, -0.1):
            n_keep_dim = int(apoz_stat_list.shape[1] * percent)
            if n_keep_dim == apoz_stat_list.shape[1]:
                max_apoz = np.max(apoz_stat_list[i])
            else:
                idx = np.argpartition(apoz_stat_list[i], n_keep_dim)
                max_apoz = np.max(apoz_stat_list[i][idx[:n_keep_dim]])
            max_apoz_list.append(max_apoz)
        max_apoz_arr.append(max_apoz_list)
    max_apoz_arr = np.array(max_apoz_arr)

    return max_apoz_arr


def plot_hist(load=True):
    if not load:
        mnist_mean_hist, mnist_std_hist = avg_hist(dataset='mnist')
        cifar10_mean_hist, cifar10_std_hist = avg_hist(dataset='cifar10')
        hist_info = {'mnist_mean_hist': mnist_mean_hist,
                     'mnist_std_hist': mnist_std_hist,
                     'cifar10_mean_hist': cifar10_mean_hist,
                     'cifar10_std_hist': cifar10_std_hist}
        pathlib.Path('./res/stats').mkdir(parents=True, exist_ok=True)
        np.save('./res/stats/hist.npy', hist_info)
    else:
        hist_info = np.load('./res/stats/hist.npy', allow_pickle=True)
        mnist_mean_hist = hist_info.item()['mnist_mean_hist']
        mnist_std_hist = hist_info.item()['mnist_std_hist']
        cifar10_mean_hist = hist_info.item()['cifar10_mean_hist']
        cifar10_std_hist = hist_info.item()['cifar10_std_hist']

    df = pd.DataFrame([mnist_mean_hist*100, cifar10_mean_hist*100],
                      columns=[np.round(i, 1) for i in np.arange(0.1, 1.1, 0.1)],
                      index=['MNIST', 'CIFAR-10']).transpose()
    std_ = np.array([mnist_std_hist, cifar10_std_hist]) * 100
    fig = df.plot(kind='bar', yerr=std_, alpha=0.5, error_kw=dict(ecolor='k'))
    fig.set_ylabel("% of node")
    fig.set_xlabel("APoZ")
    fig.grid(axis='y', linestyle='--')
    fig.get_figure().savefig("./figs/hist_mnist_cifar10.png")


def plot_hist_n_rep(dataset, method='rsa'):
    # load hist
    hist_info = np.load('./res/stats/hist.npy', allow_pickle=True)
    if dataset == 'mnist':
        mean_hist = hist_info.item()['mnist_mean_hist']
        std_hist = hist_info.item()['mnist_std_hist']
    if dataset == 'cifar10':
        mean_hist = hist_info.item()['cifar10_mean_hist']
        std_hist = hist_info.item()['cifar10_std_hist']

    # load rep and compute std
    rep_list = []
    for filename in list(glob.glob(f'./res/lenet5/{dataset}/*')):
        rep_df = pd.read_csv(filename)
        rep_df = rep_df.iloc[::-1]  # reverse the order to align with hist
        rep_list.append(rep_df[method])
    rep_list = np.array(rep_list)
    mean_rep = np.mean(rep_list, axis=0)
    std_rep = np.std(rep_list, axis=0)

    # plot
    fig = plt.figure()  # Create matplotlib figure
    ax1 = fig.add_subplot(111)  # Create matplotlib axes
    ax2 = ax1.twinx()  # Create another axes that shares the same x-axis as ax.

    color1 = sns.color_palette("Blues", n_colors=5)
    color2 = sns.color_palette("Oranges", n_colors=5)

    df = pd.DataFrame([mean_hist * 100, mean_rep],
                      columns=[np.round(i, 1) for i in np.arange(0.1, 1.1, 0.1)],
                      index=['histogram', 'Pearson $R^2$']).transpose()
    df['histogram'].plot(kind='bar', yerr=std_hist*100, width=0.3, ax=ax1, position=1, color=color1[3])
    df['Pearson $R^2$'].plot(kind='bar', yerr=std_rep, width=0.3, ax=ax2, position=0, color=color2[1])
    ax1.set_ylabel('% of node', color=color1[4])
    ax2.set_ylabel('Pearson $R^2$', color=color2[3])

    fig.savefig("./figs/hist_rsa_cifar10.png")


def plot_max_apoz_n_rep(dataset, method='rsa'):
    # load rep and compute std
    rep_list = []
    for filename in list(glob.glob(f'./res/lenet5/{dataset}/*')):
        rep_df = pd.read_csv(filename)
        rep_list.append(rep_df[method])
    rep_list = np.array(rep_list)
    mean_rep = np.mean(rep_list, axis=0)
    std_rep = np.std(rep_list, axis=0)

    # compute max apoz
    max_apoz_arr = get_highest_apoz_at_percentage_keep(dataset=dataset) * 100
    mean_apoz = np.mean(max_apoz_arr, axis=0)
    std_apoz = np.std(max_apoz_arr, axis=0)

    # plot
    fig = plt.figure()  # Create matplotlib figure
    ax1 = fig.add_subplot(111)  # Create matplotlib axes
    ax2 = ax1.twinx()  # Create another axes that shares the same x-axis as ax.

    color1 = sns.color_palette("Blues", n_colors=10)
    color2 = sns.color_palette("Oranges", n_colors=10)

    x_axis = [np.round(i, 1) for i in np.arange(1.0, 0, -0.1)]
    ax1.plot(x_axis, mean_rep, linestyle='-', label='mean_rep', color=color1[8])
    ax1.fill_between(x_axis, mean_rep - std_rep, mean_rep + std_rep, color=color1[2])
    ax1.set_ylabel('Pearson $R^2$', color=color1[8], size=15)
    ax1.set_ylim([-0.05, 1.05])
    ax1.tick_params(axis='y', colors=color1[8])
    ax1.invert_xaxis()
    ax1.set_xticks(np.arange(1.0, 0, -0.1))
    ax1.set_xticklabels([np.round(i, 2) for i in np.arange(100, 0, -10)])
    ax1.set_xlabel('% node remain', size=15)

    ax2.plot(x_axis, mean_apoz, linestyle='--', label='mean_apoz', color=color2[8])
    ax2.fill_between(x_axis, mean_apoz - std_apoz, mean_apoz + std_apoz, color=color2[2])
    ax2.set_ylabel('max PoZ', color=color2[5], size=15)
    ax2.set_ylim([-5, 105])
    ax2.tick_params(axis='y', colors=color2[5])
    # ax2.set_xticks(np.arange(1.0, 0, -0.1))
    # ax2.set_xticklabels([np.round(i, 2) for i in np.arange(1.0, 0, -0.1)])

    plt.subplots_adjust(top=0.99, bottom=0.1, left=0.1, right=0.9)
    fig.savefig("./figs/rsa_apoz_cifar10.png")


if __name__ == '__main__':
    # plot_hist()
    # plot_track_rep(method='rsa')
    # plot_hist_n_rep(dataset='cifar10', method='rsa')
    plot_max_apoz_n_rep(dataset='cifar10', method='rsa')

