import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_his_apoz(apoz_file, save_fig):
    with open(apoz_file, 'rb') as f:
        apoz_stat = pickle.load(f)[-1]
    ax = sns.histplot(data=apoz_stat, bins=np.arange(0, 1.1, 0.1), stat='percent')
    ax.tick_params(labelsize=13)
    ax.set_xlabel("APoZ", fontsize=17)
    ax.set_ylabel("% of nodes", fontsize=17)
    ax.get_figure().savefig(save_fig)

    hist, bins = np.histogram(apoz_stat, bins=np.arange(0, 1.1, 0.1), density=False)
    print(hist, bins, hist / np.sum(hist), np.sum(hist))
    fig1, ax1 = plt.subplots()
    ax1 = plt.hist(apoz_stat, bins=np.arange(0, 1.1, 0.1), density=True)
    fig1.savefig('./figures/tmp.png')


def plot_his_apoz_multi(save_fig):
    cate_list = np.load('../data/priya/hsim_peterson.npz').files
    apoz_mat = []
    for cate in cate_list:
        with open(f'apoz_stats/vgg_peterson_{cate}_apoz_fc.pkl', 'rb') as f:
            apoz_stat = pickle.load(f)[-1]
        apoz_mat.append(apoz_stat)
    apoz_mat = np.array(apoz_mat)
    apoz_df = pd.DataFrame(apoz_mat.T, columns=cate_list)
    # print(apoz_df)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))
    for i, column in enumerate(apoz_df):
        # print(column)
        # dist, bins = np.histogram(apoz_df[column], bins=np.arange(0, 1.1, 0.1))
        # print(dist/sum(dist), bins)

        sub_plot = sns.histplot(apoz_df[column],
                                binwidth=0.05,
                                ax=axes[i // (2+1), i % 3])
        sub_plot.set_title(column, fontsize=20, weight='bold')
        sub_plot.set(ylim=(0, 1200), xlabel=None, ylabel=None)
        sub_plot.tick_params(labelsize=17)
    axes[0, 0].set_ylabel("number of nodes", fontsize=17)
    axes[1, 0].set_ylabel("number of nodes", fontsize=17)
    axes[1, 0].set_xlabel("avg % of 0 (APoZ)", fontsize=17)
    axes[1, 1].set_xlabel("avg % of 0 (APoZ)", fontsize=17)
    axes[1, 2].set_xlabel("avg % of 0 (APoZ)", fontsize=17)
    # fig.suptitle('Distribution of the percentage of zero activations (APoZ) in 6 categories',
    #              fontsize=25)

    fig.savefig(save_fig)


def plot_his_apoz_multi_cumulative(save_fig):
    cate_list = np.load('../data/priya/hsim_peterson.npz').files
    cum_hist_list = []
    for cate in cate_list:
        with open(f'apoz_stats/vgg_peterson_{cate}_apoz_fc.pkl', 'rb') as f:
            apoz_stat = pickle.load(f)[-1]
        hist, bins = np.histogram(apoz_stat, bins=np.arange(0, 1.1, 0.1))
        cum_hist = np.cumsum(hist)
        cum_hist_list.append(cum_hist)
    cum_hist_list = np.array(cum_hist_list) / cum_hist_list[0][-1] * 100
    cum_hist_df = pd.DataFrame(cum_hist_list.T, columns=cate_list)
    # print(cum_hist_df)

    fig = sns.lineplot(data=cum_hist_df)
    fig.set_xticks(np.arange(0, 10, 1))
    fig.set_xticklabels([np.round(i, 2) for i in np.arange(10, 110, 10)])
    fig.set_ylabel("% of node", size=15)
    fig.set_xlabel("PoZ", size=15)
    fig.legend(fontsize=13)

    # plt.tight_layout()
    plt.subplots_adjust(top=0.99, bottom=0.1, left=0.1, right=0.99)
    fig.get_figure().savefig(save_fig)


if __name__ == '__main__':
    # plot_his_apoz(apoz_file='apoz_stats/vgg_peterson_fruits_apoz_fc.pkl',
    #               save_fig='./figures/apoz_hist_fruit.png')

    # plot_his_apoz_multi(save_fig='./figures/apoz_hist_all.png')
    plot_his_apoz_multi_cumulative(save_fig='./figures/apoz_hist_all_cum.png')
