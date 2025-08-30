import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_label_distribution(df, df_pos, df_neg, save_path="./", file_name="label_distribution.png"):
    plt.bar(np.ones((df_pos.shape[0])), df_pos.shape[0], label=f"Abnormal {round((len(df_pos)/len(df))*100)}%")
    plt.bar(np.zeros((df_neg.shape[0])), df_neg.shape[0], label="Normal")
    plt.legend()
    # df.hist(column="class", grid=False)
    plt.title(f"Class Distribution n={df.shape[0]}")
    plt.xticks([0,1])
    plt.savefig(save_path+"/"+file_name)
    plt.show()


def plot_scatter(df_pos, df_neg, independent_variables,save_path="./", file_name="scatter.png"):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14,8))
    for i, var in enumerate(independent_variables):
        pos = sorted(df_pos[var].to_numpy())
        neg = sorted(df_neg[var].to_numpy())
        x_pos = np.arange(len(pos))
        x_neg = np.arange(len(pos), len(pos) + len(neg))
        if i < 3:
            axs = axes[0,i]
        else:
            axs = axes[1,i-3]
        axs.scatter(x_pos, pos, label='1 (Abnormal)', alpha=0.7)
        axs.scatter(x_neg, neg, label='0 (Normal)', alpha=0.7)
        axs.set_title(var)
    axes[0,1].legend()
    plt.savefig(save_path+file_name)
    plt.show()


def plot_boxplot(df,independent_variables,save_path="./", file_name="boxplot.png"):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14,8))
    for i, var in enumerate(independent_variables):
        if i < 3:
            axs = axes[0,i]
        else:
            axs = axes[1,i-3]
        sns.boxplot(data=df, x="class", y=var, ax=axs)
    axes[0,1].legend()
    plt.savefig(save_path+file_name)
    plt.show()

