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
