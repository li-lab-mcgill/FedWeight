import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils_log import LogUtils
from utils.utils_time import TimeUtils
from utils.constants import *


class PlotService:

    """ Initialize """

    def __init__(self, color_palette="tab10"):
        sns.set_palette(sns.color_palette(color_palette))

    """ Public methods """

    def plot_loss_hist(self, total_seed, total_round, train_loss_hist, val_loss_hist, title, file_name, log_path):

        history_list = []
        for seed in range(total_seed):

            train_history = train_loss_hist[seed]
            val_history = val_loss_hist[seed]

            round = list(range(total_round))
            trial = [str(seed) for i in range(total_round)]

            data_preproc = pd.DataFrame({
                'round': round,
                'trial': trial,
                'train': train_history,
                'validation': val_history})
            history = pd.melt(data_preproc, id_vars=['round', 'trial'])
            history_list.append(history)

        val_history_np = np.vstack(val_loss_hist)
        val_loss_mean = np.mean(val_history_np, axis=0)
        val_loss_std = np.std(val_history_np, axis=0)

        best_val_loss_idx = np.argmin(val_loss_mean)

        best_val_loss_mean = val_loss_mean[best_val_loss_idx]
        best_val_loss_std = val_loss_std[best_val_loss_idx]

        loss_df = pd.concat(history_list, ignore_index=True, sort=False)
        loss_df.columns = ["round", "trial", "type", "loss"]

        LogUtils.instance().log_info("Best val loss: {:.5f}±{:.5f} at round: {}".format(
            best_val_loss_mean, best_val_loss_std, best_val_loss_idx))

        start_time = TimeUtils.instance().get_start_time()
        file_path = "{}/{}_{}".format(log_path, start_time, file_name)

        plt.clf()
        plt.grid(linewidth=0.5)
        sns.lineplot(x='round', y='loss', hue='type',
                     data=loss_df).set(title=title)
        plt.savefig(file_path)
        plt.close()

        return best_val_loss_mean, best_val_loss_std

    def plot_pr_hist(self, total_seed, total_round, train_pr_hist, val_pr_hist, title, file_name, log_path):

        history_list = []
        for seed in range(total_seed):

            train_history = train_pr_hist[seed]
            val_history = val_pr_hist[seed]

            round = list(range(total_round))
            trial = [str(seed) for i in range(total_round)]

            data_preproc = pd.DataFrame({
                'round': round,
                'trial': trial,
                'train': train_history,
                'validation': val_history})
            history = pd.melt(data_preproc, id_vars=['round', 'trial'])
            history_list.append(history)

        val_history_np = np.vstack(val_pr_hist)
        val_pr_mean = np.mean(val_history_np, axis=0)
        val_pr_std = np.std(val_history_np, axis=0)

        best_val_pr_idx = np.argmax(val_pr_mean)

        best_val_pr_mean = val_pr_mean[best_val_pr_idx]
        best_val_pr_std = val_pr_std[best_val_pr_idx]

        pr_df = pd.concat(history_list, ignore_index=True, sort=False)
        pr_df.columns = ["round", "trial", "type", "pr"]

        LogUtils.instance().log_info("Best val pr: {:.5f}±{:.5f} at round: {}".format(
            best_val_pr_mean, best_val_pr_std, best_val_pr_idx))

        start_time = TimeUtils.instance().get_start_time()
        file_path = "{}/{}_{}".format(log_path, start_time, file_name)

        plt.clf()
        plt.grid(linewidth=0.5)
        sns.lineplot(x='round', y='pr', hue='type',
                     data=pr_df).set(title=title)
        plt.savefig(file_path)
        plt.close()

        return best_val_pr_mean, best_val_pr_std

    def plot_hist(self, train_hist, val_hist, best_val, best_val_epoch, title, file_name, log_path):

        start_time = TimeUtils.instance().get_start_time()
        file_path = "{}/{}_{}".format(log_path, start_time, file_name)

        plt.clf()
        plt.title(title)
        plt.plot(train_hist, label='train')
        plt.plot(val_hist, label='val')
        plt.hlines(best_val, 0, best_val_epoch, linestyles='dashed')
        plt.legend()
        plt.savefig(file_path)
        plt.close()
