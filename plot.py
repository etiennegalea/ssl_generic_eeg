import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib.lines import Line2D
from matplotlib import cm
import seaborn as sns

import umap.plot
import umap.umap_ as UMAP
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
from datetime import datetime

from helper_funcs import HelperFuncs as hf



class Plot:
    
    def __init__(self, dataset_name, metadata_string, save=True, show=False):
        self.save = save
        self.show = show
        self.dataset_name = dataset_name
        self.metadata_string = metadata_string
        self.date = hf.get_datetime(dateonly=False)

    def _plot(self, figure, title, _title=''):
        hf.check_dir(f'plots/{self.dataset_name}')
        # save fig
        if self.save:
            figure.savefig(f'plots/{self.dataset_name}/{self.date}_{title+_title}_{self.metadata_string}.png')
        # show fig
        if self.show:
            figure.show()
        figure.close()

    # Extract loss and balanced accuracy values for plotting from history object
    def plot_acc(self, clf_history):
        print(':: plotting accuracy per epoch... ', end='')
        df = pd.DataFrame(clf_history)

        df['train_acc'] *= 100
        df['valid_acc'] *= 100

        ys1 = ['train_loss', 'valid_loss']
        ys2 = ['train_acc', 'valid_acc']
        styles = ['-', ':']
        markers = ['.', '.']

        plt.style.use('seaborn-talk')

        fig, ax1 = plt.subplots(figsize=(16, 6))
        ax2 = ax1.twinx()
        for y1, y2, style, marker in zip(ys1, ys2, styles, markers):
            ax1.plot(df['epoch'], df[y1], ls=style, marker=marker, ms=7,
                    c='tab:blue', label=y1)
            ax2.plot(df['epoch'], df[y2], ls=style, marker=marker, ms=7,
                    c='tab:orange', label=y2)

        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylabel('Loss', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.set_ylabel('Accuracy [%]', color='tab:orange')
        ax1.set_xlabel('Epoch')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2)

        plt.tight_layout()
        self._plot(plt, 'train_loss_acc')
        print('Done')


    def plot_confusion_matrix(self, conf_matrix, _title=''):
        print(':: plotting confusion matrix heatmap... ', end='')
        sns.heatmap(conf_matrix, annot=True, fmt='d', linewidths=2)
        self._plot(plt, 'conf_matrix', _title)
        print('Done')


    def plot_PCA(self, X, y, annotations=['W', 'N1', 'N2', 'N3', 'R']):
        print(':: plotting PCA... ', end='')

        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        
        n_stages = len(annotations)

        fig, ax = plt.subplots()
        colors = cm.get_cmap('plasma', n_stages)(range(n_stages))
        for i, stage in enumerate(annotations):
            mask = y == i
            ax.scatter(components[mask, 0], components[mask, 1], s=1, alpha=0.4,
                    color=colors[i], label=stage)
        ax.legend()

        ax.set_title('PCA')

        self._plot(plt, 'PCA')
        print('Done')


    def plot_TSNE(self, X, y, annotations=['W', 'N1', 'N2', 'N3', 'R']):
        print(':: plotting TSNE... ', end='')

        tsne = TSNE(n_components=2)
        components = tsne.fit_transform(X)

        n_stages = len(annotations)

        fig, ax = plt.subplots()
        colors = cm.get_cmap('plasma', n_stages)(range(n_stages))
        for i, stage in enumerate(annotations):
            mask = y == i
            ax.scatter(components[mask, 0], components[mask, 1], s=1, alpha=0.4,
                    color=colors[i], label=stage)
        ax.legend()

        ax.set_title('TSNE')

        self._plot(plt, 'TSNE')
        print('Done')


    def plot_UMAP(self, X, y, annotations=['W', 'N1', 'N2', 'N3', 'R']):
        print(':: plotting UMAP... ', end='')
        _umap = umap.UMAP(n_neighbors=15)
        umap_components = _umap.fit_transform(X)

        n_stages = len(annotations)

        fig, ax = plt.subplots()
        colors = cm.get_cmap('plasma', n_stages)(range(n_stages))
        for i, stage in enumerate(annotations):
            mask = y == i
            ax.scatter(umap_components[mask, 0], umap_components[mask, 1], s=1, alpha=0.4,
            color=colors[i], label=stage)
        ax.legend()

        ax.set_title('UMAP')

        self._plot(plt, 'UMAP')
        print('Done')


    # UMAP plot with connectivity
    # https://umap-learn.readthedocs.io/en/latest/plotting.html
    def plot_UMAP_connectivity(self, X, edge_bundling=False):
        print(':: plotting UMAP with connectivity... ', end='')
        title = 'UMAP_connectivity'
        mapping = umap.UMAP(n_components=2, init='random').fit(X)

        if not edge_bundling:
            umap.plot.connectivity(mapping, show_points=True)
        else:
            title += '_edge_bundled'
            umap.plot.connectivity(mapping, edge_bundling='hammer') # bundles edges
            
        self._plot(plt, title)
        print('Done')


    # 3D UMAP plot (plotly)
    def plot_UMAP_3d(self, X, y):
        print(':: plotting 3D UMAP... ', end='')
        umap_3d = UMAP(n_components=3, init='random', random_state=0)
        proj_3d = umap_3d.fit_transform(X)
        series = pd.DataFrame(y, columns=['annots'])

        fig_3d = px.scatter_3d(
            proj_3d, x=0, y=1, z=2,
            color=series.annots, labels={'color': 'annots'}
        )

        fig_3d.update_layout(
            autosize=False,
            width=850,
            height=850
        )
        fig_3d.update_traces(marker_size=3)
        hf.check_dir(f'plots/{self.dataset_name}')
        fig_3d.write_html(f'plots/{self.dataset_name}/{self.date}_UMAP_3d_{self.metadata_string}.html')
        print('Done')

    # Plot learning curves for fully-supervised and self-supervised logistic regression
    def plot_learning_curves_sklearn(self, ssl_train_sizes, raw_train_sizes, ssl_train_scores, ssl_test_scores, raw_train_scores, raw_test_scores, dataset_name, scoring='balanced_accuracy'):
        # create additional features
        ssl_train_scores_mean, ssl_test_scores_mean = np.mean(ssl_train_scores, axis=1), np.mean(ssl_test_scores, axis=1)
        ssl_train_scores_std, ssl_test_scores_std = np.std(ssl_train_scores, axis=1), np.std(ssl_test_scores, axis=1)
        raw_train_scores_mean, raw_test_scores_mean = np.mean(raw_train_scores, axis=1), np.mean(raw_test_scores, axis=1)
        raw_train_scores_std, raw_test_scores_std = np.std(raw_train_scores, axis=1), np.std(raw_test_scores, axis=1)
        

        print(':: plotting learning curves (sklearn)... ', end='')
        _, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.grid()

        # fill in std deviation for SSL
        # ax.fill_between(
        #     ssl_train_sizes,
        #     ssl_train_scores_mean - ssl_train_scores_std,
        #     ssl_train_scores_mean + ssl_train_scores_std,
        #     alpha=0.1,
        #     color="r",
        # )
        ax.fill_between(
            ssl_train_sizes,
            ssl_test_scores_mean - ssl_test_scores_std,
            ssl_test_scores_mean + ssl_test_scores_std,
            alpha=0.1,
            color="r",
        )

        # fill in std deviation for FS
        # ax.fill_between(
        #     raw_train_sizes,
        #     raw_train_scores_mean - raw_train_scores_std,
        #     raw_train_scores_mean + raw_train_scores_std,
        #     alpha=0.1,
        #     color="g",
        # )
        ax.fill_between(
            raw_train_sizes,
            raw_test_scores_mean - raw_test_scores_std,
            raw_test_scores_mean + raw_test_scores_std,
            alpha=0.1,
            color="g",
        )

        # plt.ylim(0, 1)
        ax.set_xlabel("Training examples")
        ax.set_ylabel(scoring)

        ax.set_title(f'Fully-Supervised and Self-Supervised Logistic Regression scores per Training Example for {dataset_name} dataset')
        # plt.plot(ssl_train_sizes, ssl_train_scores_mean, color='r', label='SSL Training Score')
        plt.plot(ssl_train_sizes, ssl_test_scores_mean, '-', color='r', label='SSL')
        # plt.plot(raw_train_sizes, raw_train_scores_mean, color='g', label='FS Training Score')
        plt.plot(raw_train_sizes, raw_test_scores_mean, '-', color='g', label='FS')
        plt.legend(loc="best")

        self._plot(plt, 'logit_learning_curves')
        print('Done')




    # Plot learning curves for fully-supervised and self-supervised logistic regression
    def plot_learning_curves(self, ssl_space, raw_space, ssl_train_scores, raw_train_scores, dataset_name):

        # get mean and std dev
        ssl_train_scores_avg, raw_train_scores_avg = np.mean(ssl_train_scores, axis=1),  np.mean(raw_train_scores, axis=1)
        ssl_train_scores_std, raw_train_scores_std = np.std(ssl_train_scores, axis=1), np.std(raw_train_scores, axis=1)

        print(':: plotting learning curves... ', end='')
        _, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.grid()

        # insert 0 in the beginning of each linspace
        ssl_space, raw_space = np.insert(ssl_space, 0, 0), np.insert(raw_space, 0, 0)

        # SSL
        ax.fill_between(
            ssl_space,
            ssl_train_scores_avg - ssl_train_scores_std,
            ssl_train_scores_avg + ssl_train_scores_std,
            alpha=0.1,
            color="g",
        )
        # FS
        ax.fill_between(
            raw_space,
            raw_train_scores_avg - raw_train_scores_std,
            raw_train_scores_avg + raw_train_scores_std,
            alpha=0.1,
            color="r",
        )

        ax.set_xlabel("Training examples")
        ax.set_ylabel("Accuracy")

        ax.set_title(f'Fully-Supervised and Self-Supervised Logistic Regression scores per Training Example')
        plt.plot(ssl_space, ssl_train_scores_avg, color='g', label='SSL Training Scores')
        plt.plot(raw_space, raw_train_scores_avg, color='r', label='Raw Training Scores')
        plt.legend(loc="best")

        self._plot(plt, 'logit_learning_curves')

        print('Done')