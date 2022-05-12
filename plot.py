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

    def _plot(self, figure, title, _title='', format='pdf'):
        hf.check_dir(f'plots/{self.dataset_name}')
        # save fig
        if self.save:
            figure.savefig(f'plots/{self.dataset_name}/{self.date}_{title+_title}_{self.metadata_string}.{format}', format=format)
        # show fig
        if self.show:
            figure.show()
        figure.close()

    # Extract loss and balanced accuracy values for plotting from history object
    def plot_acc(self, clf_history):
        print(':: plotting accuracy per epoch... ', end='')
        # Extract loss and balanced accuracy values for plotting from history object
        df = pd.DataFrame(clf_history)

        df['train_acc'] *= 100
        df['valid_acc'] *= 100

        ys1 = ['train_loss', 'valid_loss']
        ys2 = ['train_acc', 'valid_acc']
        styles = ['-', ':']
        markers = ['.', '.']

        plt.style.use('seaborn-talk')

        fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        for y1, y2, style, marker in zip(ys1, ys2, styles, markers):
            ax[0].plot(df['epoch'], df[y1], ls=style, marker=marker, ms=7,
                    c='tab:blue', label=y1)
            lines1, labels1 = ax[0].get_legend_handles_labels()

            ax[0].tick_params(axis='y', labelcolor='tab:blue')
            ax[0].set_ylabel('Loss', color='tab:blue')
            ax[0].legend(lines1, labels1)

        for y1, y2, style, marker in zip(ys1, ys2, styles, markers):
            ax[1].grid()
            ax[1].plot(df['epoch'], df[y2], ls=style, marker=marker, ms=7,
                    c='tab:orange', label=y2)

            lines2, labels2 = ax[1].get_legend_handles_labels()

            ax[1].tick_params(axis='y', labelcolor='tab:orange')
            ax[1].set_ylabel('Accuracy [%]', color='tab:orange')
            ax[1].set_xlabel('Epoch')
            ax[1].legend(lines2, labels2)


        plt.suptitle('Pretext task performance for pretrained sleep staging model')
        plt.tight_layout()
        plt.rc('font', size=16)
        plt.rc('xtick', labelsize=16) 
        plt.rc('ytick', labelsize=16)
        plt.rc('axes', labelsize=16)

        self._plot(plt, 'train_loss_acc')
        print('Done')



    def plot_confusion_matrix(self, conf_matrix, _title=''):
        print(':: plotting confusion matrix heatmap... ', end='')
        sns.heatmap(conf_matrix, annot=True, fmt='.1f', linewidths=2)
        self._plot(plt, 'conf_matrix', _title)
        print('Done')


    def plot_PCA(self, X, y, annotations=['W', 'N1', 'N2', 'N3', 'R'], descriptions=[]):
        print(':: plotting PCA... ', end='')

        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        
        n_stages = len(annotations)

        fig, ax = plt.subplots()

        # remove top/right axis spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # remove x/y tick labels
        plt.xticks([])
        plt.yticks([])

        colors = cm.get_cmap('plasma', n_stages)(range(n_stages))
        for i, stage in enumerate(annotations):
            mask = y == i
            ax.scatter(components[mask, 0], components[mask, 1], s=1, alpha=0.4,
                    color=colors[i], label=stage)
        ax.legend(markerscale=10, fontsize=12)

        ax.set_title(f'PCA of {self.dataset_name} dataset')

        self._plot(plt, 'PCA')
        print('Done')


    def plot_TSNE(self, X, y, annotations=['W', 'N1', 'N2', 'N3', 'R'], descriptions=[]):
        print(':: plotting TSNE... ', end='')

        tsne = TSNE(n_components=2)
        components = tsne.fit_transform(X)

        n_stages = len(annotations)

        fig, ax = plt.subplots()

        # remove top/right axis spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # remove x/y tick labels
        plt.xticks([])
        plt.yticks([])

        colors = cm.get_cmap('plasma', n_stages)(range(n_stages))
        for i, stage in enumerate(annotations):
            mask = y == i
            ax.scatter(components[mask, 0], components[mask, 1], s=1, alpha=0.4,
                    color=colors[i], label=stage)
        ax.legend(markerscale=10, fontsize=12)

        ax.set_title(f't-SNE of {self.dataset_name} dataset')

        self._plot(plt, 'TSNE')
        print('Done')


    def plot_UMAP(self, X, y, annotations, mapping=None, descriptions=[]):
        print(':: plotting UMAP... ', end='')
        _umap = umap.UMAP(n_neighbors=15)
        umap_components = _umap.fit_transform(X)

        n_stages = len(annotations)

        fig, ax = plt.subplots()

        # remove top/right axis spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # remove x/y tick labels
        plt.xticks([])
        plt.yticks([])

        colors = cm.get_cmap('plasma', n_stages)(range(n_stages))
        # if annotating feature space with descriptions instesad
        if len(descriptions) > 0 or mapping is not None:
            annotations = mapping
            y = descriptions
        for i, stage in enumerate(annotations):
            mask = y == i
            ax.scatter(umap_components[mask, 0], umap_components[mask, 1], s=1, alpha=0.4,
            color=colors[i], label=stage)
        ax.legend(markerscale=10, fontsize=12)

        ax.set_title(f'UMAP of {self.dataset_name} dataset')

        self._plot(plt, 'UMAP')
        print('Done')


    # UMAP plot with connectivity
    # https://umap-learn.readthedocs.io/en/latest/plotting.html
    def plot_UMAP_connectivity(self, X, edge_bundling=False):
        print(':: plotting UMAP with connectivity... ', end='')
        title = f'Connectivity UMAP of {self.dataset_name}'
        mapping = umap.UMAP(n_components=2, init='random').fit(X)

        if not edge_bundling:
            umap.plot.connectivity(mapping, show_points=True)
        else:
            title += ' (edge bundled)'
            umap.plot.connectivity(mapping, edge_bundling='hammer', cmap='rainbow') # bundles edges
            
        self._plot(plt, title)
        print('Done')


    # 3D UMAP plot (plotly)
    def plot_UMAP_3d(self, X, y, annotations, mapping=None, descriptions=[]):
        print(':: plotting 3D UMAP... ', end='')
        umap_3d = UMAP(n_components=3, init='random', random_state=0)
        proj_3d = umap_3d.fit_transform(X)

        series = pd.DataFrame(y, columns=['annots'])

        # if annotating feature space with descriptions instead
        if len(descriptions) > 0 or mapping is not None:
            map = {k:v for k,v in enumerate(mapping)}
            annotations = mapping
            y = descriptions
            series['labels'] = series['annots'].map(map)
            

        fig_3d = px.scatter_3d(
            proj_3d,
            x=0, y=1, z=2,
            color=series.annots,
            labels={'color': 'labels'}
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
    def plot_learning_curves_sklearn(self, train_sizes, ssl_test_scores, raw_test_scores, dataset_name, scoring='balanced_accuracy'):
        '''
            train_sizes: set sizes used for training
            ssl_test_scores: test scores obtained for the SSL method
            raw_test_scores: test scores obtained for the FS method
            dataset: dataset name to be used in title
            scoring: scoring method used (for label)
        '''
        # create additional features
        ssl_test_scores_mean = np.mean(ssl_test_scores, axis=1)
        ssl_test_scores_std = np.std(ssl_test_scores, axis=1)
        raw_test_scores_mean = np.mean(raw_test_scores, axis=1)
        raw_test_scores_std = np.std(raw_test_scores, axis=1)
        
        print(':: plotting learning curves (sklearn)... ', end='')
        plt.rc('font', size=26)
        plt.rc('xtick', labelsize=30) 
        plt.rc('ytick', labelsize=30)
        plt.rc('axes', labelsize=26)
        _, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.grid()

        ax.fill_between(    
            train_sizes,
            ssl_test_scores_mean - ssl_test_scores_std,
            ssl_test_scores_mean + ssl_test_scores_std,
            alpha=0.1,
            color="r",
        )
        ax.fill_between(
            train_sizes,
            raw_test_scores_mean - raw_test_scores_std,
            raw_test_scores_mean + raw_test_scores_std,
            alpha=0.1,
            color="#328",
        )

        # plt.ylim(0, 1)
        ax.set_xlabel("Training examples")
        ax.set_ylabel(scoring)

        # plt.title(f'Balanced accuracy per training example for {dataset_name} dataset', fontsize=26)
        plt.plot(train_sizes, ssl_test_scores_mean, '-', color='r', label='SSL')
        plt.plot(train_sizes, raw_test_scores_mean, '-', color='#328', label='FS')
        # plt.legend(loc="best")

        self._plot(plt, 'learning_curves')
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

        ax.set_title(f'Learning curves of a Self-Supervised and Fully-Supervised Logistic Regression scores per Training Example for {dataset_name} dataset')
        plt.plot(ssl_space, ssl_train_scores_avg, color='g', label='SSL Training Scores')
        plt.plot(raw_space, raw_train_scores_avg, color='r', label='Raw Training Scores')
        plt.legend(loc="best")

        self._plot(plt, 'logit_learning_curves')

        print('Done')