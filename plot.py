import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib.lines import Line2D
from matplotlib import cm

import umap.umap_ as umap
import umap.plot
from umap import UMAP
import plotly.express as px
from datetime import datetime

from helper_funcs import HelperFuncs as hf


class Plot:
    
    def __init__(self, save=True, show=True):
        self.save = save
        self.show = show

    def _plot(self, figure, title):
        dt = hf.get_datetime()
        # show fig
        if self.show:
            figure.show()
        # save fig
        if self.save:
            figure.savefig(f'plots/{dt}.{title}.png')

    # Extract loss and balanced accuracy values for plotting from history object
    def plot_acc(self, clf_history):
        print('plotting accuracy per epoch...')
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


    def plot_UMAP(self, X, y, annotations=['W', 'N1', 'N2', 'N3', 'R']):
        print('plotting UMAP...')
        _umap = umap.UMAP(n_neighbors=15)
        umap_components = _umap.fit_transform(X)

        components = [
            # pca_components,
            # tsne_components,
            umap_components
        ]
        components_titles = [
            'UMAP'
        ]

        # SINGLE plot
        fig, ax = plt.subplots(1, figsize=(20,20))
        colors = cm.get_cmap('viridis', 5)(range(5))
        for i, stage in enumerate(annotations):
            mask = y == i
            ax.scatter(umap_components[mask, 0], umap_components[mask, 1], s=40, alpha=0.7, label=stage)
        ax.legend()

        ax.set_title(components_titles[0])

        self._plot(plt, 'UMAP')


    # UMAP plot with connectivity
    # https://umap-learn.readthedocs.io/en/latest/plotting.html
    def plot_UMAP_connectivity(self, X, edge_bundling=False):
        print('plotting UMAP with connectivity...')
        title = 'UMAP_connectivity'
        mapping = umap.UMAP(n_components=2, init='random').fit(X)

        if not edge_bundling:
            umap.plot.connectivity(mapping, show_points=True)
        else:
            title += '_edge_bundled'
            umap.plot.connectivity(mapping, edge_bundling='hammer') # bundles edges
            
        self._plot(plt, title)


    # 3D UMAP plot (plotly)
    def plot_UMAP_3d(self, X, y):
        print('plotting 3D UMAP...')
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
        fig_3d.write_html(f'plots/{hf.get_datetime()}.UMAP_3d.html')
