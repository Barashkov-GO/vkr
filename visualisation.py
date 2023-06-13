FIGURES_PATH = 'out/figures/'
DATASETS_PATH = 'out/datasets/'
DICTS_PATH = 'out/dicts/'
CLUSTERS_PATH = 'out/clusters/'
import pandas as pd
from datetime import datetime, timedelta
import os
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from tqdm.notebook import tqdm
from multiprocesspandas import applyparallel
from pandarallel import pandarallel
import psutil
from sys import getsizeof
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster

from netgraph import Graph, InteractiveGraph, EditableGraph

import pickle
import gc

tqdm.pandas()
from helper import *


class Visualisation:
    def __init__(self, dists):
        self.names = {}
        self.clusters = None
        self.dists = dists

    @staticmethod
    def concat_dicts(dict1, dict2):
        ans = dict()
        for k1 in dict1.keys():
            if int(k1) in dict2:
                ans[dict1[k1]] = dict2[int(k1)]
            else:
                ans[dict1[k1]] = 'Unnamed'

        return ans

    def set_product_names(self, data_path='datasets/чеки.csv'):
        data = pd.read_csv(data_path, nrows=None)
        data = data.rename(columns={'line_item_id': 'product_id'})
        data = data[['line_article_id', 'line_article_description']].drop_duplicates()
        product_names = data.set_index('line_article_id').to_dict()['line_article_description']

        with open(DICTS_PATH + 'products.json') as json_file:
            products_dict = json.load(json_file)

        self.names = self.concat_dicts(products_dict, product_names)

    def set_clusters(self, clusters_path=None, clusters=None):
        if clusters is None:
            with open(CLUSTERS_PATH + clusters_path + '.pkl', 'rb') as f:
                self.clusters = pickle.load(f)
        else:
            self.clusters = clusters

    def clear_clusters(self):
        self.clusters = None

    def dist(self, product1, product2):
        if product1 == product2:
            return 0

        if (product1, product2) in self.dists:
            return self.dists[(product1, product2)]

        if (product2, product1) in self.dists:
            return self.dists[(product2, product1)]

        return np.inf

    def dist_product_cluster(self, product, cluster):
        s = 0.0
        for p in cluster:
            s += self.dist(product, p) ** 2
        return np.sqrt(s)

    def show(self, helping, top_in_cluster=10, top_clusters=10):
        for cluster in self.clusters[:top_clusters]:
            if len(cluster) < 2:
                continue
            edges = []
            edge_labels = dict()
            edge_length = dict()
            cluster = sorted(cluster, key=lambda x: (self.dist_product_cluster(x, cluster)))
            cluster = cluster[:top_in_cluster]

            for p1 in cluster:
                for p2 in cluster:
                    if p1 != p2:
                        d = self.dist(p1, p2)
                        if d < np.inf:
                            if p1 in self.names:
                                p1 = self.names[p1].split()[0]
                            if p2 in self.names:
                                p2 = self.names[p2].split()[0]
                                # p1, p2 = names[p1].split()[0], names[p2].split()[0]
                            edges.append((p1, p2, d))
                            edge_length[(p1, p2)] = d
                            edge_labels[(p1, p2)] = "%.1f" % d
                            edges.append((p2, p1, d))
                            edge_length[(p2, p1)] = d
                            edge_labels[(p2, p1)] = "%.1f" % d

            g = Graph(edges,
                      # node_color='white',
                      node_layout='geometric',
                      node_layout_kwargs=dict(edge_length=edge_length, tol=1e-3),
                      node_size=10,
                      node_labels=True,
                      edge_labels=edge_labels,
                      )

            plt.show()


