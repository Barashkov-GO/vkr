import pickle
from datetime import datetime
from tqdm.notebook import tqdm
from helper import *

FIGURES_PATH = 'out/figures/'
DATASETS_PATH = 'out/datasets/'
CLUSTERS_PATH = 'out/clusters/'


def default(mean, count, scatter):
    return (mean + abs(scatter)) / count


def normalize(d, target=1.0, type1=np.uint32, type2=np.float16):
    raw = sum(d.values())
    factor = target / raw
    return {(type1(key[0]), type1(key[1])): type2(value * factor) for key, value in d.items()}


def get_dists(dists, count_lower=10, dist_func=default):
    ans = dict()
    for i in dists.items():
        dist = dist_func(i[1][0], i[1][1], i[1][2])
        if (i[1][1] >= count_lower or dist != 0) and (dist >= 0):
            ans[i[0]] = dist
    return ans


class Metric:
    def __init__(self, method='euclidean', max=100):
        self.method = method
        self.max = max

    def run(self, cluster1, cluster2, dists):
        if self.method == 'euclidean':
            return self.euclidean(cluster1, cluster2, dists)
        if self.method == 'min_dist':
            return self.min_dist(cluster1, cluster2, dists)
        if self.method == 'max_dist':
            return self.max_dist(cluster1, cluster2, dists)
        if self.method == 'average':
            return self.average(cluster1, cluster2, dists)

    def _get(self, i, j, dists):
        if i == j:
            return 0.0
        if (i, j) in dists:
            return dists[(i, j)]
        if (j, i) in dists:
            return dists[(j, i)]
        return self.max

    def euclidean(self, cluster1, cluster2, dists):
        n1, n2 = len(cluster1), len(cluster2)
        s = 0.0
        for i in cluster1:
            for j in cluster2:
                s += self._get(i, j, dists) ** 2
        return np.sqrt(s)

    def min_dist(self, cluster1, cluster2, dists):
        s, mini = 0.0, self.max + 1
        for i in cluster1:
            for j in cluster2:
                s = self._get(i, j, dists)

                if s < mini:
                    mini = s
        return mini

    def max_dist(self, cluster1, cluster2, dists):
        s, maxi = 0.0, -1.0
        for i in cluster1:
            for j in cluster2:
                s = self._get(i, j, dists)

                if s > maxi:
                    maxi = s
        return maxi

    def average(self, cluster1, cluster2, dists):
        n1, n2 = len(cluster1), len(cluster2)
        s = 0.0
        for i in cluster1:
            for j in cluster2:
                s += self._get(i, j, dists)
        return s / (n1 * n2)


class Elem:
    def __init__(self, i, j, val):
        self.i = i
        self.j = j
        self.val = val

    def get_key(self):
        return self.i, self.j

    def get_all(self):
        return self.i, self.j, self.val

    def __lt__(self, other):
        return self.val < other.val

    def __str__(self):
        return f"({self.i}, {self.j}): {self.val}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j


class ClustersDict:
    def __init__(self):
        self.dists = dict()
        self.min_ind = None

    def insert(self, i, j, val):
        self.dists[i, j] = val
        if self.min_ind is None or val < self.dists[self.min_ind]:
            self.min_ind = (i, j)

    def remove(self, i, j):
        sort_again = False
        keys = list(self.dists.keys())
        for k in keys:
            if i in k or j in k:
                if self.min_ind == k:
                    sort_again = True
                    self.min_ind = None
                self.dists.pop(k)

        if sort_again:
            for k in self.dists.keys():
                if self.min_ind is None or self.dists[k] < self.dists[self.min_ind]:
                    self.min_ind = k

    def min(self):
        return self.min_ind[0], self.min_ind[1], self.dists[self.min_ind]

    def __getitem__(self, item):
        if (item[0], item[1]) in self.dists:
            return self.dists[item[0], item[1]]
        return self.dists[item[1], item[0]]


class Clustering:
    def __init__(self, get_dists=get_dists):
        self.metric = None
        self.get_dists = get_dists
        self.statistics = {
            'min_distances': [],
            'time_of_iter': [],
            'time_of_all': 0.0,
            'count_of_iters': 0.0,
        }

    def get_stats(self):
        # self.statistics['time_of_iter'] = np.array(self.statistics['time_of_iter']).mean()
        # for k in self.statistics.keys():
        #     print(f"{k} --- {self.statistics[k]}")
        return self.statistics

    def _get_matrix_from_dict(self, dists, maxi=100):
        subs = dict()
        elements = np.unique(list(dists.keys()))
        elements.sort()
        for i, el in enumerate(elements):
            subs[i] = el

        arr = np.full((len(elements), len(elements)), maxi)
        for i1, e1 in enumerate(elements):
            for i2, e2 in enumerate(elements):
                arr[i1, i2] = self.metric._get(e1, e2, dists)

        return arr, subs

    @staticmethod
    def _merge_clusters(cluster1, cluster2):
        merged_cluster = cluster1 + cluster2
        return merged_cluster

    def _dist_between_clusters(self,
                               clusters: list[list[int]],
                               v: int,
                               u: int,
                               dists: dict[tuple[int, int], float],
                               s: int = None,
                               t: int = None,
                               clusters_dists: ClustersDict = None,
                               method: str = 'min_dist') -> float:
        """
        Counting distance between two clusters by specified method

        :param clusters: current list of clusters
        :param v: cluster to count distance with
        :param u: new merged cluster
        :param dists: distances between elements
        :param s: subcluster of u
        :param t: subcluster of u
        :param clusters_dists: current distances between clusters
        :param method: method to operate (min_dist, max_dist, average, weighted, ward)
        :return: distance between clusters u and v
        """
        if method == 'min_dist':
            metric = Metric(method='min_dist')
            return metric.run(clusters[u], clusters[v], dists)

        if method == 'max_dist':
            metric = Metric(method='max_dist')
            return metric.run(clusters[u], clusters[v], dists)

        if method == 'average':
            metric = Metric(method='average')
            return metric.run(clusters[u], clusters[v], dists)

        if method == 'weighted':
            d1, d2 = clusters_dists[s, v], clusters_dists[t, v]
            return (d1 + d2) / 2

        if method == 'ward':
            whole_length = len(clusters[v]) + len(clusters[s]) + len(clusters[t])
            sum = 0.0
            sum += (len(clusters[v]) + len(clusters[s])) / whole_length * (clusters_dists[v, s] ** 2)
            sum += (len(clusters[v]) + len(clusters[t])) / whole_length * (clusters_dists[v, t] ** 2)
            sum -= len(clusters[v]) / whole_length * (clusters_dists[s, t] ** 2)

            return np.sqrt(s)

        return self.metric.run(clusters[u], clusters[v], dists)

    def run_centroid(self, dists, method, k, top_lim, max_iter=1_000):
        dists_matrix, subs = self._get_matrix_from_dict(dists)
        start0 = datetime.now()
        centroids = dists_matrix[np.random.choice(range(len(dists_matrix)), k, replace=False)]
        labels = []
        iters = 0

        for _ in tqdm(range(max_iter)):
            iters += 1
            start = datetime.now()
            distances = np.sqrt(((dists_matrix - centroids[:, np.newaxis]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            new_centroids = np.array([dists_matrix[labels == i].mean(axis=0) for i in range(k)])
            if np.all(centroids == new_centroids):
                self.statistics['time_of_iter'].append(datetime.now() - start)
                break
            centroids = new_centroids

            self.statistics['time_of_iter'].append(datetime.now() - start)
        self.statistics['count_of_iters'] = iters
        self.statistics['time_of_all'] = datetime.now() - start0

        clusters = [[] for i in range(max(labels) + 1)]
        for i, a in enumerate(labels):
            clusters[a].append(subs[i])

        clusters = [c for c in clusters if len(c) > 1]

        with open(CLUSTERS_PATH + f'{method}_{top_lim}_{k}.pkl', 'wb') as f:
            pickle.dump(clusters, f)

        return clusters

    def run_agglomerative(self, dists, method, k, top_lim):
        start0 = datetime.now()

        elements = np.unique(list(dists.keys()))
        clusters = [[i] for i in elements]
        iters = len(elements) - k

        clusters_dists = ClustersDict()
        print('Starting counting distances between clusters...')

        for v in tqdm(range(len(clusters))):
            for j in range(v + 1, len(clusters)):
                clusters_dists.insert(v, j, self._dist_between_clusters(clusters, v, j, dists))

        print('Starting collapsing closest clusters...')
        for _ in tqdm(range(iters)):
            start = datetime.now()

            s, t, min_distance = clusters_dists.min()
            merged_cluster = self._merge_clusters(clusters[s], clusters[t])

            clusters[s], clusters[t] = [], []
            clusters.append(merged_cluster)

            u = len(clusters) - 1

            for v in range(u):
                if len(clusters[v]) != 0:
                    d = self._dist_between_clusters(clusters, v, u, dists, s, t, clusters_dists, method)
                    clusters_dists.insert(u, v, d)

            clusters_dists.remove(s, t)

            self.statistics['min_distances'].append(min_distance)
            self.statistics['time_of_iter'].append(datetime.now() - start)
        self.statistics['count_of_iters'] = iters
        self.statistics['time_of_all'] = datetime.now() - start0

        clusters = [c for c in clusters if len(c) > 1]

        with open(CLUSTERS_PATH + f'{method}_{top_lim}_{k}.pkl', 'wb') as f:
            pickle.dump(clusters, f)
        return clusters

    def fit(self,
            metric: str = 'euclidean',
            method: str = 'min_dist',
            dists_path: str = 'date_distances',
            k: int = 1000,
            top_lim: int = 10_000,
            max_iter: int = 10_000,
            dists=None
            ):
        """

        :param metric: euclidean, min_dist, max_dist, average
        :param method: min_dist, max_dist, average, weighted, ward. If specified min_dist, max_dist, average, then overwrites metric
        :param dists_path: from where to take the distances
        :param k: if agglomerative, then while to stop
        :param top_lim: cutting elements to optimize time and memory
        :param max_iter: if centroid, then while to stop
        :return:
        """

        if dists is None:
            with open(DATASETS_PATH + dists_path + '.pkl', 'rb') as f:
                dists = pickle.load(f)
        dists = self.get_dists(dists)
        self.metric = Metric(method=metric)

        if method not in ['min_dist', 'max_dist', 'average', 'weighted', 'ward']:
            return self.run_centroid(dists, method, k, top_lim), dists
        return self.run_agglomerative(dists, method, k, top_lim), dists