import pickle
from datetime import timedelta

import psutil
from pandarallel import pandarallel
from tqdm.notebook import tqdm
import multiprocessing

from helper import *

FIGURES_PATH = 'out/figures/'
DATASETS_PATH = 'out/datasets/'


trans_dists = dict()


def process_batch(batch):
    res = dict()
    for key in tqdm(batch.keys()):
        arr = np.array(batch[key])
        res[key] = [arr.mean(), arr.shape[0], np.quantile(arr, 0.75) - np.quantile(arr, 0.25)]
    return res


def concat_batches(dist):
    trans_dists.update(dist)


class Distances:
    def __init__(self,
                 data_path: str = 'data_processed',
                 nrows: int = None,
                 data=None
                 ):
        """
        Initialize Distances class with the data

        :param data_path: name of file
        :param nrows: number of rows to read
        """
        if data is None:
            self.data = pd.read_csv(DATASETS_PATH + data_path + '.csv', nrows=nrows).drop(columns=['Unnamed: 0'])
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        else:
            self.data = data
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.nrows = nrows

    def get_prices(self, field='product_id'):
        return dict(self.data[[field, 'line_item_price']].groupby(by=field).apply(lambda prices: prices.mean())[
                        'line_item_price'])

    @staticmethod
    def save_dists(file, path):
        with open(DATASETS_PATH + path + '.pkl', 'wb') as f:
            pickle.dump(file, f)

    def get_helping(self, field):
        freq = dict()

        def fill_freq(x):
            nonlocal freq
            sum_of_receipt = (x['line_item_price'] * x['line_quantity']).sum()

            for ind, val in enumerate(x[field].values):
                if not np.isnan(val):
                    if val in freq:
                        freq[val][0] += 1
                        freq[val][1].append(
                            x['line_item_price'].values[ind] * x['line_quantity'].values[ind] / sum_of_receipt)
                    else:
                        freq[val] = [1, [
                            x['line_item_price'].values[ind] * x['line_quantity'].values[ind] / sum_of_receipt]]

        self.data.groupby(by='transaction_key').apply(fill_freq)
        all = self.data['transaction_key'].drop_duplicates().shape[0]

        for k in freq.keys():
            freq[k][0] /= all
            freq[k][1] = np.median(freq[k][1])

        return freq

    def top_users(self, top_lim, field='product_id'):
        data = self.data[['gid', field]]
        d = data.groupby(by='gid').apply(lambda x: len(x)).sort_values(ascending=False)
        self.data = self.data.loc[self.data['gid'].isin(d.index.values[:top_lim])]
        return data.loc[data['gid'].isin(d.index.values[:top_lim])]

    def top_products(self, top_lim, field='product_id'):
        data = self.data[[field, 'datetime', 'gid']]
        d = data.groupby(by=field).apply(lambda x: len(x)).sort_values(ascending=False)
        self.data = self.data.loc[self.data[field].isin(d.index.values[:top_lim])]
        return data.loc[data[field].isin(d.index.values[:top_lim])]

    def get_up(self,
               top_lim: int = None,
               field: str = 'product_id'
               ):
        """
        Get distances between all pairs of users by counting purchases

        :param top_lim: limit of users to get by count of buys
        :param field: field to search in dataframe
        :return: dict[user] = {product: count}
        """

        def process_batch(x):
            ans = dict()
            for i in x[field].values:
                if i in ans:
                    ans[i] += 1
                else:
                    ans[i] = 1
            return ans

        data = self.top_users(top_lim, field)
        print(len(data['gid'].drop_duplicates()))

        pandarallel.initialize(progress_bar=True, use_memory_fs=False, nb_workers=psutil.cpu_count(logical=False))
        ans = data.groupby(by='gid').parallel_apply(process_batch)

        ans = dict(ans)
        self.save_dists(ans, f'up_{field}_{self.nrows}_{top_lim}')
        return ans

    def product_product(self,
                        top_lim: int = None,
                        interval: int = None,
                        batch_size: int = 100_000,
                        field: str = 'product_id'
                        ):
        """
        Get distances between all pairs of products by date differences

        :param top_lim: limit of products to get by count of buys
        :param interval: date interval to split data with, default: None
        :param batch_size: data batching size, default: 100_000
        :param field: field to search in dataframe
        :return: dict[(product_1, product_2)] = an array of mean of date distance by one user
        """
        ans = dict()

        data = self.top_products(top_lim=top_lim, field=field)

        print(f'Top of dataset length: {data.shape[0]}')

        data.loc[:, 'datetime'] = data['datetime'].dt.date

        def data_splitting(interval):
            nonlocal data
            batches = []
            data = data.sort_values(by='datetime')
            start = data.iloc[0].at['datetime']
            end = data.iloc[-1].at['datetime']
            while start <= end:
                sub_end = start + timedelta(days=interval)
                batch = data.loc[data['datetime'] >= start].loc[data['datetime'] < sub_end]
                batches.append(batch)
                start = sub_end

            return batches

        def fill_ans(x):
            product_date = x[[field, 'datetime']]
            res = dict()
            for i1, r1 in product_date.iterrows():
                for i2, r2 in product_date.iterrows():
                    if i1 != i2:
                        p1, p2 = r1[field], r2[field]
                        timedelta = (r1['datetime'] - r2['datetime']).days

                        if (p1, p2) in res and abs(res[(p1, p2)]) > abs(timedelta):
                            res[(p1, p2)] = timedelta
                        else:
                            res[(p1, p2)] = timedelta
            return res

        def concat_dicts(res):
            nonlocal ans
            res = res.values
            for r in res:
                for key in r.keys():
                    if key in ans:
                        ans[key].append(r[key])
                    else:
                        ans[key] = [r[key]]
            return ans

        if interval is not None:
            batches = data_splitting(interval=interval)
        else:
            batches = np.array_split(data, data.shape[0] // batch_size + 1)

        pandarallel.initialize(progress_bar=False, use_memory_fs=False, nb_workers=psutil.cpu_count(logical=False))

        for batch in tqdm(batches):
            if psutil.virtual_memory().percent >= 90:
                break
            grouped_by_user = batch.groupby(by='gid')
            temp = grouped_by_user.parallel_apply(fill_ans)
            temp = temp.dropna()
            ans = concat_dicts(temp)

        # self.save_dists(ans, f'pp_{field}_{self.nrows}')
        return ans

    def get_up_matrix(self,
                      field: str = 'product_id',
                      top_lim: int = None,
                      batch_size: int = 100_000
                      ):
        """
        Get distances between all pairs of users by counting purchases

        :param top_lim: limit of users to get by count of buys
        :param field: field to search in dataframe
        :param batch_size: size of batch to split dataframe
        :return: ans[user][product] = count of buys of specified product by user
        """

        def fill_ans(x):
            nonlocal ans
            user, product = x[0], x[1]
            ans[user, product] += 1

        data = self.top_users(top_lim=top_lim, field=field)

        max_user, max_product = max(list(data['gid'].values)), max(list(data[field].values))
        print(max_user)
        print(max_product)
        ans = np.full((max_user, max_product), 0)
        for batch in tqdm(np.array_split(data, data.shape[0] // batch_size)):
            batch.apply(fill_ans, axis=1)

        self.save_dists(ans, f'up_matrix_{field}_{self.nrows}_{top_lim}')
        return ans

    def get_pp(self,
               top_lim: int = None,
               interval: int = None,
               batch_size: int = 100_000,
               field: str = 'product_id'
               ):
        """
        Get distances between all pairs of products by date differences

        :param top_lim: limit of products to get by count of buys
        :param interval: date interval to split data with, default: None
        :param batch_size: data batching size, default: 100_000
        :param field: field to search in dataframe
        :return: dict[(product_1, product_2)] = [mean, count, quartile range] of date distances
        """
        ans = self.product_product(top_lim=top_lim, interval=interval, batch_size=batch_size, field=field)

        def chunks(dictionary, size):
            items = list(dictionary.items())
            return [dict(items[i: i + size]) for i in range(0, len(items), size)]

        def custom_error_callback(error):
            print(f'Got an Error: {error}', flush=True)

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        batches = chunks(ans, len(ans) // psutil.cpu_count() // 2)

        #     del dists

        for batch in batches:
            pool.apply_async(
                process_batch,
                args=(batch,),
                callback=concat_batches,
                error_callback=custom_error_callback,
            )

        pool.close()
        pool.join()

        self.save_dists(trans_dists, f'pp_{field}_{self.nrows}_{top_lim}')

        return trans_dists
