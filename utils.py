import os
from collections import defaultdict
import numpy as np
from scipy.sparse import dok_matrix, lil_matrix
from tqdm import tqdm

def movielens(data_path="data/sample_interactions.csv", feature_h5_path=None):
    import pandas as pd
    import h5py
    from collections import defaultdict

    df = pd.read_csv(data_path, usecols=['userId', 'tmdbId'])
    df = df.drop_duplicates()

    user_unique = df['userId'].unique()
    item_unique = df['tmdbId'].unique()
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_unique)}
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_unique)}
    idx_to_item_id = {idx: iid for iid, idx in item_id_to_idx.items()}

    user_item_dict = defaultdict(set)
    for uid, iid in zip(df['userId'], df['tmdbId']):
        u_idx = user_id_to_idx[uid]
        i_idx = item_id_to_idx[iid]
        user_item_dict[u_idx].add(i_idx)

    n_users = len(user_unique)
    n_items = len(item_unique)
    user_item_matrix = dok_matrix((n_users, n_items), dtype=np.int32)
    for u_idx, items in user_item_dict.items():
        for i_idx in items:
            user_item_matrix[u_idx, i_idx] = 1

    features = dok_matrix((n_items, 0), dtype=np.float32)

    if feature_h5_path is not None:
        with h5py.File(feature_h5_path, 'r') as hf:
            h5_tmdb_ids = hf['tmdbId'][:].astype(int)
            h5_vectors = hf['vectors'][:]

        # tmdbId -> item_index
        tmdb_to_item_idx = {
            tmdb_id: item_id_to_idx[tmdb_id]
            for tmdb_id in h5_tmdb_ids
            if tmdb_id in item_id_to_idx
        }

        feature_dim = h5_vectors.shape[-1]
        features = dok_matrix((n_items, feature_dim), dtype=np.float32)

        for h5_idx, tmdb_id in enumerate(h5_tmdb_ids):
            if tmdb_id in tmdb_to_item_idx:
                item_idx = tmdb_to_item_idx[tmdb_id]
                avg_vector = h5_vectors[h5_idx].mean(axis=0)
                features[item_idx] = avg_vector

        print(f"Loaded {len(tmdb_to_item_idx)}/{len(item_unique)} items with features")

    print(f"MovieLens dataset: {n_users} users, {n_items} items")
    return user_item_matrix.tocsr(), features.tocsr()
    

def split_data(user_item_matrix, split_ratio=(3, 1, 1), seed=1):
    np.random.seed(seed)
    train = dok_matrix(user_item_matrix.shape, dtype=np.int32)
    validation = dok_matrix(user_item_matrix.shape, dtype=np.int32)
    test = dok_matrix(user_item_matrix.shape, dtype=np.int32)

    user_item_matrix = lil_matrix(user_item_matrix)

    for user in tqdm(range(user_item_matrix.shape[0]), desc="Split data into train/valid/test"):
        items = list(user_item_matrix.rows[user])
        if len(items) >= 5:
            np.random.shuffle(items)

            total = sum(split_ratio)
            train_count = int(len(items) * split_ratio[0] / total)
            valid_count = int(len(items) * split_ratio[1] / total)

            for i in items[:train_count]:
                train[user, i] = 1
            for i in items[train_count:train_count + valid_count]:
                validation[user, i] = 1
            for i in items[train_count + valid_count:]:
                test[user, i] = 1

    print(f"{len(train.nonzero()[0])}/{len(validation.nonzero()[0])}/{len(test.nonzero()[0])} train/valid/test samples")
    return train, validation, test
