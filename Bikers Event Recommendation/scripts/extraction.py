import numpy as np
import pandas as pd

from scipy.stats import mode
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm
from datetime import datetime


def LDA(data_content):
    print('Training Latent Dirichlet Allocation (LDA)..', flush=True)

    lda = LatentDirichletAllocation(n_components=data_content.number_of_topics,
                                    learning_decay=data_content.learning_decay,
                                    learning_offset=data_content.learning_offset,
                                    batch_size=data_content.batch_size,
                                    evaluate_every=data_content.evaluate_every,
                                    random_state=data_content.random_state,
                                    max_iter=data_content.max_iter).fit(data_content.X)

    print('Latent Dirichlet Allocation (LDA) trained successfully...\n', flush=True)

    return lda


def get_tour_collection(fb, cdf, typ_event):
    tour_collection = {}

    pbar = tqdm(total=fb.shape[0], bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    pbar.set_description('Step 1 of 3')

    for idx, _ in fb.iterrows():
        bik = fb.loc[idx, 'friends']
        cell = [-1, -1, -1, -1,
                -1, -1, -1, -1]

        # Looking for friends
        if len(bik) != 0:
            bik = bik.split()
            c = cdf[cdf['biker_id'].isin(bik)]
            if c.shape[0] != 0:
                for i, te in enumerate(typ_event):
                    ce = (' '.join(c[te].tolist())).split()
                    if len(ce) != 0:
                        cell[i] = ce

        # Looking for personal
        bik = fb.loc[idx, 'biker_id']
        c = cdf[cdf['biker_id'] == bik]
        if c.shape[0] != 0:
            for i, te in enumerate(typ_event):
                ce = c[te].tolist()[0].split()
                if len(c) != 0:
                    cell[len(typ_event) + i] = ce
        tour_collection[fb.loc[idx, 'biker_id']] = cell
        pbar.update(1)
    pbar.close()

    return tour_collection


def find_interest_group(temp_df, data_content):
    if temp_df.shape[0] == 0:
        return np.zeros((1, data_content.number_of_topics))
    pred = data_content.lda.transform(temp_df[data_content.cols])

    return pred


def tour_interest_group(rt, tour, data_content):
    idx = rt[rt['tour_id'] == tour].index
    h = data_content.lda.transform(rt.loc[idx, data_content.cols])

    return h


def predict_preference(dataframe, data_content, typ_event=None):
    if typ_event is None:
        typ_event = ['going', 'not_going', 'maybe', 'invited']

    bikers = dataframe['biker_id'].drop_duplicates().tolist()
    fb = data_content.bikers_network_df[data_content.bikers_network_df['biker_id'].isin(bikers)]
    all_biker_friends = bikers.copy()

    for idx, _ in fb.iterrows():
        bik = fb.loc[idx, 'friends']
        if len(bik) != 0:
            all_biker_friends += bik.split()

    cdf = data_content.convoy_df[data_content.convoy_df['biker_id'].isin(all_biker_friends)]
    tdf = []

    for te in typ_event:
        tdf += (' '.join(cdf[te].tolist())).split()
    temp_df = data_content.tours_df[data_content.tours_df['tour_id'].isin(tdf)]
    tour_collection = get_tour_collection(fb, cdf, typ_event)
    rt = data_content.tours_df[data_content.tours_df['tour_id'].isin(dataframe['tour_id'].drop_duplicates().tolist())]

    for te in typ_event:
        dataframe['fscore_' + te] = 0
        dataframe['pscore_' + te] = 0
    pbar = tqdm(total=len(bikers), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    pbar.set_description('Step 2 of 3')

    for biker in bikers:
        sdf = dataframe[dataframe['biker_id'] == biker]
        sub = tour_collection[biker]
        for i, te in enumerate(typ_event):
            frds_tur = sub[i]
            pers_tur = sub[len(typ_event) + i]
            ft, pt = False, False
            if type(frds_tur) != int:
                kdf = temp_df[temp_df['tour_id'].isin(frds_tur)]
                frds_lat = find_interest_group(kdf, data_content)
                ft = True
            if type(pers_tur) != int:
                udf = temp_df[temp_df['tour_id'].isin(pers_tur)]
                pers_lat = find_interest_group(udf, data_content)
                pt = True
            for idx, _ in sdf.iterrows():
                tour = sdf.loc[idx, 'tour_id']
                mat = tour_interest_group(rt, tour, data_content)
                if ft:
                    # noinspection PyUnboundLocalVariable
                    dataframe.loc[idx, 'fscore_' + te] = np.median(np.dot(frds_lat, mat.T).ravel())
                if pt:
                    # noinspection PyUnboundLocalVariable
                    dataframe.loc[idx, 'pscore_' + te] = np.median(np.dot(pers_lat, mat.T).ravel())
        pbar.update(1)
    pbar.close()

    return dataframe


def get_organizers(dataframe, data_content):
    bikers = dataframe['biker_id'].drop_duplicates().tolist()
    fb = data_content.bikers_network_df[data_content.bikers_network_df['biker_id'].isin(bikers)]
    rt = data_content.tours_df[data_content.tours_df['tour_id'].isin(
        dataframe['tour_id'].drop_duplicates().tolist())]
    tc = data_content.tour_convoy_df[data_content.tour_convoy_df['tour_id'].isin(
        dataframe['tour_id'].drop_duplicates().tolist())]
    lis = ['going', 'not_going', 'maybe', 'invited']

    dataframe['org_frd'] = 0
    dataframe['frd_going'] = 0
    dataframe['frd_not_going'] = 0
    dataframe['frd_maybe'] = 0
    dataframe['frd_invited'] = 0

    pbar = tqdm(total=len(bikers), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    pbar.set_description('Step 3 of 3')

    for biker in bikers:
        tmp = dataframe[dataframe['biker_id'] == biker]
        frd = fb[fb['biker_id'] == biker]['friends'].tolist()[0].split()
        for idx, _ in tmp.iterrows():
            trs = tc[tc['tour_id'] == tmp.loc[idx, 'tour_id']]
            org = rt[rt['tour_id'] == tmp.loc[idx, 'tour_id']]['biker_id'].tolist()[0]
            if org in frd:
                dataframe.loc[idx, 'org_frd'] = 1
            if trs.shape[0] > 0:
                for l in lis:
                    t = trs[l].tolist()[0]
                    if not pd.isna(t):
                        t = t.split()
                        dataframe.loc[idx, 'frd_' + l] = len(set(t).intersection(frd))
        pbar.update(1)
    pbar.close()

    return dataframe


def set_preference_score(dataframe, data_content):
    if data_content.preference_feat:
        dataframe = predict_preference(dataframe, data_content, typ_event=['going', 'not_going'])
    else:
        print('Skipping Step 1 & 2...Not required due to reduced noise...', flush=True)

    dataframe = get_organizers(dataframe, data_content)
    print('Preferences extracted...\n', flush=True)

    return dataframe


def calculate_distance(x1, y1, x2, y2):
    if np.isnan(x1):
        return 0
    else:
        R = 6373.0
        x1, y1 = np.radians(x1), np.radians(y1)
        x2, y2 = np.radians(x2), np.radians(y2)
        dlon = x2 - x1
        dlat = y2 - y1
        a = np.sin(dlat / 2) ** 2 + np.cos(x1) * np.cos(x2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c


def append_latent_factors(df, data_content):
    cam = ['w' + str(i) for i in range(1, 101)] + ['w_other']
    out = data_content.lda.transform(df[cam])
    out[out >= (1 / data_content.number_of_topics)] = 1
    out[out < (1 / data_content.number_of_topics)] = 0

    for r in range(data_content.number_of_topics):
        df['f' + str(r + 1)] = out[:, r]

    return df


def transform(df, data_content):
    tr_df = pd.merge(df, data_content.bikers_df, on='biker_id', how='left')

    # Compute membership period
    for idx, _ in tr_df.iterrows():
        tr_df.loc[idx, 'since'] = tr_df.loc[idx, 'timestamp'] - tr_df.loc[idx, 'member_since']

    tr_df['since'] = tr_df['since'].apply(lambda x: x.total_seconds() / 86400)
    tr_df.drop(['member_since'], axis=1, inplace=True)
    tr_df = pd.merge(tr_df, data_content.tdf.drop(['biker_id', 'area'], axis=1), on='tour_id', how='left')

    # TODO: Remove below try-except clause before submission.
    # noinspection PyBroadException
    try:
        tr_df['tour_date'] = tr_df['tour_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    except:
        pass

    # Compute time_delta
    for idx, _ in tr_df.iterrows():
        tr_df.loc[idx, 'time_delta'] = tr_df.loc[idx, 'tour_date'] - tr_df.loc[idx, 'timestamp']

    tr_df['time_delta'] = tr_df['time_delta'].apply(lambda x: x.total_seconds() / 86400)
    tr_df['start_timestamp'] = tr_df['biker_id'].map(tr_df.groupby('biker_id')['timestamp'].min().to_dict())
    tr_df['response_delayed'] = [1 if x1 > x2 else 0 for x1, x2 in
                                 zip(tr_df['timestamp'], tr_df['start_timestamp'])]
    tr_df.drop(['timestamp', 'tour_date', 'start_timestamp'], axis=1, inplace=True)

    # Compute Distance
    for idx, _ in tr_df.iterrows():
        x1, y1 = tr_df.loc[idx, 'latitude_y'], tr_df.loc[idx, 'longitude_y']
        x2, y2 = tr_df.loc[idx, 'latitude_x'], tr_df.loc[idx, 'longitude_x']
        tr_df.loc[idx, 'distance'] = calculate_distance(x1, y1, x2, y2)

    tr_df.drop(['latitude_x', 'latitude_y', 'longitude_x', 'longitude_y'], axis=1, inplace=True)
    # tr_df = tr_df.fillna(0)

    # Merging for fractions
    tr_df = pd.merge(tr_df, data_content.tour_convoy_df.drop(['invited', 'going', 'not_going', 'maybe'], axis=1),
                     on='tour_id', how='left')

    if data_content.time_zone_feat:
        tr_df = pd.get_dummies(tr_df, columns=['time_zone'])

    tr_df = append_latent_factors(tr_df, data_content)

    # Setting up labels
    # noinspection PyBroadException
    try:
        tr_df['like'] = tr_df['like'] - tr_df['dislike']
        bikers = tr_df['biker_id'].drop_duplicates().tolist()
        groups = []
        for biker in bikers:
            groups.append(tr_df[tr_df['biker_id'] == biker].shape[0])
        return tr_df, groups
    except:
        return tr_df, None


def train_valid_split(data_content):
    feat_cols = data_content.train_df_tr.columns.tolist()
    drop_cols = ['biker_id', 'tour_id', 'like', 'dislike', 'language'] + \
                ['w' + str(i) for i in range(1, 101)] + ['w_other']

    for dc in drop_cols:
        if dc in feat_cols:
            feat_cols.remove(dc)

    X_train = data_content.train_df_tr.loc[:data_content.split_point, feat_cols]
    X_valid = data_content.train_df_tr.loc[data_content.split_point + 1:, feat_cols]
    y_train = data_content.train_df_tr.loc[:data_content.split_point, 'like']
    y_valid = data_content.train_df_tr.loc[data_content.split_point + 1:, 'like']

    group_point = len(data_content.train_df_tr.loc[:data_content.split_point, 'biker_id'].drop_duplicates().tolist())
    train_group, valid_group = data_content.group[:group_point], data_content.group[group_point:]

    return X_train, X_valid, y_train, y_valid, train_group, valid_group, feat_cols


def prepare_test(data_content):
    for c in data_content.feat_cols:
        if c not in data_content.test_df_tr.columns.tolist():
            data_content.test_df_tr[c] = 0
    X_test = data_content.test_df_tr[data_content.feat_cols]

    return X_test


def feature_extraction(data_content):
    data_content.X = data_content.tdf[data_content.cols]
    data_content.lda = LDA(data_content)

    print('Getting Preferences for training data...', flush=True)
    data_content.train_df = set_preference_score(data_content.train_df, data_content)
    data_content.train_df_tr, data_content.group = transform(data_content.train_df, data_content)

    print('Getting Preferences for test data...', flush=True)
    data_content.test_df = set_preference_score(data_content.test_df, data_content)
    data_content.test_df_tr, _ = transform(data_content.test_df, data_content)

    data_content.X_train, data_content.X_valid, data_content.y_train, data_content.y_valid, \
    data_content.train_group, data_content.valid_group, data_content.feat_cols = train_valid_split(data_content)
    data_content.X_test = prepare_test(data_content)

    return data_content
