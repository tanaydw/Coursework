import numpy as np
import pandas as pd

from scipy.stats import mode
from tqdm import tqdm
from geopy.geocoders import Nominatim
from datetime import datetime


def handle_bornIn(x):
    skip_vals = ['16-Mar', '23-May', 'None']
    if x not in skip_vals:
        return datetime(2012, 1, 1).year - datetime(int(x), 1, 1).year
    else:
        return 23


def handle_gender(x):
    if x == 'male':
        return 1
    else:
        return 0


def handle_memberSince(x):
    skip_vals = ['--None']
    if pd.isna(x):
        return datetime(2012, 1, 1)
    elif x not in skip_vals:
        return datetime.strptime(x, '%d-%m-%Y')
    else:
        return datetime(2012, 1, 1)


def process_tours_df(data_content):
    dtype = {}
    cols = data_content.tours_df.columns[9:]

    for d in cols:
        dtype[d] = np.int16
    data_content.tours_df = data_content.tours_df.astype(dtype)

    data_content.tours_df['area'] = data_content.tours_df['city'] + ' ' + data_content.tours_df['state'] + ' ' + \
                                    data_content.tours_df['pincode'] + ' ' + data_content.tours_df['country']

    data_content.tours_df['area'] = data_content.tours_df['area'].apply(lambda x: x.lstrip() if type(x) == str else x)
    data_content.tours_df['area'] = data_content.tours_df['area'].apply(lambda x: x.rstrip() if type(x) == str else x)
    data_content.tours_df.drop(['city', 'state', 'pincode', 'country'], axis=1, inplace=True)

    data_content.tours_df['tour_date'] = data_content.tours_df['tour_date'].apply(
        lambda x: datetime(int(x.split('-')[2]), int(x.split('-')[1]), int(x.split('-')[0]), 23, 59))


def process_tour_convoy_df(data_content):
    print('Initializing tour_convoy_df...', flush=True)

    data_content.tour_convoy_df['total_going'] = 0
    data_content.tour_convoy_df['total_not_going'] = 0
    data_content.tour_convoy_df['total_maybe'] = 0
    data_content.tour_convoy_df['total_invited'] = 0
    data_content.tour_convoy_df['fraction_going'] = 0
    data_content.tour_convoy_df['fraction_not_going'] = 0
    data_content.tour_convoy_df['fraction_maybe'] = 0

    known_bikers = set()
    lis = ['going', 'not_going', 'maybe', 'invited']
    pbar = tqdm(total=data_content.tour_convoy_df.shape[0],
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    pbar.set_description("Step 1 of 2")

    for idx, _ in data_content.tour_convoy_df.iterrows():
        s = [0, 0, 0]
        for j, l in enumerate(lis):
            if not pd.isna(data_content.tour_convoy_df.loc[idx, l]):
                biker = data_content.tour_convoy_df.loc[idx, l].split()
                data_content.tour_convoy_df.loc[idx, 'total_' + l] = len(biker)
                if j != 3:
                    s[j] = len(biker)
                for bik in biker:
                    known_bikers.add(bik)
        if sum(s) != 0:
            for j in range(3):
                data_content.tour_convoy_df.loc[idx, 'fraction_' + lis[j]] = s[j] / sum(s)
        pbar.update(1)
    pbar.close()

    mean = data_content.tour_convoy_df['total_invited'].mean()
    std = data_content.tour_convoy_df['total_invited'].std()
    data_content.tour_convoy_df['fraction_invited'] = data_content.tour_convoy_df['total_invited'].apply(
        lambda x: (x - mean) / std)
    biker_tour_convoy_df = dict()

    for biker in list(known_bikers):
        biker_tour_convoy_df[biker] = [[], [], [], []]

    pbar = tqdm(total=data_content.tour_convoy_df.shape[0], bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    pbar.set_description("Step 2 of 2")

    for idx, _ in data_content.tour_convoy_df.iterrows():
        for l in lis:
            if not pd.isna(data_content.tour_convoy_df.loc[idx, l]):
                biker = data_content.tour_convoy_df.loc[idx, l].split()
                for bik in biker:
                    biker_tour_convoy_df[bik][lis.index(l)] += \
                        [data_content.tour_convoy_df.loc[idx, 'tour_id']]
        pbar.update(1)
    pbar.close()

    for key, _ in biker_tour_convoy_df.items():
        for i in range(4):
            biker_tour_convoy_df[key][i] = ' '.join(list(set(biker_tour_convoy_df[key][i])))

    biker_tour_convoy_df = pd.DataFrame.from_dict(biker_tour_convoy_df, orient='index')
    biker_tour_convoy_df.reset_index(inplace=True)
    biker_tour_convoy_df.columns = ['biker_id'] + lis
    print('tour_convoy_df ready...', flush=True)

    return biker_tour_convoy_df


def get_coordinates(locations, data_content):
    geolocation_map = {}
    locator = Nominatim(user_agent="Kolibri")

    for i in tqdm(range(len(locations)),
                  disable=False,
                  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        # noinspection PyBroadException
        try:
            location = locator.geocode(locations[i])
            geolocation_map[locations[i]] = [location.latitude, location.longitude]
        except:
            # Called when there is presumably some noise in the Address location
            # noinspection PyBroadException
            data_content.noise += [locations[i]]
            geolocation_map[locations[i]] = [np.nan, np.nan]

    location_df = pd.DataFrame({'location': list(locations),
                                'latitude': np.array(list(geolocation_map.values()))[:, 0],
                                'longitude': np.array(list(geolocation_map.values()))[:, 1]})

    return geolocation_map, location_df


def initialize_locations(data_content):
    # noinspection PyBroadException
    try:
        location_df = pd.read_csv(data_content.base_dir + 'temp/location.csv')
        location_from_csv = True
    except:
        location_df = None
        location_from_csv = False
    if location_from_csv:
        geolocation = {}
        print('Initializing Locations from DataFrame...', flush=True)
        for i, l in enumerate(location_df['location'].tolist()):
            geolocation[l] = [location_df.loc[i, 'latitude'], location_df.loc[i, 'longitude']]
    else:
        print('Initializing Locations from Nominatim...', flush=True)
        biker_location = data_content.bikers_df['area'].dropna().drop_duplicates().tolist()
        geolocation, location_df = get_coordinates(biker_location, data_content)

    return geolocation, location_df


def impute_location_from_tour_convoy(data_content):
    # From tour_convoy
    unk_loc = data_content.bikers_df[pd.isna(data_content.bikers_df['latitude'])]
    org_bik = list(set(data_content.convoy_df['biker_id'].drop_duplicates().tolist()).intersection(
        data_content.bikers_df['biker_id'].tolist()))
    groups = ['going', 'not_going', 'maybe', 'invited']
    rest_trs = data_content.tours_df[data_content.tours_df['tour_id'].isin(
        data_content.tour_convoy_df['tour_id'])]
    rest_con = data_content.convoy_df[data_content.convoy_df['biker_id'].isin(org_bik)]
    pbar = tqdm(total=unk_loc.shape[0], bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    pbar.set_description('Step ' + str(data_content.current_step) + ' of ' + str(data_content.total_steps))

    for idx, _ in unk_loc.iterrows():
        if unk_loc.loc[idx, 'biker_id'] in org_bik:
            cdf = rest_con[rest_con['biker_id'] == unk_loc.loc[idx, 'biker_id']]
            if cdf.shape[0] > 0:
                tours = []
                for g in groups:
                    tours += cdf[g].tolist()[0].split()
                tours = (' '.join(tours)).split()
                trs = rest_trs[rest_trs['tour_id'].isin(tours)]
                if trs.shape[0] > 0:
                    m, _ = mode(trs[['latitude']], axis=0)
                    if not np.isnan(m[0, 0]):
                        index = trs[trs['latitude'] == m[0, 0]].index.tolist()[0]
                        lat, long, = trs.loc[index, 'latitude'], trs.loc[index, 'longitude']
                        data_content.bikers_df.loc[idx, 'latitude'] = lat
                        data_content.bikers_df.loc[idx, 'longitude'] = long
        pbar.update(1)
    pbar.close()

    data_content.current_step += 1


def impute_location_from_tours(data_content):
    # From tours_df
    unk_loc = data_content.bikers_df[pd.isna(data_content.bikers_df['latitude'])]
    org_bik = list(set(data_content.tours_df['biker_id'].drop_duplicates().tolist()).intersection(
        data_content.bikers_df['biker_id'].tolist()))
    pbar = tqdm(total=unk_loc.shape[0], bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    pbar.set_description('Step ' + str(data_content.current_step) + ' of ' + str(data_content.total_steps))

    for idx, _ in unk_loc.iterrows():
        if unk_loc.loc[idx, 'biker_id'] in org_bik:
            tours = data_content.tours_df[data_content.tours_df['biker_id'] == unk_loc.loc[idx, 'biker_id']]
            if tours.shape[0] > 0:
                m, _ = mode(tours[['latitude']], axis=0)
                if not np.isnan(m[0, 0]):
                    index = tours[tours['latitude'] == m[0, 0]].index.tolist()[0]
                    lat, long, = tours.loc[index, 'latitude'], tours.loc[index, 'longitude']
                    if not np.isnan(lat):
                        data_content.bikers_df.loc[idx, 'latitude'] = lat
                        data_content.bikers_df.loc[idx, 'longitude'] = long
        pbar.update(1)
    pbar.close()

    data_content.current_step += 1


def impute_lcoation_from_friends(data_content):
    biker_df = pd.merge(data_content.bikers_df,
                        data_content.bikers_network_df, on='biker_id', how='left').copy()
    bikers_df_ids = set(data_content.bikers_df['biker_id'].tolist())

    # From friends
    for i in range(data_content.location_recursion):
        pbar = tqdm(total=biker_df.shape[0], bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        pbar.set_description('Step ' + str(data_content.current_step) + ' of ' + str(data_content.total_steps))
        for idx, rows in biker_df.iterrows():
            if not pd.isna(biker_df.loc[idx, 'friends']):
                bikers_known_friends = set(biker_df.loc[idx, 'friends'].split()).intersection(bikers_df_ids)
                if len(bikers_known_friends) >= data_content.member_threshold:
                    temp_df = biker_df[biker_df['biker_id'].isin(bikers_known_friends)].dropna()
                    if temp_df.shape[0] > 0:
                        m, _ = mode(temp_df[['latitude']], axis=0)
                        if not np.isnan(m[0, 0]):
                            index = temp_df[temp_df['latitude'] == m[0, 0]].index.tolist()[0]
                            lat, long, = temp_df.loc[index, 'latitude'], temp_df.loc[index, 'longitude']
                            if pd.isna(data_content.bikers_df.loc[idx, 'latitude']):
                                data_content.bikers_df.loc[idx, 'latitude'] = lat
                                data_content.bikers_df.loc[idx, 'longitude'] = long
                            elif not np.isnan(lat):
                                dist = (data_content.bikers_df.loc[idx, 'latitude'] - lat) ** 2 + \
                                       (data_content.bikers_df.loc[idx, 'longitude'] - long) ** 2
                                if (dist ** 0.5) > data_content.gps_threshold:
                                    data_content.bikers_df.loc[idx, 'latitude'] = lat
                                    data_content.bikers_df.loc[idx, 'longitude'] = long
            pbar.update(1)
        pbar.close()
        data_content.current_step += 1


def fill_missing_locations(data_content):
    impute_lcoation_from_friends(data_content)
    impute_location_from_tours(data_content)
    impute_location_from_tour_convoy(data_content)


def handle_locations(data_content):
    print('Preprocessing bikers_df..', flush=True)
    print('Initializing Locations...', flush=True)
    geolocation, location_df = initialize_locations(data_content)
    loc = set(location_df['location'].tolist())

    for i in tqdm(range(data_content.bikers_df.shape[0]),
                  disable=False, desc='Step 1 of ' + str(data_content.total_steps),
                  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        if data_content.bikers_df.loc[i, 'area'] in loc:
            data_content.bikers_df.loc[i, 'latitude'] = geolocation[data_content.bikers_df.loc[i, 'area']][0]
            data_content.bikers_df.loc[i, 'longitude'] = geolocation[data_content.bikers_df.loc[i, 'area']][1]
    data_content.current_step += 1

    # Imputing Missing Locations
    fill_missing_locations(data_content)
    print('Locations Initialized...', flush=True)
    print('bikers_df ready', flush=True)


def time_zone_converter(data_content):
    for idx, _ in data_content.bikers_df.iterrows():
        if not np.isnan(data_content.bikers_df.loc[idx, 'longitude']):
            x = data_content.bikers_df.loc[idx, 'longitude']
            data_content.bikers_df.loc[idx, 'time_zone'] = (np.floor((x - 7.500000001) / 15) + 1) * 60


def time_zone_for_location_imputation(data_content):
    timezones = np.unique(data_content.bikers_df['time_zone'].drop_duplicates().dropna())
    tz = dict()

    for time in timezones:
        df = data_content.bikers_df[data_content.bikers_df['time_zone'] == time]
        m, _ = mode(df[['latitude']], axis=0)
        if not np.isnan(m[0, 0]):
            index = df[df['latitude'] == m[0, 0]].index.tolist()[0]
            lat, long, = df.loc[index, 'latitude'], df.loc[index, 'longitude']
            tz[time] = [lat, long]
    data_content.bikers_df['time_zone'] = data_content.bikers_df['time_zone'].map(
        lambda x: x if x in timezones else np.nan)
    df = data_content.bikers_df[(pd.isna(data_content.bikers_df['latitude'])) & (
        pd.notna(data_content.bikers_df['time_zone']))]

    for idx, _ in df.iterrows():
        key = df.loc[idx, 'time_zone']
        if key in tz.keys():
            data_content.bikers_df.loc[idx, 'latitude'] = tz[key][0]
            data_content.bikers_df.loc[idx, 'longitude'] = tz[key][1]


def language_for_location_imputation(data_content):
    df = data_content.bikers_df[(pd.isna(data_content.bikers_df['latitude']))]

    for idx, _ in df.iterrows():
        location = data_content.locale_[data_content.bikers_df.loc[idx, 'language']][3]
        data_content.bikers_df.loc[idx, 'latitude'] = location[0]
        data_content.bikers_df.loc[idx, 'longitude'] = location[1]


def compute_non_pop(tdf, data_content):
    dont_pop = tdf[(pd.isna(tdf['latitude']))]['biker_id'].tolist()
    cat = ['going', 'maybe', 'invited', 'not_going']
    pbar = tqdm(total=tdf.shape[0], bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    pbar.set_description('Step 1 of 4')

    for idx, _ in tdf.iterrows():
        bik = data_content.tour_convoy_df[data_content.tour_convoy_df['tour_id'] == tdf.loc[idx, 'tour_id']]
        coll = []
        for c in cat:
            if not pd.isna(bik[c].tolist()[0]):
                coll += bik[c].tolist()[0].split()
        dont_pop += coll
        pbar.update(1)
    pbar.close()
    dont_pop = list(set(dont_pop))

    return dont_pop


def initialize_network_dict(data_content):
    network = {}
    pbar = tqdm(total=data_content.bikers_network_df.shape[0],
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    pbar.set_description("Step 2 of 4")

    for idx, _ in data_content.bikers_network_df.iterrows():
        bik_id = data_content.bikers_network_df.loc[idx, 'biker_id']
        if not pd.isna(data_content.bikers_network_df.loc[idx, 'friends']):
            bik = data_content.bikers_network_df.loc[idx, 'friends'].split()
            if bik_id in network:
                network[bik_id] = network[bik_id] + list(bik)
            else:
                network[bik_id] = list(bik)
            for biker in bik:
                if biker in network:
                    network[biker] = network[biker] + [bik_id]
                else:
                    network[biker] = [bik_id]
        pbar.update(1)
    pbar.close()

    return network


def process_bikers_network_df(tdf, data_content):
    dont_pop = compute_non_pop(tdf, data_content)
    network = initialize_network_dict(data_content)
    pop_list = list(set(network.keys()) - set(dont_pop))

    for ele in pop_list:
        network.pop(ele)

    for key, _ in network.items():
        network[key] = ' '.join(list(set(network[key])))

    network_df = pd.DataFrame.from_dict(network, orient='index')
    network_df.reset_index(inplace=True)
    network_df.columns = ['biker_id', 'friends']

    return network_df


def fill_network_df(network_df, data_content):
    network_df = pd.merge(network_df, data_content.bikers_df[['biker_id', 'latitude', 'longitude']], on='biker_id',
                          how='left')

    network_df['friends'] = network_df['friends'].apply(lambda x: x.split()[0])
    get_frnds = list(set(network_df['friends'].tolist()).intersection(data_content.bikers_df['biker_id'].tolist()))
    grouped = network_df.groupby(by='friends')
    small_df = data_content.bikers_df[data_content.bikers_df['biker_id'].isin(get_frnds)]
    pbar = tqdm(total=small_df.shape[0], bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    pbar.set_description('Step 3 of 4')

    for idx, _ in small_df.iterrows():
        i = grouped.get_group(small_df.loc[idx, 'biker_id']).index
        network_df.loc[i, 'latitude'] = small_df.loc[idx, 'latitude']
        network_df.loc[i, 'longitude'] = small_df.loc[idx, 'longitude']
        pbar.update(1)
    pbar.close()

    return network_df


def fill_location_for_tours_df(tdf, network_df, data_content):
    tid = tdf[pd.isna(tdf['latitude'])]
    pbar = tqdm(total=tid.shape[0], bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    pbar.set_description('Step 4 of 4')

    for idx, _ in tid.iterrows():
        cat = ['going', 'maybe', 'invited', 'not_going']
        bik = data_content.tour_convoy_df[data_content.tour_convoy_df['tour_id'] == tdf.loc[idx, 'tour_id']]
        coll = []
        for c in cat:
            if not pd.isna(bik[c].tolist()[0]):
                coll += bik[c].tolist()[0].split()
        g = network_df[network_df['biker_id'].isin(coll)]
        if g.shape[0] > 0:
            m, _ = mode(g[['latitude']], axis=0)
            if not np.isnan(m[0, 0]):
                index = g[g['latitude'] == m[0, 0]].index.tolist()[0]
                lat, long = g.loc[index, 'latitude'], g.loc[index, 'longitude']
                tdf.loc[idx, 'latitude'] = lat
                tdf.loc[idx, 'longitude'] = long
        pbar.update(1)
    pbar.close()

    bid = tdf[pd.isna(tdf['latitude'])]['biker_id'].drop_duplicates().tolist()
    chi = data_content.tours_df[data_content.tours_df['biker_id'].isin(bid)]
    chi = chi[pd.notna(chi['latitude'])].groupby('biker_id')[['latitude', 'longitude']].agg(
        lambda x: x.value_counts().index[0])
    chi = chi.reset_index()

    for idx, _ in tdf[pd.isna(tdf['latitude'])].iterrows():
        m = chi[chi['biker_id'] == tdf.loc[idx, 'biker_id']]
        if m.shape[0] != 0:
            tdf.loc[idx, 'latitude'] = m['latitude'].tolist()[0]
            tdf.loc[idx, 'longitude'] = m['longitude'].tolist()[0]

    # Using tour_convoy_df to find tours attended by biker organizing this tour
    # and fill location from based on that information.
    coll = []
    tid = tdf[pd.isna(tdf['latitude'])]
    sdf = data_content.convoy_df[data_content.convoy_df['biker_id'].isin(tid['biker_id'].tolist())]

    for idx, _ in tid.iterrows():
        cat = ['going', 'maybe', 'invited', 'not_going']
        bik = sdf[sdf['biker_id'] == tid.loc[idx, 'biker_id']]
        if bik.shape[0] > 0:
            for c in cat:
                if not pd.isna(bik[c].tolist()[0]):
                    coll += bik[c].tolist()[0].split()

    small_df = data_content.tours_df[data_content.tours_df['tour_id'].isin(coll)]
    for idx, _ in tid.iterrows():
        cat = ['going', 'maybe', 'invited', 'not_going']
        bik = sdf[sdf['biker_id'] == tdf.loc[idx, 'biker_id']]
        if bik.shape[0] > 0:
            coll = []
            for c in cat:
                if not pd.isna(bik[c].tolist()[0]):
                    coll += bik[c].tolist()[0].split()
            g = small_df[small_df['tour_id'].isin(coll)]
            if g.shape[0] > 0:
                m, _ = mode(g[['latitude']], axis=0)
                if not np.isnan(m[0, 0]):
                    index = g[g['latitude'] == m[0, 0]].index.tolist()[0]
                    lat, long = g.loc[index, 'latitude'], g.loc[index, 'longitude']
                    tdf.loc[idx, 'latitude'] = lat
                    tdf.loc[idx, 'longitude'] = long

    return tdf


def process_bikers_df(data_content):
    data_content.bikers_df['area'] = data_content.bikers_df['area'].apply(
        lambda x: x.replace('  ', ', ') if not pd.isna(x) else x)
    data_content.bikers_df['language'] = data_content.bikers_df['language_id'] + \
                                         '_' + data_content.bikers_df['location_id']
    data_content.bikers_df['age'] = data_content.bikers_df['bornIn'].apply(
        lambda x: handle_bornIn(x))
    data_content.bikers_df['gender'] = data_content.bikers_df['gender'].apply(
        lambda x: handle_gender(x))
    data_content.bikers_df['member_since'] = data_content.bikers_df['member_since'].apply(
        lambda x: handle_memberSince(x))

    handle_locations(data_content)
    time_zone_converter(data_content)
    time_zone_for_location_imputation(data_content)
    language_for_location_imputation(data_content)
    time_zone_converter(data_content)
    data_content.bikers_df.drop(['bornIn', 'area', 'language_id', 'location_id'], axis=1, inplace=True)

    # noinspection PyBroadException
    try:
        tdf = pd.read_csv(data_content.base_dir + 'temp/tdf.csv')
    except:
        print('Initializing bikers_network_df', flush=True)
        tdf = data_content.tours_df[data_content.tours_df['tour_id'].isin(
            data_content.tour_convoy_df['tour_id'])]
        network_df = process_bikers_network_df(tdf, data_content)
        network_df = fill_network_df(network_df, data_content)
        tdf = fill_location_for_tours_df(tdf, network_df, data_content)
        print('bikers_network_df initialized', flush=True)

    return tdf


def preprocess_dataset(data_content):
    print('Preprocessing the Data...', flush=True)
    process_tours_df(data_content)
    data_content.convoy_df = process_tour_convoy_df(data_content)
    data_content.tdf = process_bikers_df(data_content)
    print('Data processed...\n', flush=True)

    return data_content
