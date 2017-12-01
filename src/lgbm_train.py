import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb

path = '../data/'
train = pd.read_csv(path + 'train.csv',  dtype={'msno': 'category',
                                                'source_system_tab': 'category',
                                                'source_screen_name': 'category',
                                                'source_type': 'category',
                                                'target': np.uint8,
                                                'song_id': 'category'})

test = pd.read_csv(path + 'test.csv',    dtype={'id': np.uint32,
                                                'song_id': 'category',
                                                'msno': 'category',
                                                'source_system_tab': 'category',
                                                'source_screen_name': 'category',
                                                'source_type': 'category',
                                                'song_id': 'category'})

songs = pd.read_csv(path + 'songs.csv', dtype={'genre_ids': 'category',
                                               'language': 'category',
                                               'artist_name': 'category',
                                               'composer': 'category',
                                               'lyricist': 'category',
                                               'song_id': 'category'})

members = pd.read_csv(path + 'members.csv', dtype={'city': 'category',
                                                   'bd': np.uint8,
                                                   'gender': 'category',
                                                   'registered_via': 'category'})

songs_extra = pd.read_csv(path + 'song_extra_info.csv')

print('...Data preprocessing...')
# get rid of | in genre id
songs['genre_ids'] = songs['genre_ids'].apply(lambda genre: str(genre).split('|')[0]).astype('category')
#songs['song_length'] = (songs['song_length'] / 60000).astype(int)
print('Generating members features...')
members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_day'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_day'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))

members['registration_init_time'] = pd.to_datetime(members['registration_init_time'], format='%Y%m%d')
members['expiration_date'] = pd.to_datetime(members['expiration_date'], format='%Y%m%d')
members['membership_length'] = members['expiration_date'] - members['registration_init_time']
# convert timedelta object back to int
members['membership_length'] = (members['membership_length'] / np.timedelta64(1, 'D')).astype(int)
# get rid of skew data
members.loc[members.membership_length < 0, 'membership_length'] = 0
#norm_range = members['membership_length'].max() - members['membership_length'].min()
# normalize length to [0,1]
# members['membership_length'] = (members['membership_length'] - members['membership_length'].min()) / norm_range
mean_age = members['bd'].mean()
members.loc[members.bd == 0, 'bd'] = mean_age


def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan


songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis=1, inplace=True)

def get_features(data, train=True):
    """features to used"""
    featurs = [
        'source_system_tab', 'source_screen_name', 'source_type', 'song_length', 'genre_ids', 'language',
        'city', 'registered_via', 'registration_year', 'registration_month', 'registration_day',
        'expiration_year', 'expiration_month', 'expiration_day', 'song_year', 'membership_length',
        'bd', 'gender',
    ]
    label = []
    feature_vectors = pd.DataFrame()
    feature_vectors['msno'] = data['msno']
    feature_vectors['song_id'] = data['song_id']
    feature_vectors['source_system_tab'] = data['source_system_tab']
    feature_vectors['source_screen_name'] = data['source_screen_name']
    feature_vectors['source_type'] = data['source_type']
    feature_vectors = feature_vectors.merge(members, on='msno', how='left')
    feature_vectors = feature_vectors.merge(songs, on='song_id', how='left')
    feature_vectors = feature_vectors.merge(songs_extra, on='song_id', how='left')
    if train:
        label = data['target']
    return feature_vectors[featurs], label

params = {
    'task': 'train',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 2**8,
    'max_depth': 10,
    'learning_rate': 0.2,
    'verbosity': 0
}
num_round = 100

train_x, labels = get_features(train, True)
test_x, empty = get_features(test, train=False)
train_x = train_x.apply(lambda x: x.fillna(x.value_counts().index[0]))
test_x = test_x.apply(lambda x: x.fillna(x.value_counts().index[0]))
print('Building data set...')
train_data = lgb.Dataset(train_x,
                         label=labels,
                         categorical_feature=['source_system_tab', 'source_screen_name', 'source_type', 'language'])
valid_data = lgb.Dataset(train_x, labels)

print('Training......')
bst = lgb.train(params, train_data, num_round, valid_sets=valid_data, verbose_eval=5)

print('Predicting......')
pred = bst.predict(test_x)
print('Prediction Done......')
print('##############Writing output##############')
sub = pd.DataFrame()
sub['id'] = test['id']
sub['target'] = pred
sub.to_csv('out.csv', index=False, mode='w+')
