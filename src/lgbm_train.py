import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
import argparse
import time
import gc
path = '../data/'
start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--cv", help="Cross Validation Mode",
                    action="store_true")
args = parser.parse_args()
cv = False
if args.cv:
    print('###\tCross-Validation Mode')
    cv = True
else:
    print('###\tPrediction Mode')

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
                                                'source_type': 'category'})

songs = pd.read_csv(path + 'songs.csv', dtype={'genre_ids': 'category',
                                               'language': np.float32,
                                               'artist_name': 'category',
                                               'composer': 'category',
                                               'lyricist': 'category',
                                               'song_id': 'category'})

members = pd.read_csv(path + 'members.csv', dtype={'city': 'category',
                                                   'bd': np.uint8,
                                                   'gender': 'category',
                                                   'registered_via': 'category'})

songs_extra = pd.read_csv(path + 'song_extra_info.csv')
print('song features')
# filling missing values
def is_chinese(x):
    if '3.0' in str(x) or '10.0' in str(x) or '24.0' in str(x) or '59.0' in str(x):
        return 1
    return 0

def genre_id_count(x):
    if x == 'no_genre_id':
        return 0
    else:
        return x.count('|') + 1

def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';', '-', '、', '&', '+'])) + 1
    #return sum(map(x.count, ['|', '/', '\\', ';']))

def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';', '-', '、', '&', '+'])) + 1

def is_featured(x):
    if 'feat' in str(x) :
        return 1
    return 0

def artist_count(x): 
    if x == 'no_artist': 
        return 0 
    else: 
        return x.count('and') + x.count(',') + x.count('feat') + x.count('&') 

songs['lyricist'] = songs['lyricist'].cat.add_categories(['no_lyricist'])
songs['lyricist'].fillna('no_lyricist',inplace=True)
songs['genre_ids'] = songs['genre_ids'].cat.add_categories(['no_genre_id'])
songs['genre_ids'].fillna('no_genre_id',inplace=True)
songs['composer'] = songs['composer'].cat.add_categories(['no_composer'])
songs['composer'].fillna('no_composer',inplace=True)
songs.song_length.fillna(200000,inplace=True)
# songs['artist_name'] = songs['artist_name'].cat.add_categories(['no_artist'])
# songs['artist_name'].songs('no_artist',inplace=True)
songs['is_chinese'] = songs['language'].apply(is_chinese).astype(np.uint8)
songs['num_genre_ids'] = songs['genre_ids'].apply(genre_id_count).astype(np.uint8)
songs['genre_ids'] = songs['genre_ids'].apply(lambda genre: str(genre).split('|')[0]).astype('category')
songs['composer_count'] = songs['composer'].apply(composer_count).astype(np.uint8)
songs['lyricist_count'] = songs['lyricist'].apply(lyricist_count).astype(np.uint8)
songs['is_featured'] = songs['artist_name'].apply(is_featured).astype(np.uint8)
songs['artist_count'] = songs['artist_name'].apply(artist_count).astype(np.int8) 
print('\t Done')


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
# mean_age = members['bd'].mean()
# members.loc[members.bd == 0, 'bd'] = mean_age

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
    features = [
        'source_system_tab', 'source_screen_name', 'source_type', 'song_length', 'genre_ids', 'language',
        'city', 'registered_via', 'registration_year', 'registration_month', 'registration_day',
        'expiration_year', 'expiration_month', 'expiration_day', 'song_year', 'membership_length',
        'bd', 'gender', 'song_id_count', 'genre_id_count', 'artist_name_count', 'num_genre_ids', 'composer_count', 'lyricist_count', 'is_featured',
        'artist_count', 'genre_song_count', 'stype|msno', 'ssname|msno', 'sstab|msno', 'stype|sid'
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
    s = feature_vectors['song_id'].value_counts()
    feature_vectors['song_id_count'] =feature_vectors['song_id'].apply(lambda x: s[x])
    g = feature_vectors['genre_ids'].value_counts()
    feature_vectors['genre_id_count'] =feature_vectors['genre_ids'].apply(lambda x: g[x])
    a = feature_vectors['artist_name'].value_counts()
    feature_vectors['artist_name_count'] =feature_vectors['artist_name'].apply(lambda x: a[x])
    del s
    del g
    del a
    gc.collect()
    genre_count = feature_vectors.groupby(['msno','genre_ids'])['song_id'].count().to_frame().reset_index();
    genre_count.columns = ['msno', 'genre_ids', 'genre_song_count']
    feature_vectors = feature_vectors.merge(genre_count, on=['msno', 'genre_ids'], how='left')
    del genre_count

    stype_msno = feature_vectors.groupby(['msno', 'source_type'])['source_type'].count()
    stype_msno = stype_msno.rename(columns={'source_type': 'type_count'}).to_frame().reset_index()
    stype_msno.columns = ['msno', 'source_type', 'type_count']
    msno_ctr = feature_vectors['msno'].value_counts()
    stype_msno['msno_count'] = stype_msno['msno'].apply(lambda x: msno_ctr[x])
    stype_msno['stype|msno'] = stype_msno['type_count'] / stype_msno['msno_count']
    stype_msno.drop(['type_count', 'msno_count'], axis=1)
    feature_vectors = feature_vectors.merge(stype_msno, on=['msno', 'source_type'], how='left')
    del stype_msno
    ssname_msno = feature_vectors.groupby(['msno', 'source_screen_name'])['source_screen_name'].count()
    ssname_msno = ssname_msno.rename(columns={'source_screen_name': 'name_count'}).to_frame().reset_index()
    ssname_msno.columns = ['msno', 'source_screen_name', 'name_count']
    ssname_msno['msno_count'] = ssname_msno['msno'].apply(lambda x: msno_ctr[x])
    ssname_msno['ssname|msno'] = ssname_msno['name_count'] / ssname_msno['msno_count']
    ssname_msno.drop(['name_count', 'msno_count'], axis=1)
    feature_vectors = feature_vectors.merge(ssname_msno, on=['msno', 'source_screen_name'], how='left')
    del ssname_msno
    sstab_msno = feature_vectors.groupby(['msno', 'source_system_tab'])['source_system_tab'].count()
    sstab_msno = sstab_msno.rename(columns={'source_system_tab': 'tab_count'}).to_frame().reset_index()
    sstab_msno.columns = ['msno', 'source_system_tab', 'tab_count']
    sstab_msno['msno_count'] = sstab_msno['msno'].apply(lambda x: msno_ctr[x])
    del msno_ctr
    sstab_msno['sstab|msno'] = sstab_msno['tab_count'] / sstab_msno['msno_count']
    sstab_msno.drop(['tab_count', 'msno_count'], axis=1)
    feature_vectors = feature_vectors.merge(sstab_msno, on=['msno', 'source_system_tab'], how='left')
    del sstab_msno

    stype_sid = feature_vectors.groupby(['song_id', 'source_type'])['source_type'].count()
    stype_sid = stype_sid.rename(columns={'source_type': 'type_count'}).to_frame().reset_index()
    stype_sid.columns = ['song_id', 'source_type', 'type_count']
    sid_ctr = feature_vectors['song_id'].value_counts()
    stype_sid['sid_count'] = stype_sid['song_id'].apply(lambda x: sid_ctr[x])
    del sid_ctr
    stype_sid['stype|sid'] = stype_sid['type_count'] / stype_sid['sid_count']
    stype_sid.drop(['type_count', 'sid_count'], axis=1)
    feature_vectors = feature_vectors.merge(stype_sid, on=['song_id', 'source_type'], how='left')
    del stype_sid

    if train:
        label = data['target']
    return feature_vectors[features], label

params = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'learning_rate': 0.2 ,
    'verbose': 0,
    'num_leaves': 108,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': 1,
    'feature_fraction': 0.9,
    'feature_fraction_seed': 1,
    'max_bin': 256,
    'max_depth': 10,
    'metric' : 'auc',
    'num_threads': 8,
}
num_round = 826

train_x, labels = get_features(train, True)
test_x, empty = get_features(test, train=False)
train_x = train_x.apply(lambda x: x.fillna(x.value_counts().index[0]))
test_x = test_x.apply(lambda x: x.fillna(x.value_counts().index[0]))
del train
del songs
gc.collect()
print('Building data set...')
if cv: 
    train_data = lgb.Dataset(train_x[:-1475483], 
                             label=labels[:-1475483],
                             categorical_feature=['source_system_tab', 'source_screen_name', 'source_type'])
    valid_data = lgb.Dataset(train_x[-1475483: ], labels[-1475483:])

    print('Training Using Validation Set......')
    bst = lgb.train(params, train_data, 1000, valid_sets=valid_data, verbose_eval=10, early_stopping_rounds=10)
    print('Cross Validation Finished......')
else:
    train_data = lgb.Dataset(train_x, 
                             label=labels,
                             categorical_feature=['source_system_tab', 'source_screen_name', 'source_type'])
    valid_data = lgb.Dataset(train_x, labels)
    print('Predicting......')
    bst = lgb.train(params, train_data, num_round, valid_sets=valid_data, verbose_eval=10)
    pred = bst.predict(test_x)
    print('Prediction Done......')
    print('##############Writing output##############')
    sub = pd.DataFrame()
    sub['id'] = test['id']
    sub['target'] = pred
    #sub.to_csv('out.csv', index=False, mode='w+')
    sub.to_csv('lgbm_submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f') 

end = time.time()
print(str((end - start) / 60), "minutes")