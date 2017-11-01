import pandas as pd

test = pd.read_csv('../train.csv')
test_feature = []
source_dict = {}
source_dict['my library'] = 0
source_dict['discover'] = 1
source_dict['search'] = 2
source_dict['radio'] = 3
source_dict['explore'] = 4
source_dict['listen with'] = 5
print('Building Test vectors')
for index, row in test.iterrows():
    category = row.tolist()[3];
    f_vector = [0, 0, 0, 0, 0, 0, 0]
    if source_dict.get(category) is None:
        f_vector[6] = 1
    else:
        f_vector[source_dict[category]] = 1;
    # try:
    #     count = songs_count.loc[songs_count['song_id'] == row['song_id']]
    #     f_vector.append(int(count.iloc[0, -2]/4000))
    # except KeyError as e:
    #     f_vector.append(0)
    test_feature.append(f_vector)


temp = pd.DataFrame(test_feature)
temp.to_csv('train_data.csv')