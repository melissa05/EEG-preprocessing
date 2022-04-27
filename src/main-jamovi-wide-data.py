import pandas as pd

if __name__ == '__main__':

    participant_data = pd.read_csv('../data/form-results/form-results.csv')
    participant_data = participant_data[['code', 'gender']]

    rating_path = '../data/ratings-results/ratings-results.csv'
    data = pd.read_csv(rating_path, index_col=0)
    data = data[['code', 'manipulation', 'valence', 'arousal']]

    data['code'] = data['code'].astype('category')
    data['manipulation'] = data['manipulation'].astype('category')

    participant_codes = data.loc[:, 'code'].to_list()
    participant_codes = list(set(participant_codes))

    # h1: subject ID and rating

    features = ['valence', 'arousal']

    for feature in features:

        data_wide = data.groupby(['code', 'manipulation']).mean()[feature].reset_index().pivot(index='code', columns=['manipulation'])
        data_wide.columns = ["_".join(col[1:]).strip() for col in data_wide.columns.values]

        gender = []
        for idx, row in data_wide.iterrows():
            g = participant_data.loc[participant_data['code'] == idx]['gender'].values[0]
            if g == 'female':
                gender.append('f')
            else:
                gender.append('m')

        data_wide.insert(0, 'gender', gender)
        data_wide['gender'] = data_wide['gender'].astype('category')
        data_wide.to_csv('../data/jamovi-tables/h1_'+feature+'.csv')
