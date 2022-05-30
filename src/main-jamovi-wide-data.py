import json

import pandas as pd

from src.EEGAnalysis import EEGAnalysis

if __name__ == '__main__':

    participant_data = pd.read_csv('../data/form-results/form-results.csv')
    participant_data = participant_data[['code', 'gender']]

    rating_path = '../data/ratings-results/ratings-results.csv'
    data = pd.read_csv(rating_path, index_col=0)

    data = data[(data['code'] != 'maba09') & (data['code'] != 'soze31') & (data['code'] != 'nipe10') &
                (data['code'] != 'dino02')]
    data['code'] = data['code'].astype('category')
    data['img_name'] = data['img_name'].astype('category')
    data['manipulation'] = data['manipulation'].astype('category')

    # h1: original vs new ratings

    image_codes = data.loc[:, 'img_name'].to_list()
    image_codes = list(set(image_codes))
    image_codes = [code for code in image_codes if '_orig' in code]

    features = [['vm', 'valence'], ['am', 'arousal']]

    for feature in features:
        x = data.loc[data['img_name'].isin(image_codes)]
        data_wide = x.groupby(['img_name']).mean()[feature].dropna().reset_index()
        data_wide.to_csv('../data/jamovi-tables/h1_' + feature[1] + '.csv', index=0)

    # h2: subject ID and rating according to the manipulation

    participant_codes = data.loc[:, 'code'].to_list()
    participant_codes = list(set(participant_codes))

    features = ['valence', 'arousal']

    for feature in features:

        data_wide = data.groupby(['code', 'manipulation']).mean()[feature].reset_index().pivot(index='code',
                                                                                               columns=['manipulation'])
        data_wide.columns = ["_".join(col[1:]).strip() for col in data_wide.columns.values]

        gender = []
        for idx, row in data_wide.iterrows():
            gender.append(participant_data.loc[participant_data['code'] == idx]['gender'].values[0][0])

        data_wide.insert(0, 'gender', gender)
        data_wide['gender'] = data_wide['gender'].astype('category')
        data_wide.to_csv('../data/jamovi-tables/h2_' + feature + '.csv')

    # h3: subject ID and N170 amplitude according to the manipulation

    path = '../data/eeg/'
    dict_info = json.load(open('../data/eeg/info.json'))

    data_wide = pd.DataFrame(columns=['code', 'gender', 'blackwhite', 'blurring', 'circularblurringext',
                                      'circularblurringint', 'canny', 'original'])
    columns = list(data_wide.columns)[2:]

    for code in participant_codes:
        file_path = path + 'subj_' + code + '_block1.xdf'

        eeg = EEGAnalysis(file_path, dict_info=dict_info)
        eeg.create_raw()
        eeg.set_reference()
        eeg.filter_raw()
        eeg.define_annotations()
        eeg.define_epochs_raw(save_epochs=False)
        peaks = eeg.get_n170_peak()

        peaks_values = [peaks[key] for key in columns]
        g = participant_data.loc[participant_data['code'] == code]['gender'].values[0][0]

        peaks_values = [code, g] + peaks_values
        data_wide.loc[len(data_wide.index)] = peaks_values

    data_wide['code'] = data_wide['code'].astype('category')
    data_wide['gender'] = data_wide['gender'].astype('category')
    data_wide.to_csv('../data/jamovi-tables/h3.csv')

