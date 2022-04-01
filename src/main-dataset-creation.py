from os.path import exists

import pandas as pd

from EEGAnalysis import *

if __name__ == '__main__':

    # get the codes of people in the form
    path_form = '../data/form-results/form-results.csv'
    df_form = pd.read_csv(path_form, index_col=0)
    codes_form = df_form.loc[:, 'partecipant code'].tolist()
    codes_form = list(set(codes_form))

    # get the codes of people in the ratings
    path_ratings = '../data/ratings-results/ratings-results.csv'
    df_ratings = pd.read_csv(path_ratings, index_col=0)
    codes_ratings = df_ratings.loc[:, 'partecipant_code'].tolist()
    codes_ratings = list(set(codes_ratings))

    path_eeg = '../data/eeg/'

    info_dataset, signal_dataset, label_dataset = [], [], []

    for code in codes_form:
        if code not in codes_ratings:
            continue

        path_signals = path_eeg+'/subj_'+code+'_block1.xdf'
        if not exists(path_signals):
            continue

        data_form = df_form.loc[df_form['partecipant code'] == code, :].values.flatten().tolist()[1:]
        data_ratings = df_ratings.loc[df_ratings['partecipant_code'] == code]

        eeg = EEGAnalysis(path_signals)
        eeg.create_raw()
        eeg.filter_raw()
        eeg.set_reference()
        eeg.define_epochs_raw(visualize=False, only_manipulation=False)

        signals = eeg.get_epochs_dataframe()
        signals.to_csv('../data/eeg/signal.csv')

        epochs = signals.loc[:, 'epoch'].values.flatten()
        epochs = np.unique(epochs)

        for epoch in epochs:

            current_signal = signals.loc[signals['epoch'] == epoch]
            x = current_signal.iloc[:, 3:].to_numpy().T

            img_name = current_signal.iloc[0]['condition']
            img_name = img_name.split('/')[0]

            ratings = data_ratings.loc[data_ratings['img_name'] == img_name][['valence_slider.response', 'arousal_slider.response']].values[0]

            info_dataset.append(data_form)
            signal_dataset.append(x)
            label_dataset.append(ratings)

    info_dataset = np.array(info_dataset)
    signal_dataset = np.array(signal_dataset)
    label_dataset = np.array(label_dataset)

    np.save('../data/final-dataset/info_dataset.npy', info_dataset)
    np.save('../data/final-dataset/signal_dataset.npy', signal_dataset)
    np.save('../data/final-dataset/label_dataset.npy', label_dataset)
