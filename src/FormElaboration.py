import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_personality_matrix(num_personalities, num_data, personality_types):
    """
    Creation of multiplication matrix and bias vector for the computation of the personality test according to the
    definition
    :param personality_types:
    :param num_personalities: number of personalities types in the study
    :param num_data: number of data to which the subject has answered
    :return: multiplication matrix and bias vector
    """

    # empty personality matrix
    personality_matrix = np.zeros([num_personalities, num_data])

    # where to put +1 or -1 in the personality matrix for each row
    E = {'name': 'E', '+': [1, 11, 21, 31, 41], '-': [6, 16, 26, 36, 46]}
    A = {'name': 'A', '+': [7, 17, 27, 37, 42, 47], '-': [2, 12, 22, 32]}
    C = {'name': 'C', '+': [3, 13, 23, 33, 43, 48], '-': [8, 18, 28, 38]}
    N = {'name': 'N', '+': [9, 19], '-': [4, 14, 24, 29, 34, 39, 44, 49]}
    O = {'name': 'O', '+': [5, 15, 25, 35, 40, 45, 50], '-': [10, 20, 30]}

    # filling of the matrix according to the definition
    for dict in [E, A, C, N, O]:

        name = dict['name']
        plus = dict['+']
        minus = dict['-']

        index = personality_types.index(name)

        for id in plus:
            personality_matrix[index, id-1] = +1
        for id in minus:
            personality_matrix[index, id-1] = -1

    # personality bias vector definition according to the explanation
    personality_bias = [20, 14, 14, 38, 8]

    return personality_matrix, personality_bias


if __name__ == '__main__':

    # unique filepath for the form results file
    filepath = '../data/form-results/Form results - Risposte del modulo 1.csv'

    # reading of the file
    df = pd.read_csv(filepath)

    # extraction of the participant codes and columns names
    codes = df.iloc[:]['INSERT YOUR PARTECIPANT CODE'].tolist()
    cols = list(df.columns)

    # determination of the columns of interest (demographic information) and removal of non-interesting parts
    columns = df.columns[1:6]
    columns_list = list(columns)
    for idx, name in enumerate(columns_list):
        name = name.lower()
        columns_list[idx] = name.replace('insert your ', '')

    # definition of the types of personality according to the definition
    p_types = ['E', 'A', 'C', 'N', 'O']

    # creation of the new dataframe with the columns of interest
    new_columns = columns_list + p_types
    participants_results = pd.DataFrame(columns=new_columns)

    # cycle for each participant
    for code in codes:

        # extraction data corresponding to the participant of interest
        data = df.loc[df['INSERT YOUR PARTECIPANT CODE'] == code]

        # extraction of the data in the columns of interest
        results = list(data[columns].values[0])
        results = list(map(lambda x: str(x).lower(), results))

        # extraction of the columns of interest in the personality computation: all the ones that starts with 'I...'
        p = []
        for col in cols:
            res = re.findall("^I[.]{3} [\[]{1}.+[\]]{1}", str(col))
            if res:
                p.append(res[0])

        # extraction of the data regarding the columns of interest just extracted and cast to int values
        p_data = list(data.iloc[0][p])
        p_data = list(int(vote[0]) for vote in p_data)

        # creation of the personality matrix and bias vector
        p_matrix, p_bias = create_personality_matrix(len(p_types), len(p_data), p_types)

        # computation for the derivation of each personality score
        p_results = list(np.array(p_matrix.dot(p_data) + p_bias, dtype='int'))

        # concatenation of data of interest and saving of it inside the final dataframe
        results = results + p_results
        participants_results.loc[participants_results.shape[0]] = results

    # saving of the results regarding the form
    participants_results.to_csv('../data/form-results/form-results.csv')

    # COMPUTATION OF ANALYSIS FOR ONLINE IMAGE EVALUATION

    # path to the folder containing the csv files results
    rating_path = '../data/ratings-results/'
    available_ratings_paths = glob.glob(rating_path + '/*.csv')

    # definition of the columns of interest, for the input file (columns) and for the output file (new_columns)
    columns = list(['img_name', 'manipulation', 'vm', 'am', 'valence_slider.response', 'arousal_slider.response'])
    participant_code = 'partecipant_code'
    new_columns = [participant_code] + columns

    # creation of results dataframe
    responses = pd.DataFrame(columns=new_columns)

    # cycling over all participants codes (found in online form)
    for code in codes:

        # find the filepath (if exists) containing the participant code
        correct = None
        for filepath in available_ratings_paths:
            if filepath.find(code) != -1:
                correct = filepath
        if correct is None:
            continue

        # reading dataframe
        df = pd.read_csv(correct)

        # extraction of the columns of interest and subsequent saving
        data = df[columns]
        data.insert(0, participant_code, code)

        means = data.loc[:, ['vm', 'am']]
        new_ratings = data.loc[:, ['valence_slider.response', 'arousal_slider.response']]
        labels, valence_difference, arousal_difference = [], [], []
        threshold = 0.1

        for index, row in means.iterrows():
            valence = row['vm']
            arousal = row['am']

            if (np.square(valence) + np.square(arousal)) <= np.square(threshold):
                labels.append('Neutral')
            elif valence > 0 and arousal > 0:
                labels.append('HVHA')
            elif valence > 0 and arousal <= 0:
                labels.append('HVLA')
            elif valence <= 0 and arousal > 0:
                labels.append('LVHA')
            elif valence <= 0 and arousal <= 0:
                labels.append('LVLA')

            new_valence = new_ratings.loc[index, 'valence_slider.response']
            new_arousal = new_ratings.loc[index, 'arousal_slider.response']

            valence_difference.append(valence-new_valence)
            arousal_difference.append(arousal-new_arousal)

        data.insert(data.shape[1], 'label', labels, True)
        data.insert(data.shape[1], 'valence_difference', valence_difference, True)
        data.insert(data.shape[1], 'arousal_difference', arousal_difference, True)

        responses = pd.concat([responses, data])

    # saving of the csv file containing all the
    responses.to_csv('../data/ratings-results/ratings-results.csv')

    # to visually check normality for statistical tests
    valence_difference = responses.loc[:, 'valence_difference']
    plt.hist(valence_difference)
    plt.title('valence difference')
    plt.show()

    arousal_difference = responses.loc[:, 'arousal_difference']
    plt.hist(arousal_difference)
    plt.title('arousal difference')
    plt.show()
