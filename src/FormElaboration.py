import glob
import re

import numpy as np
import pandas as pd


def create_personality_matrix(num_personalities, num_data):
    """
    Creation of multiplication matrix and bias vector for the computation of the personality test according to the
    definition
    :param num_personalities: number of personalities types in the study
    :param num_data: number of data to which the subject has answered
    :return: multiplication matrix and bias vector
    """

    # empty personality matrix
    personality_matrix = np.zeros([num_personalities, num_data])

    # filling of the matrix according to the definition
    for i in range(personality_matrix.shape[0]):
        if (i % 2) == 0:
            for j in range(5):
                c = i + 10 * j
                personality_matrix[i, c] = int(1)
                c = i + 5 + 10 * j
                personality_matrix[i, c] = int(-1)
        else:
            for j in range(5):
                c = i + 10 * j
                personality_matrix[i, c] = int(-1)
                c = i + 5 + 10 * j
                personality_matrix[i, c] = int(+1)

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
    columns = df.columns[1:5]
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
        results = list(df[columns].values[0])
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
        p_matrix, p_bias = create_personality_matrix(len(p_types), len(p_data))

        # computation for the derivation of each personality score
        p_results = list(np.array(p_matrix.dot(p_data) + p_bias, dtype='int'))

        # concatenation of data of interest and saving of it inside the final dataframe
        results = results + p_results
        participants_results.loc[df.shape[0]] = results

    # saving of the results regarding the form
    print(participants_results)
    participants_results.to_csv('../data/form-results/form_results.csv')

    # COMPUTATION OF ANALYSIS FOR ONLINE IMAGE EVALUATION

    codes = ['1']

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
        responses = pd.concat([responses, data])

    # saving of the csv file containing all the
    responses.to_csv('../data/ratings-results/ratings-results.csv')
