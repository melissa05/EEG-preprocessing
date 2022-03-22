import csv
import glob
import os
import pickle
import re

import numpy as np
import pandas as pd

if __name__ == '__main__':

    # filepath = '../data/form-results/Form results - Risposte del modulo 1.csv'
    # df = pd.read_csv(filepath)
    # df = df.drop('Informazioni cronologiche', axis=1)
    #
    # codes = df.iloc[:]['INSERT YOUR PARTECIPANT CODE'].tolist()
    #
    # participants_results = {}
    #
    # for code in codes:
    #     data = df.loc[df['INSERT YOUR PARTECIPANT CODE'] == code]
    #     results = {'gender': data.iloc[0]['Insert your gender'], 'age': int(data.iloc[0]['Insert your age']),
    #                'nationality': data.iloc[0]['Insert your home country'].lower()}
    #
    #     cols = list(df.columns)
    #     p = []
    #
    #     for col in cols:
    #         res = re.findall("^I[.]{3} [\[]{1}.+[\]]{1}", str(col))
    #         if res:
    #             p.append(res[0])
    #
    #     p_data = list(data.iloc[0][p])
    #     p_data = list(int(vote[0]) for vote in p_data)
    #
    #     p_types = ['E', 'A', 'C', 'N', 'O']
    #
    #     p_matrix = np.zeros([len(p_types), len(p_data)])
    #
    #     for i in range(p_matrix.shape[0]):
    #         if (i % 2) == 0:
    #             for j in range(5):
    #                 c = i + 10 * j
    #                 p_matrix[i, c] = int(1)
    #                 c = i + 5 + 10 * j
    #                 p_matrix[i, c] = int(-1)
    #         else:
    #             for j in range(5):
    #                 c = i + 10 * j
    #                 p_matrix[i, c] = int(-1)
    #                 c = i + 5 + 10 * j
    #                 p_matrix[i, c] = int(+1)
    #
    #     p_bias = [20, 14, 14, 38, 8]
    #     p_results = np.array(p_matrix.dot(p_data) + p_bias, dtype='int')
    #
    #     for idx, p_type in enumerate(p_types):
    #         results[p_type] = p_results[idx]
    #
    #     participants_results[code] = results
    #
    # a_file = open("../data/form-results/form_data.pkl", "wb")
    # pickle.dump(participants_results, a_file)
    # a_file.close()

    codes = ['1']
    rating_path = '../data/ratings-results/'
    available_ratings_paths = glob.glob(rating_path+'/*.csv')

    for code in codes:

        correct = None
        for filepath in available_ratings_paths:
            if filepath.find(code):
                correct = filepath

        if correct is None:
            continue

        df = pd.read_csv(correct)
        print(df)

    # a_file = open("../data/form-results/form_data.pkl", "rb")
    # participants_results = pickle.load(a_file)
    # print(participants_results)
