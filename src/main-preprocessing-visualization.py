import sys
import tkinter

import matplotlib.pyplot as plt
import numpy as np

from EEGAnalysis import *
from functions import *


def get_path():

    if 'tkinter' in sys.modules:
        from tkinter import filedialog
        path_selected = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select a File",
                                                   filetypes=(("xdf files", "*.xdf*"),))
    else:
        path_selected = input(
            "Not able to use tkinter to select the file. Insert here the file path and press ENTER:\n")

    return path_selected


if __name__ == '__main__':

    # path = get_path()
    path = '../data/eeg/subj_maba09_block1.xdf'

    eeg = EEGAnalysis(path)
    eeg.create_raw()
    # eeg.visualize_raw()
    eeg.filter_raw()
    eeg.set_reference()
    # eeg.visualize_raw()
    eeg.define_epochs_raw(visualize=False)

    epochs = eeg.get_epochs_dataframe()
    conditions = epochs.iloc[:]['condition'].tolist()
    conditions = list(set(conditions))
    # print(epochs)
    # exit(1)
    x_axis = epochs.iloc[:]['time'].tolist()
    x_axis = np.sort(np.array(list(set(x_axis))))
    # print(x_axis)

    means = {}

    # questa cosa deve essere fatta su tutti i file che vengono analizzati, così da creare un pattern tra partecipanti
    # - può essere inserito come funzione nella classe, essere ripetuto per ogni elemento della lista di dati e infine
    # creare la matrice finale di medie
    # HA SENSO SOLO INTRA-SOGGETTO O ANCHE INTER-SOGGETTO?

    epoch_signals = {}
    rois_numbers = {}

    for condition in conditions:
        condition_epochs = epochs.loc[epochs['condition'] == condition, :]
        # print(condition_epochs.shape)

        number_epochs = condition_epochs.iloc[:]['epoch'].tolist()
        number_epochs = list(set(number_epochs))
        # print(number_epochs)

        for epoch_number in number_epochs:

            current_epoch = condition_epochs.loc[condition_epochs['epoch'] == epoch_number, :].values[:, 3:-2]
            current_epoch = np.array(current_epoch).T
            # print(current_epoch)
            # print(current_epoch.shape)

            list_channels = condition_epochs.columns.values[3:-2]
            rois_numbers = define_rois(list_channels)
            # print(list_channels)
            # print(rois_numbers)

            for roi in rois_numbers.keys():
                current_roi_epoch = current_epoch[rois_numbers[roi]]
                # print(current_roi_epoch.shape)

                label = condition + '/' + roi

                if label in epoch_signals:
                    epoch_signals[label] = np.concatenate((epoch_signals[label], current_roi_epoch))

                else:
                    epoch_signals[label] = current_roi_epoch

    for key in epoch_signals:

        mean_current_epochs = np.mean(epoch_signals[key], axis=0)
        means[key] = mean_current_epochs

    # print(means)

    min_value = 100
    max_value = -100
    for label in epoch_signals.keys():
        m = np.min(means[label])
        min_value = min(m, min_value)

        ma = np.max(means[label])
        max_value = max(ma, max_value)

    # print(min_value, max_value)

    for condition in conditions:
        correct_labels = [s for s in epoch_signals.keys() if condition+'/' in s]
        correct_short_labels = [s.split('/')[1] for s in correct_labels]

        for idx, label in enumerate(correct_labels):
            plt.plot(x_axis, means[label], label=correct_short_labels[idx])

        plt.vlines(170, ymin=min_value, ymax=max_value)

        path = '../images/epochs/'+condition+'.png'
        plt.title(condition)
        plt.legend()
        plt.savefig(path)
        plt.show()

    for roi in rois_numbers.keys():
        correct_labels = [s for s in epoch_signals.keys() if '/'+roi in s]
        correct_short_labels = [s.split('/')[0] for s in correct_labels]

        for idx, label in enumerate(correct_labels):
            plt.plot(x_axis, means[label], label=correct_short_labels[idx])

        plt.vlines(170, ymin=min_value, ymax=max_value)

        path = '../images/epochs/'+roi+'.png'
        plt.title(roi)
        plt.legend()
        plt.savefig(path)
        plt.show()
