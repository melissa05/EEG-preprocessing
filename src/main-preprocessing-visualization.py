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
    paths = ['../data/eeg/subj_maba09_block1.xdf']
             # ['../data/eeg/subj_maba09_block1.xdf', '../data/eeg/subj_soze31_block1.xdf', '../data/eeg/subj_nipe10_block1.xdf',
             # '../data/eeg/subj_dino02_block1.xdf']

    signals_means = {}

    for path in paths:
        eeg = EEGAnalysis(path)
        eeg.create_raw()
        # eeg.visualize_raw()
        eeg.filter_raw()
        eeg.set_reference()
        eeg.define_epochs_raw(visualize=True)
        eeg.visualize_raw()

        means = eeg.plot_mean_epochs()
        for key in means.keys():
            if key in signals_means:
                signals_means[key] = np.concatenate((signals_means[key], np.array([means[key]])), axis=0)
            else:
                signals_means[key] = np.array([means[key]])

    for key in signals_means.keys():
        signals_means[key] = np.mean(signals_means[key], axis=0)

    conditions, rois = derive_conditions_rois(labels=signals_means.keys())
    plot_mean_epochs(signals_means, conditions, rois)
