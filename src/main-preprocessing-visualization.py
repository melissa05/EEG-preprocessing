import sys
import tkinter
from importlib import reload

import matplotlib

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

    # reload(matplotlib)
    # matplotlib.use('Agg')

    # path = get_path()
    paths = ['../data/eeg/subj_moob25_block1.xdf']

    # ['../data/eeg/subj_maba09_block1.xdf', '../data/eeg/subj_soze31_block1.xdf',
    # '../data/eeg/subj_nipe10_block1.xdf', '../data/eeg/subj_dino02_block1.xdf']
    # ['../data/eeg/subj_jomo20_block1.xdf', '../data/eeg/subj_mama13_block1.xdf',
    # '../data/eeg/subj_moob25_block1.xdf', '../data/eeg/subj_vamo24_block1.xdf']

    signals_means = {}

    for path in paths:

        plt.close('all')
        print('\n\nAnalyzing file', path)

        eeg = EEGAnalysis(path)
        eeg.create_raw()
        # eeg.visualize_raw()

        eeg.set_reference()
        eeg.filter_raw()
        # eeg.ica_remove_eog()
        # eeg.visualize_raw()

        eeg.define_annotations()
        eeg.define_epochs_raw(visualize=False)
        eeg.define_ers_erd()
        exit(1)
        eeg.define_evoked()

        means = eeg.plot_mean_epochs()
        for key in means.keys():
            if key in signals_means:
                signals_means[key] = np.concatenate((signals_means[key], np.array([means[key]])), axis=0)
            else:
                signals_means[key] = np.array([means[key]])

        plt.close('all')

    for key in signals_means.keys():
        signals_means[key] = np.mean(signals_means[key], axis=0)

    conditions, rois = derive_conditions_rois(labels=signals_means.keys())
    plot_mean_epochs(signals_means, conditions, rois)
