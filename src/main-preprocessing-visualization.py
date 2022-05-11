import sys
import tkinter

from EEGAnalysis import *
from functions import *


if __name__ == '__main__':

    paths = ['../data/eeg/subj_jomo20_block1.xdf', '../data/eeg/subj_mama13_block1.xdf',
             '../data/eeg/subj_moob25_block1.xdf', '../data/eeg/subj_vamo24_block1.xdf']

    # ['../data/eeg/subj_maba09_block1.xdf', '../data/eeg/subj_soze31_block1.xdf',
    # '../data/eeg/subj_nipe10_block1.xdf', '../data/eeg/subj_dino02_block1.xdf']
    # ['../data/eeg/subj_jomo20_block1.xdf', '../data/eeg/subj_mama13_block1.xdf',
    # '../data/eeg/subj_moob25_block1.xdf', '../data/eeg/subj_vamo24_block1.xdf']

    dict_info = {'streams': {'EEGMarkers': 'BrainVision RDA Markers', 'EEGData': 'BrainVision RDA', 'Triggers': 'PsychoPy'},
                 'filtering': {'low': 1, 'high': 60, 'notch': 50},
                 'spatial_filtering': 'average',
                 'samples_remove': 0,
                 't_min': -0.5,
                 't_max': 1,
                 'rois': dict(
                     central=["Cz", "C3", "C4"],
                     frontal=["Fz", "Fp1", "F3", "F7", "FC1", "FC2", "F4", "F8", "Fp2"],
                     occipital_parietal=["O1", "Oz", "O2", "Pz", "P3", "P7", "P4", "P8"],
                     temporal=["FC6", "FC5", "T7", "T8", "CP5", "CP6", "FT9", "FT10", "TP9", "TP10"],),
                 'bad_epoch_names': ['intro', 'pause', 'end']
                 }

    signals_means = {}

    for path in paths:

        plt.close('all')
        print('\n\nAnalyzing file', path)

        eeg = EEGAnalysis(path, dict_info)
        eeg.run(visualize_raw=False, save_images=False)

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
