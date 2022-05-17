import tkinter

from EEGAnalysis import *
from functions import *

if __name__ == '__main__':

    paths = ['../data/eeg/subj_mama13_block1.xdf', '../data/eeg/subj_jomo20_block1.xdf',
             '../data/eeg/subj_moob25_block1.xdf', '../data/eeg/subj_vamo24_block1.xdf']

    # ['../data/eeg/subj_maba09_block1.xdf', '../data/eeg/subj_soze31_block1.xdf',
    # '../data/eeg/subj_nipe10_block1.xdf', '../data/eeg/subj_dino02_block1.xdf']
    # ['../data/eeg/subj_jomo20_block1.xdf', '../data/eeg/subj_mama13_block1.xdf',
    # '../data/eeg/subj_moob25_block1.xdf', '../data/eeg/subj_vamo24_block1.xdf']

    dict_info = {'streams': {'EEGMarkers': 'BrainVision RDA Markers',
                             'EEGData': 'BrainVision RDA',
                             'Triggers': 'PsychoPy'},
                 'montage': 'standard_1020',
                 'filtering': {'low': 1,
                               'high': 60,
                               'notch': 50},
                 'spatial_filtering': 'average',
                 'samples_remove': 0,
                 't_min': -0.5,
                 't_max': 1,
                 'epochs_reject_criteria': dict(eeg=200e-6,  # 200 µV
                                                eog=1e-3),  # 1 mV
                 'rois': dict(
                     central=["Cz", "C3", "C4"],
                     frontal=["Fz", "Fp1", "F3", "F7", "FC1", "FC2", "F4", "F8", "Fp2"],
                     occipital_parietal=["O1", "Oz", "O2", "Pz", "P3", "P7", "P4", "P8"],
                     temporal=["FC6", "FC5", "T7", "T8", "CP5", "CP6", "FT9", "FT10", "TP9", "TP10"], ),
                 'bad_epoch_names': ['intro', 'pause', 'end'],
                 'erp': [0, 170, 300],
                 'erds': [1, 50]
                 }

    signals_means = {}

    for path in paths:

        plt.close('all')
        print('\n\nAnalyzing file', path)

        eeg = EEGAnalysis(path, dict_info)
        eeg.run(visualize_raw=False, save_images=False)

        if len(paths) > 1:
            evoked = eeg.evoked
            for key in evoked.keys():
                if key in signals_means:
                    signals_means[key] = mne.combine_evoked([signals_means[key], evoked[key]], weights='equal')
                else:
                    signals_means[key] = evoked[key]

        exit(1)

    conditions, rois = derive_conditions_rois(labels=signals_means.keys())
    plot_mean_epochs(signals_means, conditions, rois, dict_info['erp'])
