import numpy as np


def define_rois(channel_list):
    rois = dict(
        central=["Cz", "C3", "C4"],
        frontal=["Fz", "Fp1", "F3", "F7", "FC1", "FC2", "F4", "F8", "Fp2"],
        occipital_parietal=["O1", "Oz", "O2", "Pz", "P3", "P7", "P4", "P8"],
        temporal=["FC6", "FC5", "T7", "T8", "CP5", "CP6", "FT9", "FT10", "TP9", "TP10"],
    )

    rois_numbers = dict(
        central=np.array([np.where(channel_list == i)[0][0] for i in rois['central']]),
        frontal=np.array([np.where(channel_list == i)[0][0] for i in rois['frontal']]),
        occipital_parietal=np.array([np.where(channel_list == i)[0][0] for i in rois['occipital_parietal']]),
        temporal=np.array([np.where(channel_list == i)[0][0] for i in rois['temporal']]),
    )

    return rois_numbers
