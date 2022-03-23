import os
import sys
import tkinter
from EEGPreprocessing import *


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

    path = 'C:/Users/Giulia Pezzutti/Documents/eeg-preprocessing/data/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task' \
           '-Default_run-003_eeg.xdf '
    # path = 'C:/Users/giuli/Documents/Università/Traineeship/eeg-preprocessing/data/sub-P001/ses-S001/eeg/' \
    #        'sub-P001_ses-S001_task-Default_run-003_eeg.xdf'

    eeg = EEGPreprocessing(path)
    eeg.create_raw()
    # eeg.visualize_raw()
    eeg.filter_raw()
    # eeg.visualize_raw()
    eeg.define_epochs_raw()
    eeg.visualize_raw()
