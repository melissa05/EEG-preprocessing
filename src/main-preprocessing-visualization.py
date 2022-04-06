import os
import sys
import tkinter
from EEGAnalysis import *


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
    eeg.define_epochs_raw(visualize=True)
