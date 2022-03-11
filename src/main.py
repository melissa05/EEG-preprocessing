import os
import sys
import tkinter
import numpy as np
import pyxdf
from matplotlib import pyplot as plt
from scipy.signal import butter, sosfilt, iirnotch, lfilter


def get_path():
    name = 'tkinter'

    if name in sys.modules:
        from tkinter import filedialog
        path_selected = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select a File",
                                                   filetypes=(("xdf files", "*.xdf*"),))
    else:
        path_selected = input("Not able to use tkinter to select the file. Insert here the file path and press ENTER:\n")

    return path_selected


def get_subject_name(path):
    base = os.path.basename(path)
    file = os.path.splitext(base)[0]

    x = file.split('-', 1)[1]
    x = x.split('_ses', 1)[0]

    return x


def load_xdf(path):
    dat = pyxdf.load_xdf(path)

    orn_signal, eeg_signal, marker_signal, eeg_frequency = None, None, None, None

    for i in range(len(dat[0])):
        stream_name = dat[0][i]['info']['name']

        if stream_name == ['Explore_CA46_ORN']:
            orn_signal = dat[0][i]['time_series']
        if stream_name == ['Explore_CA46_ExG']:
            eeg_signal = dat[0][i]['time_series']
            eeg_frequency = int(dat[0][i]['info']['nominal_srate'][0])
        if stream_name == ['Explore_CA46_Marker']:
            marker_signal = dat[0][i]['time_series']

    return orn_signal, eeg_signal, marker_signal, eeg_frequency


def load_channel_names(path):
    with open(path) as f:
        lines = f.readlines()

    dict_channels_name = {}
    for line in lines:
        s = line.replace(' ', '')
        s = s.replace('\n', '')

        number, name = s.split('-')
        if name != 'GND':
            dict_channels_name[number] = name

    return dict_channels_name

def butter_bandpass_filter(data, lowcut, highcut, fs, order=8):
    low = lowcut / fs
    high = highcut / fs
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    y = sosfilt(sos, data)

    b, a = iirnotch(50, Q=150, fs=fs)
    y = lfilter(b, a, y)
    return y


if __name__ == '__main__':

    # path = get_path()
    path = 'C:/Users/giuli/Documents/Università/Traineeship/eeg-preprocessing/data/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf'
    filename = get_subject_name(path)
    [_, eeg, marker, eeg_freq] = load_xdf(path)

    path_channel = 'C:/Users/giuli/Documents/Università/Traineeship/eeg-preprocessing/data/Channels - Explore_CA46.txt'
    channels_name = load_channel_names(path_channel)

    eeg = np.asmatrix(eeg)
    eeg = eeg - np.mean(eeg, axis=0)
    eeg_filt = butter_bandpass_filter(eeg, lowcut=0.1, highcut=40, fs=eeg_freq, order=8)

    plt.plot(eeg_filt)
    plt.show()
