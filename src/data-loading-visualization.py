import os
import sys
import tkinter
from pathlib import Path

import mne
import pyxdf
from scipy.fft import fft
from numpy import *
from scipy.signal import *
from numpy import *
from matplotlib import *
from scipy import *
from pylab import *
import pandas as pd


def get_path():
    if 'tkinter' in sys.modules:
        from tkinter import filedialog
        path_selected = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select a File",
                                                   filetypes=(("xdf files", "*.xdf*"),))
    else:
        path_selected = input(
            "Not able to use tkinter to select the file. Insert here the file path and press ENTER:\n")

    return path_selected


def get_info_from_path(path):
    base = os.path.dirname(path)
    folder = base.split('data/')[1]

    base = os.path.basename(path)
    file_name = os.path.splitext(base)[0]

    subject = (file_name.split('sub-')[1]).split('_')[0]
    session = (file_name.split('ses-')[1]).split('_')[0]
    run = 'R' + (file_name.split('run-')[1]).split('_')[0]

    output_folder = '/sub-' + subject + '/ses-' + session + '/run-' + run + '/'

    infos = {'folder': folder, 'file_name': file_name, 'subject': subject, 'session': session, 'run': run,
             'output_folder': output_folder}

    return infos


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

        channel_number, channel_name = s.split('-')
        if channel_name != 'GND':
            dict_channels_name[channel_number] = channel_name

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
    path = 'C:/Users/Giulia Pezzutti/Documents/eeg-preprocessing/data/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf'
    # path = 'C:/Users/giuli/Documents/Università/Traineeship/eeg-preprocessing/data/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf'
    file_info = get_info_from_path(path)

    [_, eeg, marker, eeg_freq] = load_xdf(path)

    # mio computer
    # path_channel = 'C:/Users/giuli/Documents/Università/Traineeship/eeg-preprocessing/data/Channels - Explore_CA46.txt'
    # computer lab
    path_channel = 'C:/Users/Giulia Pezzutti/Documents/eeg-preprocessing/data/Channels - Explore_CA46.txt'
    channels_name = load_channel_names(path_channel)

    see = False

    eeg = np.asmatrix(eeg)
    eeg = eeg[500:eeg.shape[0] - 500]
    eeg = eeg - np.mean(eeg, axis=0)

    for (number, channel_name) in channels_name.items():

        data = butter_bandpass_filter(eeg[:, int(number) - 1], lowcut=0.1, highcut=60, fs=eeg_freq, order=8)
        eeg[:, int(number) - 1] = data

        Path('images/' + file_info['output_folder']).mkdir(parents=True, exist_ok=True)

        plt.plot(data)
        plt.title(channel_name)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (uV)')
        plt.tight_layout()
        plt.savefig('images/' + file_info['output_folder'] + '/' + channel_name + '_signal.jpg')

        if see: plt.show()
        plt.close()

        # COMPUTATION OF POWER BANDS

        # Get real amplitudes of FFT (only in postive frequencies)
        fft_vals = np.absolute(np.fft.rfft(data))

        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(data), 1.0 / eeg_freq)

        # Define EEG bands
        eeg_bands = {'Delta': (0.5, 4),
                     'Theta': (4, 8),
                     'Alpha': (8, 12),
                     'Beta': (12, 30),
                     'Gamma': (30, 60)}

        # Take the mean of the fft amplitude for each EEG band
        eeg_band_fft = {}
        for band in eeg_bands:
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                               (fft_freq <= eeg_bands[band][1]))[0]
            eeg_band_fft[band] = np.mean(fft_vals[freq_ix])  # qui mettere power se si vuole vedere PSD e non magnitude

        df = pd.DataFrame(columns=['band', 'val'])
        df['band'] = eeg_bands.keys()
        df['val'] = [eeg_band_fft[band] for band in eeg_bands]
        ax = df.plot.bar(x='band', y='val', legend=False, color=['b', 'orange', 'g', 'r', 'purple'])
        ax.set_xlabel("EEG band")
        ax.set_ylabel("Mean band Amplitude")
        plt.title(channel_name)
        plt.tight_layout()
        plt.savefig('images/' + file_info['output_folder'] + '/' + channel_name + '_bands_power.jpg')

        if see: plt.show()
        plt.close()

    info = mne.create_info(list(channels_name.values()), eeg_freq, ["eeg"] * 8)
    raw = mne.io.RawArray(eeg.T, info)
    mne.viz.plot_raw(raw, scalings=dict(eeg=10000e-6), duration=eeg.shape[0] / eeg_freq)
    mne.viz.plot_raw_psd(raw)

    raw.filter(l_freq=0.4, h_freq=60)
    raw.notch_filter(freqs=50.0)
    mne.viz.plot_raw_psd(raw)
