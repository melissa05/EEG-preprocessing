import os
import sys
import tkinter
from pathlib import Path

import mne
import numpy as np
import pyxdf
from matplotlib import *
from pylab import *


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

    print(base.split('data/')[0])
    general_data_folder = os.path.dirname(path).split('data/')[0]

    subject = (file_name.split('sub-')[1]).split('_')[0]
    session = (file_name.split('ses-')[1]).split('_')[0]
    run = 'R' + (file_name.split('run-')[1]).split('_')[0]

    output_folder = '/sub-' + subject + '/ses-' + session + '/run-' + run + '/'

    infos = {'folder': folder, 'file_name': file_name, 'subject': subject, 'session': session, 'run': run,
             'general_data_folder': general_data_folder, 'output_folder': output_folder}

    return infos


def load_xdf(path):
    dat = pyxdf.load_xdf(path)

    orn_signal, eeg_signal, marker_ids, eeg_frequency, eeg_instants, marker_instants = None, None, None, None, None, None

    for i in range(len(dat[0])):
        stream_name = dat[0][i]['info']['name']

        if stream_name == ['Explore_CA46_ORN']:
            orn_signal = dat[0][i]['time_series']
        if stream_name == ['Explore_CA46_ExG']:
            eeg_signal = dat[0][i]['time_series']
            eeg_instants = dat[0][i]['time_stamps']
            eeg_frequency = int(dat[0][i]['info']['nominal_srate'][0])
        if stream_name == ['Explore_CA46_Marker']:
            marker_ids = dat[0][i]['time_series']
            marker_instants = dat[0][i]['time_stamps']

    return orn_signal, eeg_signal, eeg_instants, marker_ids, marker_instants, eeg_frequency


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


if __name__ == '__main__':

    # path = get_path()
    path = 'C:/Users/Giulia Pezzutti/Documents/eeg-preprocessing/data/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-003_eeg.xdf'
    # path = 'C:/Users/giuli/Documents/Università/Traineeship/eeg-preprocessing/data/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-003_eeg.xdf'
    file_info = get_info_from_path(path)

    [_, eeg, eeg_inst, marker, marker_inst, eeg_freq] = load_xdf(path)

    # mio computer
    # path_channel = 'C:/Users/giuli/Documents/Università/Traineeship/eeg-preprocessing/data/Channels - Explore_CA46.txt'
    # computer lab
    path_channel = 'C:/Users/Giulia Pezzutti/Documents/eeg-preprocessing/data/Channels - Explore_CA46.txt'
    channels_name = load_channel_names(path_channel)

    see = False

    eeg = np.asmatrix(eeg)
    eeg = eeg[500:eeg.shape[0] - 500]
    eeg_inst = eeg_inst[500:eeg.shape[0] - 500]
    eeg = eeg - np.mean(eeg, axis=0)

    marker_inst -= eeg_inst[0]
    marker_inst = marker_inst[marker_inst>=0]

    Path('images/' + file_info['output_folder']).mkdir(parents=True, exist_ok=True)

    #     plt.savefig('images/' + file_info['output_folder'] + '/' + channel_name + '_signal.jpg')
    #
    #     # COMPUTATION OF POWER BANDS
    #
    #     # Get real amplitudes of FFT (only in postive frequencies)
    #     fft_vals = np.absolute(np.fft.rfft(data))
    #
    #     # Get frequencies for amplitudes in Hz
    #     fft_freq = np.fft.rfftfreq(len(data), 1.0 / eeg_freq)
    #
    #     # Define EEG bands
    #     eeg_bands = {'Delta': (0.5, 4),
    #                  'Theta': (4, 8),
    #                  'Alpha': (8, 12),
    #                  'Beta': (12, 30),
    #                  'Gamma': (30, 60)}
    #
    #     # Take the mean of the fft amplitude for each EEG band
    #     eeg_band_fft = {}
    #     for band in eeg_bands:
    #         freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
    #                            (fft_freq <= eeg_bands[band][1]))[0]
    #         eeg_band_fft[band] = np.mean(fft_vals[freq_ix])
    #
    #     df = pd.DataFrame(columns=['band', 'val'])
    #     df['band'] = eeg_bands.keys()
    #     df['val'] = [eeg_band_fft[band] for band in eeg_bands]
    #     ax = df.plot.bar(x='band', y='val', legend=False, color=['b', 'orange', 'g', 'r', 'purple'])
    #     ax.set_xlabel("EEG band")
    #     ax.set_ylabel("Mean band Amplitude")
    #     plt.title(channel_name)
    #     plt.tight_layout()
    #     plt.savefig('images/' + file_info['output_folder'] + '/' + channel_name + '_bands_power.jpg')
    #
    #     if see: plt.show()
    #     plt.close()

    print('\n\nImporting MNE data\n')
    info = mne.create_info(list(channels_name.values()), eeg_freq, ["eeg"] * 8)
    raw = mne.io.RawArray(eeg.T, info)

    print('\n\nSetting montage\n')
    standard_montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(standard_montage)

    mne.viz.plot_raw(raw, scalings=dict(eeg=1e2), duration=eeg.shape[0] / eeg_freq)
    raw.plot_psd()

    print('\n\nApplying bandpass and notch filters\n')
    raw.filter(l_freq=1, h_freq=60, filter_length=eeg.shape[0], l_trans_bandwidth=1, h_trans_bandwidth=1)
    raw.notch_filter(freqs=50)

    mne.viz.plot_raw(raw, scalings=dict(eeg=1e2), duration=eeg.shape[0] / eeg_freq)
    raw.plot_psd(fmax=80)

    raw.plot_psd_topo()

    print('\n\nGetting epochs')
    events = []
    for idx, marker_data in enumerate(marker[0]):
        events.append(np.array([marker_inst[idx]*eeg_freq, int(0), int(marker_data)]))
    events = np.array(events).astype(int)
    epochs = mne.Epochs(raw, events)

    mapping = {1: 'auditory/left', 2: 'auditory/right', 3: 'visual/left', 4: 'visual/right', 5: 'smiley', 32: 'buttonpress'}
    annot_from_events = mne.annotations_from_events(events=events, event_desc=mapping, sfreq=raw.info['sfreq'],
                                                    orig_time=raw.info['meas_date'])
    raw.set_annotations(annot_from_events)

    mne.viz.plot_raw(raw, scalings=dict(eeg=1e2), duration=eeg.shape[0] / eeg_freq)
