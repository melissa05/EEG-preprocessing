import os
from pathlib import Path

import mne
import numpy as np
import pyxdf


class EEGPreprocessing:

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def __init__(self, path, removed_samples=500):
        self.data_path = path

        self.file_info = {}
        self.get_info_from_path()

        self.eeg_signal, self.eeg_instants, self.eeg_fs = None, None, None
        self.marker_ids, self.marker_instants = None, None

        self.load_xdf()

        self.channels_names = {}
        self.load_channels()

        self.eeg_signal = np.asmatrix(self.eeg_signal)
        self.eeg_signal = self.eeg_signal[500:self.eeg_signal.shape[0] - 500]
        self.eeg_instants = self.eeg_instants[500:self.eeg_instants.shape[0] - 500]

        self.eeg_signal = self.eeg_signal - np.mean(self.eeg_signal, axis=0)

        self.marker_instants -= self.eeg_instants[0]
        self.marker_instants = self.marker_instants[self.marker_instants >= 0]

        self.length = self.eeg_instants.shape[0]

        Path(self.file_info['general_data_folder'] + 'images/' + self.file_info['output_folder']).mkdir(parents=True,
                                                                                                        exist_ok=True)

        self.info, self.raw, self.epochs = None, None, None

    def get_info_from_path(self):
        base = os.path.dirname(self.data_path)
        folder = base.split('data/')[1]

        base = os.path.basename(self.data_path)
        file_name = os.path.splitext(base)[0]

        general_data_folder = os.path.dirname(self.data_path).split('data/')[0]

        subject = (file_name.split('sub-')[1]).split('_')[0]
        session = (file_name.split('ses-')[1]).split('_')[0]
        run = 'R' + (file_name.split('run-')[1]).split('_')[0]

        output_folder = '/sub-' + subject + '/ses-' + session + '/run-' + run + '/'

        self.file_info = {'folder': folder, 'file_name': file_name, 'subject': subject, 'session': session, 'run': run,
                          'general_data_folder': general_data_folder, 'output_folder': output_folder}

    def load_xdf(self):
        dat = pyxdf.load_xdf(self.data_path)

        for i in range(len(dat[0])):
            stream_name = dat[0][i]['info']['name']

            if stream_name == ['Explore_CA46_ORN']:
                orn_signal = dat[0][i]['time_series']
            if stream_name == ['Explore_CA46_ExG']:
                self.eeg_signal = dat[0][i]['time_series']
                self.eeg_instants = dat[0][i]['time_stamps']
                self.eeg_fs = int(dat[0][i]['info']['nominal_srate'][0])
            if stream_name == ['Explore_CA46_Marker']:
                self.marker_ids = dat[0][i]['time_series']
                self.marker_instants = dat[0][i]['time_stamps']

    def load_channels(self):
        base = os.path.dirname(self.data_path)
        path = base.split('data/')[0] + 'data/Channels - Explore_CA46.txt'

        with open(path) as f:
            lines = f.readlines()

        dict_channels_name = {}
        for line in lines:
            s = line.replace(' ', '')
            s = s.replace('\n', '')

            channel_number, channel_name = s.split('-')
            if channel_name != 'GND':
                dict_channels_name[channel_number] = channel_name

        self.channels_names = dict_channels_name

    def create_raw(self):
        self.info = mne.create_info(list(self.channels_names.values()), self.eeg_fs, ["eeg"] * 8)
        self.raw = mne.io.RawArray(self.eeg_signal.T, self.info)
        standard_montage = mne.channels.make_standard_montage('standard_1020')
        self.raw.set_montage(standard_montage)

    def filter_raw(self, l_freq=1, h_freq=60, n_freq=50):
        self.raw.filter(l_freq=l_freq, h_freq=h_freq, filter_length=self.length, l_trans_bandwidth=1, h_trans_bandwidth=1)
        self.raw.notch_filter(freqs=n_freq)

    def visualize_raw(self, signal=True, psd=True, psd_topo=True):

        if signal:
            mne.viz.plot_raw(self.raw, scalings=dict(eeg=1e2), duration=self.length / self.eeg_fs)
        if psd:
            self.raw.plot_psd()
        if psd_topo:
            self.raw.plot_psd_topo()

    def define_epochs_raw(self, topo_plot=True):
        events = []
        for idx, marker_data in enumerate(self.marker_ids[0]):
            events.append(np.array([self.marker_instants[idx] * self.eeg_fs, int(0), int(marker_data)]))
        events = np.array(events).astype(int)
        self.epochs = mne.Epochs(self.raw, events)

        if topo_plot:
            self.epochs.plot_psd_topomap()

        mapping = {1: 'auditory/left', 2: 'auditory/right', 3: 'visual/left', 4: 'visual/right', 5: 'smiley',
                   32: 'buttonpress'}
        annot_from_events = mne.annotations_from_events(events=events, event_desc=mapping, sfreq=self.raw.info['sfreq'],
                                                        orig_time=self.raw.info['meas_date'])
        self.raw.set_annotations(annot_from_events)
