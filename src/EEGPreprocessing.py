import os
from pathlib import Path

import mne
import numpy as np
import pyxdf
import sklearn
import pandas as pd


class EEGPreprocessing:

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def __init__(self, path, removed_samples=0):

        self.data_path = path

        self.file_info = {}
        self.get_info_from_path()

        self.eeg_signal, self.eeg_instants, self.eeg_fs = None, None, None
        self.marker_ids, self.marker_instants = None, None
        self.channels_names = {}
        self.channels_types = {}

        self.load_xdf()

        self.eeg_signal = np.asmatrix(self.eeg_signal)

        self.eeg_signal = self.eeg_signal[removed_samples:self.eeg_signal.shape[0] - removed_samples]
        self.eeg_instants = self.eeg_instants[removed_samples:self.eeg_instants.shape[0] - removed_samples]

        self.eeg_signal = self.eeg_signal - np.mean(self.eeg_signal, axis=0)

        self.marker_instants -= self.eeg_instants[0]
        self.marker_instants = self.marker_instants[self.marker_instants >= 0]

        self.length = self.eeg_instants.shape[0]

        Path(self.file_info['folder'] + 'images/' + self.file_info['output_folder']).mkdir(parents=True, exist_ok=True)

        self.info, self.raw, self.epochs = None, None, None

    def get_info_from_path(self):
        """
        Getting main information from file path regarding subject, session, run ids and output folder according to the
        standard of LSLRecorder
        """

        # get name of the original file
        base = os.path.basename(self.data_path)
        file_name = os.path.splitext(base)[0]

        # main folder in which data is contained
        folder = os.path.dirname(self.data_path).split('data/')[0]

        # extraction of subject, session and run indexes
        subject = (file_name.split('subj_')[1]).split('_block')[0]

        # output folder according to the standard
        output_folder = '/sub-' + subject

        self.file_info = {'folder': folder, 'file_name': file_name, 'subject': subject, 'output_folder': output_folder}

    def load_xdf(self):
        """
        Load of .xdf file from the filepath given in input to the constructor. The function automatically divides the
        different streams in the file and extract their main information, according to who data is stored with Mentalab
        """

        stream_names = {'Markers': 'BrainVision RDA Markers', 'EEG': 'BrainVision RDA', 'Triggers': 'PsychoPy'}

        # data loading
        dat = pyxdf.load_xdf(self.data_path)[0]

        # data iteration to extract the main information
        for i in range(len(dat)):
            stream_name = dat[i]['info']['name'][0]

            # if stream_name == stream_names['Markers']:
            #     orn_signal = dat[i]['time_series']

            if stream_name == stream_names['EEG']:
                self.eeg_signal = dat[i]['time_series'][:, :32]
                self.eeg_signal = self.eeg_signal * 1e-6
                self.eeg_instants = dat[i]['time_stamps']
                self.eeg_fs = int(float(dat[i]['info']['nominal_srate'][0]))
                self.load_channels(dat[i]['info']['desc'][0]["channels"][0]['channel'])

            if stream_name == stream_names['Triggers']:
                self.marker_ids = dat[i]['time_series']
                self.marker_instants = dat[i]['time_stamps']

    def load_channels(self, dict_channels):
        """
        Upload channels name from a file, contained in data file, reporting the number (from 1 to 8 for Explore_CA46),
        a dash and the corresponding channel name
        """

        # x = data[0][0]['info']['desc'][0]["channels"][0]['channel']
        # si ottiene cosi' la list adi defaultdict con i canali

        for idx, info in enumerate(dict_channels):

            if info['label'][0].find('dir') != -1 or info['label'][0] == 'MkIdx':
                continue

            self.channels_names[idx] = info['label'][0]

            if self.channels_names[idx] == 'FP2':
                self.channels_names[idx] = 'Fp2'

            self.channels_types[idx] = 'eog' if info['label'][0].find('EOG') != -1 else 'eeg'

    def create_raw(self):
        """
        Creation of MNE raw instance from the data, setting the general information and the relative montage
        """

        self.info = mne.create_info(list(self.channels_names.values()), self.eeg_fs, list(self.channels_types.values()))
        self.raw = mne.io.RawArray(self.eeg_signal.T, self.info)

        # montage setting
        standard_montage = mne.channels.make_standard_montage('standard_1020')
        self.raw.set_montage(standard_montage)

    def filter_raw(self, l_freq=0.5, h_freq=60, n_freq=50, order=8):
        """
        Filter of MNE raw instance data with a band-pass filter and a notch filter
        :param l_freq: low frequency of band-pass filter
        :param h_freq: high frequency of band-pass filter
        :param n_freq: frequency of notch filter
        :param order: order of the filter
        """

        iir_params = dict(order=order, ftype='butter')
        iir_params = mne.filter.construct_iir_filter(iir_params=iir_params, f_pass=[l_freq, h_freq],
                                                     sfreq=self.eeg_fs, btype='bandpass', return_copy=False, verbose=40)

        self.raw.filter(l_freq=l_freq, h_freq=h_freq, filter_length=self.length,
                        l_trans_bandwidth=0.1, h_trans_bandwidth=0.1,
                        method='iir', iir_params=iir_params, verbose=40)

        if n_freq is not None:
            self.raw.notch_filter(freqs=n_freq, verbose=40)

    def visualize_raw(self, signal=True, psd=True, psd_topo=True):
        """
        Visualization of the plots that could be generated with MNE
        :param signal: boolean, if the signal plot should be generated
        :param psd: boolean, if the psd plot should be generated
        :param psd_topo: boolean, if the topographic psd plot should be generated
        """

        viz_scalings = dict(eeg=1e-5, eog=1e-4, ecg=1e-4, bio=1e-7, misc=1e-5)

        if signal:
            mne.viz.plot_raw(self.raw, scalings=viz_scalings, duration=self.length / self.eeg_fs)
        if psd:
            self.raw.plot_psd()
        if psd_topo:
            self.raw.plot_psd_topo()

    def set_reference(self, type='average'):
        """
        Resetting the reference in raw data
        :param type: type of referencing to be performed
        """

        mne.set_eeg_reference(self.raw, ref_channels=type, copy=False)

    def define_epochs_raw(self, topo_plot=True):
        """
        Function to extract events from the marker data, generate the correspondent epochs and determine annotation in
        the raw data according to the events
        :param topo_plot: boolean, if the topographic plot should be generated
        """

        # generation of the events according to the definition
        triggers = {'onsets': [], 'duration': [], 'description': []}
        for idx, marker_data in enumerate(self.marker_ids):
            triggers['onsets'].append(self.marker_instants[idx])
            triggers['duration'].append(int(0))
            triggers['description'].append(marker_data[0])

        annotations = mne.Annotations(triggers['onsets'], triggers['duration'], triggers['description'])
        self.raw.set_annotations(annotations)

        events, event_mapping = mne.events_from_annotations(self.raw)

        # generation of the epochs according to the events
        t_min = -0.2  # start of each epoch (200ms before the trigger)
        t_max = 0.8  # end of each epoch (500ms after the trigger)
        self.epochs = mne.Epochs(self.raw, events, tmin=t_min, tmax=t_max)

        if topo_plot:
            self.epochs.plot_psd_topomap()

        self.epochs.plot_image()

    def get_raw_ndarray(self):
        """
        Get the entire raw signal into a numpy array of dimension
        [number of channels, number of samples]
        """

        return self.raw.get_data()

    def get_epochs_ndarray(self):
        """
        Get the raw signal divided into epochs into a numpy array of dimension
        [number of epochs, number of channels, number of samples]
        """

        return self.epochs.get_data()

    def __get_eeg_fs__(self):
        return self.eeg_fs
