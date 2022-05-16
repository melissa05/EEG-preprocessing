import os
from pathlib import Path

import mne
import numpy as np
import pyxdf
from matplotlib import pyplot as plt


class EEGAnalysis:

    def __init__(self, path, dict_info):

        self.data_path = path

        self.file_info = {}
        self.get_info_from_path()

        self.eeg_signal, self.eeg_instants, self.eeg_fs, self.length = None, None, None, None
        self.marker_ids, self.marker_instants = None, None
        self.channels_names, self.channels_types, self.evoked_rois = {}, {}, {}
        self.info, self.raw = None, None
        self.events, self.event_mapping, self.epochs, self.annotations = None, None, None, None
        self.evoked = {}

        self.input_info = dict_info

        self.t_min = self.input_info['t_min']  # start of each epoch (200ms before the trigger)
        self.t_max = self.input_info['t_max']  # end of each epoch (800ms after the trigger)
        self.load_xdf()

    def get_info_from_path(self):
        """
        Getting main information from file path regarding subject, folder and output folder according to
        LSLRecorder standard
        """

        # get name of the original file
        base = os.path.basename(self.data_path)
        file_name = os.path.splitext(base)[0]

        # main folder in which data is contained
        base = os.path.abspath(self.data_path)
        folder = os.path.dirname(base).split('data/')[0]
        folder = folder.replace('\\', '/')

        # extraction of subject, session and run indexes
        subject = (file_name.split('subj_')[1]).split('_block')[0]

        # output folder according to the standard
        output_folder = folder.rsplit('/', 2)[0] + '/images/sub-' + subject
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self.file_info = {'folder': folder, 'file_name': file_name, 'subject': subject, 'output_folder': output_folder}

    def load_xdf(self):
        """
        Load of .xdf file from the filepath given in input to the constructor. The function automatically divides the
        different streams in the file and extract their main information
        """

        stream_names = self.input_info['streams']

        # data loading
        dat = pyxdf.load_xdf(self.data_path)[0]

        orn_signal, orn_instants = [], []
        effective_sample_frequency = None

        # data iteration to extract the main information
        for i in range(len(dat)):
            stream_name = dat[i]['info']['name'][0]

            if stream_name == stream_names['EEGMarkers']:  # gets 'BrainVision RDA Markers' stream
                orn_signal = dat[i]['time_series']
                orn_instants = dat[i]['time_stamps']

            if stream_name == stream_names['EEGData']:
                self.eeg_signal = dat[i]['time_series'][:, :32]
                self.eeg_signal = self.eeg_signal * 1e-6
                self.eeg_instants = dat[i]['time_stamps']
                self.eeg_fs = int(float(dat[i]['info']['nominal_srate'][0]))
                self.load_channels(dat[i]['info']['desc'][0]['channels'][0]['channel'])
                effective_sample_frequency = float(dat[i]['info']['effective_srate'])

            if stream_name == stream_names['Triggers']:
                self.marker_ids = dat[i]['time_series']
                self.marker_instants = dat[i]['time_stamps']

        # cast to arrays
        self.eeg_instants = np.array(self.eeg_instants)
        self.eeg_signal = np.asmatrix(self.eeg_signal)

        # check lost-samples problem
        if len(orn_signal) != 0:
            self.fix_lost_samples(orn_signal, orn_instants, effective_sample_frequency)

        self.length = self.eeg_instants.shape[0]

        # remove samples at the beginning and at the end
        samples_to_be_removed = self.input_info['samples_remove']
        if samples_to_be_removed > 0:
            self.eeg_signal = self.eeg_signal[samples_to_be_removed:self.length - samples_to_be_removed]
            self.eeg_instants = self.eeg_instants[samples_to_be_removed:self.length - samples_to_be_removed]

        # reference all the markers instant to the eeg instants (since some samples at the beginning of the
        # recording have been removed)
        self.marker_instants -= self.eeg_instants[0]
        self.marker_instants = self.marker_instants[self.marker_instants >= 0]

        # remove signal mean
        self.eeg_signal = self.eeg_signal - np.mean(self.eeg_signal, axis=0)

    def load_channels(self, dict_channels):
        """
        Upload channels name from a xdf file
        """

        # x = data[0][0]['info']['desc'][0]["channels"][0]['channel']
        # to obtain the default-dict list of the channels from the original file (data, not dat!!)

        # cycle over the info of the channels
        for idx, info in enumerate(dict_channels):

            if info['label'][0].find('dir') != -1 or info['label'][0] == 'MkIdx':
                continue

            # get channel name
            self.channels_names[idx] = info['label'][0]

            if self.channels_names[idx] == 'FP2':
                self.channels_names[idx] = 'Fp2'

            # get channel type
            self.channels_types[idx] = 'eog' if info['label'][0].find('EOG') != -1 else 'eeg'

    def fix_lost_samples(self, orn_signal, orn_instants, effective_sample_frequency):

        print('BrainVision RDA Markers: ', orn_signal)
        print('BrainVision RDA Markers instants: ', orn_instants)
        print('\nNominal srate: ', self.eeg_fs)
        print('Effective srate: ', effective_sample_frequency)

        print('Total number of samples: ', len(self.eeg_instants))
        final_count = len(self.eeg_signal)
        for lost in orn_signal:
            final_count += int(lost[0].split(': ')[1])
        print('Number of samples with lost samples integration: ', final_count)

        total_time = len(self.eeg_instants) / effective_sample_frequency
        real_number_samples = total_time * self.eeg_fs
        print('Number of samples with real sampling frequency: ', real_number_samples)

        # print(self.eeg_instants)

        differences = np.diff(self.eeg_instants)
        differences = (differences - (1 / self.eeg_fs)) * self.eeg_fs
        # differences = np.round(differences, 4)
        print('Unique differences in instants: ', np.unique(differences))
        print('Sum of diff ', np.sum(differences))
        # plt.plot(differences)
        # plt.ylim([1, 2])
        # plt.show()

        new_marker_signal = self.marker_instants

        for idx, lost_instant in enumerate(orn_instants):
            x = np.where(self.marker_instants < lost_instant)[0][-1]

            missing_samples = int(orn_signal[idx][0].split(': ')[1])
            additional_time = missing_samples / self.eeg_fs

            new_marker_signal[(x + 1):] = np.array(new_marker_signal[(x + 1):]) + additional_time

    def create_raw(self):
        """
        Creation of MNE raw instance from the data, setting the general information and the relative montage
        """

        print('\n')
        self.info = mne.create_info(list(self.channels_names.values()), self.eeg_fs, list(self.channels_types.values()))
        self.raw = mne.io.RawArray(self.eeg_signal.T, self.info, first_samp=self.eeg_instants[0])

        # montage setting
        standard_montage = mne.channels.make_standard_montage(self.input_info['montage'])
        self.raw.set_montage(standard_montage)

    def visualize_raw(self, signal=True, psd=True, psd_topo=True):
        """
        Visualization of the plots that could be generated with MNE according to a scaling property
        :param signal: boolean, if the signal plot should be generated
        :param psd: boolean, if the psd plot should be generated
        :param psd_topo: boolean, if the topographic psd plot should be generated
        """

        viz_scaling = dict(eeg=1e-4, eog=1e-4, ecg=1e-4, bio=1e-7, misc=1e-5)

        if signal:
            mne.viz.plot_raw(self.raw, scalings=viz_scaling, duration=50)
        if psd:
            self.raw.plot_psd()
            plt.close()
        if psd_topo:
            self.raw.plot_psd_topo()
            plt.close()

    def set_reference(self):
        """
        Resetting the reference in raw data
        """

        mne.set_eeg_reference(self.raw, ref_channels=self.input_info['spatial_filtering'], copy=False)

    def filter_raw(self):
        """
        Filter of MNE raw instance data with a band-pass filter and a notch filter
        """

        # extract the frequencies for the filtering
        l_freq = self.input_info['filtering']['low']
        h_freq = self.input_info['filtering']['high']
        n_freq = self.input_info['filtering']['notch']

        # apply band-pass filter
        if not (l_freq is None and h_freq is None):
            iir_params = dict(order=8, ftype='butter')
            iir_params = mne.filter.construct_iir_filter(iir_params=iir_params, f_pass=[l_freq, h_freq],
                                                         sfreq=self.eeg_fs, btype='bandpass', return_copy=False, verbose=40)

            self.raw.filter(l_freq=l_freq, h_freq=h_freq, filter_length=self.length,
                            l_trans_bandwidth=0.1, h_trans_bandwidth=0.1,
                            method='iir', iir_params=iir_params, verbose=40)

        # apply notch filter
        if n_freq is not None:
            self.raw.notch_filter(freqs=n_freq, verbose=40)

    def ica_remove_eog(self):

        n_components = list(self.channels_types.values()).count('eeg')

        eeg_raw = self.raw.copy()
        eeg_raw = eeg_raw.pick_types(eeg=True)

        ica = mne.preprocessing.ICA(n_components=0.99999, method='fastica', random_state=97)

        ica.fit(eeg_raw)
        ica.plot_sources(eeg_raw)
        ica.plot_components()
        # ica.plot_properties(eeg_raw)

        # find which ICs match the EOG pattern
        eog_indices, eog_scores = ica.find_bads_eog(self.raw, h_freq=5, threshold=3)
        print(eog_indices)

        ica.exclude = ica.exclude + eog_indices

        # # barplot of ICA component "EOG match" scores
        # ica.plot_scores(eog_scores)
        # # plot diagnostics
        # # ica.plot_properties(self.raw, picks=eog_indices)
        # # plot ICs applied to raw data, with EOG matches highlighted
        # ica.plot_sources(eeg_raw)

        reconst_raw = self.raw.copy()
        ica.apply(reconst_raw)

        # viz_scaling = dict(eeg=1e-4, eog=1e-4, ecg=1e-4, bio=1e-7, misc=1e-5)
        # reconst_raw.plot(scalings=viz_scaling)
        # reconst_raw.plot_psd()
        self.raw = reconst_raw

    def define_annotations(self):

        # generation of the events according to the definition
        triggers = {'onsets': [], 'duration': [], 'description': []}

        # read every trigger in the stream
        for idx, marker_data in enumerate(self.marker_ids):

            # annotations to be rejected
            if marker_data[0] in self.input_info['bad_epoch_names']:
                continue

            # extract triggers information
            triggers['onsets'].append(self.marker_instants[idx])
            triggers['duration'].append(int(0))

            condition = marker_data[0].split('/')[-1]
            if condition == 'edges': condition = 'canny'
            triggers['description'].append(condition)

        # define MNE annotations
        self.annotations = mne.Annotations(triggers['onsets'], triggers['duration'], triggers['description'])

    def define_epochs_raw(self, visualize=True, rois=True):
        """
        Function to extract events from the marker data, generate the correspondent epochs and determine annotation in
        the raw data according to the events
        :param visualize:
        :param rois: boolean variable to select if visualize results in terms to rois or not
        """

        # set the annotations on the current raw and extract the correspondent events
        self.raw.set_annotations(self.annotations)
        self.events, self.event_mapping = mne.events_from_annotations(self.raw)

        # Automatic rejection criteria for the epochs
        reject_criteria = self.input_info['epochs_reject_criteria']

        # generation of the epochs according to the events
        self.epochs = mne.Epochs(self.raw, self.events, event_id=self.event_mapping, preload=True,
                                 baseline=(self.t_min, 0), reject=reject_criteria, tmin=self.t_min, tmax=self.t_max)

        if visualize:
            self.visualize_epochs(signal=False, topo_plot=False, conditional_epoch=True, rois=rois)

    def visualize_epochs(self, signal=True, topo_plot=True, conditional_epoch=True, rois=True):
        """
        :param signal: boolean, if visualize the whole signal with triggers or not
        :param topo_plot: boolean, if the topographic plot should be generated
        :param conditional_epoch: boolean, if visualize the epochs extracted from the events
        :param rois: boolean (only if conditional_epoch=True), if visualize the epochs according to the rois or not
        """

        self.visualize_raw(signal=signal, psd=False, psd_topo=False)

        rois_numbers = self.define_rois()
        rois_names = list(rois_numbers.keys())

        if topo_plot:
            self.epochs.plot_psd_topomap()

        if conditional_epoch:
            if rois:

                for condition in self.event_mapping.keys():

                    # generate the epochs plots according to the roi and save them
                    images = self.epochs[condition].plot_image(combine='mean', group_by=rois_numbers, show=False)
                    for idx, img in enumerate(images):
                        img.savefig(self.file_info['output_folder'] + '/' + condition + '_' + rois_names[idx] + '.png')
                        plt.close(img)

            else:

                for condition in self.event_mapping.keys():

                    # generate the epochs plots for each channel and save them
                    images = self.epochs[condition].plot_image(show=False)
                    for idx, img in enumerate(images):
                        img.savefig(self.file_info['output_folder'] + '/' + condition + '_' + self.channels_names[idx] + '.png')
                        plt.close(img)

        plt.close('all')

        for condition in self.event_mapping.keys():
            img = self.epochs[condition].plot_psd_topomap(vlim=(7, 45), show=False)
            img.savefig(self.file_info['output_folder'] + '/' + condition + '_topography.png')
            plt.close(img)

        plt.close('all')

    def define_rois(self):

        # rois dict given in input
        rois = self.input_info['rois']

        # channel numbers associated to each roi
        rois_numbers = dict(
            central=np.array([self.raw.ch_names.index(i) for i in rois['central']]),
            frontal=np.array([self.raw.ch_names.index(i) for i in rois['frontal']]),
            occipital_parietal=np.array([self.raw.ch_names.index(i) for i in rois['occipital_parietal']]),
            temporal=np.array([self.raw.ch_names.index(i) for i in rois['temporal']]),
        )

        return rois_numbers

    def define_ers_erd(self):

        # function to get the square of the signal (approximation of the power)
        def square_signal(array_data):
            return np.square(array_data)

        if len(self.input_info['erds']) != 2:
            print('No erds frequencies provided!')
            return

        # frequencies for the erds maps
        f_min = int(self.input_info['erds'][0])
        f_max = int(self.input_info['erds'][1])

        f_start = list(range(f_min, f_max, 1))
        f_end = list(range(f_min+2, f_max+2, 1))
        f_plot = list(range(f_min+1, f_max+2, 1))

        rois_numbers = self.define_rois()

        x_axis = None
        # x_axis = list(range(int(self.t_min * 1000 - 2), int(self.t_max * 1000 + 1), 2))

        signals = self.raw.copy()
        filter_bank = []

        # for each frequency band in the erds
        for idx, start in enumerate(f_start):

            # filter the signal
            filtered_signals = signals.filter(start, f_end[idx], l_trans_bandwidth=1, h_trans_bandwidth=1)

            # divide into epochs
            filtered_signals.set_annotations(self.annotations)
            events, event_mapping = mne.events_from_annotations(self.raw)
            epochs = mne.Epochs(filtered_signals, events, self.event_mapping, preload=True, baseline=(self.t_min, 0),
                                reject=self.input_info['epochs_reject_criteria'], tmin=self.t_min, tmax=self.t_max)

            # save the obtained epochs
            filter_bank.append(epochs)

        # generation of the erds images, one for each condition and for each roi

        # for each type of epoch
        for condition in self.event_mapping.keys():

            # for each roi
            for roi, roi_numbers in rois_numbers.items():

                freq_erds_results = []

                # for each frequency band (so for each set of epochs previously saved)
                for freq_band_epochs in filter_bank:

                    # take frequency band of interest
                    condition_epochs = freq_band_epochs[condition].copy()

                    if x_axis is None:
                        x_axis = condition_epochs.times
                        step = np.abs(x_axis[1]-x_axis[0])
                        x_axis = np.append(x_axis, x_axis[-1]+step)
                        print(x_axis)

                    # take channels of interest
                    condition_epochs = condition_epochs.pick(roi_numbers)

                    # square each epoch
                    condition_epochs = condition_epochs.apply_function(square_signal)

                    # extract data
                    epochs_data = condition_epochs.get_data()

                    # derive reference for each epoch and channel
                    reference = epochs_data[:, :, :int(self.t_min*self.eeg_fs)]
                    reference_power = np.mean(reference, axis=2)

                    # for each value inside the data, compute the ERDS value -> trial-individual references
                    erds_epochs = []
                    for idx_epoch, epoch in enumerate(epochs_data):
                        for idx_ch, channel in enumerate(epoch):
                            erds = np.zeros(epochs_data.shape[2])
                            for sample, power in enumerate(channel):
                                current_reference_power = reference_power[idx_epoch, idx_ch]
                                erds[sample] = (power - current_reference_power)/current_reference_power * 100
                            erds_epochs.append(erds)

                    # mean for each epoch and channel and save
                    mean_erds = np.mean(np.array(erds_epochs), axis=0)
                    freq_erds_results.append(mean_erds)

                # visualization
                freq_erds_results = np.array(freq_erds_results)
                z_min, z_max = -np.abs(freq_erds_results).max(), np.abs(freq_erds_results).max()
                fig, ax = plt.subplots()
                p = ax.pcolor(x_axis, f_plot, freq_erds_results, cmap='RdBu', snap=True, vmin=z_min, vmax=z_max)
                ax.set_xlabel('Time (\u03bcs)')
                ax.set_ylabel('Frequency (Hz)')
                ax.set_title(condition+' '+roi)
                ax.axvline(0, color='k')
                fig.colorbar(p, ax=ax)
                fig.savefig(self.file_info['output_folder'] + '/' + condition + '_' + roi + '_erds.png')
                plt.close()

    def define_evoked(self):

        rois_numbers = self.define_rois()

        evoked = self.epochs.average(picks=['eeg'], by_event_type=True)

        for evok in evoked:
            roi_evoked = mne.channels.combine_channels(evok, rois_numbers, method='mean')
            self.evoked_rois[evok.comment] = roi_evoked

    def plot_evoked(self):

        rois_numbers = self.define_rois()

        for condition in self.event_mapping.keys():
            condition_epochs = self.epochs[condition]

            for roi in sorted(rois_numbers.keys()):
                condition_roi_epoch = condition_epochs.copy()
                condition_roi_epoch = condition_roi_epoch.pick(rois_numbers[roi])

                condition_roi_epoch = condition_roi_epoch.average()
                condition_roi_epoch = mne.channels.combine_channels(condition_roi_epoch,
                                                                    groups={'mean': list(range(len(rois_numbers[roi])))})

                label = condition + '/' + roi
                self.evoked[label] = condition_roi_epoch

        # get minimum and maximum value of the mean signals
        min_value, max_value = np.inf, -np.inf
        for label in self.evoked.keys():
            data = self.evoked[label].get_data()[0]
            min_value = min(np.min(data), min_value)
            max_value = max(np.max(data), max_value)

        # Plot everything

        Path(self.file_info['output_folder'] + '/epochs/').mkdir(parents=True, exist_ok=True)

        fig, axs = plt.subplots(3, 2, figsize=(25.6, 19.2))
        path = self.file_info['output_folder'] + '/epochs/conditions.png'

        for i, ax in enumerate(fig.axes):

            condition = list(self.event_mapping.keys())[i]

            correct_labels = [s for s in self.evoked.keys() if condition + '/' in s]
            correct_short_labels = [s.split('/')[1] for s in correct_labels]

            for idx, label in enumerate(correct_labels):
                ax.plot(self.evoked[label].times*1000, self.evoked[label].get_data()[0], label=correct_short_labels[idx])

            for erp in self.input_info['erp']:
                ax.vlines(erp, ymin=min_value, ymax=max_value, linestyles='dashed')

            ax.set_title(condition)

        plt.legend(bbox_to_anchor=(1.2, 2))
        plt.savefig(path)
        plt.close()

        fig, axs = plt.subplots(2, 2, figsize=(25.6, 19.2))
        path = self.file_info['output_folder'] + '/epochs/rois.png'

        for i, ax in enumerate(fig.axes):
            roi = list(rois_numbers.keys())[i]

            correct_labels = [s for s in self.evoked.keys() if '/' + roi in s]
            correct_short_labels = [s.split('/')[0] for s in correct_labels]

            for idx, label in enumerate(correct_labels):
                ax.plot(self.evoked[label].times*1000, self.evoked[label].get_data()[0], label=correct_short_labels[idx])

            for erp in self.input_info['erp']:
                ax.vlines(erp, ymin=min_value, ymax=max_value, linestyles='dashed')

            ax.set_title(roi)

        # TODO remove anchor
        plt.legend(bbox_to_anchor=(1.2, 1.1))
        plt.savefig(path)
        plt.close()

    def run(self, visualize_raw=False, save_images=True):

        self.create_raw()

        if visualize_raw:
            self.visualize_raw()

        if self.input_info['spatial_filtering'] is not None:
            self.set_reference()

        if self.input_info['filtering'] is not None:
            self.filter_raw()

        if visualize_raw:
            self.visualize_raw()

        # self.ica_remove_eog()

        self.define_annotations()
        self.define_epochs_raw(visualize=save_images)
        self.define_ers_erd()
        self.plot_evoked()

    def __getattr__(self, name):
        return 'EEGAnalysis does not have `{}` attribute.'.format(str(name))
