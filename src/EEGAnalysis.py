import os
from pathlib import Path

import mne
import numpy as np
import pyxdf
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mne.stats import permutation_cluster_1samp_test as pcluster_test


class EEGAnalysis:

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

        self.evoked_rois = {}
        self.info, self.raw, self.events, self.event_mapping, self.epochs, self.annotations = None, None, None, None, None, None

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

        stream_names = {'Markers': 'BrainVision RDA Markers', 'EEG': 'BrainVision RDA', 'Triggers': 'PsychoPy'}

        # data loading
        dat = pyxdf.load_xdf(self.data_path)[0]

        orn_signal, orn_instants = [], []
        effective_sample_frequency = None

        # data iteration to extract the main information
        for i in range(len(dat)):
            stream_name = dat[i]['info']['name'][0]

            if stream_name == stream_names['Markers']:  # gets 'BrainVision RDA Markers' stream
                orn_signal = dat[i]['time_series']
                orn_instants = dat[i]['time_stamps']

            if stream_name == stream_names['EEG']:
                self.eeg_signal = dat[i]['time_series'][:, :32]
                self.eeg_signal = self.eeg_signal * 1e-6
                self.eeg_instants = dat[i]['time_stamps']
                self.eeg_fs = int(float(dat[i]['info']['nominal_srate'][0]))
                self.load_channels(dat[i]['info']['desc'][0]["channels"][0]['channel'])
                effective_sample_frequency = float(dat[i]['info']['effective_srate'])

            if stream_name == stream_names['Triggers']:
                self.marker_ids = dat[i]['time_series']
                self.marker_instants = dat[i]['time_stamps']

        self.eeg_instants = np.array(self.eeg_instants)
        self.eeg_signal = np.array(self.eeg_signal)

        if len(orn_signal) != 0:
            self.fix_lost_samples(orn_signal, orn_instants, effective_sample_frequency)

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

        total_time = len(self.eeg_instants)/effective_sample_frequency
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

            new_marker_signal[(x+1):] = np.array(new_marker_signal[(x+1):]) + additional_time

    def load_channels(self, dict_channels):
        """
        Upload channels name from a xdf file
        """

        # x = data[0][0]['info']['desc'][0]["channels"][0]['channel']
        # to obtain the default-dict list of the channels from the original file (data, not dat!!)

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

        print('\n')
        self.info = mne.create_info(list(self.channels_names.values()), self.eeg_fs, list(self.channels_types.values()))
        self.raw = mne.io.RawArray(self.eeg_signal.T, self.info, first_samp=self.eeg_instants[0])

        # montage setting
        standard_montage = mne.channels.make_standard_montage('standard_1020')
        self.raw.set_montage(standard_montage)

    def filter_raw(self, l_freq=1, h_freq=60, n_freq=50, order=8):
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
        Visualization of the plots that could be generated with MNE according to a scaling property
        :param signal: boolean, if the signal plot should be generated
        :param psd: boolean, if the psd plot should be generated
        :param psd_topo: boolean, if the topographic psd plot should be generated
        """

        viz_scaling = dict(eeg=1e-4, eog=1e-4, ecg=1e-4, bio=1e-7, misc=1e-5)

        if signal:
            mne.viz.plot_raw(self.raw, scalings=viz_scaling, duration=50)
            # plt.close()
        if psd:
            self.raw.plot_psd()
            plt.close()
        if psd_topo:
            self.raw.plot_psd_topo()
            plt.close()

    def set_reference(self, ref_type='average'):
        """
        Resetting the reference in raw data
        :param ref_type: type of referencing to be performed
        """

        mne.set_eeg_reference(self.raw, ref_channels=ref_type, copy=False)

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

    def define_epochs_raw(self, visualize=True, only_manipulation=True, rois=True):
        """
        Function to extract events from the marker data, generate the correspondent epochs and determine annotation in
        the raw data according to the events
        :param visualize:
        :param only_manipulation: boolean variable to select to manipulate epochs only according to the image
        manipulation type or according to the whole trigger (img_name/manipulation)
        :param rois: boolean variable to select if visualize results in terms to rois or not
        """

        # generation of the events according to the definition
        triggers = {'onsets': [], 'duration': [], 'description': []}

        for idx, marker_data in enumerate(self.marker_ids):

            if marker_data[0] == 'intro' or marker_data[0] == 'pause' or marker_data[0] == 'end':
                continue

            triggers['onsets'].append(self.marker_instants[idx])
            triggers['duration'].append(int(0))

            if only_manipulation:
                manipulation = marker_data[0].split('/')[-1]
                if manipulation == 'edges':
                    manipulation = 'canny'
                triggers['description'].append(manipulation)
            else:
                triggers['description'].append(marker_data[0])

        self.annotations = mne.Annotations(triggers['onsets'], triggers['duration'], triggers['description'])
        self.raw.set_annotations(self.annotations)

        self.events, self.event_mapping = mne.events_from_annotations(self.raw)

        # generation of the epochs according to the events

        t_min = -0.2  # start of each epoch (200ms before the trigger)
        t_max = 0.8  # end of each epoch (800ms after the trigger)

        # Automatic rejection criteria: reject epochs based on maximum peak-to-peak signal amplitude (PTP)
        reject_criteria = dict(eeg=200e-6,  # 200 µV
                               eog=1e-3)  # 1 mV

        self.epochs = mne.Epochs(self.raw, self.events, event_id=self.event_mapping, preload=True,
                                 # baseline=(None, t_min),
                                 reject=reject_criteria, tmin=t_min, tmax=t_max)

        print('\nApplying baseline')
        self.epochs = self.epochs.apply_baseline((t_min, None))

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
                    images = self.epochs[condition].plot_image(combine='mean', group_by=rois_numbers,
                                                               # vmin=-6e-9, vmax=6e-9,
                                                               # scalings=viz_scaling,
                                                               show=False)

                    for idx, img in enumerate(images):
                        img.savefig(self.file_info['output_folder'] + '/' + condition + '_' + rois_names[idx] + '.png')
                        plt.close(img)
            else:
                for condition in self.event_mapping.keys():
                    self.epochs[condition].plot_image(show=False)

        plt.close('all')

        for condition in self.event_mapping.keys():
            img = self.epochs[condition].plot_psd_topomap(vlim=(7, 45), show=False)
            img.savefig(self.file_info['output_folder'] + '/' + condition + '_topography.png')
            plt.close(img)

        frequencies = np.arange(8, 50, 3)
        combined_epochs = mne.channels.combine_channels(self.epochs, rois_numbers)

        power = mne.time_frequency.tfr_morlet(combined_epochs, freqs=frequencies, n_cycles=5, use_fft=True,
                                              return_itc=False, decim=3, average=False)

        for idx, condition in enumerate(self.event_mapping.keys()):

            images = power[idx].average().plot(mode='mean', show=False)

            for index, img in enumerate(images):
                img.savefig(self.file_info['output_folder'] + '/' + condition + '_' + list(rois_numbers.keys())[index] + '_freq.png')
            plt.close('all')

        plt.close('all')

    def define_rois(self):

        rois = dict(
            central=["Cz", "C3", "C4"],
            frontal=["Fz", "Fp1", "F3", "F7", "FC1", "FC2", "F4", "F8", "Fp2"],
            occipital_parietal=["O1", "Oz", "O2", "Pz", "P3", "P7", "P4", "P8"],
            temporal=["FC6", "FC5", "T7", "T8", "CP5", "CP6", "FT9", "FT10", "TP9", "TP10"],
        )

        rois_numbers = dict(
            central=np.array([self.raw.ch_names.index(i) for i in rois['central']]),
            frontal=np.array([self.raw.ch_names.index(i) for i in rois['frontal']]),
            occipital_parietal=np.array([self.raw.ch_names.index(i) for i in rois['occipital_parietal']]),
            temporal=np.array([self.raw.ch_names.index(i) for i in rois['temporal']]),
        )

        return rois_numbers

    def define_ers_erd(self):
        freqs = np.arange(2, 36)  # frequencies from 2-35Hz
        t_min, t_max = -0.2, 0.8
        v_min, v_max = -1, 1.5  # set min and max ERDS values in plot
        baseline = [-0.2, 0]

        cnorm = TwoSlopeNorm(vmin=v_min, vcenter=0, vmax=v_max)  # min, center & max ERDS
        kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type='mask')

        tfr = mne.time_frequency.tfr_multitaper(self.epochs, freqs=freqs, n_cycles=freqs, use_fft=True,
                                                return_itc=False, average=False, decim=4)
        # tfr.crop(t_min, t_max).apply_baseline(baseline, mode="percent")

        for event in self.event_mapping:
            # select desired epochs for visualization
            tfr_ev = tfr[event]
            print(tfr_ev)
            fig, axes = plt.subplots(1, 4, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 10, 1]})
            print(axes)
            for ch, ax in enumerate(axes[:-1]):  # for each channel
                print(len(axes[:-1]))
                # positive clusters
                _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
                # negative clusters
                _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

                print(c1, p1, c2, p2)

                # note that we keep clusters with p <= 0.05 from the combined clusters
                # of two independent tests; in this example, we do not correct for
                # these two comparisons
                c = np.stack(c1 + c2, axis=2)  # combined clusters
                p = np.concatenate((p1, p2))  # combined p-values
                mask = c[..., p <= 0.05].any(axis=-1)

                # plot TFR (ERDS map with masking)
                tfr_ev.average().plot([ch], cmap="RdBu", cnorm=cnorm, axes=ax,
                                      colorbar=False, show=False, mask=mask,
                                      mask_style="mask")

                ax.set_title(self.epochs.ch_names[ch], fontsize=10)
                ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
                if ch != 0:
                    ax.set_ylabel("")
                    ax.set_yticklabels("")
            fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
            fig.suptitle(f"ERDS ({event})")
            plt.show()

    def define_evoked(self):

        rois_numbers = self.define_rois()

        evoked = self.epochs.average(picks=['eeg'], by_event_type=True)

        for evok in evoked:

            roi_evoked = mne.channels.combine_channels(evok, rois_numbers, method='mean')
            self.evoked_rois[evok.comment] = roi_evoked

    def plot_mean_epochs(self):

        epochs = self.get_epochs_dataframe()
        conditions = epochs.iloc[:]['condition'].tolist()
        conditions = sorted(list(set(conditions)))

        x_axis = epochs.iloc[:]['time'].tolist()
        x_axis = np.sort(np.array(list(set(x_axis))))
        rois_numbers = self.define_rois()

        means, epoch_signals = {}, {}

        for condition in conditions:

            # epochs belonging only to the current condition
            condition_epochs = epochs.loc[epochs['condition'] == condition, :]

            # number of the epochs belonging to the current condition
            number_epochs = condition_epochs.iloc[:]['epoch'].tolist()
            number_epochs = list(set(number_epochs))

            for epoch_number in number_epochs:

                # current epoch (in current condition), extracted and made as matrix: #channels x #samples
                current_epoch = condition_epochs.loc[condition_epochs['epoch'] == epoch_number, :].values[:, 3:-2]
                current_epoch = np.array(current_epoch).T

                for roi in sorted(rois_numbers.keys()):

                    # signals of the current epoch and in the current roi: #channels (in ROI) x #number samples
                    current_roi_epoch = current_epoch[rois_numbers[roi]]

                    label = condition + '/' + roi

                    # save in a matrix containing all signals coming from same condition and same roi
                    if label in epoch_signals:
                        epoch_signals[label] = np.concatenate((epoch_signals[label], current_roi_epoch))
                    else:
                        epoch_signals[label] = current_roi_epoch

        # mean of the signals in same condition and same roi
        for key in epoch_signals:
            mean_current_epochs = np.mean(epoch_signals[key], axis=0)
            means[key] = mean_current_epochs

        # get minimum and maximum value of the mean signals
        min_value, max_value = 100, -100
        for label in epoch_signals.keys():
            min_value = min(np.min(means[label]), min_value)
            max_value = max(np.max(means[label]), max_value)

        Path(self.file_info['output_folder'] + '/epochs/').mkdir(parents=True, exist_ok=True)

        fig, axs = plt.subplots(3, 2, figsize=(25.6, 19.2))
        path = self.file_info['output_folder'] + '/epochs/conditions.png'

        for i, ax in enumerate(fig.axes):

            condition = conditions[i]

            correct_labels = [s for s in epoch_signals.keys() if condition + '/' in s]
            correct_short_labels = [s.split('/')[1] for s in correct_labels]

            for idx, label in enumerate(correct_labels):
                ax.plot(x_axis, means[label], label=correct_short_labels[idx])

            ax.vlines(0, ymin=min_value, ymax=max_value, linestyles='dashed')
            ax.vlines(170, ymin=min_value, ymax=max_value, colors='r', linestyles='dashed')
            ax.vlines(300, ymin=min_value, ymax=max_value, colors='g', linestyles='dashed')
            ax.set_title(condition)

        plt.legend(bbox_to_anchor=(1.2, 2))
        plt.savefig(path)
        plt.close()

        fig, axs = plt.subplots(2, 2, figsize=(25.6, 19.2))
        path = self.file_info['output_folder'] + '/epochs/rois.png'

        for i, ax in enumerate(fig.axes):
            roi = list(rois_numbers.keys())[i]

            correct_labels = [s for s in epoch_signals.keys() if '/' + roi in s]
            correct_short_labels = [s.split('/')[0] for s in correct_labels]

            for idx, label in enumerate(correct_labels):
                ax.plot(x_axis, means[label], label=correct_short_labels[idx])

            ax.vlines(0, ymin=min_value, ymax=max_value, linestyles='dashed')
            ax.vlines(170, ymin=min_value, ymax=max_value, colors='r', linestyles='dashed')
            ax.vlines(300, ymin=min_value, ymax=max_value, colors='g', linestyles='dashed')

            ax.set_title(roi)

        plt.legend(bbox_to_anchor=(1.2, 1.1))
        plt.savefig(path)
        plt.close()

        return means

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

    def get_epochs_dataframe(self):

        return self.epochs.to_data_frame()

    def __get_eeg_fs__(self):
        return self.eeg_fs
