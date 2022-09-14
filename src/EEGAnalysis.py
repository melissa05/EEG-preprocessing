import os
import pathlib
import pickle
from pathlib import Path

import mne
import numpy as np
import pyxdf
from matplotlib import pyplot as plt
from src.ERDS import compute_erds


class EEGAnalysis:
    """
    Class implemented for the EEG preprocessing and visualization from the reading of .xdf files. This class is
    intended with the use of BrainVision products, LSL and Psychopy.

    Giulia Pezzutti

    Assoc.Prof. Dr. Selina Christin Wriessnegger
    M.Sc. Luis Alberto Barradas Chacon

    Institute for Neural Engineering, @ TUGraz

    :ivar xdf_paths: Filepaths of xdf files to be read.
    :vartype xdf_paths: list[str]
    :ivar file_info: Holds information regarding input and output files and directories.
    :vartype file_info: dict
    :ivar eeg_fs: Sampling frequency of EEG signals.
    :vartype eeg_fs: int
    :ivar eeg_signals: List of EEG time series; one array per xdf file.
    :vartype eeg_signals: list(:class:`numpy.ndarray`)
    :ivar eeg_instants: List of EEG timestamps; one array per xdf file.
    :vartype eeg_instants: list(:class:`numpy.ndarray`)
    :ivar marker_ids: List of (PsychoPy) markers; one list of lists per xdf file (each innermost list contains one
        marker).
    :vartype marker_ids: list(list(list(str)))
    :ivar marker_instants: List of marker timestamps; one array per xdf file.
    :vartype marker_instants: list(:class:`numpy.ndarray`)
    :ivar channels_names: Dictionary of all channels with numbers as keys and names as values.
    :vartype channels_names: dict
    :ivar channels_types: Dictionary of channel types with numbers as keys and types as values.
    :vartype channels_types: dict
    :ivar evoked_rois:
    :vartype evoked_rois:
    :ivar info: Info object with information about sensors and methods of measurement.
    :vartype info: :class:`mne.Info`
    :ivar raw: Instance of mne RawArray object, holding eeg raw data and related information, like channels names.
    :vartype raw: :class:`mne.io.RawArray`
    :ivar bad_channels: Names of all the bad channels.
    :vartype bad_channels: list[str]
    :ivar events: Array of events as returned by :func:`mne.events_from_annotations`.
    :vartype events: :class:`numpy.ndarray` [int]
    :ivar event_mapping: Event ids as returned by :func:`mne.events_from_annotations`.
    :vartype event_mapping: dict
    :ivar epochs: Epochs extracted from attribute :attr:`raw`.
    :vartype epochs: :class:`mne.Epochs`
    :ivar annotations: List of time series annotations; one instance per xdf file.
    :vartype annotations: list[:class:`mne.Annotations`]
    :ivar evoked:
    :vartype evoked: dict
    :ivar rois_numbers: Dictionary of different rois with roi names as keys and lists of channel indices as values.
    :vartype rois_numbers: dict
    :ivar input_info: The parameter `dict_info` passed to the class constructor.
    :vartype input_info: dict
    :ivar t_min:
    :vartype t_min:
    :ivar t_max:
    :vartype t_max:
    """

    def __init__(self, paths, dict_info, per_run_inspection=False):
        """
        *Constructor* of the class: it initializes all the necessary variables, loads the xdf file with all the
        correspondent information.

        Calls :meth:`get_info_from_path` and :meth:`load_raws_from_xdfs`.

        :param paths: paths to .xdfs file containing the data. The filepath must be with following structure:
            <folder>/sub-<sub-code>_block<num>.xdf (e.g. "/data/subj_001_block1.xdf").
        :type paths: list[str] | str
        :param dict_info: dict containing the main information about the processing. It must contain the following keys:
            streams, montage, filtering, spatial_filtering, samples_remove, t_min, t_max, full_annotation,
            annotation_durations, epochs_reject_criteria, rois, bad_epoch_names, erp, erds. See the documentation for
            further info about the dict.
        :type dict_info: dict
        """

        # Initialize variables:
        self.xdf_paths = paths if isinstance(paths, list) else [paths]
        self.file_info = {}
        self.eeg_fs = None
        self.eeg_signals, self.eeg_instants = [], []
        self.marker_ids, self.marker_instants = [], []
        self.channels_names, self.channels_types, self.evoked_rois = {}, {}, {}
        self.info, self.raw, self.bad_channels = None, None, None
        self.events, self.event_mapping, self.epochs, self.annotations = None, None, None, None
        self.evoked = {}
        self.rois_numbers = {}

        # Extract info from the dict:
        self.input_info = dict_info
        # Extract info from the path:
        self.get_info_from_path()
        self.t_min = self.input_info['t_min']  # start of each epoch
        self.t_max = self.input_info['t_max']  # end of each epoch
        # Load xdf files into raw variable:
        self.load_raws_from_xdf(per_run_inspection)

    def get_info_from_path(self):
        """
        Gets main information from file path regarding subject, folder and output folder and saves that information in
        attribute :attr:`file_info`.
        """

        # Get names of the xdf files to be loaded:
        file_names = [os.path.splitext(f)[0] for f in self.xdf_paths]

        # Get main folder in which data is contained:
        base = os.path.abspath(self.xdf_paths[0])
        folder = os.path.dirname(base).split('data/')[0]
        folder = folder.replace('\\', '/')
        
        project_folder = str(pathlib.Path(__file__).parent.parent.absolute())

        # Extraction of subject number and task:
        if self.input_info['lsl-version'] == '1.12':
            subject = (file_names[0].split('subj_')[1]).split('_block')[0]
        elif self.input_info['lsl-version'] == '1.16':
            if '_task-' in file_names[0]:
                subject = (file_names[0].split('sub-')[2]).split('_task')[0]
            else:
                subject = (file_names[0].split('sub-')[1]).split('_ses')[0]
        else:
            subject = ''

        if '_task-' in file_names[0]:
            task = (file_names[0].split('_task-')[1]).split('_run')[0]
        else:
            task = ''

        # Output folder according to the standard:
        output_folder = str(pathlib.Path(__file__).parent.parent.absolute()) + '/images/sub-' + subject
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self.file_info = {'input_folder': folder, 'file_names': file_names, 'subject': subject, 'task': task,
                          'output_images_folder': output_folder, 'project_folder': project_folder}

    def load_xdf(self, xdf_path):
        """
        Loads xdf file from the filepath given in input.

        The function automatically divides the different streams in the file according to the stream names given in
        attribute :attr:`input_info` and extracts their main information. EEG signals are converted from µV to V:
        Saves the sampling frequency of the EEG acquisition in attribute :attr:`eeg_fs`. Appends EEG and marker data
        of given file to attributes :attr:`eeg_signals`, :attr:`eeg_instants`, :attr:`marker_ids`,
        and :attr:`marker_instants`.

        Calls :meth:`load_channels`. Optionally calls :meth:`_fix_lost_samples`.

        :param xdf_path: Path to the xdf file to be loaded (whole path from project directory).
        :type xdf_path: str
        """

        stream_names = self.input_info['streams']

        # Load data:
        dat = pyxdf.load_xdf(xdf_path)[0]

        orn_signal, orn_instants = [], []
        effective_sample_frequency = None

        # Data iteration to extract the main information:
        for i in range(len(dat)):
            stream_name = dat[i]['info']['name'][0]

            # Gets 'BrainVision RDA Markers' stream:
            if stream_name == stream_names['EEGMarkers']:
                orn_signal = dat[i]['time_series']
                orn_instants = dat[i]['time_stamps']

            # Gets 'BrainVision RDA Data' stream:
            if stream_name == stream_names['EEGData']:
                # EEG signal:
                eeg_signal = dat[i]['time_series'][:, :32]  # Number of EEG channels should not be hardcoded.
                eeg_signal = eeg_signal * 1e-6  # Convert to Volt
                # EEG time instants:
                eeg_instants = dat[i]['time_stamps']
                # Sampling frequencies:
                self.eeg_fs = int(float(dat[i]['info']['nominal_srate'][0]))
                effective_sample_frequency = float(dat[i]['info']['effective_srate'])
                # Load the channels from the data:
                self.load_channels(dat[i]['info']['desc'][0]['channels'][0]['channel'])

            # Gets 'PsychoPy' (cues) stream:
            if stream_name == stream_names['Triggers']:
                marker_ids = dat[i]['time_series']
                marker_instants = dat[i]['time_stamps']

        # Cast to array:
        eeg_instants = np.array(eeg_instants)
        # eeg_signal = np.asmatrix(eeg_signal)

        # Check lost-samples problem:
        if len(orn_signal) != 0:
            print('\n\nATTENTION: some samples have been lost during the acquisition!!\n\n')
            self._fix_lost_samples(orn_signal, orn_instants, effective_sample_frequency)

        # Remove samples at the beginning and at the end of EEG:
        length = eeg_instants.shape[0]
        samples_to_be_removed = self.input_info['samples_remove']
        if samples_to_be_removed > 0:
            eeg_signal = eeg_signal[samples_to_be_removed:(length - samples_to_be_removed)]
            eeg_instants = eeg_instants[samples_to_be_removed:(length - samples_to_be_removed)]

        # Reference all the marker instants to the eeg instants (since some samples at the beginning of the
        # recording have been removed)
        marker_instants -= eeg_instants[0]
        marker_instants = marker_instants[marker_instants >= 0]

        # Remove signal mean:
        eeg_signal = eeg_signal - np.mean(eeg_signal, axis=0)

        # Save data in attributes:
        self.eeg_signals.append(eeg_signal)
        self.eeg_instants.append(eeg_instants)
        self.marker_ids.append(marker_ids)
        self.marker_instants.append(marker_instants)

    def load_raws_from_xdf(self, per_run_inspection=False):
        """
        Loads one or multiple xdf files into a raw object which is saved in the attribute :attr:`raw`. Populates
        attribute :attr:`info`.

        If parameter `per_run_inspection` is True, the data of each xdf file (i.e. run) is plotted individually,
        allowing the user to select bad channels. These bad channels will be interpolated as soon as the plot is closed.

        Gets called by constructor.

        Calls :meth:`load_xdf`, :meth:`create_annotations`, and :meth:`create_raw`.

        :param per_run_inspection: Whether each run should be plotted individually and bad channels interpolated.
        :type per_run_inspection: bool
        """

        # Load file after file:
        for path in self.xdf_paths:
            self.load_xdf(path)

        # Create mne.Info object.
        self.info = mne.create_info(list(self.channels_names.values()), self.eeg_fs, list(self.channels_types.values()))

        self.create_annotations(full=self.input_info['full_annotation'])  # Cues (from PsychoPy).

        # Create one Raw object for each xdf file, then concatenate them in the end:
        raws = []
        for i in range(len(self.eeg_signals)):
            raw = self.create_raw(self.eeg_signals[i], self.eeg_instants[i])
            raw.set_annotations(self.annotations[i])
            if per_run_inspection:
                mne.viz.plot_raw(raw, title=f'Run {i}', block=True, show_first_samp=True)
                raw.interpolate_bads()
            raws.append(raw)
        self.raw = mne.concatenate_raws(raws)

    def load_channels(self, dict_channels):
        """
        Loads channel names and types and saves them in attributes :attr:`channels_names` and :attr:`channels_types`.
        Also checks for bad channels in attribute :attr:`input_info` and saves their names in attribute
        :attr:`bad_channels`.

        :param dict_channels: Channel information loaded from an xdf file.
        :type dict_channels: dict
        """

        # x = data[0][0]['info']['desc'][0]["channels"][0]['channel']
        # to obtain the default-dict list of the channels from the original file (data, not dat!!)

        # Iterate over the info of the channels:
        for idx, info in enumerate(dict_channels):

            if info['label'][0].find('dir') != -1 or info['label'][0] == 'MkIdx':
                continue

            # Get channel name:
            self.channels_names[idx] = info['label'][0]

            # Solve problem with MNE and BrainProduct incompatibility:
            if self.channels_names[idx] == 'FP2':
                self.channels_names[idx] = 'Fp2'

            # Get channel type:
            self.channels_types[idx] = 'eog' if info['label'][0].find('EOG') != -1 else 'eeg'

        # Get bad channels for given subject:
        if self.file_info['subject'] in self.input_info['bad_channels'].keys():
            self.bad_channels = self.input_info['bad_channels'][self.file_info['subject']]
        else:
            self.bad_channels = []

    def _fix_lost_samples(self, orn_signal, orn_instants, effective_sample_frequency):
        # I am not sure this method is finished? There is no return and also no attribute changed?? Plus it wouldn't
        # work with my (Melissa's) changes now, as I have changed self.eeg_instants as well as self.eeg_signals.

        print('BrainVision RDA Markers: ', orn_signal)
        print('BrainVision RDA Markers instants: ', orn_instants)
        print('\nNominal srate: ', self.eeg_fs)
        print('Effective srate: ', effective_sample_frequency)

        print('Total number of samples: ', len(self.eeg_instants))
        final_count = len(self.eeg_signals)
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

    def create_raw(self, eeg_signal, eeg_instants):
        """
        Creates MNE RawArray instance from the data, setting the general information and the relative montage.
        Also creates the dictionary for the regions of interest according to the current data file and saves is in
        attribute :attr:`rois_numbers`.

        :param eeg_signal: EEG time series.
        :type eeg_signal: :class:`numpy.ndarray`
        :param eeg_instants: EEG timestamps.
        :type eeg_instants: :class:`numpy.ndarray`
        :returns: Instance of mne RawArray object which was created with the given input.
        :rtype: :class:`mne.io.RawArray`
        """

        # Create RawArray with MNE for the data:
        # I think there is something fishy here. Since I don't remove any samples in the beginning of my data, setting
        # first_samp=0 works for me. The other approach gave me a time delay and my markers did not match my data.
        # However, I do not think that my "fix" (first_samp=0) works for everyone.
        # raw = mne.io.RawArray(eeg_signal.T, self.info, first_samp=eeg_instants[0])
        raw = mne.io.RawArray(eeg_signal.T, self.info, first_samp=0)

        # Set montage setting according to the input:
        standard_montage = mne.channels.make_standard_montage(self.input_info['montage'])
        raw.set_montage(standard_montage)

        # Check for bad channels:
        if len(self.bad_channels) > 0:
            raw.info['bads'] = self.bad_channels
            raw.interpolate_bads(reset_bads=True)

        # Channel numbers associated to each roi:
        rois = self.input_info['rois']
        for roi in rois.keys():
            self.rois_numbers[roi] = np.array([raw.ch_names.index(i) for i in rois[roi]])

        return raw

    def visualize_eeg(self, signal=True, psd=True, psd_topo=False, sensors=False):
        """
        Visualization of raw data using different plots generated with MNE.

        :param signal: Whether the raw time signal plot should be generated.
        :type signal: bool
        :param psd: Whether the psd plot should be generated.
        :type psd: bool
        :param psd_topo: Whether the topographic psd plot should be generated.
        :type psd_topo: bool
        :param sensors: Whether the sensors plot (electrode positions) should be generated.
        :type sensors: bool
        """

        if signal:
            viz_scaling = dict(eeg=5e-5, eog=5e-5, ecg=1e-4, bio=1e-7, misc=1e-5)  # Custom scaling for signals.
            mne.viz.plot_raw(self.raw, scalings=viz_scaling, duration=20, show_first_samp=True)
        if psd:
            self.raw.plot_psd()
        if psd_topo:
            self.raw.plot_psd_topo()
            plt.show()
        if sensors:
            self.raw.plot_sensors(kind='topomap', ch_type='eeg', show_names=True)
            plt.show()

    def raw_spatial_filtering(self):
        """
        Resets the reference in raw data in attribute :attr:`raw` according to the spatial filtering type in the input
        dict.
        """

        mne.set_eeg_reference(self.raw, ref_channels=self.input_info['spatial_filtering'], copy=False)

    def raw_time_filtering(self):
        """
        Filter raw data in attribute :attr:`raw` with a band-pass filter and a notch filter.
        """

        # Extract the frequencies for the filters:
        l_freq = self.input_info['filtering']['low']
        h_freq = self.input_info['filtering']['high']
        n_freq = self.input_info['filtering']['notch']

        # Apply band-pass filter (single segments that were concatenated are filtered separately):
        if not (l_freq is None and h_freq is None):
            self.raw.filter(l_freq=l_freq, h_freq=h_freq, l_trans_bandwidth=0.1, h_trans_bandwidth=0.1,
                            skip_by_annotation=('edge', 'bad_acq_skip'), verbose=40)

        # Apply notch filter:
        if n_freq is not None:
            self.raw.notch_filter(freqs=n_freq, verbose=40)

    def raw_inspect_ica_components(self):
        """
        Plots ICA components of attribute :attr:`raw`.

        This function is intended to be used for the inspection of ICA components. See :meth:`raw_perform_ica` for
        applying ICA.
        """

        eeg_raw = self.raw.copy().filter(l_freq=1, h_freq=None)

        ica = mne.preprocessing.ICA(n_components=0.9, method='fastica', random_state=97)

        ica.fit(eeg_raw)
        ica.plot_sources(eeg_raw)
        ica.plot_components()

    def raw_perform_ica(self):
        """
        Removes ICA components according to ICA indices specified in attribute :attr:`input_info`.

        Before applying ICA, :meth:`raw_inspect_ica_components` should be used to inspect the ICA components and choose
        the ones to be removed.
        """

        eeg_raw = self.raw.copy().filter(l_freq=1, h_freq=None)

        ica = mne.preprocessing.ICA(n_components=0.9, method='fastica', random_state=97)

        ica.fit(eeg_raw)
        if isinstance(self.input_info['ica_exclude'][self.file_info['subject']], dict):
            ica.exclude = self.input_info['ica_exclude'][self.file_info['subject']][self.file_info['task']]
        else:
            ica.exclude = self.input_info['ica_exclude'][self.file_info['subject']]

        reconst_raw = self.raw.copy()
        ica.apply(reconst_raw)

        self.raw = reconst_raw

    def raw_ica_remove_eog(self):
        """
        Old method from Giulia. Deprecated. Will be removed with next commit.

        Automatic removal of EOG components, as suggested by MNE, just does not work.
        """

        n_components = list(self.channels_types.values()).count('eeg')

        eeg_raw = self.raw.copy()
        eeg_raw = eeg_raw.pick_types(eeg=True)

        ica = mne.preprocessing.ICA(n_components=0.99999, method='fastica', random_state=97)

        ica.fit(eeg_raw)
        ica.plot_sources(eeg_raw)
        ica.plot_components()
        # ica.plot_properties(eeg_raw)

        # # find which ICs match the EOG pattern
        # eog_indices, eog_scores = ica.find_bads_eog(self.raw, h_freq=5, threshold=3)
        # print(eog_indices)
        # ica.exclude = ica.exclude + eog_indices

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

    def create_annotations(self, full=False):
        """
        Annotations creation according to MNE definition. Annotations are extracted from markers stream data (onset,
        duration and description) and saved in attribute :attr:`annotations`.

        :param full: Annotations can be made of just one word or more than one. In 'full' case the whole annotation is
            considered, otherwise only the second word is kept.
        :type full: bool
        """

        annotations = []

        # Read every trigger in the stream:
        for idx_out, marker_ids in enumerate(self.marker_ids):

            # Generation of the events according to the definition:
            triggers = {'onsets': [], 'duration': [], 'description': []}
            for idx, marker_data in enumerate(marker_ids):

                # Annotations to be rejected:
                if marker_data[0] in self.input_info['bad_epoch_names']:
                    continue

                # Extract triggers information:
                triggers['onsets'].append(self.marker_instants[idx_out][idx])
                triggers['duration'].append(self.input_info['annotation_durations'][marker_data[0]])

                # According to 'full' parameter, extract the correct annotation description:
                if not full:
                    condition = marker_data[0].split('/')[-1]
                else:
                    condition = marker_data[0]
                if condition == 'edges': condition = 'canny'
                triggers['description'].append(condition)

            annotations.append(mne.Annotations(triggers['onsets'], triggers['duration'], triggers['description'],
                                               orig_time=None))

        # Save MNE annotations:
        self.annotations = annotations

    def create_epochs(self, visualize_epochs=False, rois=True):
        """
        Creates mne epochs and saves them in attribute :attr:`epochs`.

        In order to do so, events and event mapping are created from the annotated raw data in attribute :attr:`raw`
        and are saved in the attributes :attr:`events` and :attr:`event_mapping`.

        Optionally calls :meth:`visualize_epochs`.

        :param visualize_epochs: Whether to generate epochs plots or not.
        :type visualize_epochs: bool
        :param rois: Whether to visualize results according to the rois or for each channel. Only has an effect if
            `visualize_epochs` is `True`.
        :type rois: bool
        """

        # Create events and the event mapping from annotations already stored in raw:
        self.events, self.event_mapping = mne.events_from_annotations(self.raw, event_id=None, use_rounding=True)

        # Automatic rejection criteria for the epochs
        if 'eeg' in self.input_info['epochs_reject_criteria'].keys():
            reject_criteria = self.input_info['epochs_reject_criteria']
        else:
            reject_criteria = self.input_info['epochs_reject_criteria'][self.file_info['subject']]

        # Generation of the epochs according to the events:
        self.epochs = mne.Epochs(self.raw, self.events, event_id=self.event_mapping, preload=True,
                                 baseline=(self.t_min, 0), reject=reject_criteria, tmin=self.t_min, tmax=self.t_max)

        # Plot epochs, if wanted:
        if visualize_epochs:
            self.visualize_epochs(conditional_epoch=True, rois=rois)

    # def save_epochs(self):
    #     foldername = os.path.join((self.file_info['project_folder'], 'data', f'sub-{self.file_info["subject"][1:]}'))
    #     fname = os.path.join((foldername, f'sub-{self.file_info["subject"]}_task-{self.file_info["task"]}-epo.fif'))
    #     self.epochs.save(fname=fname, fmt='double', overwrite=True)

    def visualize_epochs(self, conditional_epoch=True, rois=True):
        """
        Saves plots of mean epoch signal.

        Still needs to be checked properly.

        :param conditional_epoch: boolean, if visualize the epochs extracted from the events or the general mean epoch
        :type conditional_epoch: bool
        :param rois: Whether to visualize the epochs according to the rois; only if `conditional_epoch` = `True`.
        :type rois: bool
        """

        rois_names = list(self.rois_numbers.keys())

        # Generate the mean plots according to the condition in the annotation value:
        if conditional_epoch:
            # Generate the epochs plots according to the roi and save them:
            if rois:
                for condition in self.event_mapping.keys():
                    images = self.epochs[condition].plot_image(combine='mean', group_by=self.rois_numbers, show=False)
                    for idx, img in enumerate(images):
                        img.savefig(self.file_info['output_images_folder'] + '/' + condition + '_' + rois_names[idx]
                                    + '.png')
                        plt.close(img)

            # Generate the epochs plots for each channel and save them:
            else:
                for condition in self.event_mapping.keys():
                    images = self.epochs[condition].plot_image(show=False)
                    for idx, img in enumerate(images):
                        img.savefig(self.file_info['output_images_folder'] + '/' + condition + '_' +
                                    self.channels_names[idx] + '.png')
                        plt.close(img)

        # Generate the mean plot considering all the epochs conditions:
        else:
            # Generate the epochs plots according to the roi and save them:
            if rois:
                images = self.epochs.plot_image(combine='mean', group_by=self.rois_numbers, show=False)
                for idx, img in enumerate(images):
                    img.savefig(self.file_info['output_images_folder'] + '/' + rois_names[idx] + '.png')
                    plt.close(img)

            # Generate the epochs plots for each channel and save them:
            else:
                images = self.epochs.plot_image(show=False)
                for idx, img in enumerate(images):
                    img.savefig(self.file_info['output_images_folder'] + '/' + self.channels_names[idx] + '.png')
                    plt.close(img)

        plt.close('all')

    def create_evoked(self, rois=True):
        """
        Function to define the evoked variables starting from the epochs. The evoked will be considered separately for
        each condition present in the annotation and for each ROI (otherwise, in general for the whole dataset).

        :param rois: Whether to visualize results according to the rois or just the general results.
        :type rois: bool
        """

        # for each condition
        for condition in self.event_mapping.keys():

            # get only the epochs of interest
            condition_epochs = self.epochs[condition]

            if rois:
                # for each roi of interest
                for roi in sorted(self.rois_numbers.keys()):
                    # extract only the channels of interest
                    condition_roi_epoch = condition_epochs.copy()
                    condition_roi_epoch = condition_roi_epoch.pick(self.rois_numbers[roi])

                    # average for each epoch and for each channel
                    condition_roi_epoch = condition_roi_epoch.average()
                    condition_roi_epoch = mne.channels.combine_channels(condition_roi_epoch, groups={'mean': list(range(len(self.rois_numbers[roi])))})

                    # define the label for the current evoked and save it
                    label = condition + '/' + roi
                    self.evoked[label] = condition_roi_epoch

            else:

                # average for each epoch and for each channel
                condition_epochs = condition_epochs.average()
                condition_epochs = mne.channels.combine_channels(condition_epochs, groups={'mean': list(
                    range(len(self.epochs.ch_names)))})

                # save the current evoked
                self.evoked['mean'] = condition_epochs

    def visualize_evoked(self):
        """
        Function to plot the computed evoked for each condition and for each region of interest.
        """

        # get minimum and maximum value of the mean signals
        min_value, max_value = np.inf, -np.inf
        for label in self.evoked.keys():
            data = self.evoked[label].get_data()[0]
            min_value = min(np.min(data), min_value)
            max_value = max(np.max(data), max_value)

        # path for images saving
        Path(self.file_info['output_images_folder'] + '/epochs/').mkdir(parents=True, exist_ok=True)

        number_conditions = len(list(self.event_mapping.keys()))
        path = self.file_info['output_images_folder'] + '/epochs/conditions.png'
        fig, axs = plt.subplots(int(np.ceil(number_conditions/2)), 2, figsize=(25.6, 19.2))

        for i, ax in enumerate(fig.axes):

            if i >= number_conditions:
                break

            condition = list(self.event_mapping.keys())[i]

            # extract the roi from the key name of the dictionary containing the evoked
            correct_labels = [s for s in self.evoked.keys() if condition + '/' in s]
            correct_short_labels = [s.split('/')[1] for s in correct_labels]

            # correctly plot all evoked
            for idx, label in enumerate(correct_labels):
                ax.plot(self.evoked[label].times * 1000, self.evoked[label].get_data()[0],
                        label=correct_short_labels[idx])

            # draw ERP vertical lines to see the peak of interest
            for erp in self.input_info['erp']:
                ax.vlines(erp, ymin=min_value, ymax=max_value, linestyles='dashed')

            ax.set_title(condition)
            ax.legend()

        plt.savefig(path)
        plt.close()

        path = self.file_info['output_images_folder'] + '/epochs/rois.png'
        number_rois = len(list(self.rois_numbers.keys()))
        fig, axs = plt.subplots(int(np.ceil(number_rois/2)), 2, figsize=(25.6, 19.2))

        for i, ax in enumerate(fig.axes):

            if i >= number_rois:
                break

            roi = list(self.rois_numbers.keys())[i]

            # extract the condition from the key name of the dictionary containing the evoked
            correct_labels = [s for s in self.evoked.keys() if '/' + roi in s]
            correct_short_labels = [s.split('/')[0] for s in correct_labels]

            # correctly plot all evoked
            for idx, label in enumerate(correct_labels):
                ax.plot(self.evoked[label].times * 1000, self.evoked[label].get_data()[0],
                        label=correct_short_labels[idx])

            # draw ERP vertical lines to see the peak of interest
            for erp in self.input_info['erp']:
                ax.vlines(erp, ymin=min_value, ymax=max_value, linestyles='dashed')

            ax.set_title(roi)
            ax.legend()

        plt.savefig(path)
        plt.close()

    def get_peak(self, t_min, t_max, peak, mean=True, channels=None):
        """
        Function to extract the peaks' amplitude from the epochs separately for each condition found and returns them
        or the mean value.

        :param t_min: lower bound of the time window in which the algorithm should look for the peak
        :param t_max: upper bound of the time window in which the algorithm should look for the peak
        :param peak: +1 for a positive peak, -1 for a negative peak
        :param mean: boolean value, if the return value should be the mean value or the list of amplitudes
        :param channels: list of channels name to be investigated. If None, all the channels are considered
        :return: if mean=True, mean amplitude value; otherwise list of detected peaks' amplitude and list of the
            correspondent annotations
        """

        if channels is None:
            channels = self.raw.ch_names
        peaks = {}
        annotations = {}

        # extraction of the data of interest and of the correspondent annotations
        epochs_interest = self.epochs.copy()
        epochs_interest = epochs_interest.pick_channels(channels)
        labels = np.squeeze(np.array(epochs_interest.get_annotations_per_epoch())[:, :, 2])

        # get the unique conditions of interest
        if len(list(self.event_mapping.keys())[0].split('/')) > 1:
            conditions_interest = [ann.split('/')[1] for ann in self.event_mapping.keys()]
        else:
            conditions_interest = self.event_mapping.keys()
        conditions_interest = list(set(conditions_interest))

        # for each condition of interest
        for condition in conditions_interest:

            # get the correspondent epochs and crop the signal in the time interval for the peak searching
            condition_roi_epoch = epochs_interest[condition]
            data = condition_roi_epoch.crop(tmin=t_min, tmax=t_max).get_data()

            # if necessary, get the annotation correspondent at each epoch
            condition_labels = []
            if not mean:
                condition_labels = [label for label in labels if '/' + condition in label]

            peak_condition, latency_condition, annotation_condition = [], [], []

            # for each epoch
            for idx, epoch in enumerate(data):

                # extract the mean signal between channels
                signal = np.array(epoch).mean(axis=0)

                # find location and amplitude of the peak of interest
                peak_loc, peak_mag = mne.preprocessing.peak_finder(signal, thresh=(max(signal) - min(signal)) / 50,
                                                                   extrema=peak, verbose=False)
                peak_mag = peak_mag * 1e6

                # reject peaks too close to the beginning or to the end of the window
                if len(peak_loc) > 1 and peak_loc[0] == 0:
                    peak_loc = peak_loc[1:]
                    peak_mag = peak_mag[1:]
                if len(peak_loc) > 1 and peak_loc[-1] == (len(signal) - 1):
                    peak_loc = peak_loc[:-1]
                    peak_mag = peak_mag[:-1]

                # select peak according to the minimum or maximum one and convert the location from number of sample
                # (inside the window) to time instant inside the epoch
                if peak == -1:
                    peak_loc = peak_loc[np.argmin(peak_mag)] / self.eeg_fs + t_min
                    peak_mag = np.min(peak_mag)
                if peak == +1:
                    peak_loc = peak_loc[np.argmax(peak_mag)] / self.eeg_fs + t_min
                    peak_mag = np.max(peak_mag)

                # save the values found
                peak_condition.append(peak_mag)
                latency_condition.append(peak_loc)

                # in the not-mean case, it's necessary to save the correct labelling
                if not mean:
                    annotation_condition.append(condition_labels[idx].split('/')[0])

            # compute output values or arrays for each condition
            if mean:
                peaks[condition] = np.mean(np.array(peak_condition))
            else:
                peaks[condition] = np.array(peak_condition)
                annotations[condition] = annotation_condition

        if not mean:
            return peaks, annotations

        return peaks

    def save_pickle(self):
        """
        Function to save epochs, labels and main information into pickle files. The first two are saved as numpy arrays,
        the last one is saved as dictionary.
        """

        epochs = np.array(self.epochs.get_data())
        labels = [annotation[0][2] for annotation in self.epochs.get_annotations_per_epoch()]
        info = {'fs': self.eeg_fs, 'channels': self.epochs.ch_names, 'tmin': self.t_min, 'tmax': self.t_max}

        Path(self.file_info['project_folder'] + 'data/pickle/').mkdir(parents=True, exist_ok=True)

        with open(self.file_info['project_folder'] + '/data/pickle/' + self.file_info['subject'] + '_data.pkl', 'wb') as f:
            pickle.dump(epochs, f)
        with open(self.file_info['project_folder'] + '/data/pickle/' + self.file_info['subject'] + '_labels.pkl', 'wb') as f:
            pickle.dump(labels, f)
        with open(self.file_info['project_folder'] + '/data/pickle/' + self.file_info['subject'] + '_info.pkl', 'wb') as f:
            pickle.dump(info, f)

        print('Pickle files correctly saved')

    def run_raw_epochs(self, visualize_raw=False, save_images=True, ica_analysis=False, create_evoked=True,
                       save_pickle=True):
        """
        Function to run all the methods previously reported. Attention: ICA is for now not used.

        :param visualize_raw: Whether raw signals should be visualized.
        :type visualize_raw: bool
        :param save_images: Whether epoch plots should be saved. (note: they are never visualized)
        :type save_images: bool
        :param ica_analysis: Whether ICA analysis should be performed.
        :type ica_analysis: bool
        :param create_evoked: Whether Evoked computation is necessary.
        :type create_evoked: bool
        :param save_pickle: Whether the pickles with data, label and info should be saved.
        :type save_pickle: bool
        """

        if visualize_raw:
            self.visualize_eeg()

        if self.input_info['spatial_filtering'] is not None:
            self.raw_spatial_filtering()

        if self.input_info['filtering'] is not None:
            self.raw_time_filtering()

        if visualize_raw:
            self.visualize_eeg()

        if ica_analysis:
            self.raw_ica_remove_eog()

        self.create_annotations(full=self.input_info['full_annotation'])
        self.create_epochs(visualize_epochs=save_images)
        if save_images:
            compute_erds(epochs=self.epochs, rois=self.input_info['rois'], fs=self.eeg_fs, t_min=self.t_min,
                         f_min=self.input_info['erds'][0], f_max=self.input_info['erds'][1],
                         path=self.file_info['output_images_folder'])
        if create_evoked:
            self.create_evoked()
            if save_images:
                self.visualize_evoked()
        if save_pickle:
            self.save_pickle()

    def preprocess_raw(self, visualize_raw=False, ica_analysis=False, reference=None):
        """
        Perform some preprocessing to the raw data stored in attribute :attr:`raw`.
        
        Sets the reference channel if parameter `reference` is given.

        Optionally calls (depending on settings) :meth:`raw_spatial_filtering`, :meth:`raw_time_filtering`,
        :meth:`visualize_eeg`, and :meth:`raw_perform_ica`.

        :param visualize_raw: Whether to visualize the data after processing steps.
        :type visualize_raw: bool
        :param ica_analysis: Whether to perform EOG removal via ICA.
        :type ica_analysis: bool
        :param reference: List of channels to use as reference, or 'average' to use a virtual reference.
        :type reference: str or list[str]
        """

        if reference:
            self.raw.set_eeg_reference(ref_channels=reference, projection=True)
            self.raw.apply_proj()

        if ica_analysis:
            self.raw_perform_ica()

        # Perform spatial and time filtering if user specified that in input info dict:
        if self.input_info['spatial_filtering'] is not None:
            self.raw_spatial_filtering()
        if self.input_info['filtering'] is not None:
            self.raw_time_filtering()

        if visualize_raw:
            self.visualize_eeg()

    def __getattr__(self, name):
        return 'EEGAnalysis does not have `{}` attribute.'.format(str(name))
