import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from more_itertools import locate


def define_ers_erd_spectrogram(self, epochs, rois_numbers, fs, t_min, f_max=50):

    signals = epochs.get_data(picks='eeg')  # epochs x channels x instants

    annotations = [annotation[0][2] for annotation in epochs.get_annotations_per_epoch()]
    conditions = list(set(annotations))

    freq, time, spectrogram = scipy.signal.spectrogram(signals, fs=fs, nperseg=250, noverlap=225, scaling='spectrum')

    time = time + t_min
    spectrogram = np.squeeze(spectrogram[:, :, np.argwhere(freq < f_max), :])
    freq = np.squeeze(freq[np.argwhere(freq < f_max)])

    last_time = time[-1] + (time[-1] - time[-2])
    x_axis = np.insert(time, -1, last_time)
    last_freq = freq[-1] + (freq[-1] - freq[-2])
    y_axis = np.insert(freq, -1, last_freq)

    for roi in rois_numbers.keys():
        numbers = rois_numbers[roi]
        roi_epochs = spectrogram[:, numbers, :, :]

        reference = np.squeeze(roi_epochs[:, :, :, np.argwhere(time < 0)])
        power_reference = np.mean(reference, axis=-1)

        # todo da fare in modo più efficente
        for idx_epoch, epoch in enumerate(roi_epochs):
            for idx_channel, channel in enumerate(epoch):
                for idx_freq, freq_value in enumerate(channel):
                    reference = power_reference[idx_epoch, idx_channel, idx_freq]
                    for idx_sample, sample in enumerate(freq_value):
                        roi_epochs[idx_epoch, idx_channel, idx_freq, idx_sample] = (sample - reference) / reference

        for condition in conditions:

            roi_condition_epochs = roi_epochs[list(locate(annotations, lambda x: x == condition))]

            roi_condition_epochs = np.mean(roi_condition_epochs, axis=0)
            roi_condition_epochs = np.mean(roi_condition_epochs, axis=0)

            roi_condition_epochs = np.array(roi_condition_epochs)
            z_min, z_max = -np.abs(roi_condition_epochs).max(), np.abs(roi_condition_epochs).max()
            fig, ax = plt.subplots()
            p = ax.pcolor(x_axis, y_axis, roi_condition_epochs, cmap='RdBu', snap=True, vmin=z_min, vmax=z_max)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(condition + ' ' + roi)
            ax.axvline(0, color='k')
            fig.colorbar(p, ax=ax)
            fig.savefig(self.file_info['output_folder'] + '/' + condition + '_' + roi + '_erds.png')
            plt.close()

