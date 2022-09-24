import matplotlib.pyplot as plt
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs
# from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.stats import permutation_cluster_test as pcluster_test
from scipy.stats import t
from mne_connectivity.viz import plot_sensors_connectivity
from mne_connectivity.viz import plot_connectivity_circle


subjects = ['001', '002', '003', '004', '005', '007', '008', '009', '010', '011', '012', '013', '014',
            '015', '016', '017', '018', '019', '021', '022', '023', '024', '025', '026', '028',
            '029', '030', '031']  # Without 006, 020 and 027

# subjects_left = ['003', '005', '008', '011', '012', '013', '018', '021', '023', '024', '026', '029', '030', '031']
subjects_left = ['003', '005']

# subjects_right = ['001', '002', '004', '007', '009', '010', '014', '015', '016', '017', '019', '022', '025', '028']
subjects_right = ['001', '002']

subject_ids = {"left": subjects_left, "right": subjects_right}

tmin, tmax = 1, 6  # Activity period
# fmin, fmax = (1, 4, 8, 15), (4, 7, 13, 30)  # Frequency bands
f_bands = {"delta": (1, 4), "theta": (4, 7), "alpha": (8, 13), "beta": (13, 30)}
events = ['left', 'both', 'right']
sfreq = 500

kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None, out_type='mask')  # for cluster test

# for task in ['ME', 'MI']:
for task in ['MI']:
    data_participants_dicts = {"left": {}, "right": {}}

    for i, event in enumerate(events):
        # for i, event in enumerate(['left']):
        for j, fband in enumerate(f_bands.keys()):
            # for j, fband in ['delta']:
            fmin, fmax = f_bands[fband][0], f_bands[fband][1]

            data_participants = None
            for laterality in subject_ids.keys():
                for s, subject in enumerate(subject_ids[laterality]):
                    epochs = mne.read_epochs(f'data/sub-P{subject}/sub-P{subject}_task-{task}-epo.fif', preload=True)
                    epochs.load_data().pick_types(eeg=True)
                    epochs_event = epochs[event]
                    con = spectral_connectivity_epochs(epochs_event, method='coh', mode='multitaper', sfreq=sfreq,
                                                       fmin=fmin, fmax=fmax, faverage=False, tmin=tmin, tmax=tmax,
                                                       mt_adaptive=False, n_jobs=1)

                    data_ = con.get_data(output='dense')
                    data = data_ + np.transpose(data_, axes=[1, 0, 2])
                    if data_participants is None:
                        data_participants = data
                        print(f'First Participant. Shape of data_participants is {data_participants.shape}')
                    else:
                        data_participants = np.append(data_participants, data, axis=-1)
                        print(f'Participant #{s}. Shape of data_participants is {data_participants.shape}')

                data_participants_dicts[laterality][f"{event}-{fband}"] = data_participants


for task in ['MI']:
    # Figure setup
    fig, axes = plt.subplots(len(f_bands), len(events), figsize=(12, 16))

    for laterality in subject_ids.keys():
        for i, event in enumerate(events):
            for j, fband in enumerate(f_bands.keys()):
                data_participants = data_participants_dicts[laterality][f"{event}-{fband}"]

                # Got all particpiants for given event and fband, so now we can mean and plot:
                data_mean = np.mean(data_participants, axis=2)

                data_clustertest_left = np.moveaxis(data_participants_dicts["left"][f"{event}-{fband}"], 2, 0)
                data_clustertest_right = np.moveaxis(data_participants_dicts["right"][f"{event}-{fband}"], 2, 0)

                ax = axes[j, i]

                # _, c, p, _ = pcluster_test([data_clustertest_left, data_clustertest_right], tail=1, **kwargs)
                # mask = c[..., p <= 0.05].any(axis=-1)
                # data_mean_masked = np.ma.masked_where(mask, data_mean)
                # im = ax.imshow(data_mean_masked, vmin=0, vmax=1)

                im = ax.imshow(data_mean, vmin=0, vmax=1)

                # # Colorbar
                # cbar = ax.figure.colorbar(im, ax=ax)
                # cbar.ax.set_ylabel('Connectivity', rotation=-90, va="bottom")

                # Show all ticks and label them with the respective list entries
                labels = con.names
                ax.set_xticks(range(len(labels)), labels=labels)
                ax.set_yticks(range(len(labels)), labels=labels)

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                         rotation_mode="anchor")

        pad = 5
        for ax, event in zip(axes[0], events):
            ax.annotate(event, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')
        for ax, fband in zip(axes[:, 0], f_bands.keys()):
            ax.annotate(fband, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')

        fig.suptitle(f'Connectivity by Coherence {laterality} handed')
        fig.tight_layout()
        plt.show()
