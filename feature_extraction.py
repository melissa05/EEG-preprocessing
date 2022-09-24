import os
import csv
import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def square(arr):
    return arr ** 2


def calc_ERDS(epochs, channel=None, avg_roi=False):
    ERDS_dict = {}
    bandpower_dict = {}
    for event in epochs.event_id.keys():
        epochs_event = epochs[event]

        epochs_base = epochs_event.copy()
        epochs_base.crop(-1.5, 0)

        epochs_active = epochs_event.copy()
        epochs_active.crop(2, 6)

        ERDS_sum = 0
        A_sum = 0
        for e in range(epochs_event.events.shape[0]):
            if channel:
                R = epochs_base.get_data(picks=[channel], item=e).mean()
                A = epochs_active.get_data(picks=[channel], item=e).mean()
            elif avg_roi:
                # Mean over all channels and all epochs of event:
                R = epochs_base.get_data(item=e).mean()
                A = epochs_active.get_data(item=e).mean()
            A_sum += A
            ERDS_sum += (A - R) / R
        ERDS = round(ERDS_sum * 100 / epochs_event.events.shape[0], 2)
        A = A_sum / epochs_event.events.shape[0]

        ERDS_dict[event] = ERDS
        bandpower_dict[event] = A

    return ERDS_dict, bandpower_dict


def plot_erds_topoplot(epochs, f_bands=((8, 13, 'Alpha'), (16, 24, 'Beta')), fig_title='ERDS Topoplots', show=False):
    channels = epochs.ch_names
    events = ['left', 'both', 'right']
    data_dict = {}
    vmin_alpha, vmax_alpha = -70, 200
    vmin_beta, vmax_beta = -50, 50
    cnorm_alpha = TwoSlopeNorm(vmin=vmin_alpha, vcenter=0, vmax=vmax_alpha)
    cnorm_beta = TwoSlopeNorm(vmin=vmin_beta, vcenter=0, vmax=vmax_beta)

    for fband in f_bands:
        fmin, fmax, fband_name = fband
        data_dict[fband_name] = {
            "left": [],
            "both": [],
            "right": []
        }

        epochs_filt = epochs.copy()
        epochs_filt.filter(fmin, fmax)
        epochs_filt.apply_function(square)

        for ch in channels:
            erds_dict, _ = calc_ERDS(epochs_filt, ch)
            data_dict[fband_name]["left"].append(erds_dict["left"])
            data_dict[fband_name]["both"].append(erds_dict["both"])
            data_dict[fband_name]["right"].append(erds_dict["right"])

    # Figure setup
    fig, axes = plt.subplots(len(f_bands), len(events), figsize=(12, 8))

    for i, event in enumerate(events):
        for j, fband in enumerate(f_bands):
            data = data_dict[fband[-1]][event]
            ax = axes[j, i]

            if fband[-1] == 'Alpha':
                cnorm = cnorm_alpha
                im_alpha, cn = mne.viz.plot_topomap(data, epochs.info, cmap='RdBu', cnorm=cnorm, axes=ax, show=False)
            else:
                cnorm = cnorm_beta
                im_beta, cn = mne.viz.plot_topomap(data, epochs.info, cmap='RdBu', cnorm=cnorm, axes=ax, show=False)

    pad = 5
    for ax, event in zip(axes[0], events):
        ax.annotate(event, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for ax, fband in zip(axes[:, 0], f_bands):
        ax.annotate(f'{fband[-1]}\n{fband[0]}-{fband[1]} Hz', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    fig.subplots_adjust(right=0.85)
    cbar_alpha_ax = fig.add_axes([0.9, 0.55, 0.02, 0.35])  # Left, bottom, width, height
    cbar_beta_ax = fig.add_axes([0.9, 0.15, 0.02, 0.35])  # Left, bottom, width, height
    fig.colorbar(im_alpha, cax=cbar_alpha_ax)
    fig.colorbar(im_beta, cax=cbar_beta_ax)
    fig.suptitle(fig_title)
    # fig.tight_layout()
    if show:
        plt.show()
    else:
        return fig


if __name__ == '__main__':
    subjects = ['001', '002', '003', '004', '005', '007', '008', '009', '010', '011', '012', '013', '014',
                '015', '016', '017', '018', '019', '021', '022', '023', '024', '025', '026', '028',
                '029', '030', '031']  # Without 006, 020 and 027
    # subjects = ['001', '002']

    # Creating the csv file to keep track of dropped epochs:
    dirname = 'C:\\Users\\Melissa\\PycharmProjects\\EEG-preprocessing\\data'

    # # ERDS and bandpower for C3 and C4:
    # # ERDS values:
    # if not os.path.isfile(f'{dirname}/erds.csv'):
    #     with open(f'{dirname}/erds.csv', 'w') as csvfile:
    #         writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    #         writer.writerow(['participant',
    #                          'ME_both_alpha_C3', 'ME_both_beta_C3', 'ME_both_alpha_C4', 'ME_both_beta_C4',
    #                          'ME_left_alpha_C3', 'ME_left_beta_C3', 'ME_left_alpha_C4', 'ME_left_beta_C4',
    #                          'ME_right_alpha_C3', 'ME_right_beta_C3', 'ME_right_alpha_C4', 'ME_right_beta_C4',
    #                          'MI_both_alpha_C3', 'MI_both_beta_C3', 'MI_both_alpha_C4', 'MI_both_beta_C4',
    #                          'MI_left_alpha_C3', 'MI_left_beta_C3', 'MI_left_alpha_C4', 'MI_left_beta_C4',
    #                          'MI_right_alpha_C3', 'MI_right_beta_C3', 'MI_right_alpha_C4', 'MI_right_beta_C4'])
    #
    # # Bandpower values:
    # if not os.path.isfile(f'{dirname}/bandpower.csv'):
    #     with open(f'{dirname}/bandpower.csv', 'w') as csvfile:
    #         writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    #         writer.writerow(['participant',
    #                          'ME_both_alpha_C3', 'ME_both_beta_C3', 'ME_both_alpha_C4', 'ME_both_beta_C4',
    #                          'ME_left_alpha_C3', 'ME_left_beta_C3', 'ME_left_alpha_C4', 'ME_left_beta_C4',
    #                          'ME_right_alpha_C3', 'ME_right_beta_C3', 'ME_right_alpha_C4', 'ME_right_beta_C4',
    #                          'MI_both_alpha_C3', 'MI_both_beta_C3', 'MI_both_alpha_C4', 'MI_both_beta_C4',
    #                          'MI_left_alpha_C3', 'MI_left_beta_C3', 'MI_left_alpha_C4', 'MI_left_beta_C4',
    #                          'MI_right_alpha_C3', 'MI_right_beta_C3', 'MI_right_alpha_C4', 'MI_right_beta_C4'])
    #
    # for subject in subjects:
    #     results = {}
    #     for task in ['ME', 'MI']:
    #         epochs = mne.read_epochs(f'data/sub-P{subject}/sub-P{subject}_task-{task}-epo.fif', preload=True)
    #         epochs_lap = mne.preprocessing.compute_current_source_density(epochs)
    #         epochs_lap.pick_channels(['C3', 'C4'])
    #
    #         # Alpha band:
    #         epochs_alpha = epochs_lap.copy()
    #         epochs_alpha.filter(8, 13)
    #         epochs_alpha.apply_function(square)
    #
    #         ERDS_alpha_c3, bandpower_alpha_c3 = calc_ERDS(epochs_alpha, 'C3')
    #         print(f'P{subject} {task} alpha ERDS: {ERDS_alpha_c3}')
    #         print(f'P{subject} {task} alpha Bandpower: {bandpower_alpha_c3}')
    #         ERDS_alpha_c4, bandpower_alpha_c4 = calc_ERDS(epochs_alpha, 'C4')
    #         print(f'P{subject} {task} alpha ERDS: {ERDS_alpha_c4}')
    #         print(f'P{subject} {task} alpha Bandpower: {bandpower_alpha_c4}')
    #
    #         # Beta band:
    #         epochs_beta = epochs_lap.copy()
    #         epochs_beta.filter(16, 24)
    #         epochs_beta.apply_function(square)
    #
    #         ERDS_beta_c3, bandpower_beta_c3 = calc_ERDS(epochs_beta, 'C3')
    #         print(f'P{subject} {task} beta ERDS: {ERDS_beta_c3}')
    #         print(f'P{subject} {task} beta Bandpower: {bandpower_beta_c3}')
    #         ERDS_beta_c4, bandpower_beta_c4 = calc_ERDS(epochs_beta, 'C4')
    #         print(f'P{subject} {task} beta ERDS: {ERDS_beta_c4}')
    #         print(f'P{subject} {task} beta Bandpower: {bandpower_beta_c4}')
    #
    #         results[task] = {}
    #         results[task]["ERDS"] = [ERDS_alpha_c3["both"], ERDS_beta_c3["both"], ERDS_alpha_c4["both"], ERDS_beta_c4["both"],
    #                                  ERDS_alpha_c3["left"], ERDS_beta_c3["left"], ERDS_alpha_c4["left"], ERDS_beta_c4["left"],
    #                                  ERDS_alpha_c3["right"], ERDS_beta_c3["right"], ERDS_alpha_c4["right"], ERDS_beta_c4["right"]]
    #         results[task]["bandpower"] = [bandpower_alpha_c3["both"], bandpower_beta_c3["both"], bandpower_alpha_c4["both"], bandpower_beta_c4["both"],
    #                                       bandpower_alpha_c3["left"], bandpower_beta_c3["left"], bandpower_alpha_c4["left"], bandpower_beta_c4["left"],
    #                                       bandpower_alpha_c3["right"], bandpower_beta_c3["right"], bandpower_alpha_c4["right"], bandpower_beta_c4["right"]]
    #
    #     with open(f'{dirname}/erds.csv', 'a') as csvfile:
    #         writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    #         row = [subject] + results["ME"]["ERDS"] + results["MI"]["ERDS"]
    #         writer.writerow(row)
    #
    #     with open(f'{dirname}/bandpower.csv', 'a') as csvfile:
    #         writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    #         row = [subject] + results["ME"]["bandpower"] + results["MI"]["bandpower"]
    #         writer.writerow(row)

    a=0
    # ERDS values for ROIs:
    rois = {"FL": ["Fp1", "F7", "F3"], "FR": ["Fp2", "F4", "F8"],
            "CL": ["FC5", "FC1", "C3", "CP5", "CP1"], "CR": ["FC2", "FC6", "C4", "CP2", "CP6"],
            "PL": ["P7", "P3", "O1"], "PR": ["P4", "P8", "O2"]
            }
    # Generate csv file if not exists:
    if not os.path.isfile(f'{dirname}/erds_rois.csv'):
        with open(f'{dirname}/erds_rois.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            writer.writerow(['participant',
                             'ME_both_alpha_FL', 'ME_both_beta_FL',
                             'ME_left_alpha_FL', 'ME_left_beta_FL',
                             'ME_right_alpha_FL', 'ME_right_beta_FL',
                             'ME_both_alpha_FR', 'ME_both_beta_FR',
                             'ME_left_alpha_FR', 'ME_left_beta_FR',
                             'ME_right_alpha_FR', 'ME_right_beta_FR',
                             'ME_both_alpha_CL', 'ME_both_beta_CL',
                             'ME_left_alpha_CL', 'ME_left_beta_CL',
                             'ME_right_alpha_CL', 'ME_right_beta_CL',
                             'ME_both_alpha_CR', 'ME_both_beta_CR',
                             'ME_left_alpha_CR', 'ME_left_beta_CR',
                             'ME_right_alpha_CR', 'ME_right_beta_CR',
                             'ME_both_alpha_PL', 'ME_both_beta_PL',
                             'ME_left_alpha_PL', 'ME_left_beta_PL',
                             'ME_right_alpha_PL', 'ME_right_beta_PL',
                             'ME_both_alpha_PR', 'ME_both_beta_PR',
                             'ME_left_alpha_PR', 'ME_left_beta_PR',
                             'ME_right_alpha_PR', 'ME_right_beta_PR',
                             'MI_both_alpha_FL', 'MI_both_beta_FL',
                             'MI_left_alpha_FL', 'MI_left_beta_FL',
                             'MI_right_alpha_FL', 'MI_right_beta_FL',
                             'MI_both_alpha_FR', 'MI_both_beta_FR',
                             'MI_left_alpha_FR', 'MI_left_beta_FR',
                             'MI_right_alpha_FR', 'MI_right_beta_FR',
                             'MI_both_alpha_CL', 'MI_both_beta_CL',
                             'MI_left_alpha_CL', 'MI_left_beta_CL',
                             'MI_right_alpha_CL', 'MI_right_beta_CL',
                             'MI_both_alpha_CR', 'MI_both_beta_CR',
                             'MI_left_alpha_CR', 'MI_left_beta_CR',
                             'MI_right_alpha_CR', 'MI_right_beta_CR',
                             'MI_both_alpha_PL', 'MI_both_beta_PL',
                             'MI_left_alpha_PL', 'MI_left_beta_PL',
                             'MI_right_alpha_PL', 'MI_right_beta_PL',
                             'MI_both_alpha_PR', 'MI_both_beta_PR',
                             'MI_left_alpha_PR', 'MI_left_beta_PR',
                             'MI_right_alpha_PR', 'MI_right_beta_PR'])

    results = {}
    for subject in subjects:
        for task in ['ME', 'MI']:
            results[task] = {}
            epochs_orig = mne.read_epochs(f'data/sub-P{subject}/sub-P{subject}_task-{task}-epo.fif', preload=True)
            for roi in rois.keys():
                epochs = epochs_orig.copy()
                epochs.pick_channels(rois[roi])

                # Alpha band:
                epochs_alpha = epochs.copy()
                epochs_alpha.filter(8, 13)
                epochs_alpha.apply_function(square)

                ERDS_alpha, _ = calc_ERDS(epochs_alpha, avg_roi=True)
                print(f'P{subject} {task} {roi} alpha ERDS: {ERDS_alpha}')

                # Beta band:
                epochs_beta = epochs.copy()
                epochs_beta.filter(16, 24)
                epochs_beta.apply_function(square)

                ERDS_beta, _ = calc_ERDS(epochs_beta, avg_roi=True)
                print(f'P{subject} {task} {roi} beta ERDS: {ERDS_beta}')

                results[task][roi] = [ERDS_alpha["both"], ERDS_beta["both"],
                                      ERDS_alpha["left"], ERDS_beta["left"],
                                      ERDS_alpha["right"], ERDS_beta["right"]]

        with open(f'{dirname}/erds_rois.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            row = [subject] + \
                  results["ME"]["FL"] + results["ME"]["FR"] + results["ME"]["CL"] + results["ME"]["CR"] + results["ME"]["PL"] + results["ME"]["PR"] + \
                  results["MI"]["FL"] + results["MI"]["FR"] + results["MI"]["CL"] + results["MI"]["CR"] + results["MI"]["PL"] + results["MI"]["PR"]
            writer.writerow(row)


    # # ERDS Topoplots:
    # for subject in subjects:
    #     for task in ['MI', 'ME']:
    #         epochs = mne.read_epochs(f'data/sub-P{subject}/sub-P{subject}_task-{task}-epo.fif')
    #         epochs = epochs.pick_types(eeg=True)
    #         # plot_erds_topoplot(epochs, fig_title=f'ERDS Topoplots - {task}', show=True)
    #         fig = plot_erds_topoplot(epochs, fig_title=f'ERDS Topoplots - {task}')
    #         fig.savefig(f'images/sub-P{subject}/P{subject}_{task}_erds_topoplots.png')
