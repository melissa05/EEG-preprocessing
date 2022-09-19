import os
import csv
import mne


def square(arr):
    return arr ** 2


def calc_ERDS(epochs, channel):
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
            R = epochs_base.get_data(picks=[channel], item=e).mean()
            A = epochs_active.get_data(picks=[channel], item=e).mean()
            A_sum += A
            ERDS_sum += (A - R) / R
        ERDS = ERDS_sum / epochs_event.events.shape[0]
        A = A_sum / epochs_event.events.shape[0]

        # R_sum = epochs_base.get_data(picks=[channel], item=0)
        # A_sum = epochs_active.get_data(picks=[channel], item=0)
        # for e in range(1, epochs_event.events.shape[0]):
        #     R_sum += epochs_base.get_data(picks=[channel], item=e)
        #     A_sum += epochs_active.get_data(picks=[channel], item=e)
        # R = R_sum.mean() / epochs_event.events.shape[0]
        # A = A_sum.mean() / epochs_event.events.shape[0]
        # ERDS = (A - R) / R

        ERDS_dict[event] = ERDS * 100
        bandpower_dict[event] = A
    return ERDS_dict, bandpower_dict


if __name__ == '__main__':
    subjects = ['001', '002', '003', '004', '005', '007', '008', '009', '010', '011', '012', '013', '014',
                '015', '016', '017', '018', '019', '021', '022', '023', '024', '025', '026', '027', '028',
                '029', '030', '031']  # Without 006 and 020
    # subjects = ['001', '002']

    # Creating the csv file to keep track of dropped epochs:
    dirname = 'C:\\Users\\Melissa\\PycharmProjects\\EEG-preprocessing\\data'
    if not os.path.isfile(f'{dirname}/erds.csv'):
        with open(f'{dirname}/erds.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            writer.writerow(['participant',
                             'ME_both_alpha_C3', 'ME_both_beta_C3', 'ME_both_alpha_C4', 'ME_both_beta_C4',
                             'ME_left_alpha_C3', 'ME_left_beta_C3', 'ME_left_alpha_C4', 'ME_left_beta_C4',
                             'ME_right_alpha_C3', 'ME_right_beta_C3', 'ME_right_alpha_C4', 'ME_right_beta_C4',
                             'MI_both_alpha_C3', 'MI_both_beta_C3', 'MI_both_alpha_C4', 'MI_both_beta_C4',
                             'MI_left_alpha_C3', 'MI_left_beta_C3', 'MI_left_alpha_C4', 'MI_left_beta_C4',
                             'MI_right_alpha_C3', 'MI_right_beta_C3', 'MI_right_alpha_C4', 'MI_right_beta_C4'])
    if not os.path.isfile(f'{dirname}/bandpower.csv'):
        with open(f'{dirname}/bandpower.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            writer.writerow(['participant',
                             'ME_both_alpha_C3', 'ME_both_alpha_C4', 'ME_both_beta_C3', 'ME_both_beta_C4',
                             'ME_left_alpha_C3', 'ME_left_alpha_C4', 'ME_left_beta_C3', 'ME_left_beta_C4',
                             'ME_right_alpha_C3', 'ME_right_alpha_C4', 'ME_right_beta_C3', 'ME_right_beta_C4',
                             'MI_both_alpha_C3', 'MI_both_alpha_C4', 'MI_both_beta_C3', 'MI_both_beta_C4',
                             'MI_left_alpha_C3', 'MI_left_alpha_C4', 'MI_left_beta_C3', 'MI_left_beta_C4',
                             'MI_right_alpha_C3', 'MI_right_alpha_C4', 'MI_right_beta_C3', 'MI_right_beta_C4'])

    for subject in subjects:
        print(f'Subject: P{subject}')
        results = {}
        for task in ['ME', 'MI']:
            epochs = mne.read_epochs(f'data/sub-P{subject}/sub-P{subject}_task-{task}-epo.fif', preload=True)
            epochs_lap = mne.preprocessing.compute_current_source_density(epochs)
            epochs_lap.pick_channels(['C3', 'C4'])

            # Alpha band:
            epochs_alpha = epochs_lap.copy()
            epochs_alpha.filter(8, 14)
            epochs_alpha.apply_function(square)

            ERDS_alpha_c3, bandpower_alpha_c3 = calc_ERDS(epochs_alpha, 'C3')
            print(f'P{subject} {task} alpha ERDS: {ERDS_alpha_c3}')
            print(f'P{subject} {task} alpha Bandpower: {bandpower_alpha_c3}')
            ERDS_alpha_c4, bandpower_alpha_c4 = calc_ERDS(epochs_alpha, 'C4')
            print(f'P{subject} {task} alpha ERDS: {ERDS_alpha_c4}')
            print(f'P{subject} {task} alpha Bandpower: {bandpower_alpha_c4}')

            # Beta band:
            beta = epochs_lap.copy()
            beta.filter(16, 24)
            beta.apply_function(square)

            ERDS_beta_c3, bandpower_beta_c3 = calc_ERDS(beta, 'C3')
            print(f'P{subject} {task} beta ERDS: {ERDS_beta_c3}')
            print(f'P{subject} {task} beta Bandpower: {bandpower_beta_c3}')
            ERDS_beta_c4, bandpower_beta_c4 = calc_ERDS(beta, 'C4')
            print(f'P{subject} {task} beta ERDS: {ERDS_beta_c4}')
            print(f'P{subject} {task} beta Bandpower: {bandpower_beta_c4}')

            results[task] = {}
            results[task]["ERDS"] = [ERDS_alpha_c3["both"], ERDS_alpha_c4["both"], ERDS_beta_c3["both"], ERDS_beta_c4["both"],
                                       ERDS_alpha_c3["left"], ERDS_alpha_c4["left"], ERDS_beta_c3["left"], ERDS_beta_c4["left"],
                                       ERDS_alpha_c3["right"], ERDS_alpha_c4["right"], ERDS_beta_c3["right"], ERDS_beta_c4["right"]]
            results[task]["bandpower"] = [bandpower_alpha_c3["both"], bandpower_alpha_c4["both"], bandpower_beta_c3["both"], bandpower_beta_c4["both"],
                                            bandpower_alpha_c3["left"], bandpower_alpha_c4["left"], bandpower_beta_c3["left"], bandpower_beta_c4["left"],
                                            bandpower_alpha_c3["right"], bandpower_alpha_c4["right"], bandpower_beta_c3["right"], bandpower_beta_c4["right"]]

        with open(f'{dirname}/erds.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            row = [subject] + results["ME"]["ERDS"] + results["MI"]["ERDS"]
            writer.writerow(row)

        with open(f'{dirname}/bandpower.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            row = [subject] + results["ME"]["bandpower"] + results["MI"]["bandpower"]
            writer.writerow(row)
