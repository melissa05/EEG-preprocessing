"""
Custom to my (Melissa) data.
"""


import os
import pyxdf


def sanity_check(task, eeg_source_path):
    """
    Performs basic sanity checks on the raw data of the Handedness MI study.

    Checks performed are:

    * 3 different cues in markers
    * 30 trials in total per file
    * right number of channels (33 for BrainAmp, 36 for LiveAmp) in EEG data
    * effective sampling rate does not deviate more than 0.03 Hz from 500 Hz

    :param task: The task performed during the runs. Either 'ME' or 'MI'.
    :type task: str
    :param eeg_source_path: Path to directory which contains xdf files of raw data of one participant.
    :type eeg_source_path: str
    :return: True if no issues were detected, else False.
    :rtype: bool
    """
    # Expectations per input (xdf) file:
    marker_channel_name = 'PsychoPy'
    expected_n_classes = 3
    expected_n_trials = 30
    expected_n_channels = 33

    # Find filenames of all files of given participant (path) and task:
    xdf_files = [f for f in os.listdir(eeg_source_path)
                 if os.path.isfile(os.path.join(eeg_source_path, f)) and f[-4:] == '.xdf' and f'task-{task}' in f]

    # Go over each file and perform basic checks:
    all_good = True
    for i, f in enumerate(xdf_files):
        data, header = pyxdf.load_xdf(os.path.join(eeg_source_path, f))

        # Check markers:
        markers = data[next((j for (j, d) in enumerate(data) if d['info']['name'] == [marker_channel_name]), None)]
        marker_types = set(cue[0] for cue in markers['time_series'])
        # Check number of different cues:
        if len(marker_types) != expected_n_classes:
            print(f"Warning: Expected {expected_n_classes} different cues, but found "
                  f"{len(tuple(markers['time_series']))} in file {f}.")
            all_good = False
        # Check number of trials:
        if len(markers['time_series']) != expected_n_trials:
            print(f"Warning: Expected {expected_n_trials} trials, but found {len(markers['time_series'])} "
                  f"in file {f}.")
            all_good = False

        # Check for lost samples (RDA markers):
        markers_rda = data[next((j for (j, d) in enumerate(data) if d['info']['name'] == ['BrainVision RDA Markers']),
                                  None)]
        n_lost_samples = len(markers_rda['time_stamps'])
        if n_lost_samples > 0:
            print(f"Warning: {n_lost_samples} samples were lost in file {f}.")
            all_good = False

        # Check eeg:
        eeg_data = data[next((i for (i, d) in enumerate(data) if d['info']['name'] == ['BrainVision RDA']), None)]
        # Number of channels (info):
        if eeg_data['info']['channel_count'][0] != f'{expected_n_channels}':
            print(f"Warning: Expected number of channels is {expected_n_channels}, but is "
                  f"{eeg_data['info']['channel_count'][0]} in file {f}.")
            all_good = False
        # Number of channels (time series):
        if eeg_data['time_series'].shape[1] != expected_n_channels:
            print(f"Warning: Expected number of channels is {expected_n_channels}, but found "
                  f"{eeg_data['time_series'].shape[1]} in the time series of file {f}.")
            all_good = False
        # Sampling rate:
        print(f"Effective sampling rate is {eeg_data['info']['effective_srate']} Hz.")
        if abs(eeg_data['info']['effective_srate'] - 500) >= 0.03:
            print(f"Warning: Expected a sampling rate of 500 Hz, but is {eeg_data['info']['effective_srate']}.")
            all_good = False

    return all_good


if __name__ == '__main__':
    # subjects = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014',
    #             '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027']
    subjects = []
    tasks = ['ME', 'MI']

    for subject in subjects:
        eeg_source_path = f'../data/sourcedata/sub-{subject}/eeg'
        for task in tasks:
            print(f"Checking participant {subject}, task {task} ...")
            if sanity_check(task=task, eeg_source_path=eeg_source_path):
                print(f"All good for participant {subject} and task {task}.\n")
            else:
                print(f"Watch out! Something wrong with participant {subject} and task {task}.\n")
