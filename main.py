import json
import os
import matplotlib
import csv
import mne

from src.EEGAnalysis import EEGAnalysis


if __name__ == '__main__':

    matplotlib.use('TKAgg')

    dict_info = json.load(open('data/sourcedata/info.json'))

    # Load multiple files:
    participant = '031'
    task = 'MI'
    # task = 'ME'
    # ica_mode = 'inspect'
    ica_mode = 'apply'

    eeg_source_path = f'data/sourcedata/sub-{participant}/eeg'
    eegfiles = [f for f in os.listdir(eeg_source_path)
                if os.path.isfile(os.path.join(eeg_source_path, f)) and f[-4:] == '.xdf' and f'task-{task}' in f]
    paths = [os.path.join(eeg_source_path, f) for f in eegfiles]

    eeg = EEGAnalysis(paths, dict_info, per_run_inspection=False)  # Load raw data without any pre-processing

    if ica_mode == 'inspect':
        eeg.visualize_eeg()
        eeg.raw_inspect_ica_components()
    elif ica_mode == 'apply':
        eeg.preprocess_raw(visualize_raw=True, ica_analysis=True, reference='average')  # Perform pre-processing (filtering, ICA).
        # eeg.preprocess_raw(visualize_raw=False, ica_analysis=True, rereference='average)  # Perform pre-processing (filtering, ICA).
        eeg.create_epochs(visualize_epochs=False)  # Create epochs with epoch rejection.
        eeg.save_epochs()
    a = 0

    # eeg_target_path = f'data/sub-{participant}/eeg'
    #
    # os.makedirs(eeg_target_path, exist_ok=True)
    #
    # mne.export.export_raw(fname=os.path.join(eeg_target_path, f'sub-{participant}_task-{task}_eeg.edf'),
    #                       raw=eeg.raw,
    #                       overwrite=True)
    #
    # with open(f'{eeg_target_path}/sub-{participant}_task-{task}_events.csv', 'w') as tsvfile:
    #     writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
    #     writer.writerow(['onset', 'duration', 'value', 'stim_file'])
    #     for i, cue in enumerate(eeg.raw.annotations.description):
    #         if cue in ['left', 'right', 'both']:
    #             writer.writerow([eeg.raw.annotations.onset[i], 7, cue, f'{cue}.jpg'])
