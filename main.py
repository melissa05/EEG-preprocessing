import json
import os
import matplotlib
import csv
import mne

from src.EEGAnalysis import EEGAnalysis


if __name__ == '__main__':
    subjects = ['001', '002', '003', '004', '005', '007', '008', '009', '010', '011', '012', '013', '014',
                '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028',
                '029', '030', '031']

    matplotlib.use('TKAgg')

    dict_info = json.load(open('data/sourcedata/info.json'))

    # participant = '023'
    # task = 'MI'
    # task = 'ME'
    # ica_mode = 'inspect'
    ica_mode = 'apply'

    for participant in subjects:
        print(f'Subject: P{participant}')
        for task in ['ME', 'MI']:

            eeg_source_path = f'data/sourcedata/sub-{participant}/eeg'
            eegfiles = [f for f in os.listdir(eeg_source_path)
                        if os.path.isfile(os.path.join(eeg_source_path, f)) and f[-4:] == '.xdf' and f'task-{task}' in f]
            paths = [os.path.join(eeg_source_path, f) for f in eegfiles]

            if participant in '018':
                eeg = EEGAnalysis(paths, dict_info, per_run_inspection=True)  # Load raw data without any pre-processing
            else:
                eeg = EEGAnalysis(paths, dict_info, per_run_inspection=False)  # Load raw data without any pre-processing

            if ica_mode == 'inspect':
                eeg.visualize_eeg()
                eeg.raw_inspect_ica_components()
            elif ica_mode == 'apply':
                # Creating the csv file to keep track of dropped epochs:
                dirname = os.path.join(eeg.file_info['project_folder'], 'data')
                if not os.path.isfile(f'{dirname}/dropped_events.csv'):
                    with open(f'{dirname}/dropped_events.csv', 'w') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
                        writer.writerow(['participant', 'task', 'both', 'left', 'right', 'total'])  # Header (first row)

                # eeg.preprocess_raw(visualize_raw=True, ica_analysis=True, reference='average')  # Perform pre-processing (filtering, ICA).
                # eeg.preprocess_raw(visualize_raw=False, ica_analysis=True, reference='average')  # Perform pre-processing (filtering, ICA).
                # eeg.create_epochs(visualize_epochs=False)  # Create epochs with epoch rejection.
                # eeg.save_epochs(track_dropped=True)
