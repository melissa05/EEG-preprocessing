# EEGAnalysis class

This class is intended to perform in a quick way the first steps of the EEG preprocessing. 
The class implements the main preprocessing steps that can be applied to an EEG signal

* data loading 
* spatial filtering
* temporal filtering
* signal segmentation 

and some methods for the signal characteristics visualization

* raw visualization
* power spectral density
* mean epoch amplitude
* ERDS maps

The methods are implemented thanks to Python 3.7 and MNE package. 

## EEGAnalysis

### How to run

The repository can be cloned in the computer and the requirements can be found in `requirements.txt` file.

Have a look at the documentation (`eeg-preprocessing.pdf`) to understand what the code can do and how.

To run the whole analysis, it's necessary to create an instance of the class and to call a run function. 
The constructor of the class takes in input the path to the .xdf file and a dictionary containing the following 
information (in quotes the keys of the different data are reported):

* "lsl-version": string for the lsl version used for the acquisition. It can be "1.12" or "1.16"
* "streams": a dictionary containing the names of the streams of interest contained in the xdf file
    * "EEGMarkers": name of the marker stream sent by the EEG acquisition device
    * "EEGData": name of the EEG stream sent by the EEG acquisition device
    * "Triggers": name of the marker stream sent by the program used for the paradigm
* "montage": string containing the name of the montage used for the EEG acquisition
* "filtering": dictionary containing the frequencies for the temporal filtering (if one value is None, that filtering is not performed)
    * "low": low bound for the band-pass filtering
    * "high": high bound for the band-pass filtering
    * "notch": frequency for the notch filtering
* "spatial_filtering": string containing the name of the type of spatial filtering to be performed. It can be 'average' or 'REST'. It can also contain a list of string containing the name(s) of the channel(s) used to construct the reference 
* "samples_remove": int value for the number of samples to be removed at the beginning and at the end of the raw signal
* "t_min": time instant (in seconds) of the beginning of the epochs with respect to the marker
* "t_max": time instant (in seconds) of the end of the epochs with respect to the marker
* "full_annotation": 0 if just the last part of the annotation is considered (in case it's of type "value1/value2"), 1 if the whole annotation should be considered
* "annotation_durations": dictionary of the annotations (keys) and their durations in seconds (values)
* "epochs_reject_criteria": dictionary containing the threshold values for the epoch rejection according to the peak-to-peak amplitude
    * "eeg": amplitude for rejection according to eeg amplitude
    * "eog": amplitude for rejection according to eog amplitude
* "rois": dictionary containing the regions of interest to be applied. The key must contain the name of the roi, the value the list of the channels name contained in the roi
* "bad_epoch_names": list containing the annotation names of the epochs to be excluded
* "erp": list containing the time instants (in milliseconds) to be highlighted with a vertical line in ERP plot
* "erds": list containing the lower and the higher frequencies of the frequency range to be visualized in ERDS plots 
* "bad_channels": dict containing keys correspondent to the participant code, values containing a list of channels labelled as bad

The dictionary can be previously saved in a json file and loaded every time it is needed. An example of the json file is given by `info_example.json`.
All the generated plots will be saved in `images/` folder, inside a sub-folder named 'subj-X', where X is the name of the participant extracted from the input path.  

An example of the usage of the class can be found in `example.py`. To run the script it's necessary to download the data file at this [link](https://drive.google.com/file/d/1QJACAUq3nOzYe69RH_6mlofJCYqt4ZJf/view?usp=sharing) and save it in `data/eeg/`.

### Implemented functions

The following methods have been implemented inside the class:
* `get_info_from_path`: to extract the main information from the filepath
* `load_xdf`: to properly read the streams and the info of xdf files
* `load_channels`: to read the list of channels names from the EEG data stream
* `fix_lost_samples`: to try to fix the data adding fake samples where missing
* `create_raw`: to generate MNE Raw instance for the EEG data
* `visualize_raw`: to visualize the signal and the psd of the loaded Raw data
* `raw_spatial_filtering`: to perform spatial filtering
* `raw_time_filtering`: to perform frequency filtering
* `raw_ica_remove_eog`: to perform ICA and reject bad components
* `create_annotations`: to extract the Annotation instance from marker data
* `create_epochs`: to segment the signal according to the defined annotations (and eventually roi)
* `visualize_epochs`: to generate epochs plots
* `create_evoked`: to generate the Evoked for the different annotations (and eventually roi)
* `visualize_evoked`: to plot the generated evoked
* `get_peak`: to extract peak amplitude in a time window of interest
* `save_pickle`: to save the data of the current acquisition divided into signals, labels and info

The first four methods are automatically called by the class constructor. The other methods can be called individually 
or can be called thanks to the following functions:
* `run_raw_epochs`: to perform the whole analysis (raw creation, filtering, annotation defining, epochs division, evoked definition and pickle saving)
* `run_raw`: to perform the analysis just regarding the raw (raw creation, filtering, annotation defining)
* `run_combine_raw_epochs`: to perform the loading of a file, creating the correspondent filtered Raw, generate the annotations and append to it other raw files given in input. The remaining analysis is performed on the whole concatenated signal

### Work in progress

* Spatial filtering with Laplacian
* ICA for EOG
* fix the problem of missing samples in the acquisition
* check on the input values
* Proper documentation

## ERDS

The code for ERD/S maps can be found in `src/ERDS.py`. The function takes in input:
* epochs: Epochs MNE variable of interest
* rois: dict object containing a set of key-value pairs. The key must be the name of the region of interest, the value is a list containing the channels belonging to that ROI
* fs: sampling frequency of the EEG acquisition
* t_min: time instant of the epoch (with respect to the stimuli instant)
* f_min: minimum frequency for which the ERDS maps are visualized (50 Hz by default)
* f_max: maximum frequency for which the ERDS maps are visualized (50 Hz by default)
* path: folder path where to save the computed ERDS maps. If None (default), the maps are just shown

ERDS maps are also implemented for epochs in numpy matrix (#epochs x #channels x #samples) variables and annotations in numpy arrays (#epochs).

## Main solution to numpy-MNE problems

When running the code, if you get the following error: 

``mean() gor an unexpected keyword argument 'keepdims'``

it's due to incompatibility problems between MNE and numpy. To fix it, you can add a cast to np.array to the variable on which the mean function is called inside the library. 

If you get instead the following error: 

``all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has 2 dimension(s)``

you can fix the problem substituting inside the library the call of np.concatenate with the following one 

``np.concatenate([l_z_pad, np.array(2* x[0] - x[n_pad[0]:0:-1]).flatten(), np.array(x).flatten(), np.array(2 * x[-1] - x[-2:-n_pad[1] - 2:-1]).flatten(), r_z_pad])``


## Contacts

For any problem, suggestion or help, feel free to contact me at the following email address:

giulia.pezzutti@studenti.unipd.it