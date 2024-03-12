# eegClassification
Classification of EEG data based on the Kaggle competition [HMS - Harmful Brain Activity Classification
](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview)


<img src="https://github.com/AmyRouillard/eegClassification/blob/main/files/eFig2.png" width="700">
Fig. 1: Examples of the data as seen by the labeling expert.

## Overview

There are six harmful brain activity patterns of interest: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or “other”.

The goal is to predict the probability of each of the six classes give 50-second window of EEG data and spectrogram data corresponding to 10 mins centered around the 50-second window. Several experts were asked to  provided a label (vote for a single classification) for the central 10 seconds of the EEG sample, see Fig. 1 for examples. 

One of the challenges of this dataset is that the number of experts voting on each window varies, see Fig. 2. The distribution of the class labels is balanced, see Fig. 3.

<img src="https://github.com/AmyRouillard/eegClassification/blob/main/files/vote_dist.png" width="750">

Fig. 2: Distribution of the number of expert votes for the training, validation and test sets.

<img src="https://github.com/AmyRouillard/eegClassification/blob/main/files/class_dist.png" width="750">

Fig. 3: Distribution of the class labels for the training, validation and test sets.


## EEG dataset

The data provided includes the raw EEG data and spectrograms. The EEG data records the electrical activity of the brain. The samples provided record a 50 second window, while the spectrograms cover a 10 minute window. 

The spectrograms are constructed from EEG data using [multitaper spectral estimation](https://en.wikipedia.org/wiki/Multitaper)[1], and represent a visualization of the Fourier spectrum of the EEG signals over time. 

The EEG data is collected using several electrodes, see Fig. 4, and the four spectrograms are constructed from 4 different regions of the scalp as follows: Left lateral (Fp1, F7, T3, T5, O10); Right lateral (Fp2, F8, T4, T6, O2); Left Parasagittal (Fp1, F3, C3, P3, O1); Right Parasagittal (Fp2, F4, C4, P4, O2). 

<img src="https://github.com/AmyRouillard/eegClassification/blob/main/files/eegmelb.gif" width="400">

Fig. 4: EEG electrode placements[2]

### Data format

After processing the raw data the following features are available for training. In addition to the training data, a single test sample is supplied for testing code before making a submission to the competition.

#### Target

* `class_prob` - list of length 6, probability of each class, see `labels`

#### Features

* `data_spec_*` - four arrays of shape (299, 100) - cropped from the raw spectrogram
* `data_eeg_*` - twenty arrays of shape (9800,) - cropped from the raw EEG data
* `data_eeg_*_spec` - twenty arrays of shape (129, 43) - spectrogram of the the cropped EEG data

#### Metadata (deprecated)

* `spec_freq (Hz)` - array of shape (100,) contain the frequency values for the spectrogram
* `spec_time (s)` - array of shape (299,) contain the time values for the spectrogram
* `eeg_time (s)` - array of shape (9800,) contain the time values for the EEG data
* `freq` - tuple of (eeg frequency, spectrogram frequency ) = (0.5 Hz, 200 Hz)
* `class_label` - string, consensus label for the central 10 seconds given by the class with the most votes. 
* `class_votes` - list of votes for each class, length 6.
* `class_probs` - list of probabilities for each class, length 6
* `labels` - list ['seizure_vote', 'lpd_vote','gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote'].


## Contents

|Folder | File |Description|
|--|--|--|
|**`utils/`**| |Utility files used across the project. |
|| `utils/CustomDataset` | Custom Pytorch dataset class with example of creating a data loader.  |
||||
|**`notebooks/EDA/`**| | Exploratory data analysis and planning for data preprocessing. |
||`first_look.ipynb`|  First look at the data. Creation of train, validation and test splits. |
||`EDA_votes.ipynb`| Exploratory data analysis of the class distributions.  |
||`Working_with_spectrograms.ipynb`| Exploring the spectrogram data.  |
||`preprocessing.ipynb`| First draft of preprocessing pipeline, streamlined in `notebooks/Data processing/`. Consistency of the data confirmed and frequencies of data collection checked. |
|**`notebooks/Data processing/`**| |Preprocessing the data. |
||`data_scaling.ipynb`| Train a standard scaler and a min max scaler using partial fit on batch data.  |
|**`notebooks/Models/`**| | Notebooks for model training. |
||`.ipynb`|   |
||||
|**`files/`**|| Folder containing output files. |
|| `*_processed.csv`| Meta data corresponding to the train, validation and test splits created from full training data. |
||`*.png`,`*.gif`| Images for the readme. |
||`train_val_test_info_dicts`| Pickled dictionaries to be passed to the dataloader split into train, validation and test sets according to  `*_processed.csv`. |
||||
|**`sample_data/`**|| Competition data, excluding full training data. |
|| `*_eegs/` | Eeg files, a sample of the full dataset. Contains parquet files.  |
|| `*_spectrograms/` | Spectrogram files, a sample of the full dataset.  Contains parquet files. |
|| `train.csv` | Metadata for the full training set.  |
|| `sample_submission.csv` | Sample submission file.   |
|| `test.csv` | Example metadata for the test set, corresponds to  `sample_submission.csv` |
||||
|**`models/`** ||Folder containing trained models. |
||||
||||

## Current tasks

### Week 1:
- Explorator data analysis
	- labels distribution (expect consensus)  :heavy_check_mark: 
	- EEG data format  :heavy_check_mark:
	- view spectrogram(s) (uniform size/missing data?)  :heavy_check_mark: 
	- check test data if it has the same format as train data :heavy_check_mark:
- Clarify goal (classify 50 second window or a whole EEG)  :heavy_check_mark: 
- Compute the KL score for a given annotator (if possible) - unfortunately, the data is not available  :heavy_check_mark:
- Create a Train/Validation/Test split with no data leakage (group by eeg_id)  :heavy_check_mark:
- Create preprocessing pipeline :heavy_check_mark:
- check preprocessing pipeline works on Kaggle test sample for submission :heavy_check_mark:

### Week 2:
- Is there any missing data (NaN etc?)
- Rescaling? (e.g. min-max scaling, log scaling for spectrogram data)
	- log scaling for spectrogram data :heavy_check_mark: 
	- min-max scaling/standard :heavy_check_mark: 
- Understand Kullback-Leibler Divergence :heavy_check_mark: 
- Apply preprocessing pipeline to all data :heavy_check_mark: 
- Batch the data  :heavy_check_mark: 
- clean up readme :heavy_check_mark: 
- train scalers on 1%-2% of the data (107000 samples) :heavy_check_mark:
- update markdown in notebooks, documentation 

### Week 3:
- compute an estimate of the KL divergence given that 1/6 is predicted for each class. A score worse than this, is worse than guessing.
- Baseline model (random forest?)
- Make a first submission to leader board  
- Transfer learning (resnet50, imagenet, dino)  
	- Data preparation
	- Augmentation
	- Learning scheduler

### Week 4:

- Explainablitiy

## Refereces

[1] Zafar, S.F., Amorim, E., Williamsom, C.A., Jing, J., Gilmore, E.J., Haider, H.A., Swisher, C., Struck, A., Rosenthal, E.S., Ng, M. and Schmitt, S., 2020. A standardized nomenclature for spectrogram EEG patterns: inter-rater agreement and correspondence with common intensive care unit EEG patterns. Clinical Neurophysiology, 131(9), pp.2298-2306.
[2] https://paulbourke.net/dataformats/eeg/