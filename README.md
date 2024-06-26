# eegClassification

Classification of EEG data based on the Kaggle competition [HMS - Harmful Brain Activity Classification
](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview)


<img src="https://github.com/AmyRouillard/eegClassification/blob/main/files/eFig2.png" width="700">
Fig. 1: Examples of the data as seen by the labeling expert.

## Overview

The dataset provided is a collection of EEG data and spectrograms. The goal is to classify the EEG data into one of six classes. The data is collected from patients and annotated by experts.

There are six harmful brain activity patterns of interest: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or “other”.

The goal is to predict the probability of each of the six classes given 50-second window of EEG data and 10-minute window of spectrogram data centered at the 50-second window. Several experts were asked label the central 10 seconds of the EEG sample, see Fig. 1 for examples. 

The number of experts voting on each window varies, see Fig. 2. The distribution of the class labels is balanced, see Fig. 3.

<img src="https://github.com/AmyRouillard/eegClassification/blob/main/files/vote_dist.png" width="750">

Fig. 2: Distribution of the number of expert votes for the training, validation and test sets.

<img src="https://github.com/AmyRouillard/eegClassification/blob/main/files/class_dist.png" width="750">

Fig. 3: Distribution of the class labels for the training, validation and test sets.


## EEG dataset

The EEG data records the electrical activity of the brain at serval points on the scalp. Signals are collected using several electrodes, see Fig. 4, and the four spectrograms are constructed from 4 different regions of the scalp as follows: Left lateral (Fp1, F7, T3, T5, O1); Right lateral (Fp2, F8, T4, T6, O2); Left Parasagittal (Fp1, F3, C3, P3, O1); Right Parasagittal (Fp2, F4, C4, P4, O2). 

The spectrograms are constructed from EEG data using [multitaper spectral estimation](https://en.wikipedia.org/wiki/Multitaper)[1], and represent a visualization of the Fourier spectrum of the EEG signals over time. 

<img src="https://github.com/AmyRouillard/eegClassification/blob/main/files/eegmelb.gif" width="400">

Fig. 4: EEG electrode placements[2]

### Data format

After processing the raw data the following features are available for training.

#### Features

* `*data*_spec_*` - four arrays of shape (299, 100) - cropped from the raw spectrogram
* `*_eeg_*` - twenty arrays of shape (9800,) - cropped from the raw EEG data
* `*_eeg_spec_*` - twenty arrays of shape (129, 43) - spectrogram of the the cropped EEG data. These are sometimes cropped to a 10 second window so that the shape become (129,7)

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
|**`pipeline.py`**| | Training pipeline used to train batches of models for comparison. |
|**`inference.py`**| | Inference pipeline. |
|**`utils/`**| |Utility files used across the project. Contains helper functions for training and inference. |
|| `utils/CustomDataset` | Custom Pytorch dataset used to pre-process that data.  |
|| `utils/CustomDatasetNPY` | Custom Pytorch dataset used to load -preprocess that data from .npy files.  |
|| `utils/model_architectures.py` | Custom pytorch CNN actictectures used.   |
|| `utils/ultils` | Helper functions.   |
||||
|**`notebooks/EDA/`**| | Exploratory data analysis and planning for data preprocessing. |
|**`notebooks/Data processing/`**| |Preprocessing  of the data. |
|**`notebooks/Model training/`**| | Notebooks draft for model training. |
|**`notebooks/Inference/`**| | Notebooks draft for model inference. |
||||
|**`files/`**|| Folder containing output files. |
|| `*_processed.csv`| Meta data corresponding to the train, validation and test splits created from full training data. (depreciated) |
||`*.png`,`*.gif`| Images for the readme. |
||||
|**`sample_data/`**|| Competition data, excluding full training data. |
|| `*_eegs/` | Eeg files, a sample of the full dataset. Contains parquet files.  |
|| `*_spectrograms/` | Spectrogram files, a sample of the full dataset.  Contains parquet files. |
|| `train.csv` | Metadata for the full training set.  |
|| `sample_submission.csv` | Sample submission file.   |
|| `test.csv` | Example metadata for the test set, corresponds to  `sample_submission.csv` |
||||
|**`models/`** ||Folder containing information about the trained models. |
||||
||||

## Tasks

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

- Is there any missing data (NaN etc?)
- Rescaling? (e.g. min-max scaling, log scaling for spectrogram data)
	- log scaling for spectrogram data :heavy_check_mark: 
	- min-max scaling/standard :heavy_check_mark: 
- Understand Kullback-Leibler Divergence :heavy_check_mark: 
- Apply preprocessing pipeline to all data :heavy_check_mark: 
- Batch the data  :heavy_check_mark: 
- clean up readme :heavy_check_mark: 
- train scalers on 1%-2% of the data (107000 samples) :heavy_check_mark:
- update markdown in notebooks, documentation  :heavy_check_mark:

- compute an estimate of the KL divergence given that 1/6 is predicted for each class. A score worse than this, is worse than guessing. :heavy_check_mark:
- Baseline model (random forest?) - XGBoost on features generated by pretrained model :heavy_check_mark:
- Transfer learning (resnet50, imagenet, dino)  
- Augmentation :heavy_check_mark:
- Learning scheduler :heavy_check_mark:

- Custom CNN architecture :heavy_check_mark:
- model ensemble :heavy_check_mark:

### Resources:

https://www.kaggle.com/code/awsaf49/hms-hbac-kerascv-starter-notebook
https://www.tensorflow.org/api_docs/python/tf/tile
https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b2.html
https://keras.io/api/keras_cv/models/backbones/efficientnetv2/

https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
https://pytorch.org/vision/main/models/efficientnet.html


## Refereces

[1] Zafar, S.F., Amorim, E., Williamsom, C.A., Jing, J., Gilmore, E.J., Haider, H.A., Swisher, C., Struck, A., Rosenthal, E.S., Ng, M. and Schmitt, S., 2020. A standardized nomenclature for spectrogram EEG patterns: inter-rater agreement and correspondence with common intensive care unit EEG patterns. Clinical Neurophysiology, 131(9), pp.2298-2306.

[2] https://paulbourke.net/dataformats/eeg/

