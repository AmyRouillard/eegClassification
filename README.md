# eegClassification
Classification of EEG data based on the Kaggle competition [HMS - Harmful Brain Activity Classification
](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview)

## Overview

There are six patterns of interest: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or “other”.

The goal is to predict the probability of each of the six classes for each 50-second window of EEG data.

One of the challenges is that the number of experts voting on each window varies. 



## Contents

|Folder/File |Description|
|--|--|
|**`notebooks/`**|  Python notebooks. |
|`notebooks/first_look.ipynb`|  First look at the data. Creation of train, validation and test splits. |
|`notebooks/EDA_votes.ipynb`| Exploratory data analysis of the class distributions.  |
|`notebooks/Working_with_spectrograms.ipynb`|   |
|||
|**`files/`**| Folder containing output files. |
| `files/*_processed.csv`| Meta data corresponding to the train, validation and test splits created from training data. Some postprocessing is done, vote are converted to probabilities, total votes and file path to parquet files are recorded. |
|||
|**`sample_data/`**| Competition data, excluding full training data. |
| `sample_data/train_eegs` | One EEG from the training data form each class.  |
|||
|**`models/`** |Folder containing trained models. |


## Current tasks

- Explorator data analysis
	- labels distribution (expect consensus)
	- EEG data format
	- view spectrogram(s) (uniform size/missing data?)
		- spectrogram stacking possibilities
- Clarify goal (classify 50 second window or a whole EEG)  
- Understand Kullback-Leibler Divergence
	- Compute the KL score for a given annotator (if possible) 
- Create a Train/Validation/Test split with no data leakage (group by eeg_id) 
- Baseline model (random forest?)
- Make a first submission to leader board  
- Transfer learning (resnet50, imagenet, dino)  
	- Data preparation
	- Augmentation
	- Learning scheduler
- Explainablitiy

