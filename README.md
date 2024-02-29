# eegClassification
Classification of EEG data based on the Kaggle competition [HMS - Harmful Brain Activity Classification
](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview)

## Overview

## Files

|File|Description|
|--|--|
|`notebooks/`| |
|`models/`| Folder containing the models|


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

