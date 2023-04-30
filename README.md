
# Tackling Word-Level Sign Language Recognition over space and time


This repository contains the code for our WLASL project, which aims to classify American Sign Language (ASL) signs using a combination of ResNetRNN and Dense Graph Convolutional Network (DGCN) models.


## Dataset

The Word-Level American Sign Language Recognition dataset, published in 2018, is a large-scale video dataset featuring 2000 common ASL wordrepresentations (glosses), signed by 100 different signers of varying ASL proficiency. It was developed for the purposes of facilitating research in the sign language recognition domain. It leverages the youtube-dl tool to download the videos in the dataset from their urls, which are described in the WLASL vx.x.json file, containing metadata on all the 20,000 video instances it encompasses. Link to the dataset is provided at the bottom of this README.

## Installation

Clone this repository and install the required packages using the following commands:

```bash
git clone https://github.com/shankhsuri/Sign-Language-Decoding.git
cd Sign-Language-Decoding
conda create -n wlasl_env --file requirements.txt
conda activate wlasl_env
```

## Methods

Our approach involves the development of a comprehensive pipeline for WLASL video training, integrating data preprocessing, feature extraction, and model training and evaluation for a stacked ensemble of a ResNetRNN and different configurations of the DGCN model. The approach taken is a combination of both image appearance-based and pose-based methods. Both models are implemented using PyTorch.

Image appearance-based method: ResNetRNN leverages appearance-based techniques, using 18 deep convolutional layers to create a representation of each input frame, followed by a modeling of the temporal dynamics of the video using a sequence of LSTMs.

Pose-based method: A Graph Convolutional Network (GCN) is chosen to model spatial dependencies. GCNs are a type of neural network designed to work with graph data, where the human body is modeled as a fully connected graph.

## Usage

To run the project, simply execute the following command:
```bash
python main.py
```
This will start the entire pipeline for the WLASL project, including data preprocessing, feature extraction, model training, and evaluation.

## Contributors

- Yasmin Farhan
- Shankh Suri

## 🔗 Links

[Kaggle Videos used in our project](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed?select=videos/)
