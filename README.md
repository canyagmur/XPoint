
<p align="center">
  <h1 align="center"><ins>CS-REG-NET</ins> <br>Cross-Spectral Registration of Thermal and Optical Imagery</h1>
  <p align="center">
      TODO Add author info
  </p>
  <h2 align="center">
    <p>ECCV 2024</p>
    TODO: Add paper link
  </h2>
  
</p>
<p align="center">
    <img src="assets/match-table.png" alt="example" width=80%></a>
    <br>
    <em>To mitigate the challenges of optical-thermal spectral differences, we introduce CS-REG-NET, a Transformer-based framework with a hierarchical attention mechanism for feature detection and description in multispectral images, addressing spectral differences through a combined global and local context understanding. Our approach incorporates a multi-task learning strategy with a lightweight homography regression head, significantly improving feature-based alignment capabilities.</em>
</p>

##

# CS-REG-NET
This is a PyTorch implementation of "CS-REG-NET: A Self-Supervised Swin-Transformer based Network for Cross-Spectral Registration of Thermal and Optical Imagery"

## Installation
This software requires Python 3.6 or higher (Tested on 3.6.13).

Requirements can be installed with:
```
pip install -r requirements.txt
```

The repository includes pre-trained models for CS-REG-NET. However, to train the models, you need to download the dataset separately (see [Dataset](#dataset)).

The "csregnet" python package can be locally installed by executing:
```
pip install -e . --user
```
(You can remove the `--user` flag if operating in a virtual environment)

## Dataset
### Multispectral Image Pair Dataset
The dataset is hosted on the [Autonomous Systems Lab dataset website](https://projects.asl.ethz.ch/datasets/doku.php?id=corl2020-multipoint), which also offers basic information about the data.

The dataset can be downloaded by running (from the csregnet directory):
```python
python download_multipoint_data.py
```
A different target directory can be specified with the `-d` flag.
You can force overwrite existing files by setting the -f flag. Please note that the dataset files are quite large (over 36 GB total), so the download process may take some time.

### VEDAI Dataset
The VEDAI dataset can be downloaded from the [official website](https://downloads.greyc.fr/vedai/). The dataset is used for the evaluation of the CS-REG-NET on the VEDAI dataset.

## Dataset Structure
The dataset is expected to be structured as one of the following examples:
### 1- HDF5 Files (set "filename" parameter in config files):
```
data
├── MULTIPOINT
│   ├── training.hdf5
│   └── test.hdf5
└── VEDAI
    ├── training.hdf5
    └── test.hdf5
```
### 2- Image Files (set "foldername" parameter in config files):
```
data
├── MULTIPOINT
│   ├── training
│   │   ├── optical
│   │   │   ├── 0001.png
│   │   │   ├── 0002.png
│   │   │   └── ...
│   │   └── thermal
│   │       ├── 0001.png
│   │       ├── 0002.png
│   │       └── ...
│   └── test
│       ├── optical
│       │   ├── 0001.png
│       │   ├── 0002.png
│       │   └── ...
│       └── thermal
│           ├── 0001.png
│           ├── 0002.png
│           └── ...
```
As it can be seen, the dataset is expected to be structured in a way that the training and test data are separated into different directories. The optical and thermal images are expected to be in separate directories. The image pairs are expected to have the same name in the optical and thermal directories.





## Pre-trained Models
Pre-trained models for CS-REG-NET can be downloaded from the [anonymous drive link](https://drive.google.com/drive/folders/1vcqUdqoRa0Qj5KqMOP_9y2VTfcC_BjKN?usp=sharing). The pre-trained models include "csregnet","csregnet-mp512", "csregnet-vedai256", and "csregnet-vedai512". Download those models and store them in the `model_weights` directory.

The base model is "csregnet" and it is trained on the multispectral image pair dataset with a resolution of 256x256. Then "csregnet-mp512" model is finetuned for only 30 epochs on the multispectral image pair dataset with a resolution of 512x512. The "csregnet-vedai256" and "csregnet-vedai512" models are finetuned for only 10 epochs on the VEDAI dataset with a resolution of 256x256 and 512x512, respectively. Finetuning the models is important for learning positional encodings as stated in the SwinTransformerV2 paper.

### Ground truth keypoints file
The ground truth keypoints file for the multispectral image pair dataset can be downloaded from the same link [anonymous drive link](https://drive.google.com/drive/folders/1vcqUdqoRa0Qj5KqMOP_9y2VTfcC_BjKN?usp=sharing) under "csregnet_labels" folder.The ground truth keypoints file can be used to train your own model.

## Usage
In the following section the scripts to train and visualize the results of CS-REG-NET are explained. For each script, additional help on the input paramaters and flags can be found using the `-h` flag (e.g. `python show_keypoints.py -h`).


#### Benchmark on Predicting Keypoints and Homography
The performance of the trained CS-REG-NET can be evaluated by executing the `benchmark.py` script.

Example benchmark on multipoint's dataset,256x256 resolution:
```
python benchmark.py -y configs/cipdp.yaml -m model_weights/csregnet -v latest -e -p
```
Example benchmark on multipoint's dataset,512x512 resolution:
```
python benchmark.py  -y configs/cipdp_mp512.yaml -m model_weights/csregnet-mp512 -v latest -e -p 
```
Example benchmark on VEDAI dataset, 256x256 resolution:
```
python benchmark.py -y configs/cipdp_vedai256.yaml -m model_weights/csregnet-vedai256 -v latest -e -p 

```
Example benchmark on VEDAI dataset, 512x512 resolution:
```
python benchmark.py -y configs/cipdp_vedai512.yaml -m model_weights/csregnet-vedai512 -v latest -e -p 
```


Here the '-y' flag specifies yaml file ,the `-m` flag specifies the model weights, the `-v` flag the version of the model, the `-e` flag computes the metrics for the whole dataset, and the `-p` flag plots the results of some samples.

#### Individually Predicting Keypoints
Predicting only keypoints can be done executing the `predict_keypoints.py` script.
The results are plotted by adding the `-p` flags and the metrics for the whole dataset are computed by adding the `-e` flag.

#### Individually Predicting the Homography
Predicting the alignment of an image pair can be done using the `predict_align_image_pair.py` script.
The resulting keypoints and matches can be visualized by adding the `-p` flag.
The metrics over the full dataset are computed when adding the `-e` flag.

#### Generating Keypoint Labels
Keypoint labels for a given set of image pairs can be generated using:

```
python export_keypoints.py -o tmp/labels.hdf5
```

where the `-o` flag defines the output filename. The base detector and the export settings can be modified by making a copy of the `configs/config_export_keypoints.yaml` config file, editing the desired parameters, and specifying your new config file with the `-y` flag.
```
python export_keypoints.py -y configs/custom_export_keypoints.yaml -o tmp/labels.hdf5
```


#### Visualizing Keypoint Labels
The generated keypoint labels can be inspected by executing the `show_keypoints.py` script:

```
python show_keypoints.py -d data/MULTIPOINT/training.hdf5 -k tmp/labels.hdf5 -n 100
```

The `-d` flag specifies the dataset file, the `-k` flag the labels file, and the `-n` flag the index of the sample which is shown.


#### Visualizing Samples from Datasets
By executing the following command:
```
python show_image_pair_sample.py -i tmp/test.hdf5 -n 100
```

the 100th image pair of the `tmp/test.hdf5` dataset is shown.

#### Training CS-REG-NET
CS-REG-NET can be trained by executing the `train.py` script. All that script requires is a path to a yaml file with the training parameters:

```
python train.py -y configs/cmt.yaml
```

The hyperparameter for the training, e.g. learning rate, model parameters, can be modified in the yaml file.




## Citing
If you use this code in your research, please consider citing the following paper:
```
TODO: Add citation
```


## Credits
TODO Add credits
