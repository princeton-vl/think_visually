# Training and Evaluation Code 

[**Think Visually: Question Answering through Virtual Imagery**](http://bit.ly/think_visually_paper)  
[Ankit Goyal](http://imankgoyal.github.io), [Jian Wang](http://jianwang.me/), [Jia Deng](https://www.cs.princeton.edu/~jiadeng/)  
*Annual Meeting of the Association for Computational Linguistics (ACL), 2018*

## Getting Started

First download/clone the repository. We would refer to the directory containing the code as `<think_visually dir>`.

```
git clone git@github.com:umich-vl/think_visually.git
```

#### Requirements
Our current implementation only supports GPU so you need a GPU and need to have CUDA installed on your machine. We used Python version **3.5.3**, CUDA version **8.0.44** and cuDNN version **8.0-v5**.

#### Install Libraries
We recommend to first install [Anaconda](https://anaconda.org/) and create a virtual environment.
```
conda create --name think_visually python=3.5
```

Activate the virtual environment and install the libraries. Make sure you are in `<think_visually dir>`.
```
source activate think_visually
pip install -r requirements.txt
```

#### Download Datasets and Pre-trained Models
Download all the folders [here](http://bit.ly/think_visually_acl_2018). Unzip them and put them in `<think_visually dir>`.

## Code Organization

- `<think_visally dir>/model.py`: The main python script for creating model graph, training and testing.

- `<think_visally dir>/configs`: It contains various sample config files. `model.py` uses a config file to decide the model (`DSMN`/`DMN+`), the dataset used (`FloorPlanQA`/```ShapeIntersection```), various model parameters (like learning rate) etc. More information about the configuration files is present in `<think_visually dir>/configs/README.md`. 

- `<think_visally dir>/results`: It contains all the pretrained models as well as training curves for the pre-trained models.

- `<think_visally dir>/utils`: It contains various utility files for data loading, preprecessing and common neural-net layers.  

- `<think_visally dir>/data_FloorPlanQA`: It contains all the FloorPlanQA dataset. More information about various files in that folder is in `<think_visually dir>/data_FloorPlanQA/README.md`.

- `<think_visally dir>/data_ShapeIntersection`: It contains all the ShapeIntesection dataset. More information about various files in that folder is in `<think_visually dir>/data_ShapeIntesection/README.md`.

## Running Experiments

To train and evaluate a model use the `model.py` script with a config file.
```
python model.py <relative path to config file>
```

For example, to load the pretrained `DSMN` model on the `FloorPlanQA` dataset and evaluate it, use the following command.
```
python model.py configs/DSMN_FloorPlanQA.yml
```

Similarly to load the pretrained `DSMN` model on the `FloorPlanQA` dataset with 0.78125% partial suprevision, use the following command.
```
python model.py configs/DSMN_FloorPlanQA_sup_0.0078125.yml
```

Note that in order to train from scratch you need to set the `pretrained` flag in the config file to 0. More information about how to set up a config file is in `<think_visually dir>/configs/README.md`.

**ADVICE**: As mentioned in the paper we found the `DMN+`/`DSMN` models to be unstable across runs. For consistent results, we recommend running the same model (with random initialization) atleast 10 / 20 times (you can use the run flag in the config file). The `DSMN#` model (i.e. `DSMN` with intermediate supervision) is relatively stable and requires less runs.

**UPDATE**: We reran all models on ShapeInterection so the results of the pretrained models are `+- 2%` of reported in the paper.  
