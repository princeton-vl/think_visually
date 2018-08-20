## Config Files

A config file is a `.yaml` file which contains all the information about the current experiment. It is used by the `<think_visually dir>/model.py` to  to decide the model (`DSMN`/`DMN+`), the dataset used (`FloorPlanQA`/`ShapeIntersection`), various model parameters (like `learning rate`) etc. To run an new experiment, just create a new config file wiht appropriate fields and run `python model.py <path to config file>`.

The various fields of a config file are explained here:

- `pretrained`: Whether to load the pretrained model or train from scratch. `1` would load the pretrained model and `0` would train from scratch.

- `dataset`: Which dataset to use. Possible values are `'FloorPlanQA'` and `'ShapeIntersection'`. 

- `model`: Which model to use. Possible values are `'DMN'` and `'DSMN'`. For `DSMN*` model use `'DSMN'`. Adjust `reg` (described below) to use `DSMN*`
  
- `batch_size`: Recommended value `128`.

- `run`: Like an ID for an experiemnt. Can used to run mutiple experiments with same parameters and differnet random initialization. For example two experiment wih only differnce in `run` would differ only in parameter initialization. 
  
- `num_epochs`: Maximum number of epochs an experiment would run.

- `early_stopping_epoch`: Number of epochs to run the training if there is no increase in performance on the validation set. This determines the early stopping criteria.

- `embedding_size`: The dimension of vector representation of words. The dimension of the hidden layers of various `LSTMs` in the model is same as the `embedding size`.
 
- `learning_rate`:  Recommended value `0.001`.
  
- `num_hops`: Number of hops in `DSMN` and `DMN`. Can be used to reproduce results in `Table 2c` of the paper. Recommended value `3`.
  
- `save_path`: Path of the folder relative to `<think_visually dir>/model.py` where all the models weights and curves are stored.
 
- `l2_reg`: Weight for the l2 regalarization of the model parameters.
  
- `dropout_reg`: Dropouts are applied on the fully-connected layer in the model. It determines the probability with which features are kept during training. Recommended value `0.9`.

- `abl_number`: To reproduce the experiments in `Table 2b` of the paper. Possible values are `1`, `2` and `3`.  Recommended value `1`. `1` is for `f = [En_f(M^(T)); m^(T); q ]`; `2` is for `f = [m^(T); q ]`; and `3` is for `f = [En_f(M^(T)); q ]`.

- `data_path`: Path of the folder relative to `<think_visually dir>/model.py` where all the data is stored.

- `reg`: Only relevant to `DSMN` model. Equivalent to `$\lambda_{vi}$` in `Sec 4.2` of the paper. It is always between `0` and `1`. When it is `1` the model is `DSMN`. When it is less than `1`, model is `DSMN*`. Recommended value for `DSMN*` is `0.1`.

- `per_inter_sup`: Portion of sample with visual representations. To reproduce `Fig 4` in the paper. Possible values are `1`, `0.5`, `0.25`, `0.125`, `0.0625`, `0.03125`, `0.015625`, `0.0078125` and `0.00390625` . Use `1` for vanilla `DSMN*`.
