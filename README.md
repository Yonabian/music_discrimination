# Music Discrimination
- Report : The Exploration of The Musical Discrimination Between Real Music and Generated Music
- Artifact Producer : Yona Bian


## Project Overview
1. This repository will involve training discriminator neural networks to explore whether music is created by real humans or fake music generated. 
2. This repository is perfectly cometible with pytorch


## Downloading Data
```bash
$ ssh -J [your uId]@bulwark.cecs.anu.edu.au [your name]@plasticpal.gpu
```
* The datasets are on charles plasticpal machine. Please get account for access before download.

* Download the folders:

            - train
            - test
            - train_LSTM
            - test_LSTM
            - pop_midi

## How to Use

* Run `python train.py` to train and test the first model.
* Run `python train_LSTM.py` to train and test the second model.
* The `data_process` folder contains all the scripts for generating and processing the data.


## References
* MusicTransformer [here](https://github.com/jason9693/MusicTransformer-pytorch)
* LSTM FCN for Time Series Classification [here](https://github.com/titu1994/LSTM-FCN)