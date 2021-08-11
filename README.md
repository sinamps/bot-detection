# bot-detection

This repository contains the codes and the link to the data of our paper "Automatic Identification of Social Media Bots using Deepfake Text Detection" for reproducibility purposes.

The dataset we have used in this work is the "TweepFake" dataset published by Fagni et. al. ([link to their paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0251415), [link to the dataset files](https://www.kaggle.com/mtesconi/twitter-deep-fake-text)).

The architecture of our model and the full details of our work are explained in our paper.


If you use our work, please cite our paper:


To use our codes, please first read the script "t_bi_nv.py" no matter what model you want to run. This file includes comments that are required for running all of our scripts. Please also make sure imports and data set file addresses are corrected in the code.

Association between our published scripts and the model configurations from our paper is as follows:
 * For configurations Cfg 1, Cfg 2, Cfg 4, Cfg 7, Cfg 8, and Cfg 10, use "t_bi_nv.py".
 * For configurations Cfg 3 and Cfg 9, use "t.py"
 * For configuration Cfg 5, use "t_bi_ap.py"
 * For configuration Cfg 6, use "t_bi_mp.py"
 * For "LSTM on GloVe (twitter-glove-200)", use "glove_lstm.py"
In script names, _t_ stands for transformers, _bi_ for BiLSTM, _nv_ for NeXtVLAD, _ap_ for average pooling, and _mp_ for maximum pooling.

Some of the dependencies you need to have installed on the Python environment that you want to use for running our code:
 * torch
 * numpy
 * json
 * transformers
 * sklearn
 * pandas

For running the GloVe model you also need:
 * gensim
 * tensorflow.keras
 * preprocessor

You can install all of these dependencies using the pip package manager.
For the fastText model, we just used the train_supervised function from fastText's text classification problem from [here](https://fasttext.cc/docs/en/supervised-tutorial.html).
