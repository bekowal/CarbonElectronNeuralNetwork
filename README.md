-----
# Empirical fits to inclusive electron-carbon scattering data obtained by deep-learning methods

see [arxiv paper](https://arxiv.org/abs/2312.xxxxx)

## Info

We introduce the deep neural network fits to inclusive electron-carbon scattering data. There are two models available, A and B.

* Model A:
    It is based on an ensemble of 50 neural networks, which fit clone datasets.
* Model B:
    It is a single neural network with dropout layers. The Monte Carlo dropout mechanism is used to make predictions.

## Making Predictions

To run the model:
* Install jax package ()
* to make model A predictions for electron energy E, scattering angle theta, and zzz.
    execute `xxx yyy zzz`
* to make model B predictions for electron energy E, scattering angle theta, and zzz.
    execute `main.py yyy zzz`

* when one executes ` main.py` it corresponds to xxxx

* the model's outpu is in the format: xxx ??? ???

## Citation
    @article{Kowal:2023,
    author = "Kowal, Beata E. and Graczyk, Krzysztof M. and Ankowski, Artur M. and Banerjee, Rwik Dharmapal and Prasad, Hemant and Sobczyk, Jan T.",
    title = "{Empirical fits to inclusive electron-carbon scattering data obtained by deep-learning methods}",
    eprint = "2312.xxxxx",
    archivePrefix = "arXiv",
    primaryClass = "nucl-ex",
    month = "12",
    year = "2023"}
   
