-----
# Empirical fits to inclusive electron-carbon scattering data obtained by deep-learning methods

* [arXiv:2312.xxxxx](https://arxiv.org/abs/2312.xxxxx)

## Introduction

We introduce the deep neural network fits to inclusive electron-carbon scattering data. There are two models available, A and B.

* Model A:
    It is based on an ensemble of 50 neural networks, which fit clone datasets.
* Model B:
    It is a single neural network with dropout layers. The Monte Carlo dropout mechanism is used to make predictions.

## Making Predictions 

To run the model:
* Install numpy, jax and flax packages
* to make model A predictions for electron energy $E$ [GeV], scattering angle $\theta$ [degree], and a range of the energy transfer $\omega \in[\omega_1,\omega_2]$ [GeV], number of points $p$, number of variants of the neural network $v\in [1,50]$,
    execute 
  ```
  $ python main.py energy=E  theta=θ  min=ω₁ max=ω₂ nop=p nov=v clones
  ```
* to make model B predictions for electron energy $E$ [GeV], scattering angle $\theta$ [degree], and a range of the energy transfer $\omega \in[\omega_1,\omega_2]$ [GeV], number of points $p$, number of variants of the neural network $v \geq 1 $,
    execute 
  ```
  $ python main.py energy=E  theta=θ  min=ω₁ max=ω₂ nop=p nov=v dropout
  ```
* when one executes  `python main.py clones` or `python main.py dropout` it corresponds to $E=0.68$ GeV, $\theta=60^{\circ}$, $\omega \in [0,0.68]$ GeV, number of points $p=100$, number of variants of the neural network $v=50$

* the output is saved in the directory Results_Clones/Results_Dropout directory respectively.
* the output is in the format .txt file with three columns:
  * energy transfer value [GeV],
  * cross section $d^2\sigma/d\omega/d\Omega$ [nb/sr/GeV] - a mean value of the $v$ variants of the neural network predictions,
  * uncertainty [nb/sr/GeV] - a standard deviation of the $v$ variants of the neural network predictions.


## Citation
    @article{Kowal:2023,
    author = "Kowal, Beata E. and Graczyk, Krzysztof M. and Ankowski, Artur M. and Banerjee, Rwik Dharmapal and Prasad, Hemant and Sobczyk, Jan T.",
    title = "{Empirical fits to inclusive electron-carbon scattering data obtained by deep-learning methods}",
    eprint = "2312.xxxxx",
    archivePrefix = "arXiv",
    primaryClass = "nucl-ex",
    month = "12",
    year = "2023"}
   
