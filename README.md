-----
# Empirical fits to inclusive electron-carbon scattering data obtained by deep-learning methods

* [Phys.Rev.C 110 (2024) 2, 025501](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.110.025501), **Models A and B**
* [Phys.Rev.C 112 (2025) 5, 5](https://journals.aps.org/prc/abstract/10.1103/ydvc-2567), **Model C - the most recent one**

## Introduction

We introduce the deep neural network fits to inclusive electron-carbon scattering data. There are three models available, A, B and C.

* Model A (Bootstrap):
    It is based on an ensemble of 50 neural networks, which fit clone datasets.
* Model B (Dropout):
    It is a single neural network with dropout layers. The Monte Carlo dropout mechanism is used to make predictions.
* Model C (Bootstrap, a new fit):
    It is based on an ensemble of 50 neural networks, which fit clone datasets.
  
## Making Predictions 

To run the model:
* Install numpy, jax and flax packages (Model A and B) or install numpy and keras (Model C)
* to make model A predictions for electron energy $E$ [GeV], scattering angle $\theta$ [degree], and a range of the energy transfer $\omega \in[\omega_1,\omega_2]$ [GeV], number of points $p$, number of variants of the neural network $v\in [1,50]$,
    execute 
  ```
  $ python main.py energy=E  theta=θ  min=ω₁ max=ω₂ nop=p nov=v bootstrap
  ```
* to make model B predictions for electron energy $E$ [GeV], scattering angle $\theta$ [degree], and a range of the energy transfer $\omega \in[\omega_1,\omega_2]$ [GeV], number of points $p$, number of variants of the neural network $v \geq 1 $,
    execute 
  ```
  $ python main.py energy=E  theta=θ  min=ω₁ max=ω₂ nop=p nov=v dropout
  ```
* to make model C predictions for electron energy $E$ [GeV], scattering angle $\theta$ [degree], and a range of the energy transfer $\omega \in[\omega_1,\omega_2]$ [GeV], number of points $p$, number of variants of the neural network $v\in [1,50]$,
    execute 
  ```
  $ python main.py energy=E  theta=θ  min=ω₁ max=ω₂ nop=p nov=v newfit
  ```
* when one executes  `python main.py bootstrap` or `python main.py dropout` or `python main.py newfit` it corresponds to $E=0.68$ GeV, $\theta=60^{\circ}$, $\omega \in [0,0.68]$ GeV, number of points $p=100$, number of variants of the neural network $v=50$

* the output is saved in the directory Results_Bootstrap/Results_Dropout directory respectively.
* the output is in the format .txt file with three columns:
  * energy transfer value [GeV],
  * cross section $d^2\sigma/d\omega/d\Omega$ [nb/sr/GeV] - a mean value of the $v$ variants of the neural network predictions,
  * uncertainty [nb/sr/GeV] - a standard deviation of the $v$ variants of the neural network predictions.


## Citation
    @article{Kowal:2023dcq,
    author = "Kowal, Beata E. and Graczyk, Krzysztof M. and Ankowski, Artur M. and Banerjee, Rwik Dharmapal and Prasad, Hemant and Sobczyk, Jan T.",
    title = "{Empirical fits to inclusive electron-carbon scattering data obtained by deep-learning methods}",
    eprint = "2312.17298",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1103/PhysRevC.110.025501",
    journal = "Phys. Rev. C",
    volume = "110",
    number = "2",
    pages = "025501",
    year = "2024"}

    @article{Kowal:2025qkk,
    author = "Kowal, Beata E. and Graczyk, Krzysztof M. and Ankowski, Artur M. and Banerjee, Rwik Dharmapal and Bonilla, Jose L. and Prasad, Hemant and Sobczyk, Jan T.",
    title = "{Re-optimization of a deep neural network model for electron-carbon scattering using new experimental data}",
    eprint = "2508.00996",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1103/ydvc-2567",
    journal = "Phys. Rev. C",
    volume = "112",
    number = "5",
    pages = "055504",
    year = "2025"}

## Supplemental Material
* [suplemental_material to Phys.Rev.C 110 (2024) 2, 025501](https://github.com/bekowal/CarbonElectronNeuralNetwork/blob/main/supplemantal_material.pdf)

## ACKNOWLEDGMENTS

Work supported by the National Science Centre under grant UMO-2021/41/B/ST2/ 02778.
