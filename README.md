# JMD-Diversity-in-Bayesian-Optimization
Code to support the experiments published in "How Diverse Initial Samples Help and Hurt Bayesian Optimizers" in the Journal of Mechanical Design

# How Diverse Initial Samples Help and Hurt Bayesian Optimizers

by
Eesh Kamrah,
Seyede Fatemeh Ghoreishi,
Zijian “Jason” Ding,
Joel Chan,
Mark Fuge

This paper has been submitted for publication in *Journal of Mechanical Design*.

> This paper presents results to show that initializing Bayesian optimizers with diverse examples isn't always the best strategy.

![](results/experiment3-2-3/Experiment3-2-3.png)

*Optimality gap plot showing effects of diversity when the optimizer is not allowed to fit the hyper-parameters for the Gaussian
Process and the hyper-parameters are instead fixed to the values found in Experiment 2. The results from this plot show positive NCOG values for all
families of wildcat wells function, showing that once the Model Building advantage’ is taken away the diverse samples outperform non-diverse samples.*


## Abstract

> Design researchers have struggled to produce quantitative predictions for exactly why and when diversity might help or hinder design search efforts.
This paper addresses that problem by studying one ubiquitously used search strategy\textemdash Bayesian Optimization (BO)\textemdash on a 2D test problem with modifiable convexity and difficulty.
Specifically, we test how providing diverse versus non-diverse initial samples to BO affects its performance during search and introduce a fast ranked-DPP method for computing diverse sets, which we need to detect sets of highly diverse or non-diverse initial samples.

We initially found, to our surprise, that diversity did not appear to affect BO, neither helping nor hurting the optimizer's convergence. However, follow-on experiments illuminated a key trade-off. Non-diverse initial samples hastened posterior convergence for the underlying model hyper-parameters\textemdash a \textit{Model Building} advantage. In contrast, diverse initial samples accelerated exploring the function itself\textemdash a \textit{Space Exploration} advantage. Both advantages help BO, but in different ways, and the initial sample diversity directly modulates how BO trades those advantages. Indeed, we show that fixing the BO hyper-parameters removes the Model Building advantage, causing diverse initial samples to always outperform models trained with non-diverse samples.
These findings shed light on why, at least for BO-type optimizers, the use of diversity has mixed effects and cautions against the ubiquitous use of space-filling initializations in BO.
To the extent that humans use explore-exploit search strategies similar to BO, our results provide a testable conjecture for why and when diversity may affect human-subject or design team experiments.


## Software implementation

>The code from this paper presents a DPP based fast diverse and non-diverse rank based sampler. This has been described in the supplementary part of the paper. 
>This sampler is also present in the source code.

All source code used to generate the results and figures in the paper are in
the `src` folder.
The data used in this study is provided in `data`.
Results generated by the code are saved in `results`.


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/IDEALLab/JMD-Diversity-in-Bayesian-Optimization

A copy of the repository is also archived at *insert DOI here*


## Dependencies

You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `environment.yml`.

We use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command in the repository folder (where `environment.yml`
is located) to create a separate environment and install all required
dependencies in it:

    conda env create

## Reproducing the results

Before running any code you must activate the conda environment:

    source activate ENVIRONMENT_NAME

or, if you're on Windows:

    conda activate ENVIRONMENT_NAME

This will enable the environment for your current terminal session.
Any subsequent commands will use software that is installed in the environment.

To produce all results and figures, and compile run individual scripts found in the scripts
directory. For further details on each script, check the readme file in the scripts directory.

If all goes well, the figures will be in their respective results folder in `results/Experiment#`.


## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is not open source. The authors reserve the rights to the
article content, which is currently submitted for publication in the
Journal of Mechanical Design.
