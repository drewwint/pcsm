![PCSM package image](https://github.com/drewwint/drewEwinters.site/blob/master/content/software/pcsm/featured.png)

# Probabilistic Cognitive State Modeling (PCSM)

 

PCSM combines Finite Impulse Response (FIR) modeling of BOLD activity with a Gaussian Mixture Model-Hidden Markov Model (GMM-HMM) to quantify dynamic, spatially distributed brain states. From these posteriors, PCSM derives interpretable, emergent properties including serial-parallel processing, cognitive demand, resource level, and serial bottleneck.

 

This repository holds the code with links to the simulated data and derivatives for evaluating PCSM.

 

## PCSM Materials and Extras

|Open Science|Presentations|Data/Code|Examples|

|:-----------|:-----------|:-----------|:-----------|

|[Preregistration](https://doi.org/10.17605/OSF.IO/DFJSB)  |[University of Zurich](https://www.drewewinters.com/talk/probabilistic-cognitive-state-modeling-pcsm-a-framework-for-quantifying-information-processing-in-fmri/)|  [Data](https://osf.io/bp3gn)| [Human Data Example](https://github.com/drewwint/pcsm/blob/main/code/30_notebooks/PCSM_example_human_data_reduced.ipynb)|

|[BioRxiv Preprint](https://doi.org/10.1101/2025.10.31.685855)  |[American College of Neuropsychopharmocology]()  |[GitHub](https://github.com/drewwint/pcsm)| |

|[OSF Repository](https://osf.io/bp3gn)  |   |[Initial Package](https://github.com/drewwint/pcsm/tree/main/code/10_functions)| |

 

 

## PCSM Pipeline Image

![Image of PCSM Pipeline](https://github.com/drewwint/pcsm/blob/main/PCSM_figure.png)

*Depicting the PCSM pipeline. (A) Modeling BOLD from task-based fMRI using a Finite Impulse Response (FIR) model. (B) FIR-derived timeseries are inputs into the GMM-HMM with PCSM alignment. The emergent properties from these dynamic outputs are then decoded in the following steps. (C) The number of responding nodes at each timepoint is decoded to estimate parallel and serial processing periods that are then projected to brain space. (D) Cognitive demand and resource levels are computed from these dynamics. (E) From these metrics, PCSM derives a scalar index of serial bottleneck severity.*

 

This image outlines the workflow for computations in PCSM - These steps are outlined in more detail in the [preprint](https://doi.org/10.1101/2025.10.31.685855) and the upcoming peer-reviewed publication.

 

Examples of using PCSM with human data can be found in this repository [here](https://github.com/drewwint/pcsm/blob/main/code/30_notebooks/PCSM_example_human_data_reduced.ipynb) or under `code/30_notebooks/PCSM_example_human_data_reduced.ipynb`

 

## Data links

[The Open Science Framework (OSF) repository for PCSM](https://osf.io/bp3gn) holding the simulations can be found [here](https://osf.io/bp3gn/files).

 

The human data used for informing simulations can be found on OpenNeuro [here](https://openneuro.org/datasets/ds000030/versions/00001). The derivatives for this dataset can be imported via [Nilearn](https://nilearn.github.io/stable/index.html) using [fetch_open_neuro](https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_ds000030_urls.html) with an example of how to do this in a tutorial [here](https://nilearn.github.io/dev/auto_examples/04_glm_first_level/plot_bids_features.html#sphx-glr-auto-examples-04-glm-first-level-plot-bids-features-py).

 

### Notes on simulated data conversion

The simulated data and derivatives - comprised of 15,000 simulations with 200 node timeseries each for ~132 trials with a TR of 2s - were substantially larger than what is allowed for OSF. Therefore I compressed these files and split them for easier storage. This requires proper concatenation and unpacking that can be done with the code below.

 

```

# Bash------------------------------------------------

# change directory into where files were downloaded

cd ~/<name of download filename>

 

# concatenating

cat simulated_data.part-* > simulated_data.tar.gz

 

# decompressing

tar -xzf simulated_data.tar.gz

```

 

## PCSM Functions

Functions the scripts folder uses are in the 'code/10_functions/projlib' folder. This can be downloaded as a package and imported into your coding framework for immediate use.

 

### Note on upcoming package

While under review, I anticipate some changes to be made - upon acceptance, I will place these functions in a user-friendly format and upload to PyPI for easy import via pip.

 

For now, PCSM can be tested and used by cloning the repository and installing it into your session using the following code example:

 

```

#Bash-----------------------------------------

git clone https://github.com/drewwint/pcsm

 

cd ~/<location repository is cloned at>

 

pip install .

 

#Python---------------------------------------

import projlib

 

from projlib import metric_calculation as mc

# from projlib import <etc..>

```

 

Then you can import modules and functions to run PCSM for yourself. Until the PCSM package is finalized and in PyPI - you will  need to install and import the dependencies outlined in the .txt file  in the /code/env folder, including 'hmmlearn' and 'nilearn'. For the final package, the required portions of these packages will be forked so that the PCSM package will be self-contained and robust to external dependency changes in the future.

 

 
