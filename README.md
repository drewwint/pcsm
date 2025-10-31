# Probabalistic Cognitive State Modeling (PCSM) repository

![Image of PCSM Pipeline](https://github.com/drewwint/pcsm/PCSM_figure.png)



This repository holds the code with links to the  simulated data and simulated derivatives for evaluating PCSM.

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
Functions the scripts folder uses are in the code/10_functions/joblib folder. This can be downloaoded as a package and imported into your coding framework for immediate use. 

### Note on upcomming package
While under review I anticipate some changes to be made - upon acceptance I will place these functions in a user friendly format and upload to PyPI for easy import via pip.

For now, PCSM can be tested and used by cloning the repository and installing into your session using the following code example:

```
#Bash-----------------------------------------
git clone https://github.com/drewwint/pcsm

cd ~/<location repository is cloned at>

pip install .

#Python---------------------------------------
import projlib

from projlib import metric_calculation as mc

#-- ect.. 
``` 

Then you can import modules and functions to run PCSM for yourself. Until PCSM package is finalized and  in PyPI - you will  need to install and import the dependencies outlined in the .txt file  in the /code/env folder including 'hmmlearn' and 'nilearn'. For the final package the required portions of these packages will be forked so that the PCSM package will be self contained and robust to external dependency changes in the future. 


## PCSM links
- [Preregistration](https://doi.org/10.17605/OSF.IO/DFJSB)
- [Preprint](www.drewEwinters.com)
- [OSF](https://doi.org/10.17605/OSF.IO/BP3GN)
- [github](https://github.com/drewwint/pcsm)
 
