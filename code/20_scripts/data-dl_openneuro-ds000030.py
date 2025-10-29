 # !/bin/python

import nilearn
import nilearn.datasets
import numpy as np
import os
import nibabel as nib

from nilearn.datasets import (
fetch_ds000030_urls,
fetch_openneuro_dataset
)

_, urls = fetch_ds000030_urls(data_dir="/home/dwinters/ascend-lab/data/data-PCSM")

data_dir, _ = fetch_openneuro_dataset(
    urls=urls, 
    data_dir="/home/dwinters/ascend-lab/data/data-PCSM/ds000030"
    )

