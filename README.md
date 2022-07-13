# dynPCA

Python implementation of a new dynamic Principal Component Analysis algorithm [[1](#1),[2](#2)] for multidimensional time series input.

In its current state, the code belongs to the initial state of the project but fixes were applied to work with Python 3.8 inside the Conda environment built by the provided `.yml` file. You can create a conda environment by

`conda env create -f dynPCA_env.yml`

Then activate the environment:

`conda activate dynPCA`

The core of the module is `lowrankappr.py`

A jupyter notebook is also available to test the code: `dynPCA_tutorial.ipynb`

A comparison with static PCA is included the jupyter notebook: `dynPCA_errors.ipynb`

## References:

<a id="1">[1]</a>  M. Bolla, T. Szabados, M. Baranyi, and F. Abdelkhalek, ‘Block circulant matrices and the spectra of multivariate stationary sequences’, Special Matrices, vol. 9, no. 1, pp. 36–51, Jan. 2021, doi: [10.1515/spma-2020-0121](https://doi.org/10.1515/spma-2020-0121).

<a id="2">[2]</a>  M. Baranyi, M. Bolla, and Gy. Kocsisné Szilágyi, ‘A novel dynamic Principal Component Analysis method, applied to ECG signals’, in 2021 55th Asilomar Conference on Signals, Systems, and Computers, Pacific Grove, CA, USA, Oct. 2021, pp. 265–269. doi: [10.1109/IEEECONF53345.2021.9723158](https://doi.org/10.1109/IEEECONF53345.2021.9723158).



