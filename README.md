# dynPCA

Python implementation of a new dynamic Principal Component Analysis algorithm [[1](#1),[2](#2)] for multidimensional time series input.

In its current state, the code belongs to the initial state of the project but fixes were applied to work with Python 3.8 inside the Conda environment built by the provided `.yml` file. You can create a conda environment by

`conda env create -f dynPCA_env.yml`

Then activate the environment:

`conda activate dynPCA`

The core of the module is `lowrankappr.py`. Some docstrings need to updated here.

A jupyter notebook is also available to test the code: `dynPCA_tutorial.ipynb`

A comparison with static PCA is included in the jupyter notebook: `dynPCA_errors.ipynb`

In the repository, two datasets are available for testing: 

  * `VARMA_ts_dim6_AR12_MA12.csv`: a randomly generated 6-dimensional ARMA(12,12) process of length 10.000
  * `Energy_Stock_Prices.xlsx`: dataset from Paul, Samit (2020), ‘Multivariate dependence and portfolio management strategy of energy stocks: An EVT-vine copula approach - Energy Economics’, Mendeley Data, V1, doi: [10.17632/zyytsd86d7.1](https://doi.org/10.17632/zyytsd86d7.1)

## References:

<a id="1">[1]</a>  M. Bolla, T. Szabados, M. Baranyi, and F. Abdelkhalek, ‘Block circulant matrices and the spectra of multivariate stationary sequences’, Special Matrices, vol. 9, no. 1, pp. 36–51, Jan. 2021, doi: [10.1515/spma-2020-0121](https://doi.org/10.1515/spma-2020-0121).

<a id="2">[2]</a>  M. Baranyi, M. Bolla, and Gy. Kocsisné Szilágyi, ‘A novel dynamic Principal Component Analysis method, applied to ECG signals’, in 2021 55th Asilomar Conference on Signals, Systems, and Computers, Pacific Grove, CA, USA, Oct. 2021, pp. 265–269, doi: [10.1109/IEEECONF53345.2021.9723158](https://doi.org/10.1109/IEEECONF53345.2021.9723158).



