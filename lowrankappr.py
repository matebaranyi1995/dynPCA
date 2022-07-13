from numpy.lib.type_check import _real_dispatcher
import pandas as pd
import numpy as np
from scipy import signal
from scipy import fft
import matplotlib.pyplot as plt
from cycler import cycler


__author__ = "Máté Baranyi"
__email__ = "baranyim@math.bme.hu"
__version__ = "0.31"
__date__ = "2022-Jul-13"
__status__ = "Development"


##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

def multidim_spectral_density_estimate(data, use_periodogram=False,
                                       nperseg=None, window="hann", fs=1.0,
                                       return_onesided=False, debug=True,
                                       *args, **kwargs):
    """
    Estimates the spectral density matrix process of a given multidimensional time-series
    with scipy's built-in csd() method which uses Welch's windowed peridogram method.
    This matrix can be used as a faster alternative to the bolla_M_fast() function.
    It is theoretically not justified but it is faster.

    Parameters
    ----------
    data : pd.DataFrame (ToDo: extend to np.ndarray)
        The multidimensional time-series as a pandas DataFrame where the columns and rows should
        indicate the dimensions and timestamps, respectively. Missing values are not supported.
    use_periodogram : bool, optional
        If `True` the function uses the simple multidimensional version of the periodogram
        instead of Welch's method, by default `False`.
    nperseg : int, optional
        Size of the time-series segments on the density will be estimated. This argument is also 
        forwarded to the inner `csd()` function calls. For further details see its docs.
        It will be equal to the size of the output, by default it is the length of the time-series. 
    window : str or tuple or array_like, optional
        Determines the window method to be used, this argument is also forwarded to the inner 
        `csd()` function calls. For further details see its docs. By default "hann".
    fs : float, optional
        Sampling frequency of the time series, by default 1.0.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `False`, but for
        complex data, a two-sided spectrum is always returned. (Todo: check if `True` is buggy)
    debug : bool, optional
        Checks some theoretical properties of the matrix series, by default True.

    Returns
    -------
    omeg : ndarray
        Array of sample frequencies on [-\pi * fs, \pi * fs].
    spectral_array : ndarray
        The spectral density matrix series of the time-series in `data`.
        Its shape is (nperseg, data.shape[0], data.shape[0]).
    """

    dim = data.shape[1]

    if nperseg is None:
        nperseg = data.shape[0]

    if use_periodogram:
        nperseg = data.shape[0]
        window = np.ones(data.shape[0])

    #freqs = fft.fftfreq(nperseg, 1 / fs) * 2 * np.pi
    if return_onesided and not np.iscomplexobj(data.values):
        num_of_ff = int(np.floor(nperseg/2)+1)
        #freqs = freqs[:num_of_ff]
        spectral_array = np.empty(shape=(num_of_ff, dim, dim), dtype=np.csingle)
    else:
        spectral_array = np.empty(shape=(nperseg, dim, dim), dtype=np.csingle)

    omeg = None
    data = data.to_numpy().astype(np.float32)
    # if np.iscomplexobj(data):
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            omeg, f_ij = signal.csd(data[:,i], data[:,j],
                                    return_onesided=return_onesided,
                                    nperseg=nperseg, fs=fs,
                                    window=window,
                                    *args, **kwargs)
            spectral_array[:, i, j] = f_ij
        print(f"column {i} is done")

    # else:
    #     for i in range(data.shape[1]):
    #         # print(data.columns[i:])
    #         for j in range(i,data.shape[1]):
    #             omeg, f_ij = signal.csd(data[:,i], data[:,j],
    #                                         return_onesided=return_onesided,
    #                                         nperseg=nperseg, fs=fs,
    #                                         window=window,
    #                                         *args, **kwargs)
    #             spectral_array[:, i, j] = f_ij
    #         for j in range(i):
    #             spectral_array[:, j, i] = np.conj(spectral_array[:, i, j])
    #         print(f"column {i} is done")

    omeg = omeg * 2 * np.pi

    if debug:
        np.testing.assert_array_almost_equal(
            spectral_array[1:], np.conj(np.flip(spectral_array[1:], axis=0)), decimal=4)  # symmetry
        np.testing.assert_array_almost_equal(
            spectral_array, np.conj(np.transpose(spectral_array, (0, 2, 1))), decimal=4)  # self-adjoint

    return omeg, spectral_array


# low rank approximation of Bolla + spectral densify by Me

def spectral_densify(sp_array, desired_len, mode="linear"):
    """
    The spectral density is usually estimated at a smaller number of frequencies than the length
    of the time-series. This function inter/extrapolates the given spectral density estimate to 
    a higher number of frequencies. It is supported by the continuity property of the 
    spectral density matrxi function. 

    Parameters
    ----------
    sp_array : np.ndarray
        An array of matrices with shape (orig_len, dim1, dim2). This function assumes that the matrix 
        series in `sp_array` is a matrix function known at equidistant Fourier frequencies of number`orig_len`. 
    desired_len : int
        The desire number of equidistant Fourier frequencies at we want to inter/extrapolate `sp_array`
    mode : str, optional
        The method used for inter/extrapolation. "closest" means inter/extrapolation with the closest
        known neighbor from `sp_array` while "linear" means complex linear inter/extrapolation, 
        by default "linear".

    Returns
    -------
    dense_array : np.ndarray
        An array of matrices with shape (desired_len, dim1, dim2): the inter/extrapolated matrix 
        series of the input array at equidistant Fourier frequencies.
    """

    orig_len = sp_array.shape[0]
    sp_array = fft.fftshift(sp_array, axes=0)
    dense_array = np.empty(
        shape=(desired_len, sp_array.shape[1], sp_array.shape[2]), dtype=np.csingle)
    freqs_spare = fft.fftshift(fft.fftfreq(orig_len, 1) * 2 * np.pi, axes=0)
    freqs_dense = fft.fftshift(fft.fftfreq(desired_len, 1) * 2 * np.pi, axes=0)
    dict_dense = dict(zip(freqs_dense, range(desired_len)))

    if mode == "closest":
        j, k = 0, 1
        for f, v in dict_dense.items():
            if k < orig_len:
                if np.abs(f - freqs_spare[j]) <= np.abs(f - freqs_spare[k]):
                    dense_array[v, :, :] = sp_array[j, :, :]
                elif np.abs(f - freqs_spare[j]) > np.abs(f - freqs_spare[k]):
                    dense_array[v, :, :] = sp_array[k, :, :]
                    j += 1
                    k += 1
            else:
                dense_array[v, :, :] = sp_array[j, :, :]

    elif mode == "linear":
        for i in range(sp_array.shape[1]):
            for j in range(sp_array.shape[2]):
                dense_array[:, i, j] = np.interp(
                    freqs_dense, freqs_spare, sp_array[:, i, j])

    dense_array = fft.ifftshift(dense_array, axes=0)

    return dense_array


def eigen_decomp(spectral_array, fs=1.0):
    """
    A simple wrapper around numpy's efficient eigh() function to get the eigendecomposition of
    an array of Hermitian matrices. The function also assumes that the input array is a matrix 
    function at equidistant Fourier frequencies.

    Parameters
    ----------
    spectral_array : np.ndarray
        An array of matrices with shape (len, dim, dim). This function assumes that the matrix 
        series in `spectral_array` is a matrix function known at equidistant Fourier frequencies
        of number `len`; and that the matrices are Hermitian.
    fs : float, optional
        Sampling frequency of the time series, by default 1.0.

    Returns
    -------
    omeg : np.ndarray
        Array of sample frequencies on [-\pi * fs, \pi * fs] .
    sp_eig : np.ndarray
        Array of vectors containing the eigenvalues descendingly.
    sp_vect : np.ndarray
        Array of matrices containing the eigenvectors descendingly by the eigenvalues.
        sp_vect[:,:,i] is the eigenvector corresponding to sp_eig[:,i]
    """

    omeg = fft.fftfreq(spectral_array.shape[0], 1/fs) * 2 * np.pi

    sp_eig, sp_vect = np.linalg.eigh(spectral_array)

    # checking that the eigenvalues are in ascending order
    assert np.all(np.diff(sp_eig) >= 0)

    # reverse the eigenvalues to be in descending order
    sp_eig = np.flip(sp_eig, axis=-1)
    sp_vect = np.flip(sp_vect, axis=-1)

    return omeg, sp_eig, sp_vect


def lra_bolla(data, k, sp_array, densify="linear"):
    """
    Low rank approximation of a multidimensional time-series as described in the [SPMA] paper.
    It is extended by an inter/extrapolation in the case when the spectral estimation is conducted 
    on less frequencies then the length of the time-series.
    The output of this function is mainly used in comparison with the time-series reconstructed from
    the dynamic PCs because these are theoretically the same.

    Parameters
    ----------
    data : pd.DataFrame (ToDo: extend to np.ndarray)
        The multidimensional time-series as a pandas DataFrame where the columns and rows should
        indicate the dimensions and timestamps, respectively. Missing values are not supported.
    k : int
        The desired rank of the low rank estimation. It should be less than the number of dimensions 
        (data.shape[1]) in the input.
    sp_array : np.ndarray
        An array of Hermitian matrices with shape (len, dim, dim). This function assumes that this
        is a matrix function known at equidistant Fourier frequencies of number `len`; in the order 
        suited for scipy's fft routines. This array will be inter/extrapolated so `len` will match
        the length of the time-series. The first `k` eigenvectors (in decreasing order of the 
        eigenvalues) will be used for the low rank approximation.  
    densify : str, optional
        The method used for inter/extrapolation. See docs of `spectral_densify()`, be default "linear".

    Returns
    -------
    T_series : np.ndarray
        The Fourier transform of the time-series.
    T_proj : np.ndarray
        The projection (in the freq. domain) onto the space spanned by the first `k` eigenvectors.
    data_back : np.ndarray
        The low-rank approximation of the time-series. (The inverse Fourier of `T_proj`.)
        It has the same shape as the input data. 
    """

    if sp_array.shape[0] != data.shape[0]:
        sp_array = spectral_densify(
            sp_array, data.shape[0], mode=densify)

    omeg, sp_eig, sp_vect = eigen_decomp(sp_array)

    tilde_big_u = sp_vect[:, :, :k]  # V

    tilde_big_u_H = np.transpose(tilde_big_u.conj(), (0, 2, 1))  # V*
    print("eigv.: ", tilde_big_u_H.shape)
    proj_mat = np.einsum('ijk,ikl->ijl', tilde_big_u,
                         tilde_big_u_H)  # VxV*
    print("projmat.: ", proj_mat.shape)

    data_array = np.expand_dims(data.values, axis=-1)
    print("data.: ", data_array.shape)
    T_series = fft.fftn(data_array, axes=(
        0, ), s=(data.shape[0], ), norm="ortho")
    print("T_series: ", T_series.shape)
    T_proj = np.einsum('ijk,ikl->ijl', proj_mat, T_series)
    print("T_proj: ", T_proj.shape)
    data_back = fft.ifftn(T_proj, axes=(
        0, ), s=(data.shape[0], ), norm="ortho")
    print("data_back: ", data_back.shape)

    T_series = fft.fftshift(T_series, axes=0)
    T_proj = fft.fftshift(T_proj, axes=0)

    return T_series, T_proj, data_back


# dynamic PCA in the freq. domain

def time_series_projection(data, k, sp_array, densify="linear"):
    """
    Dynamic principal components (PCs) of a multidimensional time-series as described in 
    the [SPMA] paper. The input is a multi-dimensional time-series, and the output is a lower-
    dimensional time-series representation of the series. In real scenarios the first couple of
    dynamic PCs encompass a large portion of the variability in the original time-series'. 
    This function is extended by an inter/extrapolation in the case when the spectral estimation was
    conducted on less frequencies then the length of the time-series.

    Parameters
    ----------
    data : pd.DataFrame (ToDo: extend to np.ndarray)
        The multidimensional time-series as a pandas DataFrame where the columns and rows should
        indicate the dimensions and timestamps, respectively. Missing values are not supported.
    k : int
        The desired number of returned dynamic PCs. 
    sp_array : np.ndarray
        An array of Hermitian matrices with shape (len, dim, dim). This function assumes that this
        is a matrix function known at equidistant Fourier frequencies of number `len`; in the order 
        suited for scipy's fft routines. This array will be inter/extrapolated so `len` will match
        the length of the time-series. The first `k` eigenvectors (in decreasing order of the 
        eigenvalues) will be used for the low rank approximation.  
    densify : str, optional
        The method used for inter/extrapolation. See docs of `spectral_densify()`, be default "linear".

    Returns
    -------
    T_series : np.ndarray
        The Fourier transform of the time-series.
    Z_series : np.ndarray
        The complex PCs (in the freq. domain).
    data_back : np.ndarray
        The dynamic PCs (in the time domain) of the time-series. (The inverse Fourier of `Z_series`.)
        It has the same length as the input data but the number of dimensions is only `k`.
    tilde_big_u : np.ndarray
        The first `k` eigenvectors of the inter/extrapolated matrix series of `sp_array` 
        at equidistant Fourier frequencies.
    """

    if sp_array.shape[0] != data.shape[0]:
        sp_array = spectral_densify(
            sp_array, data.shape[0], mode=densify)

    omeg, sp_eig, sp_vect = eigen_decomp(sp_array)

    tilde_big_u = sp_vect[:, :, :k]  # V

    data_array = np.expand_dims(data.values, axis=-1)
    print("data: ", data_array.shape)
    T_series = fft.fftn(data_array, axes=(
        0,), s=(data.shape[0],), norm="ortho")
    # print("T_series: ", T_series.shape)
    Z_series = np.einsum(
        'ijk,ikl->ijl', np.transpose(tilde_big_u.conj(), (0, 2, 1)), T_series)
    # print("Z_series: ", Z_series.shape)
    data_back = fft.ifftn(Z_series, axes=(
        0,), s=(data.shape[0],), norm="ortho")
    print("PC_dim: ", data_back.shape)

    T_series = fft.fftshift(T_series, axes=0)
    Z_series = fft.fftshift(Z_series, axes=0)

    return T_series, Z_series, data_back, tilde_big_u


def time_series_restore(reduced_ts, sp_array, densify="linear"):
    """
    The aim of this function is to restore a version of the original time-series from the dynamic PC
    representation of it. Using all PCs would restore the original time-series without loss of
    information. Theoretically restoring the time-series from the first `k` PCs results in a `k` rank
    approximation of the original time-series only but usually `k` was chosen so that the first `k`
    PCs explain a large portion of the variance of the original time-series.
    This function is extended by an inter/extrapolation in the case when the spectral estimation was 
    conducted on less frequencies then the length of the time-series.

    Parameters
    ----------
    reduced_ts : np.ndarray
        The dynamic PCs (in the time domain) of the time-series.
        It has the same length as the input data but the number of dimensions is only `k`.
    sp_array : np.ndarray
        The same array that was used in the dynamic PC transformation, see the docs of
        time_series_projection() and lra_bolla()
    densify : str, optional
        The method used for inter/extrapolation. See docs of `spectral_densify()`, be default "linear".

    Returns
    -------
    restored : np.ndarray
        The time-series restored from the PCs. It has the same length as the input data but the 
        number of dimensions is determined by the shape of `sp_array`.
    """

    print("sp_array: ", sp_array.shape)
    datalen, ncomps = reduced_ts.shape[0], reduced_ts.shape[1]
    reduced_ts = np.expand_dims(reduced_ts.values, axis=-1)
    print("reduced_dim: ", reduced_ts.shape)
    Z_series = fft.fftn(reduced_ts, axes=(0,), s=(datalen,), norm="ortho")
    # print("Z_series: ", Z_series.shape)

    if sp_array.shape[0] != datalen:
        sp_array = spectral_densify(
            sp_array, datalen, mode=densify)

    omeg, sp_eig, sp_vect = eigen_decomp(sp_array)

    tilde_big_u = sp_vect[:, :, :ncomps]

    T_series = np.einsum('ijk,ikl->ijl', tilde_big_u, Z_series)
    # print("T_series: ", T_series.shape)
    restored = fft.ifftn(T_series, axes=(0,), s=(datalen,), norm="ortho")
    print("restored_dim: ", restored.shape)
    return restored


##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
# Bolla M matrices

def bolla_autocovs(data, length=None, debug=True):
    """
    Estimates the autocovariance matrix sequence/function of a multidimensional time-series data
    of length n until a given lag h with m=floor(h/2):
        * for even n: C(-(m-1)),... C(0),... C(m)
        * for odd n: C(-m),... C(0),... C(m)
        * max h = floor(n/2):

    Parameters
    ----------
    data : pd.DataFrame (ToDo: extend to np.ndarray)
        The multidimensional time-series as a pandas DataFrame where the columns and rows should
        indicate the dimensions and timestamps, respectively. Missing values are not supported.
    length : int, optional
        The desired lag until the autocovariances are estimated and returned, by default it is
        the length of the time-series.
    debug : bool, optional
        If `True` it tests that C(i) = C(-i)^T for i=1...m, by default True.

    Returns
    -------
    acovs : np.ndarray
        The array of autocovariances for the given length.
    """

    size, dims = data.shape[0], data.shape[1]   # n, d

    if length is None:
        length = size

    acovs = np.empty(shape=(length, dims, dims), dtype=np.csingle)

    half_size = int(np.ceil(size/2)-1)   # argmax, c(0)
    half_length = int(np.ceil(length/2)-1)  # argmax, c(0)

    if length % 2 == 1:
        lb, rb = half_size-half_length, half_size+half_length+1
    else:
        lb, rb = half_size-half_length, half_size+half_length+2

    for ii in data.columns:
        mean_i = np.mean(data[ii].values)
        data[ii] -= mean_i
    
    data = data.to_numpy().astype(np.single)

    # np.einsum('ij,ik->i', data, np.transpose(data, (1, 0)))

    for i in range(data.shape[1]):
        # print(data.columns[i:])
        for j in range(data.shape[1]):
            # j += i
            # mean_i = np.mean(data[ii].values)
            # mean_j = np.mean(data[jj].values)
            f_ij = signal.correlate(data[:, i],#-mean_i,
                                    data[:, j],#-mean_j,
                                    mode="same", method="fft")
            f_ij = np.flip(f_ij)
            f_ij = f_ij[lb:rb]
            # for k in range(nperseg):
            acovs[:, i, j] = f_ij# / size
            # if i!= j:
            #     acovs[:, j, i] = f_ij
        print(f"dim {i} done")
    
    acovs /= size



    # if length < size:
    #     if length % 2 == 1:
    #         acovs = acovs[half_size-half_length:half_size+half_length+1, :, :]
    #     else:
    #         acovs = acovs[half_size-half_length:half_size+half_length+2, :, :]

    if debug:
        print(f"length: {length}")
        assert acovs.shape[0] == length
        print(acovs)
        assert np.argmax(acovs[:, 0, 0]) == half_length

        if length % 2 == 1:
            np.testing.assert_array_almost_equal(
                acovs, np.conj(np.transpose(np.flip(acovs, axis=0), (0, 2, 1))), decimal=4)
        else:
            np.testing.assert_array_almost_equal(
                acovs[:-1], np.conj(np.transpose(np.flip(acovs[:-1], axis=0), (0, 2, 1))), decimal=4)

    return acovs


def bolla_Ms(data, nperseg=None, fs=1.0, debug=True):
    """
    Estimates the M matrices of the [SPMA] paper form sample covariance matrices coming 
    ou of `bolla_autocovs()`. The resulting matrix sequence estimates the spectral density matrix 
    function of the multidimensional time-series for large enough data.

    This is the theoretically justified sequence of matrices to be used for dynamic PCs.

    This function is deprecated due to being extremely slow. It is kept only since it is readable
    and can be used to test if newer alternatives give the same results.

    Parameters
    ----------
    data : pd.DataFrame (ToDo: extend to np.ndarray)
        The multidimensional time-series as a pandas DataFrame where the columns and rows should
        indicate the dimensions and timestamps, respectively. Missing values are not supported.
    nperseg : int, optional
        Number of the Fourier frequencies at the M matrices will be estimated. 
        It also determines the number of lags until the autocovariances will be estimated.
        It will be equal to the size of the output, by default it is the length of the time-series.
    fs : float, optional
        Sampling frequency of the time series, by default 1.0.
    debug : bool, optional
        Checks some theoretical properties of the matrix series, by default True.

    Returns
    -------
    freqs : ndarray
        Array of sample frequencies on [-\pi * fs, \pi * fs].
    spectral_array : ndarray
        The M matrix series of the time-series in `data`.
        Its shape is (nperseg, data.shape[0], data.shape[0]).
    """

    size, dims = data.shape[0], data.shape[1]   # n, d
    acovs = bolla_autocovs(data, length=nperseg, debug=debug)

    if nperseg is None:
        nperseg = size

    half_nperseg = int(np.ceil(nperseg/2)-1)  # argmax, c(0)

    if nperseg % 2 == 1:
        acovs_half = acovs[half_nperseg:, :, :]
        assert acovs_half.shape[0] == half_nperseg+1
    else:
        acovs_half = acovs[half_nperseg:, :, :]
        assert acovs_half.shape[0] == half_nperseg+2

    # freqs = np.linspace(0, 2*np.pi, nperseg, endpoint=False)

    freqs = fft.fftfreq(nperseg, 1/fs) * 2 * \
        np.pi  # for consistency with scipy

    halfrange = np.arange(0, acovs_half.shape[0])

    if nperseg % 2 == 1:
        M = np.array([
            np.sum(acovs_half[halfrange, :, :] * np.exp(1j * omeg_j * halfrange)[:, None, None], axis=0) +
            np.sum(np.transpose(acovs_half[halfrange[1:], :, :], (0, 2, 1))
                   * np.exp(-1j * omeg_j * halfrange[1:])[:, None, None], axis=0)
            for omeg_j in freqs])
    else:  # same as the odd case but kept in case it shouldn't be
        M = np.array([
            np.sum(acovs_half[halfrange[:], :, :] * np.exp(1j * omeg_j * halfrange[:])[:, None, None], axis=0) +
            np.sum(np.transpose(acovs_half[halfrange[1:], :, :], (0, 2, 1))
                   * np.exp(-1j * omeg_j * halfrange[1:])[:, None, None], axis=0)
            for omeg_j in freqs])

    if debug:
        if np.iscomplexobj(data.values) == False:
            np.testing.assert_array_almost_equal(
                M[1:], np.conj(np.flip(M[1:], axis=0)))  # symmetry
            # if nperseg % 2 == 1:
            np.testing.assert_array_almost_equal(
                M, np.conj(np.transpose(M, (0, 2, 1))))  # self-adjoint (maybe it should be true only in the odd case)

    return freqs, M


def bolla_Ms_fast(data, nperseg=None, fs=1.0, debug=True):
    """
    Estimates the M matrices of the [SPMA] paper form sample covariance matrices coming 
    ou of `bolla_autocovs()`. The resulting matrix sequence estimates the spectral density matrix 
    function of the multidimensional time-series for large enough data.

    This is the theoretically justified sequence of matrices to be used for dynamic PCs.

    This function is a newer, faster version of `bolla_Ms()`. It utilizes that the definition of
    M matrices is a Fourier transformation of the covariance function with an extra rotation.

    Parameters
    ----------
    data : pd.DataFrame (ToDo: extend to np.ndarray)
        The multidimensional time-series as a pandas DataFrame where the columns and rows should
        indicate the dimensions and timestamps, respectively. Missing values are not supported.
    nperseg : int, optional
        Number of the Fourier frequencies at the M matrices will be estimated. 
        It also determines the number of lags until the autocovariances will be estimated.
        It will be equal to the size of the output, by default it is the length of the time-series.
    fs : float, optional
        Sampling frequency of the time series, by default 1.0.
    debug : bool, optional
        Checks some theoretical properties of the matrix series, by default True.

    Returns
    -------
    freqs : ndarray
        Array of sample frequencies on [-\pi * fs, \pi * fs].
    spectral_array : ndarray
        The M matrix series of the time-series in `data`.
        Its shape is (nperseg, data.shape[0], data.shape[0]).
    """

    size, dims = data.shape[0], data.shape[1]   # n, d
    acovs = bolla_autocovs(data, length=nperseg, debug=debug)

    if nperseg is None:
        nperseg = size

    half_nperseg = int(np.ceil(nperseg/2)-1)  # k, argmax, c(0)

    freqs = fft.fftfreq(nperseg, 1/fs) * 2 * np.pi

    assert acovs.shape[0] == nperseg

    transf = fft.ifftn(acovs, axes=(0,), s=(nperseg,)) * nperseg

    if nperseg % 2 == 0:  # maybe some simplification is possible here
        extra = np.exp(-1j * half_nperseg * freqs)
        last_extra = np.exp(-1j * freqs)
        last_term = np.einsum('i,jk->ijk', last_extra,
                              np.transpose(acovs[-1, :, :], (1, 0)))
        transf = np.add(transf, last_term)
    else:
        extra = np.exp(-1j * half_nperseg * freqs)

    M = np.einsum('ijk,i->ijk', transf, extra)

    if debug:
        if np.iscomplexobj(data.values) == False:
            # for i in range(M.shape[0]):
            np.testing.assert_array_almost_equal(
                M[1:], np.conj(np.flip(M[1:], axis=0)), decimal=4)  # symmetry
            # if nperseg % 2 == 1:
            np.testing.assert_array_almost_equal(
                M, np.conj(np.transpose(M, (0, 2, 1))), decimal=4)  # self-adjoint (maybe it should be true only in the odd case)

    return freqs, M


##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

class DynPCA:
    """
    Dynamic principal component analysis (DynPCA).

    Linear dimensionality reduction of a multidimensional time-series data using the eigenvalue 
    decomposition of the M matrices, as outlined in the [SPMA] paper.
    The result is a multidimensional time-series of same length as the input but the number
    of dimensions is less. Each new dimension corresponds to a dynamic PC of the original 
    time-series.

    The input data should be centered (but the neccesity of scaling depends on the data) for 
    each feature before fiting. The algorithm works best on stationary time-series data. 

    The structure of this class mimics the sci-kit learn API standards.
    """

    def __init__(self, n_components, densif_method="linear", n_perseg=None):
        self.n_components = n_components
        self.densif_method = densif_method
        self.n_perseg = n_perseg

        self.cols = None
        self.omeg = None
        self.sp_array = None
        self.sp_eig = None
        self.sp_vect = None
        self.fit_dim = None

    def fit(self, data, use_M_matrices=False, n_perseg=None, *args, **kwargs):
        """
        Fit the model with `data`.

        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            The multidimensional time-series as a pandas DataFrame where the columns and rows should
            indicate the dimensions and timestamps, respectively. Missing values are not supported.
        use_M_matrices : bool, optional
            If `True` the eigenvectors of the M matrices will be used from the [SPMA] paper,
            if `False` the eigenvectors of the spectral density estimation will be used for fiting,
            by default False
        n_perseg : int, optional
            Size (length of axis 0) of the spectral array used for the PC transformation.
            This parameter will be used only if the same parameter wasn't given to init().
        *args, **kwargs :
            Forwarded to the spectral array estimation method
            bolla_Ms_fast() or multidim_spectral_density_estimate()

        Returns
        -------
        self : DynPCA object
            Returns the instance itself.
        """
        if self.n_perseg is None:
            self.n_perseg = n_perseg

        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        self.cols = data.columns.tolist()
        self.fit_dim = data.shape

        if use_M_matrices:
            self.omeg, self.sp_array =\
                bolla_Ms_fast(data, nperseg=self.n_perseg, *args, **kwargs)
        else:
            self.omeg, self.sp_array =\
                multidim_spectral_density_estimate(
                    data, nperseg=self.n_perseg, *args, **kwargs)

        omeg, self.sp_eig, self.sp_vect = eigen_decomp(self.sp_array)
        # Note: these arrays are sparse, can't be used directly for transformation
        np.testing.assert_array_equal(omeg, self.omeg)

        return self

    def transform(self, data, n_components=None, return_pcs_only=True):
        """
        Apply the dimensionality reduction on `data`.

        Parameters
        ----------
        data : pd.DataFrame (ToDo: extend to np.ndarray)
            The multidimensional time-series as a pandas DataFrame where the columns and rows should
            indicate the dimensions and timestamps, respectively. Missing values are not supported.
        n_components : int
            The number of principal components to calculate, by default `self.n_components`
        return_pcs_only : bool, optional
            Return the principal components (time-domain) only, by default True

        Returns
        -------
        T_series : 
            see the docs of time_series_projection()
        Z_series : 
            see the docs of time_series_projection()
        princomps : pd.DataFrame
            see the docs of time_series_projection()
        tilde_big_u : 
            see the docs of time_series_projection()
        """
        if n_components is None:
            n_components=self.n_components
        T_series, Z_series, princomps, tilde_big_u =\
            time_series_projection(data,
                                   k=n_components,
                                   sp_array=self.sp_array,
                                   densify=self.densif_method)

        pcols = ["PC_{}".format(i+1) for i in range(n_components)]
        princomps = pd.DataFrame(np.squeeze(np.real(princomps)),
                                 columns=pcols)

        if return_pcs_only:
            return princomps
        else:
            return T_series, Z_series, princomps, tilde_big_u

    def fit_transform(self, data, use_M_matrices=False, n_perseg=None, return_pcs_only=True, *args, **kwargs):
        """
        This function is not tested!!!
        Fit the model with `data`, and apply the dimensionality reduction on `data`.

        Parameters
        ----------
        data : pd.DataFrame (ToDo: extend to np.ndarray)
            The multidimensional time-series as a pandas DataFrame where the columns and rows should
            indicate the dimensions and timestamps, respectively. Missing values are not supported.
        use_M_matrices : bool, optional
            If `True` the eigenvectors of the M matrices will be used from the [SPMA] paper,
            if `False` the eigenvectors of the spectral density estimation will be used for fiting,
            by default False
        return_pcs_only : bool, optional
            Return the principal components (time-domain) only, by default True
        *args, **kwargs :
            Forwarded to the spectral array estimation method
            bolla_Ms_fast() or multidim_spectral_density_estimate()

        Returns
        -------
        T_series : 
            see the docs of time_series_projection()
        Z_series : 
            see the docs of time_series_projection()
        princomps : pd.DataFrame
            see the docs of time_series_projection()
        tilde_big_u : 
            see the docs of time_series_projection()
        """
        self.fit(data=data, use_M_matrices=use_M_matrices, n_perseg=n_perseg)

        if return_pcs_only:
            princomps =\
                self.transform(data=data, return_pcs_only=return_pcs_only)
            return princomps
        else:
            T_series, Z_series, princomps, tilde_big_u =\
                self.transform(data=data, return_pcs_only=return_pcs_only)
            return T_series, Z_series, princomps, tilde_big_u

    def inverse_transform(self, princomps):
        """
        Transform princomps back to its original space. 
        In other words, return an input data whose transform would be these PCs.
        The returned data is close to the low-rank approximation of the time-series.

        Parameters
        ----------
        princomps : pd.DataFrame
            The dynamic PCs (in the time domain) of the time-series.
            It has the same length as the input data but the number of dimensions is only `n_comps`.

        Returns
        -------
        restored_data : pd.DataFrame
            The time-series restored from the PCs. It has the same length as the input data but the 
            number of dimensions is determined by the shape of `sp_array`.
        """
        restored_data = time_series_restore(princomps,
                                            sp_array=self.sp_array,
                                            densify=self.densif_method)

        restored_data = pd.DataFrame(np.squeeze(np.real(restored_data)),
                                     columns=self.cols)

        return restored_data

    def low_rank_transform(self, data, n_components=None):
        """
        Low rank approximation of a multidimensional time-series as described in the [SPMA] paper.
        The output of this function is mainly used in comparison with the time-series reconstructed from
        the dynamic PCs because these are theoretically the same.
        For more details see the docs of lra_bolla().

        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            The multidimensional time-series as a pandas DataFrame where the columns and rows should
            indicate the dimensions and timestamps, respectively. Missing values are not supported.
        n_components : int
            The number of principal components to calculate, by default `self.n_components`

        Returns
        -------
        lra_data : pd.DataFrame
            The low-rank approximation of the time-series.
            It has the same shape as the input data. 
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        if n_components is None:
            n_components = self.n_components

        lra_data = lra_bolla(data,
                             k=n_components,
                             sp_array=self.sp_array,
                             densify=self.densif_method)[-1]

        lra_data = pd.DataFrame(np.squeeze(np.real(lra_data)),
                                columns=data.columns)

        return lra_data

    def plot_eig_vals(self, log_scale=False, *args, **kwargs):
        """
        Simple function to plot the eigenvalue-process of the estimate spectral matrix process.

        Parameters
        ----------
        log_scale : bool, optional
            If `True` the logarithm of the eigenvalues will be plotted, by default False
        *args, **kwargs :
            Parameters forwarded to plt.subplots()
        """
        f, ax = plt.subplots(1, 1, *args, **kwargs)

        omeg_sh = fft.fftshift(self.omeg, axes=0)
        sp_eig_sh = fft.fftshift(self.sp_eig, axes=0)
        cc = (cycler(linestyle=['-', '--', '-.']) *
              cycler(color=plt.get_cmap("Paired").colors))
        ax.set_prop_cycle(cc)

        if log_scale:
            for i in range(self.sp_eig.shape[1]):
                # ax.set_title("spd log-eigvals of the time-series")
                ax.plot(omeg_sh, np.log(sp_eig_sh[:, i]),
                        label=str(i+1), linewidth=1)
                ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
            plt.ylabel('log-eigenvalues')

        else:
            for i in range(self.sp_eig.shape[1]):
                # ax.set_title("spd eigvals of the diff time-series")
                ax.plot(omeg_sh, sp_eig_sh[:, i],
                        label=str(i+1), linewidth=1)
                ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
            plt.ylabel('eigenvalues')

        plt.tight_layout()
        # plt.show()

    def plot_pcs(self, princomps, til=None, *args, **kwargs):
        """
        Small function to plot the principal components.

        Parameters
        ----------
        princomps : pd.DataFrame
            The PCs to plot
        *args, **kwargs :
            Parameters forwarded to plt.subplots()
        """
        if til is None:
            til=princomps.shape[0]
        f, ax = plt.subplots(princomps.shape[1], 1, *args, **kwargs)

        colorz = plt.get_cmap("Paired").colors

        if princomps.shape[1] == 1:
            ax.plot(princomps.iloc[:til, 0],
                    label="PCA_{}".format(1), color=colorz[0], linewidth=.5)
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

        else:
            for i in range(princomps.shape[1]):
                ax[i].plot(princomps.iloc[:til, i],
                           label="PCA_{}".format(i+1), color=colorz[i], linewidth=.5)
                ax[i].legend(bbox_to_anchor=(1, 1), loc='upper left')

        plt.tight_layout()
        # plt.show()

    def plot_pc_weights(self, i=1, absolute_only=True):
        """
        Small function to plot the weights/loadings of each original dimension of the time-series
        inside a chosen principal component in the frequency domain.
        These are complex weights, so the more interesting are the absolute weights.

        Parameters
        ----------
        i : int, optional
            The i'th principal component, by default 1
        """

        omeg_sh = fft.fftshift(self.omeg, axes=0)
        sp_vec_sh = fft.fftshift(self.sp_vect, axes=0)

        halflen = int(len(omeg_sh)/2)

        cc = (cycler(linestyle=['-', '--', '-.']) *
              cycler(color=plt.get_cmap("Paired").colors))

        if absolute_only:
            f, ax = plt.subplots(1, 1, figsize=(12, 4))

            ax.set_prop_cycle(cc)
            ax.set_title("absolute weight inside PC_{}".format(i))
            for j, c in zip(range(sp_vec_sh.shape[1]), self.cols):
                ax.plot(omeg_sh[halflen:], np.absolute(sp_vec_sh[halflen:, j, i-1]),
                        label=c, linewidth=1)
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

            plt.tight_layout()

        else:
            f, ax = plt.subplots(3, 1, figsize=(12, 12))

            ax[0].set_prop_cycle(cc)
            ax[0].set_title("absolute weight inside PC_{}".format(i))
            for j, c in zip(range(sp_vec_sh.shape[1]), self.cols):
                ax[0].plot(omeg_sh[halflen:], np.absolute(sp_vec_sh[halflen:, j, i-1]),
                           label=c, linewidth=1)
            ax[0].legend(bbox_to_anchor=(1, 1), loc='upper left')

            ax[1].set_prop_cycle(cc)
            ax[1].set_title("real weight inside PC_{}".format(i))
            for j, c in zip(range(sp_vec_sh.shape[1]), self.cols):
                ax[1].plot(omeg_sh[halflen:], np.real(sp_vec_sh[halflen:, j, i-1]),
                           label=c, linewidth=1)
            ax[1].legend(bbox_to_anchor=(1, 1), loc='upper left')

            ax[2].set_prop_cycle(cc)
            ax[2].set_title("imag weight inside PC_{}".format(i))
            for j, c in zip(range(sp_vec_sh.shape[1]), self.cols):
                ax[2].plot(omeg_sh[halflen:], np.imag(sp_vec_sh[halflen:, j, i-1]),
                           label=c, linewidth=1)
            ax[2].legend(bbox_to_anchor=(1, 1), loc='upper left')

            plt.tight_layout()
        # plt.show()

    def pc_weights_at_max(self):
        """
        Returns dataframes containing the weights/loadings of each PCs at the frequency where the 
        corresponding eigenvalue process is maximal.

        Returns
        -------
        df_abs : pd.DataFrame
            Absolute values of the weights
        df_real : pd.DataFrame
            Real part of the weights
        df_imag : pd.DataFrame
            Imaginary part of the weights
        """
        halflen = int(np.ceil(self.n_perseg/2)-1)

        omeg_half = fft.fftshift(self.omeg, axes=0)[halflen:]
        sp_vec_half = fft.fftshift(self.sp_vect, axes=0)[halflen:]
        sp_eig_half = fft.fftshift(self.sp_eig, axes=0)[halflen:]

        dim = sp_eig_half.shape[1]
        inds = ["max_freq_at", *list(self.cols)]
        colls = ["PC_{}".format(i) for i in range(1, dim+1)]
        df_abs = pd.DataFrame(columns=colls, index=inds)
        df_real = pd.DataFrame(columns=colls, index=inds)
        df_imag = pd.DataFrame(columns=colls, index=inds)

        for i in range(sp_eig_half.shape[1]):
            amax = np.argmax(sp_eig_half[:, i])
            df_abs.loc["max_freq_at", colls[i]] = omeg_half[amax]
            df_real.loc["max_freq_at", colls[i]] = omeg_half[amax]
            df_imag.loc["max_freq_at", colls[i]] = omeg_half[amax]
            # print("max frequency at {} on scale [0,pi]\n".format(omeg_half[amax]))

            # print("\nthe absolute weights inside PC_{} at max eigenvalue per col:".format(i+1))
            for j, c in zip(range(sp_vec_half.shape[1]), self.cols):
                df_abs.loc[c, colls[i]] = np.absolute(sp_vec_half[amax, j, i])
                # print("{}: {}".format(c, np.absolute(sp_vec_half[amax,j,i])))

            # print("\nreal part of the weights inside PC_{} at max eigenvalue per col:".format(i+1))
            for j, c in zip(range(sp_vec_half.shape[1]), self.cols):
                df_real.loc[c, colls[i]] = np.real(sp_vec_half[amax, j, i])
                # print("{}: {}".format(c, np.real(sp_vec_half[amax,j,i])))

            # print("\nimag part of the weights inside PC_{} at max eigenvalue per col".format(i+1))
            for j, c in zip(range(sp_vec_half.shape[1]), self.cols):
                df_imag.loc[c, colls[i]] = np.imag(sp_vec_half[amax, j, i])
                # print("{}: {}".format(c, np.imag(sp_vec_half[amax,j,i])))

        return df_abs, df_real, df_imag

    def compression_ratio(self, n_components=None, n=None):
        """
        The theoretical compression ratio achiveable with the chosen parameters.
        It assumes that the data and the output of the DynPCA are of the same datatype.
        An n * d shaped time-series can be approximated with 
        n * n_comp + nperseg * d * n_comps * 2 parameters. 

        Parameters
        ----------
        n_components : int
            Number of estimated PCs, by default self.n_components
        n : int, optional
            length of the time-series, by default it will be length of the data the parameters were fitted.

        Returns
        -------
        float
            The compression ratio
        """
        if n_components is None:
            n_components = self.n_components
        if n is None:
            n = self.fit_dim[0]
        nomi = n * n_components + self.n_perseg * \
            self.fit_dim[1] * n_components * 2
        denomi = n * self.fit_dim[1]

        return nomi/denomi
