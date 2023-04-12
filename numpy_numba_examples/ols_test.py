#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022 Mulya Agung

import time

import numpy as np
import numba
import psutil


def sim_phen(phen_arr, start, end, num_samples, noise_dist='norm', t_dof=2):
    for rep in range(start, end):
        np.random.seed(num_samples * rep)
        if noise_dist == 't':
            errors = np.random.standard_t(t_dof, num_samples)
        elif noise_dist == 'f':
            errors = np.random.f(1, num_samples - 2, num_samples)
        else:
            errors = np.random.normal(0, 1, num_samples)

        phen_arr[rep, :] = errors


@numba.njit(cache=True, parallel=True)
def nb_lstsq(snps, phen_arr, m, results, nthreads):

    snps_per_thread, rem = divmod(m, nthreads)

    for tid in numba.prange(nthreads):
        start, end = tid * snps_per_thread, (tid+1) *snps_per_thread
        if tid == nthreads-1:
            end += rem

        for sid in range(start, end):
            slopes, residuals, rank, s = np.linalg.lstsq(snps[sid].reshape((-1, 1)), phen_arr)
            results[sid, :] = slopes.ravel()
            #print(slopes.shape)


@numba.njit(cache=True, fastmath=True)
def ols_t(x, y):
    M = x.size

    x_sum = 0.
    y_sum = 0.
    x_sq_sum = 0.
    x_y_sum = 0.

    for i in range(M):
        x_sum += x[i]
        y_sum += y[i]
        x_sq_sum += x[i] ** 2
        x_y_sum += x[i] * y[i]

    slope = (M * x_y_sum - x_sum * y_sum) / (M * x_sq_sum - x_sum ** 2)
    intercept = (y_sum - slope * x_sum) / M

    x_bar = x_sum / M

    ssx = 0.0
    ssres = 0.0
    for i in range(M):
        ssx += (x[i] - x_bar) ** 2
        ssres += (y[i] - (intercept + x[i] * slope)) ** 2

    msres = ssres / (M - 2)
    se = np.sqrt(msres / ssx)
    t = slope / se
    return slope, se, t


@numba.njit
def gen_ranges_by_num_splits(n, num_splits, offset=0):
    if n < num_splits:
        split_size = 1
        num_splits = n
        rem = 0
    else:
        split_size, rem = divmod(n, num_splits)

    # Array: [[start, end]]
    splits = np.empty((num_splits, 2), dtype='uint32')

    splits[:, 1] = split_size
    for i in range(rem):
        splits[i, 1] += 1
    # Adjust start
    splits[0, 0] = offset


    for i in range(1, num_splits):
        splits[i, 0] = splits[i-1, 0] + splits[i-1, 1]

    for i in range(0, num_splits):
        splits[i, 1] += splits[i, 0]

    return splits


@numba.njit(parallel=True)
def ols_t_y_mat(x, yy, thread_ranges):
    for tid in numba.prange(len(thread_ranges)):
        start, end = thread_ranges[tid, 0], thread_ranges[tid, 1]

        for i in range(start, end):
            ols_t(x, yy[i])


@numba.njit(parallel=True)
def ols_t_y_mat_batch(xx, yy, n_snps, thread_ranges):
    for tid in numba.prange(len(thread_ranges)):
        start, end = thread_ranges[tid, 0], thread_ranges[tid, 1]

        for i in range(start, end):
            for sid in range(n_snps):
                ols_t(xx[sid], yy[i])


def run_ols():
    n_snps = 8192
    n_samples = 200
    x_arr = np.random.randint(0, 4, n_snps*n_samples).astype('f4').reshape((n_snps, -1))
    #print(x_arr)

    n_reps = 100
    y_arr = np.empty((n_samples,  n_reps), dtype='f4')
    t0 = time.perf_counter()
    sim_phen(y_arr, 0, n_samples, n_reps)
    t1 = time.perf_counter()
    print("Simulated phenotypes ({:.4f} s)".format(t1-t0))
    #print(y_arr)

    # y_0 = y_arr[:, 0].ravel()
    # print(y_0)
    # slopes_0 = np.linalg.lstsq(x_arr[0].reshape((-1, 1)), y_0)[0]
    # print(slopes_0)
    #
    # t0 = time.perf_counter()
    # results = np.empty((n_snps, n_reps), dtype='f4')
    # for snp in range(n_snps):
    #     slopes, residuals, rank, s = np.linalg.lstsq(x_arr[snp].reshape((-1, 1)), y_arr, rcond=None)
    #     results[snp, :] = slopes
    #     #np.linalg.lstsq(x_arr[snp].reshape((-1, 1)), y_arr)
    #     #results[:, snp] = b
    #     #print(slopes, slopes.shape)
    #     #print("--")
    #
    # t1 = time.perf_counter()
    # print("-- Elapsed time = {}".format(t1 - t0))

    try:
        """ Set MKL threads to 1 is necessary to run Numba threads due to conflicting resources.
                Otherwise, Numba may not be able allocate its threads, raising invalid OS proc id errors.
        """
        import mkl
        mkl.set_num_threads(12)
    except ModuleNotFoundError:
        pass

    # t0 = time.perf_counter()
    # results = np.empty((n_snps, n_reps), dtype='f4')
    # #eye = np.ones(n_samples)
    # for snp in range(n_snps):
    #     # M = np.stack([[x_arr[:, snp], eye]]).T
    #     #print(M)
    #     # Gelsy use parallel if sample size is
    #     p, res, rnk, s = lstsq(x_arr[snp].reshape((-1, 1)), y_arr, lapack_driver='gelsy')
    #     # slopes, residuals, rank, s = np.linalg.lstsq(x_arr[snp].reshape((-1, 1)), y_arr, rcond=None)
    #     # results[snp, :] = slopes
    #     # np.linalg.lstsq(x_arr[snp].reshape((-1, 1)), y_arr)
    #     # results[:, snp] = b
    #     # print(slopes, slopes.shape)
    #     # print("--")
    #
    # t1 = time.perf_counter()
    # print("-- Elapsed time (scipy) = {}".format(t1 - t0))

    yy = np.ascontiguousarray(y_arr.T)
    thread_ranges = gen_ranges_by_num_splits(n_reps, psutil.cpu_count(logical=False))
    t0 = time.perf_counter()
    results = np.empty((n_snps, n_reps), dtype='f4')
    # for snp in range(n_snps):
    #     ols_t_y_mat(x_arr[snp], yy, thread_ranges)

    ols_t_y_mat_batch(x_arr, yy, n_snps, thread_ranges)

    t1 = time.perf_counter()
    print("-- Elapsed time (custom) = {}".format(t1 - t0))

    try:
        """ Set MKL threads to 1 is necessary to run Numba threads due to conflicting resources.
            Otherwise, Numba may not be able allocate its threads, raising invalid OS proc id errors.
        """
        import mkl
        mkl.set_num_threads(1)
    except ModuleNotFoundError:
        pass

    nb_results = np.empty_like(results)
    nb_lstsq(x_arr, y_arr, n_snps, nb_results, nthreads=12)
    print("-- Elapsed time (numba) = {}".format(time.perf_counter() - t1))

    #print(results)


if __name__ == "__main__":
    run_ols()
