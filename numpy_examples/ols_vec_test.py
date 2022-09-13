import time

import numpy as np
import numba
import llvmlite.binding as llvm
llvm.set_option('', '--debug-only=loop-vectorize')


#@numba.jit
def jit_sim_phen(phen_arr, start, end, num_samples, noise_dist='norm', t_dof=2):
    for rep in range(start, end):
        np.random.seed(num_samples * rep)
        if noise_dist == 't':
            errors = np.random.standard_t(t_dof, num_samples)
        elif noise_dist == 'f':
            errors = np.random.f(1, num_samples - 2, num_samples)
        else:
            errors = np.random.normal(0, 1, num_samples)

        phen_arr[rep, :] = errors

#@numba.jit
def run_vec_ols():
    n_snps = 1000
    n_samples = 100000
    #x_arr = np.random.randint(0, 4, n_snps*n_samples).astype('f4').reshape((n_samples, -1))
    x_f_arr = np.random.randint(0, 4, n_snps * n_samples).astype('f4')

    x_arr = x_f_arr.reshape((n_snps, -1))
    #print(x_arr)

    n_reps = 100
    #y_arr = np.empty((n_samples,  n_reps), dtype='f4')
    y_samples = n_samples
    y_arr = np.empty((n_reps, y_samples), dtype='f4')
    jit_sim_phen(y_arr, 0, n_reps, y_samples)
    #print(y_arr)

    # y_0 = y_arr[:, 0].ravel()
    # print(y_0)
    # slopes_0 = np.linalg.lstsq(x_arr[0].reshape((-1, 1)), y_0)[0]
    # print(slopes_0)
    #

    y_x_arr = np.tile(y_arr[0], n_snps)

    vec_results = np.empty((n_snps * n_samples), dtype='f4')
    accums = np.empty(n_snps, dtype='f4')
    t0 = time.perf_counter()
    #results = np.empty((n_snps, n_reps), dtype='f4')
    #for snp in range(n_snps):
        #results[snp, :] = slopes
    vec_ols(x_arr, y_arr[0], n_snps, y_samples, vec_results)

    #vec_ols_2(x_f_arr, y_x_arr, len(x_f_arr), vec_results)
    #test(x_arr)

    t1 = time.perf_counter()

    vec_ols_2(x_f_arr, y_x_arr, len(x_f_arr), n_snps, n_samples, vec_results, accums)
    print("-- Elapsed time (numba) = {}".format(time.perf_counter() - t1))
    print("-- Elapsed time = {}".format(t1 - t0))

    #print(results)


@numba.njit
def test(a):
    for i in range(a.shape[0]):
        a[i] = a[i] * 2


@numba.njit(fastmath=True)
def vec_ols(x, y, x_rows, m, results):
    #M = x.size
    # x_y_sum = 0.0
    #
    # for i in range(m):n_snps
    #     x_y_sum += x[i] * y[i]

    for i in range(x_rows):
        for j in range(m):
            results[i] += x[i, j] * y[j]

    #return x_y_sum


@numba.njit(fastmath=True)
def vec_ols_2(x, y, x_n, n_snps, n_samples, results, accums):

    for i in range(x_n):
        results[i] = x[i] * y[i]

    # for i in range(n_snps):
    #     results[i] = sum(results[i*n_samples:(i+1)*n_samples])

    for i in range(n_snps):
        #for j in range(n_samples):
        accums[i] = sum(results[i*n_samples:(i+1)*n_samples])


@numba.njit(fastmath=True)
def vec_3(x, y, z, n, results):

    for i in range(n):
        results[i] = x[i] * y[i] * z[i]


def run_vec_3():
    n = 100000
    x = np.arange(n)
    y = np.random.normal(size=n)
    z = np.random.normal(size=n*2)
    results = np.empty(n, dtype='f4')

    t0 = time.perf_counter()
    vec_3(x, y, z[n*2:], n, results)
    print("Elapsed time = {}".format(time.perf_counter() - t0))


if __name__ == "__main__":
    #run_vec_ols()
    #print(test(np.asarray([1.0, 2.0, 3.0])))
    run_vec_3()

