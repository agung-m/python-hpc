import time

import numpy as np
import numba
import psutil


@numba.njit(parallel=True)
def run_ids_shuffle(num_samples, num_perms, nthreads):
    split_size, rem = divmod(num_perms, nthreads)

    perm_ids = np.empty((num_perms, num_samples), dtype='int')

    for tid in numba.prange(nthreads):
        start, end = tid*split_size, (tid+1)*split_size
        if tid == nthreads - 1:
            end += rem

        for p in range(start, end):
            perm_ids[p] = np.random.permutation(num_samples)

    return perm_ids


@numba.njit
def reorder_arr_ids(arr, dest_arr, ids, n):
    #for i in range(n):
    #    arr[i], arr[ids[i]] = arr[ids[i]], arr[i]
    # for i in range(n):
    #     dest_arr[i] = arr[ids[i]]

    return arr[ids]


num_samples = 500000
num_perms = 1000

n_cores = psutil.cpu_count(logical=False)
phen_arr = np.random.normal(size=num_samples)

t0 = time.perf_counter()
perm_ids = run_ids_shuffle(num_samples, num_perms, n_cores)
t_elapsed = time.perf_counter() - t0
print(" Shuffle idx (numba): ({:.4f} s)".format(t_elapsed))
print(perm_ids)

t0 = time.perf_counter()
for perm in range(num_perms):
    shuffle_phen = phen_arr[perm_ids[perm]]
t_elapsed = time.perf_counter() - t0
print(" Reorder array: ({:.4f} s)".format(t_elapsed))

thread_phens = np.empty((n_cores, num_samples), dtype='f4')
t0 = time.perf_counter()
for perm in range(num_perms):
    shuffle_phen = reorder_arr_ids(phen_arr, thread_phens[0], perm_ids[perm], num_samples)
t_elapsed = time.perf_counter() - t0
print(" Reorder array (numba+swap): ({:.4f} s)".format(t_elapsed))

