from joblib import Parallel, delayed

import numpy as np
from scipy.sparse import csr_matrix


def ps(nrows, nchoices, nsamples, nthreads, rseed) :
    np.random.seed(rseed)
    rows = np.array(
        Parallel(n_jobs=nthreads)(
            delayed(np.random.choice)(nchoices, size=nsamples, replace=False)
            for i in range(nrows)
        )
    )
    return rows
# -- end function

def ps2(batches, nchoices, nsamples, nthreads, rseed) :
    np.random.seed(rseed)
    sample_rows = lambda nrows : [
        np.random.choice(nchoices, size=nsamples, replace=False) for i in range(nrows)
    ]
    
    tmp_rows = Parallel(n_jobs=nthreads)(
        delayed(sample_rows)(batches[i])
        for i in range(nthreads)
    )

    all_rows = []
    for l in tmp_rows :
        all_rows.extend(l)
    rows = np.array(all_rows)
    return rows
# -- end function

def split_for_parallelism(n, t) :
    ibatch = int(n / t)
    leftover = n % t
    batches = [ ibatch ] * t
    for i in range(leftover) :
        batches[i % t] += 1
    return batches
# -- end function

if __name__ == '__main__' :
    nrows = 8
    nchoices = 10
    nsamples = 3
    rseed = 5489

    for nthreads in range(3) :
        print('With %i threads ...' % (nthreads + 1))
        print('Sampling scheme 1')
        print('='*30)
        ret = ps(
            nrows = nrows,
            nchoices = nchoices,
            nsamples = nsamples,
            nthreads = nthreads + 1,
            rseed = rseed,
        )
        print(ret.shape)
        print (ret)
        print('='*30)
        print('Sampling scheme 2')
        b = split_for_parallelism(nrows, nthreads + 1)
        assert len(b) == nthreads + 1
        print('='*30)
        ret = ps2(
            batches = b,
            nchoices = nchoices,
            nsamples = nsamples,
            nthreads = nthreads + 1,
            rseed = rseed,
        )
        print(ret.shape)
        print (ret)
        print('='*30)
# -- end function
