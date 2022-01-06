import logging
logger = logging.getLogger('SBFC')
from math import floor

import numpy as np
from joblib import Parallel, delayed

from utils.parproc import split_for_parallelism


class SimHashBloomFilter:
    def __init__(self, **kwargs):
        self.expansion_factor = kwargs['expansion_factor']
        self.bloom_filters = None
        self.kwargs = kwargs
        self.classes_ = None
        self.n_classes_ = None
        self.rnd_proj_mat = None
        self.batch_size = (
            kwargs['batch_size'] if 'batch_size' in kwargs else 128
        )
        self.kwargs['batch_size'] = self.batch_size
        self.nthreads = kwargs['nthreads'] if 'nthreads' in kwargs else 1
        self.kwargs['nthreads'] = self.nthreads

    def set_params(self, **kwargs):
        self.bloom_filters = None
        for k in kwargs:
            self.kwargs[k] = kwargs[k]
        self.classes_ = None
        self.n_classes_ = None
        self.rnd_proj_mat = None

    def fit(self, X, y):
        # Generate simhash
        nrows, ncols_in = X.shape
        self.ncols_in = ncols_in
        # Set up the SimHash
        # Get the output dimensionality
        ncols_out = max(int(ncols_in * self.expansion_factor), 2)
        assert self.rnd_proj_mat is None, (
            'projection matrix should be None, method might be already fit'
        )
        self.rnd_proj_mat = np.random.normal(0., 1., size=(ncols_in, ncols_out))
        # Get number of classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        l2i = {l: i for i, l in enumerate(self.classes_)}
        # Split data for parallel processing
        th_idxs = np.insert(
            np.cumsum(split_for_parallelism(nrows, self.nthreads)), 0, 0
        )
        # process each thread batch in parallel
        feval = lambda sidx, eidx: per_thread_job(
            bf_nrows=self.n_classes_,
            bf_ncols=ncols_out,
            th_start_idx=sidx,
            th_end_idx=eidx,
            batch_size=self.batch_size,
            X=X,
            y=y,
            rp_mat=self.rnd_proj_mat,
            l2i=l2i
        )
        all_th_bfs = Parallel(n_jobs=self.nthreads)(
            delayed(feval)(th_idxs[i], th_idxs[i+1])
            for i in range(self.nthreads)
        )
        # consolidate results from all threads
        bloom_filters = np.zeros((self.n_classes_, ncols_out), dtype=np.bool)
        for bf in all_th_bfs:
            bloom_filters += bf
        # invert bloom filter & transpose
        self.bloom_filters = np.invert(bloom_filters)
        self.tbf = np.transpose(self.bloom_filters)
        return self

    def partial_fit(X, y):
        pass

    def _bf_scores(self, X):
        nrows, ncols = X.shape
        assert self.ncols_in == ncols, (
            'NBF trained with %i columns, trying prediction with %i columns'
            % (self.ncols_in, ncols)
        )
        assert not self.bloom_filters is None, ('Method not fit yet')
        assert not self.rnd_proj_mat is None, ('Method not fit yet')
        # Process points in batches
        nbatches = (
            floor(nrows / self.batch_size)
            + int((nrows % self.batch_size) > 0)
        )
        start_idx = 0
        fX = []
        for j in range(nbatches):
            end_idx = min(start_idx + self.batch_size, nrows)
            # Generate simhash
            batch_shX = simhash(self.rnd_proj_mat, X[start_idx : end_idx, :])
            # For bloom filters from each class, compute
            #   - ((W . X)^\top 1) / self.k
            batch_fX = batch_shX.astype(np.int) @ self.tbf
            fX.extend(batch_fX.tolist())
            start_idx = end_idx
        assert len(fX) == nrows, ('Expected %i, obtained %i' % (nrows, len(fX)))
        return np.array(fX)

    def predict(self, X):
        nrows, ncols = X.shape
        th_idxs = np.insert(
            np.cumsum(split_for_parallelism(nrows, self.nthreads)), 0, 0
        )
        fX = np.vstack(
            Parallel(n_jobs=self.nthreads)(
                delayed(self._bf_scores)(X[th_idxs[i] : th_idxs[i + 1], :])
                for i in range(self.nthreads)
            )
        )
        # Return the class with minimum value
        # Break ties randomly, currently it choses minimum index with min value
        min_bf_scores = np.min(fX, axis=1)
        y = []
        nties = 0
        for min_bf_score, fx in zip(min_bf_scores, fX):
            y_set = self.classes_[fx == min_bf_score]
            l = None
            if len(y_set) > 1:
                nties += 1
                l = y_set[np.random.randint(0, len(y_set))]
            else:
                l = y_set[0]
            y.append(l)
        logging.debug('%i / %i points have ties' % (nties, nrows))
        return np.array(y)

    def predict_proba(self, X):
        nrows, ncols = X.shape
        th_idxs = np.insert(np.cumsum(split_for_parallelism(nrows, self.nthreads)), 0, 0)
        fX = np.vstack(
            Parallel(n_jobs=self.nthreads)(
                delayed(self._bf_scores)(X[th_idxs[i] : th_idxs[i + 1], :])
                for i in range(self.nthreads)
            )
        ).astype(float)
        exp_neg_fX = np.exp(-fX)
        probs = exp_neg_fX / np.sum(exp_neg_fX, axis=1)[:, None]
        return probs

    def get_params(self, deep=False):
        return self.kwargs


def simhash(rnd_proj_mat, X):
    ncols_in, ncols_out = rnd_proj_mat.shape
    assert ncols_in == X.shape[1]
    projX = X @ rnd_proj_mat
    assert ncols_out == projX.shape[1]
    assert projX.shape[0] == X.shape[0]
    retX = np.zeros_like(projX, dtype=np.bool)
    retX[projX > 0.] = True
    return retX


def per_thread_job(
        bf_nrows, bf_ncols, th_start_idx,
        th_end_idx, batch_size, X, y, rp_mat, l2i
):
    bloom_filters = np.zeros((bf_nrows, bf_ncols), dtype=np.bool)
    nrows = th_end_idx - th_start_idx
    assert bf_ncols == rp_mat.shape[1]
    # Process points in batches
    nbatches = floor(nrows / batch_size) + int((nrows % batch_size) > 0)
    start_idx = th_start_idx
    for j in range(nbatches):
        end_idx = min(start_idx + batch_size, th_end_idx)
        shX = simhash(rp_mat, X[start_idx : end_idx, :])
        nrows_batch, ncols_out_batch = shX.shape
        assert nrows_batch == (end_idx - start_idx), (
            'The number of rows batch do not match: %i vs. %i'
            % (end_idx - start_idx, nrows_batch)
        )
        assert ncols_out_batch == bf_ncols, (
            'Hashed X %i != BF %i' % (ncols_out_batch, bf_ncols)
        )
        # For each class, compute W = Complement(X_1 V X_2 V ....)
        for features, label in zip(shX, y[ start_idx : end_idx ]) :
            bloom_filters[l2i[label]] += features
        # Update batch start idx
        start_idx = end_idx
    assert start_idx == th_end_idx
    return bloom_filters


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    rnd_proj_mat = np.random.normal(0., 1., size=(X.shape[1], 10))
    batch = 5
    hashed_X = simhash(rnd_proj_mat, X[:batch, :])
    print(hashed_X)
    kwargs = { 'expansion_factor': 20. }
    sbf = SimHashBloomFilter(**kwargs)
    sbf.fit(X, y)
    print('RP matrix:', sbf.rnd_proj_mat.shape)
    print('BF matrix:', sbf.tbf.shape)
    print('Preds: ', sbf.predict(X[:batch, :]))
    print('Probs: ', sbf.predict_proba(X[:batch, :]))
