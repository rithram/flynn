from math import floor
import numpy as np
from joblib import Parallel, delayed

from utils.parproc import split_for_parallelism

from .flyhash import FlyHash

import logging
logger = logging.getLogger('FlyNN')

class FHBloomFilter:
    def __init__(self, **kwargs):
        self.k = None
        self.fh = FlyHash(**kwargs)
        self.bloom_filters = None
        self.kwargs = kwargs
        self.classes_ = None
        self.n_classes_ = None
        self.batch_size = (
            kwargs['batch_size'] if 'batch_size' in kwargs else 128
        )
        self.kwargs['batch_size'] = self.batch_size
        self.nthreads = kwargs['nthreads'] if 'nthreads' in kwargs else 1
        self.kwargs['nthreads'] = self.nthreads
        # hyperparameters for binary vs. numeric bloom filters
        # Default binary bloom filters
        self.kwargs['binary'] = (
            kwargs['binary'] if 'binary' in kwargs else True
        )
        self.binary_bf = self.kwargs['binary']
        # The 'c' hyperparameter in (0,1) for non-binary bloom filters
        assert self.binary_bf or ('c' in kwargs)
        if 'c' in kwargs:
            if self.binary_bf:
                logger.info(
                    'Ignoring the \'c\' hyperparameter,'
                    ' only used in non-binary NBF'
                )
            else:
                assert kwargs['c'] > 0. and kwargs['c'] < 1.
                self.kwargs['c'] = kwargs['c']
                self.nonbinary_bf_c = self.kwargs['c']

    def set_params(self, **kwargs):
        self.k = None
        self.bloom_filters = None
        for k in kwargs:
            self.kwargs[k] = kwargs[k]
        self.classes_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        # Generate flyhash
        nrows, ncols_in = X.shape
        self.ncols_in = ncols_in
        # Fitting the FlyHash
        self.fh.fit(X)
        # Get the value of k, the NNZ per row of X
        self.k = self.fh.k
        # Get the output dimensionality
        ncols_out = self.fh.m
        # Get number of classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        l2i = {l: i for i, l in enumerate(self.classes_)}
        # Split data for parallel processing
        th_idxs = np.insert(np.cumsum(
            split_for_parallelism(nrows, self.nthreads)
        ), 0, 0)
        # bloom filter type
        bf_type = np.bool if self.binary_bf else np.int64
        # process each thread batch in parallel
        feval = lambda sidx, eidx: per_thread_job(
            bf_nrows=self.n_classes_,
            bf_ncols=ncols_out,
            th_start_idx=sidx,
            th_end_idx=eidx,
            batch_size=self.batch_size,
            X=X,
            y=y,
            flyhash=self.fh,
            l2i=l2i,
            bf_type=bf_type,
        )
        all_th_bfs = Parallel(n_jobs=self.nthreads)(
            delayed(feval)(th_idxs[i], th_idxs[i+1])
            for i in range(self.nthreads)
        )
        # consolidate results from all threads
        bloom_filters = np.zeros((self.n_classes_, ncols_out), dtype=bf_type)
        for bf in all_th_bfs:
            bloom_filters += bf
        # invert bloom filter & transpose
        self.bloom_filters = (
            np.invert(bloom_filters) if self.binary_bf
            else (1.0 - self.nonbinary_bf_c)**bloom_filters
        )
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
        # Process points in batches
        nbatches = floor(nrows / self.batch_size) + int(
            (nrows % self.batch_size) > 0
        )
        # print(
        #     'Testing NBF with %i batches of size %i ...' % (
        #         nbatches, self.batch_size
        #     )
        # )
        start_idx = 0
        fX = []
        for j in range(nbatches):
            end_idx = min(start_idx + self.batch_size, nrows)
            # Generate flyhash
            batch_fhX = self.fh.transform(X[start_idx : end_idx, :])
            # For bloom filters from each class, compute
            #   - ((W . X)^\top 1) / self.k
            batch_fX = batch_fhX.astype(np.int) @ self.tbf
            fX.extend(batch_fX.tolist())
            start_idx = end_idx
        assert len(fX) == nrows, (
            'Expected %i, obtained %i' % (nrows, len(fX))
        )
        return np.array(fX)

    def decision_function(self, X):
        return self._bf_scores(X)

    def predict(self, X):
        nrows, ncols = X.shape
        th_idxs = np.insert(np.cumsum(
            split_for_parallelism(nrows, self.nthreads)
        ), 0, 0)
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
        return np.array(y)

    def predict_proba(self, X):
        nrows, ncols = X.shape
        th_idxs = np.insert(np.cumsum(
            split_for_parallelism(nrows, self.nthreads)
        ), 0, 0)
        fX = np.vstack(
            Parallel(n_jobs=self.nthreads)(
                delayed(self._bf_scores)(X[th_idxs[i] : th_idxs[i + 1], :])
                for i in range(self.nthreads)
            )
        ).astype(float) / self.k
        exp_neg_fX = np.exp(-fX)
        probs = exp_neg_fX / np.sum(exp_neg_fX, axis=1)[:, None]
        return probs

    def get_params(self, deep=False):
        return self.kwargs


def per_thread_job(
        bf_nrows, bf_ncols, th_start_idx, th_end_idx,
        batch_size, X, y, flyhash, l2i, bf_type
):
    bloom_filters = np.zeros((bf_nrows, bf_ncols), dtype=bf_type)
    nrows = th_end_idx - th_start_idx
    # Process points in batches
    nbatches = floor(nrows / batch_size) + int((nrows % batch_size) > 0)
    start_idx = th_start_idx
    for j in range(nbatches):
        end_idx = min(start_idx + batch_size, th_end_idx)
        fhX = flyhash.transform(X[start_idx : end_idx, :])
        nrows_batch, ncols_out_batch = fhX.shape
        assert nrows_batch == (end_idx - start_idx), (
            'The number of rows batch do not match: %i vs. %i'
            % (end_idx - start_idx, nrows_batch)
        )
        assert ncols_out_batch == bf_ncols
        # For each class, compute W = Complement(X_1 V X_2 V ....)
        for features, label in zip(fhX, y[start_idx : end_idx]):
            bloom_filters[l2i[label]] += features
        # Update batch start idx
        start_idx = end_idx
    assert start_idx == th_end_idx
    return bloom_filters
