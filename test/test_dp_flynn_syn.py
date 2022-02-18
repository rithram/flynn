import argparse
import contextlib
from itertools import product
from math import floor
import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import numpy as np
np.set_printoptions(precision=3)
import pandas as pd
from joblib import Parallel, delayed
import sys
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split

from fh.flyhash import FlyHash
from utils.parproc import split_for_parallelism

from eval_synthetic_data import create_dataset

import logging
logger = logging.getLogger('DP-FlyNN-FAST')

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class DPFlyNNFL:
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
        self.dp = kwargs['DP'] if 'DP' in kwargs else None
        self.all_th_bfs = None

    def set_params(self, **kwargs):
        self.k = None
        self.bloom_filters = None
        for k in kwargs:
            self.kwargs[k] = kwargs[k]
        self.classes_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        assert self.all_th_bfs is None
        # Generate flyhash
        nrows, ncols_in = X.shape
        self.ncols_in = ncols_in
        # Fitting the FlyHash
        self.fh.fit(X)
        # Get the value of k, the NNZ per row of X
        self.k = self.fh.k
        # Get the output dimensionality
        self.ncols_out = self.fh.m
        # Get number of classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        l2i = {l: i for i, l in enumerate(self.classes_)}
        logger.debug(f'Training with {nrows}x{ncols_in} and {self.n_classes_} classes')
        # Split data for parallel processing
        th_idxs = np.insert(np.cumsum(
            split_for_parallelism(nrows, self.nthreads)
        ), 0, 0)
        logger.debug(f'Thread chunks:{th_idxs}')
        # bloom filter type
        self.bf_type = np.bool if self.binary_bf else np.int64
        # process each thread batch in parallel
        feval = lambda sidx, eidx: per_thread_job(
            bf_nrows=self.n_classes_,
            bf_ncols=self.ncols_out,
            th_start_idx=sidx,
            th_end_idx=eidx,
            batch_size=self.batch_size,
            X=X,
            y=y,
            flyhash=self.fh,
            l2i=l2i,
            bf_type=self.bf_type,
        )
        logger.debug(f'Initiating parallel BF collection with {self.nthreads} workers')
        self.all_th_bfs = Parallel(n_jobs=self.nthreads)(
            delayed(feval)(th_idxs[i], th_idxs[i+1])
            for i in range(self.nthreads)
        )
        self.reduce_bfs()
        return self


    def reduce_bfs(self):
        assert (
            self.all_th_bfs is not None and
            self.ncols_out is not None and
            self.bf_type is not None and
            len(self.all_th_bfs) == self.nthreads
        )
        # consolidate results from all threads
        logger.debug(f'Consolidating {len(self.all_th_bfs)} BFs')
        bloom_filters = np.zeros((self.n_classes_, self.ncols_out), dtype=self.bf_type)
        for bf in self.all_th_bfs:
            dp_bf = self._DP(bf).astype(self.bf_type)
            logger.debug(f'{bloom_filters.dtype}, {bf.dtype}, {dp_bf.dtype}')
            bloom_filters += dp_bf
        # invert bloom filter & transpose
        self.bloom_filters = (
            np.invert(bloom_filters) if self.binary_bf
            else (1.0 - self.nonbinary_bf_c)**bloom_filters
        )
        self.tbf = np.transpose(self.bloom_filters)
        return self


    def _DP(self, W):
        if self.dp is None:
            return W
        logger.debug(f'In DP module with W {W.shape}')
        L, m = W.shape
        eps = self.dp['eps']
        tau = self.nthreads
        T = min(self.dp['T'], L * m)
        c0 = self.dp['c0']
        logger.debug(f'Appropriate %-tile: {np.percentile(W, axis=1, q=1. - float(T)/(m*L))}')
        ret = c0 * np.ones(W.shape)
        laplace_param = 2 * T * tau / eps
        logger.debug(f'Laplace distribution with param: {laplace_param:.4f}')
        weights = np.hstack([
            np.array([np.exp(c / laplace_param) for c in W[l, :]])
            for l in range(L)
        ])
        V = np.arange(m * L)
        assert len(weights) == len(V) == m * L
        logger.debug(f'Obtained sampling weights for each entry of the per-class FBFs')
        samples = []
        logger.debug(f'Sampling for {T} rounds ...')
        for t in range(T):
            weights /= np.sum(weights)
            i = np.random.choice(V, size=1, replace=True, p=weights)[0]
            noise = np.random.laplace(loc=0., scale=laplace_param, size = 1)[0]
            ridx = i // m
            cidx = i % m
            ret[ridx, cidx] = max(W[ridx, cidx] + noise, 0.)
            samples += [i]
            weights[i] = 0
            if T < 20:
                logger.debug(
                    f'Sample: {i} in [{m}], value: {W[ridx, cidx]},'
                    f' noise: {noise:.4f}, updated: {ret[ridx, cidx]:.1f}')
        assert len(np.unique(samples)) == T, (
            f'# samples: {len(samples)}, unique: {len(np.unique(samples))}, T: {T}, L: {L}'
        )
        return ret


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

def save_list2file(rdict, fname, cols):
    res_list = []
    for k, v in rdict.items():
        if k == 'seed' or len(v) == 0:
            continue
        logger.debug(f'{k}: {np.mean(v):.4f} +- {np.std(v):.4f}')
        res_list += [(
            *k, np.mean(v), np.std(v), np.min(v), np.max(v), 
        )]
    logger.info(f'Saving results in {fname} ...')
    pd.DataFrame(res_list, columns=cols).to_csv(fname, header=True, index=False)
    logger.info('... done')
    

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout)
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--nthreads', help='Number of parallel workers', type=int,
        default=1
    )
    parser.add_argument(
        '-S', '--nrows', help='Number of rows in the data set', type=int,
        default=1000
    )
    parser.add_argument(
        '-d', '--ncols', help='Number of columns in the data set', type=int,
        default=30
    )
    parser.add_argument(
        '-L', '--nclasses', help='Number of classes in the data set', type=int,
        default=2
    )
    parser.add_argument(
        '-C', '--nclusters_per_class', help='Number of clusters per class',
        type=int, default=5
    )
    parser.add_argument(
        '-R', '--nreps', help='Number of repetitions', type=int, default=5
    )
    parser.add_argument(
        '-f', '--exp_factor', help='the expansion factor HP',
        type=float, default=20.,
    )
    parser.add_argument(
        '-k', '--wta_ratio', help='the WTA ratio',
        type=float, default=0.025,
    )
    parser.add_argument(
        '-c', '--conn_sparsity', help='the connection sparsity',
        type=float, default=0.1,
    )
    rounds_option = ['geom', 'lin']
    parser.add_argument(
        '-r', '--rounds_option', help='Values of #rounds to try',
        choices=rounds_option, default='geom',
    )
    parser.add_argument(
        '-T', '--step_size_for_T', help='Step size for T values',
        type=int, default=25,
    )
    eval_opts = ['non-private', 'all']
    parser.add_argument(
        '-e', '--evaluation', help='What methods to evaluate',
        choices=eval_opts, default='all',
    )

    args = parser.parse_args()
    nrows = args.nrows
    ncols = args.ncols
    nclasses = args.nclasses
    EPS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    NRNDS = [
        4, 8, 16, 32, 64, # , 128, 256, 512
    ] if args.rounds_option == 'geom' else np.arange(
        args.step_size_for_T, 601, step=args.step_size_for_T, dtype=np.int
    ).tolist() + [4, 8, 16, 32]
    gamma_list = np.arange(0.1, 1.0, step=0.1).tolist()
    nthreads = args.nthreads
    NREPS = args.nreps
    EF = args.exp_factor
    WTAR = args.wta_ratio
    CS = args.conn_sparsity
    TEST_SIZE = 5000
    res_list = []
    res_file = (
        f'DP-eff-{nrows}x{ncols}x{nclasses}-classes-{nthreads}-parties'
        f'-{NREPS}-reps-EF-{EF:.1f}-CS-{CS:.1f}-WTAR-{WTAR:.4f}.csv'
    )
    res_cols = ['gamma', 'EPS', 'T', 'MEAN', 'STD', 'MIN', 'MAX']
    ## xub = 100
    ## X = np.random.randint(0, xub, size=(nrows, ncols))
    ## y = np.random.randint(0, 2, size=nrows)
    with temp_seed(5489):
        X, y = create_dataset(
            nrows + TEST_SIZE,
            ncols,
            nclasses=nclasses,
            nclusters_per_class=args.nclusters_per_class
        )
    X1, X2, y1, y2 = train_test_split(X, y, test_size=TEST_SIZE, random_state=5489, stratify=y)
    fargs = {
        'expansion_factor': EF,
        'connection_sparsity': CS,
        'wta_ratio': WTAR,
        'wta_nnz': int(WTAR * EF * ncols),
        'binary': False,
        'c': 1.0 - gamma_list[0],
        'nthreads': nthreads,
    }
    baccs = []
    res_dict = {
        'seed': [],
        **{(g, np.inf, EF*ncols): [] for g in gamma_list},
        **{(g, eps, T): [] for g, eps, T in product(gamma_list, EPS, NRNDS)}
    }
    
    with temp_seed(5489):
        fargs['random_state'] = np.random.randint(99999)
    res_dict['seed'] += [fargs['random_state']]
    logger.debug(f"Training flynn with random seed {fargs['random_state']} ...")
    flynn = DPFlyNNFL(**fargs)
    flynn.fit(X1, y1)
    for g in gamma_list:
        flynn.nonbinary_bf_c = 1.0 - g
        flynn.reduce_bfs()
        bacc = get_scorer('balanced_accuracy')(flynn, X2, y2)
        res_dict[(g, np.inf, EF*ncols)] += [bacc]
        logger.info(f'- Non-private @ gamma-{g:.1f}: {bacc:.4f}')
    save_list2file(res_dict, res_file, res_cols)
    if args.evaluation == 'non-private':
        sys.exit(0)
    for eps, T in product(EPS, NRNDS):
        for g in gamma_list:
            flynn.nonbinary_bf_c = 1.0 - g
            for rep in range(NREPS):
                flynn.dp = {'eps': eps, 'T': T, 'c0': 0.}
                flynn.reduce_bfs()
                bacc = get_scorer('balanced_accuracy')(flynn, X2, y2)
                res_dict[(g, eps, T)] += [bacc]
                logger.info(f'- Private(eps: {eps:.2f}, T={T}) @ gamma-{g:.1f}: {bacc:.4f}')
        save_list2file(res_dict, res_file, res_cols)
    logger.info('Experiment complete')

