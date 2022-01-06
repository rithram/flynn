import numpy as np
from scipy.sparse import csr_matrix

from utils.parproc import split_for_parallelism, ps2


class FlyHash:
    def __init__(self, **kwargs):
        try:
            self.expansion_factor = kwargs['expansion_factor']
            assert self.expansion_factor > 1, (
                '\'expansion_factor\' should be greater than 1, is %i'
                % self.expansion_factor
            )
            self.m = None # To be set during fit as (expansion_factor * data_dim)
            self.connection_sparsity = kwargs['connection_sparsity']
            assert self.connection_sparsity < 1.0, (
                '\'connection_sparsity\' should be (much) lesser than 1.0, is %g'
                % self.connection_sparsity
            )
            self.s = None # To be set during fit as (connection_sparsity * data_dim)
            self.wta_ratio = kwargs['wta_ratio']
            assert self.wta_ratio < 1.0, (
                '\'wta_ratio\' should be (much) lesser than 1.0, is %g'
                % self.wta_ratio
            )
            self.k = None # To be set during fit as (wta_ratio * m)
            self.binary_proj_matrix = None # To be filled in during fit
            self.sbpm = None
            self.ndims = None # To be filled in during fit
            # Setting the random seed
            self.random_seed = (
                kwargs['random_state'] if 'random_state' in kwargs
                else np.random.randint(9999)
            )
            self.kwargs = kwargs
            self.nthreads = kwargs['nthreads'] if 'nthreads' in kwargs else 1
            self.kwargs['nthreads'] = self.nthreads
            self.argsort = kwargs['argsort'] if 'argsort' in kwargs else False
            self.kwargs['argsort'] = self.argsort
        except Exception as e:
            print('Exception encountered during initialization:\n%s', repr(e))
            raise e

    def fit(self, X, y=None):
        nrows, ndims = X.shape
        self.ndims = ndims
        self.m = int(self.expansion_factor * ndims) # embedding dimension
        self.s = max(2, int(self.connection_sparsity * ndims)) # number of dims samples
        self.k = max(2, int(self.wta_ratio * self.m)) # hash length
        # Setting the randomness seed
        np.random.seed(self.random_seed)
        # Generate binary projection matrix (self.m x ndims) with self.s NNZ per row
        cols = np.arange(self.m)[None, :]
        rows = None
        if self.argsort:
            weights = np.random.random(size=(ndims, self.m))
            rows = weights.argsort(axis=0)[-self.s:, :]
        else:
            rows = np.transpose(ps2(
                batches = split_for_parallelism(self.m, self.nthreads),
                nchoices = ndims,
                nsamples = self.s,
                nthreads = self.nthreads,
                rseed = self.random_seed,
            ))
        assert not (rows is None)
        # Storing the binary projection matrix as a binary *sparse* matrix
        self.sbpm = csr_matrix(
            (
                np.array([True] * (self.m * self.s)),
                (rows.flatten(), np.repeat(cols, self.s, axis=0).flatten())
            ),
            shape=(ndims, self.m),
            dtype=np.bool
        )
        return self

    def transform(self, X):
        assert not self.sbpm is None, ('Fit not called on this instance yet')
        nrows, ndims = X.shape
        assert ndims == self.ndims, (
            'Method fit on data with %i dims, prediction attempted on data with %i dims'
            % (self.ndims, ndims)
        )
        rows = np.arange(nrows)[:, None]
        # perform the projection
        projX = (X @ self.sbpm)
        # perform WTA_k
        cols = np.argpartition(projX, -self.k, axis=1)[:, -self.k:]
        # Returning transformed X as a sparse matrix
        retX = csr_matrix(
            (
                np.array([True] * (self.k * nrows)),
                (
                    np.repeat(rows, self.k, axis=1).flatten(),
                    cols.flatten()
                )
            ),
            shape=(nrows, self.m),
            dtype=np.bool
        )
        return retX

    def get_params(self):
        return self.kwargs
