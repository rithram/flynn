#!/usr/bin/env python

import argparse
from datetime import datetime
import logging
logger = logging.getLogger('SYN+REAL')
import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from fh.flyhash import FlyHash
from eval_synthetic_data import get_all_methods, get_relative_perf


def create_dataset(nrows, ncols, nnz_per_row, nclasses, nclusters_per_class):
    exp_factor = 10
    ncols_for_Rd = int(ncols/exp_factor)
    X, y = make_classification(
        n_samples=nrows,
        n_features=ncols_for_Rd,
        n_classes=nclasses,
        n_clusters_per_class=nclusters_per_class,
        n_informative=ncols_for_Rd,
        n_redundant=0,
        n_repeated=0,
        class_sep=1.2,
        hypercube=True,
    )
    fh = FlyHash(**{
        'expansion_factor': exp_factor,
        'connection_sparsity': 0.5,
        'wta_ratio': float(nnz_per_row)/float(ncols)
    })
    bX = fh.fit(X).transform(X).toarray().astype(np.int)
    print('Dataset: (%i x %i)' % (bX.shape[0], bX.shape[1]))
    return bX, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--n_parallel', help='Number of parallel workers', type=int,
        default=1
    )
    parser.add_argument(
        '-F', '--n_folds', help='Number of folds', type=int, default=10
    )
    parser.add_argument(
        '-c', '--max_calls', help='Maximum number of calls for GP', type=int,
        default=100
    )
    parser.add_argument(
        '-r', '--n_random_starts', help='Number of random start points in GP',
        type=int, default=30
    )
    parser.add_argument(
        '-f', '--exp_factor_ub', help='Upper bound on the expansion factor HP',
        type=float,
    )
    parser.add_argument(
        '-s', '--conn_spar_ub',
        help='Upper bound on the connection sparsity HP', type=float,
    )
    parser.add_argument(
        '-w', '--wta_nnz_ub',
        help='Upper bound on the winner-take-all ratio HP', type=int,
    )
    parser.add_argument(
        '-S', '--n_rows', help='Number of rows in the data set', type=int,
        default=1000
    )
    parser.add_argument(
        '-d', '--n_cols', help='Number of columns in the data set', type=int,
        default=30
    )
    parser.add_argument(
        '-L', '--n_classes', help='Number of classes in the data set', type=int,
        default=5
    )
    parser.add_argument(
        '-C', '--n_clusters_per_class', help='Number of clusters per class',
        type=int, default=8
    )
    parser.add_argument(
        '-W', '--nnz_per_row', help='Number of NNZ per row in the binary data set', type=int,
    )
    parser.add_argument(
        '-R', '--n_reps', help='Number of repetitions', type=int, default=5
    )
    args = parser.parse_args()
    today = datetime.today()
    timestamp = (
        str(today.year) + str(today.month) + str(today.day) + '.' +
        str(today.hour) + str(today.minute) + str(today.second)
    )
    print('-'*30)
    print('Experiment timestamp: %s' % timestamp)
    print('-'*30)
    res_dir = 'results/synthetic-binary/' + timestamp
    print('Saving results/logs in \'%s\' ...' % res_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    warnings_file = res_dir + '/warnings.log'
    nfolds = args.n_folds
    results_dict = {
        'method': [],
        'accuracy': [],
        'f1macro': [],
        'f1micro': [],
    }
    methods = get_all_methods(args)
    def process_rep(rep_idx):
        dset = create_dataset(
            args.n_rows, args.n_cols, args.nnz_per_row,
            args.n_classes, args.n_clusters_per_class,
        )
        ret_dict = {
            'method': [],
            'accuracy': [],
            'f1macro': [],
            'f1micro': [],
        }
        for m, efunc in methods:
            print('[R%i] Processing \'%s\' ...' % (rep_idx, m))
            res, warns = efunc(dset, nfolds)
            ret_dict['method'].append(m)
            ret_dict['accuracy'].append(res[0])
            ret_dict['f1macro'].append(res[1])
            ret_dict['f1micro'].append(res[2])
        return pd.DataFrame.from_dict(ret_dict)
    # --
    res_df = pd.concat(Parallel(n_jobs=args.n_parallel)(
        delayed(process_rep)(i) for i in range(args.n_reps)
    ))
    print(res_df[['method', 'accuracy']])
    rel_perf = get_relative_perf(res_df, methods, 'KNNC')
    out_filename = (
        '%s/synthetic-binary.F%i.c%i.r%i.f%g.s%g.w%g.S%i.d%i.W%i.L%i.C%i.R%i.%s.csv'
        % (
            res_dir,
            nfolds,
            args.max_calls,
            args.n_random_starts,
            args.exp_factor_ub,
            args.conn_spar_ub,
            args.wta_nnz_ub,
            args.n_rows,
            args.n_cols,
            args.nnz_per_row,
            args.n_classes,
            args.n_clusters_per_class,
            args.n_reps,
            timestamp,
        )
    )
    print('Saving relative performance in \'%s\'' % out_filename)
    pd.DataFrame.from_dict(rel_perf).to_csv(out_filename, index=False)
