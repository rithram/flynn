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

from knn_baseline import eval_knnc_on_dataset_kfold_cv
from sbf_baseline import eval_sbf_on_dataset_kfold_cv
from flynn_hpo import gp_hpo, get_blank_history

def create_dataset(nrows, ncols, nclasses, nclusters_per_class):
    X, y = make_classification(
        n_samples=nrows,
        n_features=ncols,
        n_classes=nclasses,
        n_clusters_per_class=nclusters_per_class,
        n_informative=ncols,
        n_redundant=0,
        n_repeated=0,
        class_sep=1.2,
        hypercube=True,
    )
    print('Dataset: (%i x %i)' % (X.shape[0], X.shape[1]))
    return X, y


def eval_nbf_on_dataset_kfold_cv(
        dset, nfolds, search_props, hp_props, non_binary=False
):
    history = get_blank_history()
    if non_binary:
        history['c'] = []
    warnings = [ ]
    # run HPO
    acc, f1a, f1i = gp_hpo(
        dset, nfolds, history, warnings, search_props, hp_props, non_binary
    )
    best_hpo_idx = np.argsort(-np.array(history['acc']))[:5]
    assert acc == history['acc'][best_hpo_idx[0]]
    mef = np.max([ history['exp_factor'   ][i] for i in best_hpo_idx ])
    mcs = np.max([ history['conn_sparsity'][i] for i in best_hpo_idx ])
    mwr = np.max([ history['wta_nnz'    ][i] for i in best_hpo_idx ])
    if non_binary:
        mc = np.max([ history['c'][i] for i in best_hpo_idx ])
        return ((acc, (mef, mcs, mwr, mc)), (f1a, None), (f1i, None)), warnings
    return ((acc, (mef, mcs, mwr)), (f1a, None), (f1i, None)), warnings


def get_all_methods(cmd_args):
    search_properties = {
        'max_calls': cmd_args.max_calls,
        'n_random_starts': cmd_args.n_random_starts,
    }
    hp_properties = {
        'exp_factor_ub': cmd_args.exp_factor_ub,
        'conn_spar_ub': cmd_args.conn_spar_ub,
        'wta_nnz_ub': cmd_args.wta_nnz_ub,
    }
    # methods
    return [
        (
            'KNNC',
            eval_knnc_on_dataset_kfold_cv
        ),
        (
            '1NNC',
            lambda d, n: eval_knnc_on_dataset_kfold_cv(d, n, k_ub=1)
        ),
        (
            'SBFC',
            lambda d, n: eval_sbf_on_dataset_kfold_cv(d, n, ef_ub=2049.0)
        ),
        (
            'FlyNN-bin',
            lambda d, n: eval_nbf_on_dataset_kfold_cv(
                d, n, search_props=search_properties, hp_props=hp_properties
            )
        ),
        (
            'FlyNN',
            lambda d, n: eval_nbf_on_dataset_kfold_cv(
                d, n, search_props=search_properties, hp_props=hp_properties,
                non_binary=True
            )
        ),
    ]


def get_relative_perf(results_df, all_methods, base_method):
    base_res = np.array([
        r[0] for r in results_df[
            results_df['method'] == base_method
        ]['accuracy']
    ])
    print('KNNC:', base_res)
    rel_perf = {'method': [], 'rel_diff': []}
    for m, _ in all_methods:
        if m == base_method: continue
        print('Generating results for \'%s\' ...' % m)
        m_res = np.array(
            [r[0] for r in results_df[results_df['method'] == m]['accuracy']]
        )
        print(m, ':', m_res)
        rel_diff = np.divide((base_res - m_res), base_res)
        for r in rel_diff:
            rel_perf['method'  ].append(m)
            rel_perf['rel_diff'].append(r)
    return rel_perf


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
    res_dir = 'results/synthetic/' + timestamp
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
            args.n_rows, args.n_cols, args.n_classes, args.n_clusters_per_class
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
        '%s/synthetic.F%i.c%i.r%i.f%g.s%g.w%g.S%i.d%i.L%i.C%i.R%i.%s.csv'
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
            args.n_classes,
            args.n_clusters_per_class,
            args.n_reps,
            timestamp,
        )
    )
    print('Saving relative performance in \'%s\'' % out_filename)
    pd.DataFrame.from_dict(rel_perf).to_csv(out_filename, index=False)
