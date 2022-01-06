#!/usr/bin/env python

import argparse
from datetime import datetime
from glob import glob
import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)
import sys
from pprint import pprint
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline
from openml.datasets import get_dataset
from joblib import Parallel, delayed
from tqdm import tqdm
from skopt.space import Integer, Real
from skopt.callbacks import DeadlineStopper
from skopt import gp_minimize

from fh.flynn import FHBloomFilter as FlyNN
from utils.densify import Densify
from utils.evaluations import eval_est_kfold_cv, get_openml_data_list


def gen_pipeline_from_val(
        x, dim, nthreads=1, skip_densify=False,
        c=None, max_batch_size=4096
):
    assert len(x) == 4
    # 0. Densify
    assert x[0] == 0 or x[0] == 1
    # 1. expansion factor
    assert x[1] > 1.0
    exp_factor = x[1]
    # 2. connection sparsity
    assert x[2] < 1.0
    conn_sparsity = x[2]
    # 3. wta nnz
    assert isinstance(x[3], np.int64) or isinstance(x[3], int), (
        'Val %s of type %s' % (str(x[3]), str(type(x[3])))
    )
    wta_ratio = min(0.5, (float(x[3]) / (exp_factor * dim)))
    flynn_args = {
        'expansion_factor': exp_factor,
        'connection_sparsity': conn_sparsity,
        'wta_ratio': wta_ratio,
        'batch_size': max(1, int(max_batch_size / int(max(1, exp_factor / 200)))),
        'nthreads': nthreads,
    }
    if c is not None:
        flynn_args['binary'] = False
        flynn_args['c'] = c
    flynn = FlyNN(**flynn_args)
    if x[0] == 0 or skip_densify:
        return flynn
    else:
        return make_pipeline(
            Densify(**{'nthreads': nthreads}),
            flynn
        )


def eval_function(
        x, dset, skfold, history, warnings,
        non_binary=False, max_batch_size=4096
):
    if non_binary: assert len(x) == 5
    else: assert len(x) == 4
    # generate estimator
    m = (
        gen_pipeline_from_val(
            x[:4], dset[0].shape[1], c=x[4],
            max_batch_size=max_batch_size
        ) if non_binary else gen_pipeline_from_val(
            x, dset[0].shape[1], max_batch_size=max_batch_size
        )
    )
    # evaluate estimator
    (acc, f1a, f1i), w = eval_est_kfold_cv(m, dset, skfold)
    if acc is None:
        return 1.0
    # save evaluation history
    history['densify'].append(True if x[0] == 1 else False)
    history['exp_factor'].append(x[1])
    history['conn_sparsity'].append(x[2])
    history['wta_nnz'].append(x[3])
    if non_binary: history['c'].append(x[4])
    history['acc'].append(acc)
    history['f1macro'].append(f1a)
    history['f1micro'].append(f1i)
    # save warnings
    warnings.extend(w)
    return 1.0 - acc


def get_blank_history(non_binary=False):
    ret = {
        'densify': [],
        'exp_factor': [],
        'conn_sparsity': [],
        'wta_nnz': [],
        'acc': [],
        'f1macro': [],
        'f1micro': [],
    }
    if non_binary: ret['c'] = []
    return ret


def gp_hpo(
        dset, nfolds, hist, warns, search_props, hp_props,
        non_binary=False, max_batch_size=4096
):
    # get CV splitter
    skf = SKFold(n_splits=nfolds, shuffle=True, random_state=5489)
    # objective
    obj = lambda x: eval_function(
        x,
        dset=dset,
        skfold=skf,
        history=hist,
        warnings=warns,
        non_binary=non_binary,
        max_batch_size=max_batch_size,
    )
    # variables
    space = []
    # 0. densify
    l, u = 0, 1
    space.append(Integer(low=l, high=u))
    # 1. expansion factor
    l, u = 2.0, hp_props['exp_factor_ub']
    space.append(Real(low=l, high=u))
    # 2. connection sparsity
    l, u = 0.0, hp_props['conn_spar_ub']
    space.append(Real(low=l, high=u))
    # 3. wta NNZ
    l, u = 8, hp_props['wta_nnz_ub']
    space.append(Integer(low=l, high=u))
    if non_binary:
        # 4. c
        l, u = 0.2, 0.9
        space.append(Real(low=l, high=u))
    res = gp_minimize(
        obj,
        space,
        acq_func='EI',
        n_calls=search_props['max_calls'],
        n_random_starts=search_props['n_random_starts'],
        random_state=5489,
    )
    return (
        np.max(hist['acc']),
        np.max(hist['f1macro']),
        np.max(hist['f1micro'])
    )


def eval_HPO_openml_did(
        did, nfolds, search_props, hp_props, done_dsets=None,
        non_binary=False, max_batch_size=4096
):
    history = get_blank_history(non_binary)
    warnings = []
    try:
        d = get_dataset(did)
        dname = d.name + '.' + str(did)
        if not (done_dsets is None) and (dname in set(done_dsets)):
            print('%s already done ...' % dname)
            return (None, None, None)
        X, y, c, a = d.get_data(
            target=d.default_target_attribute, dataset_format='array'
        )
        # run HPO
        res = gp_hpo(
            (normalize(X), y),
            nfolds,
            history,
            warnings,
            search_props,
            hp_props,
            non_binary=non_binary,
            max_batch_size=max_batch_size,
        )
        # Save HPO runs
        hpo_res_file = res_dir + '/' + dname + '.hpo.csv'
        pd.DataFrame.from_dict(history).to_csv(hpo_res_file, index=False)
        return (dname, res, warnings)
    except (BaseException, MemoryError, SystemExit) as e:
        print('Following exception encountered for {}'.format(dname))
        exceptions = traceback.format_exception_only(type(e), e)
        for exc in exceptions:
            print(exc)
        # Save HPO runs
        if len(history['acc']) > 0:
            hpo_res_file = res_dir + '/' + dname + '.hpo.csv'
            pd.DataFrame.from_dict(history).to_csv(hpo_res_file, index=False)
            return (
                dname,
                (
                    np.max(history['acc']),
                    np.max(history['f1macro']),
                    np.max(history['f1micro'])
                ),
                warnings
            )
        return (None, None, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--n_parallel', help='Number of parallel workers',
        type=int, default=1
    )
    parser.add_argument(
        '-F', '--n_folds', help='Number of folds', type=int, default=10
    )
    parser.add_argument(
        '-c', '--max_calls', help='Maximum number of calls for GP',
        type=int, default=100
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
        help='Upper bound on the projection density HP', type=float,
    )
    parser.add_argument(
        '-w', '--wta_nnz_ub', help='Upper bound on the winner-take-all NNZ HP',
        type=int,
    )
    parser.add_argument(
        '-D', '--done_set_re',
        help='Regex for the files corresponding to datasets already processed',
        type=str, default=''
    )
    parser.add_argument(
        '-N', '--non_binary',
        help='Whether to use gamma=0 (binary, default) or gamma>0 (non-binary)',
        type=bool, default=False
    )
    parser.add_argument(
        '-n', '--min_data_dim', help='Minimum data dimensionality on OpenML',
        type=int,
    )
    parser.add_argument(
        '-x', '--max_data_dim', help='Maximum data dimensionality on OpenML',
        type=int,
    )
    parser.add_argument(
        '-S', '--max_data_samples',
        help='Maximum number of samples in data on OpenML', type=int,
    )
    parser.add_argument(
        '-B', '--max_batch_size', help='Maximum batch size for FlyNN training',
        type=int, default=4096,
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
    res_dir = 'results/flynn+hpo/' + timestamp
    print('Saving results/logs in \'%s\' ...' % res_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    warnings_file = res_dir + '/warnings.log'
    nfolds = args.n_folds
    results_dict = {
        'dataset': [],
        'accuracy': [],
        'f1macro': [],
        'f1micro': [],
    }
    search_properties = {
        'max_calls': args.max_calls,
        'n_random_starts': args.n_random_starts,
    }
    hp_properties = {
        'exp_factor_ub': args.exp_factor_ub,
        'conn_spar_ub': args.conn_spar_ub,
        'wta_nnz_ub': args.wta_nnz_ub,
    }
    X, y = load_digits(return_X_y=True)
    if (
            X.shape[1] >= args.min_data_dim
            and X.shape[1] <= args.max_data_dim
            and X.shape[0] <= args.max_data_samples
    ):
        print('Processing \'digits\' ...')
        history = get_blank_history(args.non_binary)
        warnings = []
        # run HPO
        res = gp_hpo(
            (normalize(X), y),
            nfolds,
            history,
            warnings,
            search_properties,
            hp_properties,
            non_binary=args.non_binary,
        )
        acc, f1a, f1i = res
        results_dict['dataset'].append('digits')
        results_dict['accuracy'].append(acc)
        results_dict['f1macro'].append(f1a)
        results_dict['f1micro'].append(f1i)
        with open(warnings_file, 'w') as f:
            for i, w in enumerate(warnings):
                f.write(
                    '[digits %i] [%s]\t %s\n'
                    % (i + 1, w.category.__name__, w.message)
                )
        # Save HPO runs
        hpo_res_file = res_dir + '/' + 'digits' + '.hpo.csv'
        pd.DataFrame.from_dict(history).to_csv(hpo_res_file, index=False)
    # Get OpenML dataset list
    val_dsets = get_openml_data_list(
        args.min_data_dim, args.max_data_dim, args.max_data_samples
    )
    done_dsets = [
        f.split('/')[-1].replace('.hpo.csv', '')
        for f in glob(args.done_set_re)
    ]
    if len(done_dsets) > 0:
        print(
            'Following %i sets already processed:\n%s'
            % (len(done_dsets), str(done_dsets))
        )
    dids = val_dsets['did'].values.astype(int).tolist()
    edfunc = lambda did: eval_HPO_openml_did(
        did,
        nfolds=nfolds,
        search_props=search_properties,
        hp_props=hp_properties,
        done_dsets=done_dsets,
        non_binary=args.non_binary,
        max_batch_size=args.max_batch_size,
    )
    print('Executing with %i jobs' % args.n_parallel)
    results = Parallel(n_jobs=args.n_parallel)(
        delayed(edfunc)(did) for did in tqdm(
            dids, desc='Dataset', total=len(dids)
        )
    )
    assert len(results) == len(dids)
    for dname, res, warns in results:
        if dname is None:
            assert res is None
            continue
        acc, f1a, f1i = res
        results_dict['dataset'].append(dname)
        results_dict['accuracy'].append(acc)
        results_dict['f1macro'].append(f1a)
        results_dict['f1micro'].append(f1i)
        with open(warnings_file, 'a') as f:
            for i, w in enumerate(warns):
                f.write(
                    '[%s %i] [%s]\t %s\n'
                    % (dname, i + 1, w.category.__name__, w.message)
                )
    results_df = pd.DataFrame.from_dict(results_dict)
    print(results_df.head())
    fname = res_dir + '/flynn.hpo.results.' + timestamp + '.csv'
    print('Saving results in %s' % fname)
    results_df.to_csv(fname, index=False)
