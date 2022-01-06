import logging
import os
import sys
import warnings


import numpy as np
from openml.datasets import list_datasets, get_dataset
from sklearn.base import clone
from sklearn.datasets import load_digits
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist, cifar100

from .get_data import get_tf_data

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('utils.evaluations')

def eval_est_kfold_cv(est, dset, skfold):
    X, y = dset
    acc = 0.0
    f1a = 0.0
    f1i = 0.0
    with warnings.catch_warnings(record=True) as w:
        for tr, val in skfold.split(X, y):
            try:
                m = clone(est)
                m.fit(X[tr], y[tr])
                preds = m.predict(X[val])
                acc += balanced_accuracy_score(y[val], preds)
                f1a += f1_score(y[val], preds, average='macro')
                f1i += f1_score(y[val], preds, average='micro')
            except BaseException as e:
                logger.debug(
                #print(
                    'Estimator %s failed with HP\n%s\nException:%s'
                    % (est.__class__.__name__, str(est.get_params()), str(e))
                )
                acc, f1a, f1i = None, None, None
                break
    if acc is None:
        return (None, None, None), w
    acc /= float(skfold.get_n_splits())
    f1a /= float(skfold.get_n_splits())
    f1i /= float(skfold.get_n_splits())
    return (acc, f1a, f1i), w


def eval_est_hv(est, dset) :
    X, y, vX, vy = dset
    acc = 0.0
    f1a = 0.0
    f1i = 0.0
    with warnings.catch_warnings(record=True) as w :
        try:
            m = clone(est)
            m.fit(X, y)
            preds = m.predict(vX)
            acc = balanced_accuracy_score(vy, preds)
            f1a = f1_score(vy, preds, average='macro')
            f1i = f1_score(vy, preds, average='micro')
        except BaseException as e:
            logger.debug(
                'Estimator %s failed with HP\n%s\nException:%s'
                % (est.__class__.__name__, str(est.get_params()), str(e))
            )
            acc, f1a, f1i = None, None, None
    return (acc, f1a, f1i), w


def get_openml_data_list(min_data_dim, max_data_dim, max_data_samples):
    openml_df = list_datasets(output_format='dataframe')
    val_dsets = openml_df.query(
        'NumberOfInstancesWithMissingValues == 0 & '
        'NumberOfMissingValues == 0 & '
        'NumberOfClasses > 1 & '
        'NumberOfClasses <= 30 & '
        'NumberOfSymbolicFeatures == 1 & '
        'NumberOfInstances > 999 &'
        'NumberOfFeatures >= ' + str(min_data_dim + 1) + ' & '
        'NumberOfFeatures <= ' + str(max_data_dim + 1) + ' & '
        'NumberOfInstances <= ' + str(max_data_samples)
    )[[
        'name', 'did', 'NumberOfClasses', 'NumberOfInstances',
        'NumberOfFeatures'
    ]]
    print(
        'Found %s/%s datasets'
        % (len(val_dsets.index), len(openml_df.index))
    )
    print(val_dsets[['name', 'did']].head(5))
    print(val_dsets.describe())
    return val_dsets


def get_datasets(dname, need_val_set=False, prec='double'):
    X, y = None, None
    vX, vy = None, None
    openml_dsets = {
        'letter': 6, 'higgs': 23512, 'numerai28.6': 23517,
        'connect-4': 40668, 'APSFailure': 41168,
    }
    if dname == 'digits' :
        X, y = load_digits(return_X_y=True)
    elif dname in openml_dsets: # == 'letter':
        d = get_dataset(openml_dsets[dname])
        X, y, _, _ = d.get_data(
            target=d.default_target_attribute, dataset_format='array'
        )
    # 41168, 23517, 23512, 40668
    elif dname == 'mnist' :
        X, y, vX, vy = get_tf_data(mnist, collapse_color_channels=False)
    elif dname == 'fashion_mnist' :
        X, y, vX, vy = get_tf_data(fashion_mnist, collapse_color_channels=False)
    elif dname == 'cifar10' :
        X, y, vX, vy = get_tf_data(cifar10, collapse_color_channels=True)
    elif dname == 'cifar100' :
        X, y, vX, vy = get_tf_data(cifar100, collapse_color_channels=True)
    else :
        raise Exception('Unknown data set \'{}\''.format(dname))
    assert (X is not None) and (y is not None)
    if vX is None and need_val_set:
        assert vy is None
        X, vX, y, vy = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=5489
        )
    return X, y, vX, vy


def get_app_data(data_name, labels, prec='double'):
    X, y, X1, y1 = get_datasets(
        data_name, need_val_set=True, prec=prec,
    )
    print('Full train X: {}, test X: {}'.format(X.shape, X1.shape))
    print('Full train y: {}, test y: {}'.format(y.shape, y1.shape))
    binary = labels[0] != -1 and labels[1] != -1
    # filter training data
    valid_idxs = np.append(
        np.argwhere(y == labels[0]),
        np.argwhere(y == labels[1])
    ) if binary else np.arange(len(y))
    print('Valid idxs:', valid_idxs.shape)
    X = X[valid_idxs, :]
    y = y[valid_idxs]
    print('{} train X: {}'.format('Bin' if binary else 'MC', X.shape))
    print('{} train y: {}'.format('Bin' if binary else 'MC', y.shape))
    assert len(np.unique(y)) == 2 or (not binary)
    if binary:
        print(' - Label {}: {}'.format(
            labels[0], len(np.argwhere(y == labels[0]))
        ))
        print(' - Label {}: {}'.format(
            labels[1], len(np.argwhere(y == labels[1]))
        ))
    # filter testing data
    valid_idxs = np.append(
        np.argwhere(y1 == labels[0]),
        np.argwhere(y1 == labels[1])
    ) if binary else np.arange(len(y1))
    print('Valid idxs:', valid_idxs.shape)
    X1 = X1[valid_idxs, :]
    y1 = y1[valid_idxs]
    print('{} test X: {}'.format('Bin' if binary else 'MC', X1.shape))
    print('{} test y: {}'.format('Bin' if binary else 'MC', y1.shape))
    assert len(np.unique(y1)) == 2 or (not binary)
    if binary:
        print(' - Label {}: {}'.format(
            labels[0], len(np.argwhere(y1 == labels[0]))
        ))
        print(' - Label {}: {}'.format(
            labels[1], len(np.argwhere(y1 == labels[1]))
        ))
    return X, y, X1, y1, binary
