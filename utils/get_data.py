import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist, cifar100


def print_stats(X, y):
    print('X:', X.shape)
    print('y:', y.shape)
    print('Labels:', np.unique(y).tolist())


def get_digits():
    X, y = load_digits(return_X_y=True)
    print('-'*10)
    print('Full data:')
    print('-'*10)
    print_stats(X, y)
    print('-'*10)
    return X, y, None, None


def get_tf_data(data, collapse_color_channels=False):
    (X0, y), (vX0, vy) = data.load_data()
    if len(X0.shape) > 3 and collapse_color_channels:
        a = X0.shape
        X0 = np.sum(X0, axis=3)
        vX0 = np.sum(vX0, axis=3)
        print(a, '-->', X0.shape)
    # Flattening image to a single long vector
    od = 1
    for i, d in enumerate(X0.shape):
        if i > 0:
            od *= d
    print(X0.shape, '-->', '(', X0.shape[0], ',', od, ')')
    X = np.reshape(X0, (-1, od))
    vX = np.reshape(vX0, (-1, od))
    # squeezing the labels vector
    y = np.squeeze(y)
    vy = np.squeeze(vy)
    print('-'*10)
    print('Train data:')
    print('-'*10)
    print_stats(X, y)
    print('-'*10)
    print('Validation data:')
    print('-'*10)
    print_stats(vX, vy)
    print('-'*10)
    return X, y, vX, vy


if __name__ == '__main__':
    print('=' * 30)
    print('Digits')
    print('=' * 30)
    _, _, _, _ = get_digits()
    print('=' * 30)
    print('MNIST')
    print('=' * 30)
    _, _, _, _ = get_tf_data(mnist)
    print('=' * 30)
    print('Fashion MNIST')
    print('=' * 30)
    _, _, _, _ = get_tf_data(fashion_mnist)
    print('=' * 30)
    print('CIFAR10')
    print('=' * 30)
    _, _, _, _ = get_tf_data(cifar10, False)
    print('=' * 30)
    print('CIFAR10*')
    print('=' * 30)
    _, _, _, _ = get_tf_data(cifar10, True)
    print('=' * 30)
    print('CIFAR100')
    print('=' * 30)
    _, _, _, _ = get_tf_data(cifar100, False)
