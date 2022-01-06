import sys
import numpy as np
import json

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def build_model(X_train, y_train):
    pipe = make_pipeline(
        TfidfVectorizer(
            stop_words='english',
        ),
        MultinomialNB(alpha=0.1)
    ).fit(X_train, y_train)
    return pipe


def process_data(data):
    if 'city' in data.keys():
        cc = np.array(data['city'])
        ss = np.array(data['section'])
        hh = np.array(data['heading'])

        spaces = np.array([' '] * len(data['city']))
        X = np.core.defchararray.add(cc, spaces)
        X = np.core.defchararray.add(X, ss)
        X = np.core.defchararray.add(X, spaces)
        X = np.core.defchararray.add(X, hh)
    else:
        X = None

    Y = None if 'category' not in data.keys() else np.array(data['category']).reshape(-1, 1).ravel()
    return X, Y


def load_full_data(filename, skipFirstLine=False):
    data = {
        'city': list(),
        'category': list(),
        'section': list(),
        'heading': list()
    }
    with open(filename, 'r') as file:
        firstLine = True
        for line in file:

            if firstLine and skipFirstLine:
                firstLine = False
                continue

            d = json.loads(line)
            data['city'].append(d['city'])
            data['section'].append(d['section'])
            data['heading'].append(d['heading'])
            if 'category' in d.keys():
                data['category'].append(d['category'])

    if len(data['category']) == 0:
        data.pop('category')
    return data


def load_input_data():
    data = {
        'city': list(),
        'section': list(),
        'heading': list()
    }
    n_data = None
    for line in sys.stdin:
        if n_data is None:
            n_data = int(line)
        else:
            d = json.loads(line)
            data['city'].append(d['city'])
            data['section'].append(d['section'])
            data['heading'].append(d['heading'])

    return data


def load_output_data(filename):
    data = {
        'category': list()
    }
    with open(filename, 'r') as file:
        for line in file:
            data['category'].append(line.replace('\n', ''))

    return data


def main():

    # Training
    data_train = load_full_data('training.json', skipFirstLine=True)
    X_train, y_train = process_data(data_train)

    label_encoder = LabelEncoder().fit(y_train)
    y_train = label_encoder.transform(y_train)
    clf = build_model(X_train, y_train)

    # acc = clf.score(X_train, y_train)
    # print('Training Accuracy: {}'.format(acc))

    # Input data
    data_input = load_input_data()
    # data_input = load_full_data('sample-test.in.json', skipFirstLine=True)

    X_input, _ = process_data(data_input)

    y_pred = clf.predict(X_input)
    y_out = label_encoder.inverse_transform(y_pred)

    # Printing out results
    for y in y_out:
        print(y)

    # # Extra for testing
    # data_output = load_output_data('sample-test.out.json')

    # _, y_output = process_data(data_output)
    # y_output = label_encoder.transform(y_output)

    # acc_test = accuracy_score(y_pred, y_output)
    # print('Test Accuracy: {}'.format(acc_test))


if __name__ == '__main__':
    main()
