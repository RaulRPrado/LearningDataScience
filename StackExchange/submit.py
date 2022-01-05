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
        MultinomialNB(alpha=0.3)
    ).fit(X_train, y_train)
    return pipe


def process_data(data):
    X = None if 'question' not in data.keys() else np.array(data['question'])
    Y = None if 'topic' not in data.keys() else np.array(data['topic']).reshape(-1, 1)
    return X, Y


def load_training_data():
    data_train = {'topic': list(), 'question': list()}
    with open('training.json', 'r') as file:
        n_data = None
        for line in file:

            if n_data is None:
                n_data = int(line)
            else:
                d = json.loads(line)
                data_train['topic'].append(d['topic'])
                data_train['question'].append(d['question'])
    return data_train


def load_input_data():
    data_questions = {'question': list()}
    n_data = None
    for line in sys.stdin:
        if n_data is None:
            n_data = int(line)
        else:
            d = json.loads(line)
            data_questions['question'].append(d['question'])

    return data_questions


def load_output_data():
    data_topics = {'topic': list()}
    with open('output00.txt', 'r') as file:
        for line in file:
            data_topics['topic'].append(line.replace('\n', ''))

    return data_topics


def main():

    # Training
    data_train = load_training_data()
    X_train, y_train = process_data(data_train)

    label_encoder = LabelEncoder().fit(y_train)
    y_train = label_encoder.transform(y_train)
    clf = build_model(X_train, y_train)

    acc = clf.score(X_train, y_train)
    print('Training Accuracy: {}'.format(acc))

    # Input data
    data_input = load_input_data()
    X_input, _ = process_data(data_input)

    y_pred = clf.predict(X_input)
    y_out = label_encoder.inverse_transform(y_pred)

    # Printing out results
    # for y in y_out:
    #     print(y)

    # Extra for testing
    data_output = load_output_data()

    _, y_output = process_data(data_output)
    y_output = label_encoder.transform(y_output)

    acc_test = accuracy_score(y_pred, y_output)
    print('Test Accuracy: {}'.format(acc_test))


if __name__ == '__main__':
    main()
