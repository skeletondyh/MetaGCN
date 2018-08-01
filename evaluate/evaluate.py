import numpy as np
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def evaluate(label_prefix, embedding_prefix, test_size):
    author_labels = dict()

    fp = open(label_prefix + 'author_labels.txt')
    for line in fp:
        aid = line.split()[0]
        label = int(line.split()[1])
        author_labels[aid] = label
    fp.close()

    embedding = np.load(embedding_prefix + 'val.npy')
    embedding_id_fp = open(embedding_prefix + 'val.txt')
    embedding_ids = embedding_id_fp.readlines()

    target = np.array([author_labels[line.strip()] for line in embedding_ids])

    x_train, x_test, y_train, y_test = train_test_split(embedding, target, test_size=test_size, random_state=int(time.time()))

    logisticRegr = LogisticRegression()

    logisticRegr.fit(x_train, y_train)

    predictions = logisticRegr.predict(x_test)

    macro_f1_score = metrics.f1_score(y_test, predictions, average='macro')
    micro_f1_score = metrics.f1_score(y_test, predictions, average='micro')

    print("Macro F1: ", macro_f1_score, "Micro F1: ", micro_f1_score)


label_path = sys.argv[1]
embedding_path = sys.argv[2]
test_size = float(sys.argv[3])


if __name__ == '__main__':
    evaluate(label_path, embedding_path, test_size)