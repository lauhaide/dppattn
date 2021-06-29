import argparse
import sys
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

def complex_metric(preds, labels, prefix=""):
    return {
        prefix + "bacc": balanced_accuracy_score(y_true=labels, y_pred=preds),
        prefix + "f1":   f1_score(y_true=labels, y_pred=preds, average="micro")
    }


def main(args):
    print("Hello World!")

    predicted = {}
    fin = open(args.preds, 'r')
    for line in fin.readlines():
        parts = line.split('\t')
        ex = parts[2].split('-')[0]
        sent = parts[2].split('-')[2]
        if not ex in predicted.keys():
            predicted[ex] = {}
        if not sent in predicted[ex].keys():
            predicted[ex][sent] = 0
        predicted[ex][sent] =  predicted[ex][sent] or int(parts[1])

    preds = []
    aggAcc = []
    for k in predicted.keys():
        expred = []
        for s in predicted[k].keys():
            preds.append(predicted[k][s])
            expred.append(predicted[k][s])
        ret = complex_metric(expred, [1]*len(expred))
        aggAcc.append(ret["bacc"])

    labels = [1] * len(preds)

    print("All sentences:", complex_metric(preds, labels))
    print('All sentences Len:', len(labels))
    aggScores = np.array(aggAcc)
    print("Nb. Sentences Mean {} (total ex:{})\n".format(aggScores.mean(), len(aggAcc)))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--preds', help="factCC predictions file (output after running factcc-eval.sh)", required=True)


    args = parser.parse_args(sys.argv[1:])
    main(args)
