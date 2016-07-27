
from constants import *
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
import scipy as sp
import json
import numpy as np

import matplotlib.pyplot as plt

plt.style.use('seaborn-pastel')


class SafeNumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(SafeNumpyArrayEncoder, self).default(obj)


class Evaluator(object):

    def __init__(self, clf, label_encoder, params, output_filepath):
        self.clf = clf

        self.label_encoder = label_encoder

        self.output_filepath = output_filepath

        self.results = {}

        self._set_params(params)

    def _set_params(self, params):
        self.params = params

        self.validate_params(self.params)

        self.results["params"] = {}

        for key, value in self.params.iteritems():
            self.results["params"][key] = value

    def validate_params(self, params):
        if params is None:
            raise Exception('Missing params')

        if ParamKeys.TRAINING_SIZE not in params:
            raise Exception("Missing params parameter 'train_size'")

        if ParamKeys.TEST_SIZE not in params:
            raise Exception("Missing params parameter 'test_size'")

        if ParamKeys.NUM_CLUSTERS not in params:
            raise Exception("Missing params parameter 'num_clusters'")

        if ParamKeys.CLASSIFIER not in params:
            raise Exception("Missing params parameter 'classifier'")

    def evaluate(self, X, y):
        print "Evaluating {} images".format(len(y))

        self.results["classes"] = self.label_encoder.classes_
        self.results["encoded_classes"] = self.label_encoder.transform(self.label_encoder.classes_)

        prediction_probs = None
        if "predict_proba" in dir(self.clf):
            prediction_probs = self.clf.predict_proba(X)

        predictions = self.clf.predict(X)

        """ log loss """
        # where a value closer to 0 is more desirable
        if prediction_probs is not None:
            ll = log_loss(y, prediction_probs)
            self.results["log_loss"] = ll
            print "log_loss = {}".format(ll)

        """ accuracy """
        accuracy = accuracy_score(y, predictions)

        print "accuracy_score {}".format(accuracy)
        self.results["accuracy_score"] = accuracy

        soft_accuracy = self.soft_accuracy_score(y, prediction_probs, top=4)
        print "soft_accuracy_score {}".format(
            soft_accuracy
        )
        self.results["soft_accuracy_score"] = soft_accuracy

        precision_score = metrics.precision_score(y_true=y, y_pred=predictions, average='weighted')
        self.results["precision_score"] = precision_score

        recall_score = metrics.recall_score(y_true=y, y_pred=predictions, average='weighted')
        self.results["recall_score"] = recall_score

        f1_score = metrics.f1_score(y_true=y, y_pred=predictions, average='weighted')
        self.results["f1_score"] = f1_score

        # REF: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
        cm = metrics.confusion_matrix(y_true=y, y_pred=predictions)
        self.results["confusion_matrix"] = cm

        confusion_matrix_results = {}

        for r in range(cm.shape[0]):
            r_label = self.label_encoder.classes_[r]
            c_sum = cm[r, :].sum()
            correct_count = 0

            max_confused_count = 0
            max_confused_label = ""

            for c in range(cm.shape[1]):
                c_label = self.label_encoder.classes_[c]
                c_count = cm[r, c]
                if r == c:
                    correct_count = c_count
                else:
                    if c_count > max_confused_count:
                        max_confused_count = c_count
                        max_confused_label = c_label

            confusion_matrix_results[r_label] = {
                "correct": round(float(correct_count)/float(c_sum) * 100., 2)
            }

            if max_confused_count > 0:
                confusion_matrix_results[r_label]["most_confused_with"] = {
                    "label": max_confused_label,
                    "perc": round(float(max_confused_count)/float(c_sum) * 100., 2)
                }

        self.results["confusion_matrix_results"] = confusion_matrix_results

        if prediction_probs is not None:
            k_accuracies_percs = []
            k_accuracies = self.top_k_accuracy(y, prediction_probs)
            class_count = len(self.label_encoder.classes_)
            top_k = min(10, class_count)
            for i in range(1, top_k+1):
                k_count = float(len(filter(lambda x: x <= i, k_accuracies)))
                k_perc = k_count/float(len(k_accuracies))
                k_accuracies_percs.append(round(k_perc, 3) * 100.)

            self.results["k_prediction_perc"] = k_accuracies_percs

            print "\ntop k Predictions"
            for i in range(len(self.results["k_prediction_perc"])):
                print "{} = {}".format(i+1, self.results["k_prediction_perc"][i])

        if self.output_filepath is not None:
            with open(self.output_filepath, 'w') as outfile:
                _results = json.dumps(self.results, cls=SafeNumpyArrayEncoder)
                outfile.write(_results)

        # output results to console
        #print("Confusion Matrix \n{}".format(cm))

        print confusion_matrix_results

        #self.plot_confusion_matrix(cm)

        return self.results

    def top_k_accuracy(self, y, prediction_probs):
        top_k_predictions = []

        label_prediction_probs = [
            [(int(x), labels_nums_item[x]) for x in range(len(labels_nums_item))] for labels_nums_item in
            prediction_probs]

        for idx in range(len(label_prediction_probs)):
            prectiions = label_prediction_probs[idx]
            sorted_prectiions = sorted(prectiions, key=lambda tup: tup[1], reverse=True)
            true_value = y[idx]

            for jdx in range(len(sorted_prectiions)):
                if sorted_prectiions[jdx][0] == true_value:
                    top_k_predictions.append(jdx + 1)
                    break

        return top_k_predictions

    def soft_accuracy_score(self, y, prediction_probs, top=2):
        if prediction_probs is None:
            return 0.

        score_degrade = 1./float(top)
        N = float(len(prediction_probs))

        label_prediction_probs = [
            [(int(x), labels_nums_item[x]) for x in range(len(labels_nums_item))] for labels_nums_item in prediction_probs]

        total_score = 0

        for idx in range(len(label_prediction_probs)):
            prectiions = label_prediction_probs[idx]
            sorted_prectiions = sorted(prectiions, key=lambda tup: tup[1], reverse=True)
            true_value = y[idx]
            available_score = 1.0
            for jdx in range(min(len(sorted_prectiions), top)):
                if sorted_prectiions[jdx][0] == true_value:
                    total_score += available_score
                    break
                available_score -= score_degrade

        return total_score / N

    def plot_confusion_matrix(self, cm, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        :param cm:
        :param labels:
        :param title:
        :param cmap:
        :return:
        """

        labels = self.label_encoder.classes_.tolist()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def logloss(self, y, pred):
        epsilon = 1e-15
        pred = sp.maximum(epsilon, pred)
        pred = sp.minimum(1-epsilon, pred)
        ll = sum(y*sp.log(pred) + sp.subtract(1,y)*sp.log(sp.subtract(1,pred)))
        ll = ll * -1.0/len(y)
        return ll


def get_label_correct_percentage(results):
    labels = results["classes"]
    labels_performance = results["confusion_matrix_results"]
    top_k_predictions = results["k_prediction_perc"]

    label_perc_correct = []
    for label, value in labels_performance.iteritems():
        correct = float(value["correct"])
        label_perc_correct.append((label, correct))

    label_perc_correct.sort(key=lambda item: item[1], reverse=True)

    return label_perc_correct


def get_labels_most_confussed_for_labels(results, labels):

    res = []

    for label in labels:
        if label not in results["confusion_matrix_results"]:
            continue

        value = results["confusion_matrix_results"][label]
        most_confused_dict = value["most_confused_with"]

        res.append((label, (most_confused_dict["label"], most_confused_dict["perc"])))

    return res

def evl_1():
    filename = "results/evaluation_sift_k500_win0.125_fv324_imgcnt0_cls0-113__1_subset_label_test.json"

    with open(filename, "r") as f:
        results = json.load(f)

    results_perc = get_label_correct_percentage(results)

    print results_perc

if __name__ == '__main__':
    print __file__

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    evl_1()

    # with open("results/evaluation_sift_k500_win0.125_fv324_imgcnt0_cls100.json", "r") as f:
    #     results_a = json.load(f)
    #
    # with open("results/evaluation_sift_k500_win0.125_fv324_imgcnt0_cls100-200.json", "r") as f:
    #     results_b = json.load(f)
    #
    # with open("results/evaluation_sift_k500_win0.125_fv324_imgcnt0_cls200-250.json", "r") as f:
    #     results_c = json.load(f)
    #
    # results_a_perc = get_label_correct_percentage(results_a)
    # results_b_perc = get_label_correct_percentage(results_b)
    # results_c_perc = get_label_correct_percentage(results_c)
    #
    # results_perc = results_a_perc + results_b_perc + results_c_perc
    #
    # results_perc.sort(key=lambda item: item[1], reverse=True)
    #
    # threshold = 55
    # threshold_count = len(filter(lambda x: x[1] > threshold, results_perc))
    #
    # print "{} above {}".format(threshold_count, threshold)
    #
    # print results_perc[:len(filter(lambda x: x[1] > threshold, results_perc))]
    #
    # print results_perc[-20:]
    #
    # print get_labels_most_confussed_for_labels(results_a, ['dog', 'basket', 'axe'])
    #
    # #print len(filter(lambda item: item[1] > 55, results_perc[:100]))
    #
    # print "min of subset {}".format(min(map(lambda x: x[1], results_perc[:threshold_count])))
    #
    # print "unique labels = {} from {}".format(
    #     len(set(map(lambda x: x[0], results_perc[:threshold_count]))),
    #     len(results_perc)
    # )
    #
    # above_threshold = map(lambda x: x[0], results_perc[:threshold_count])
    # below_threshold = map(lambda x: x[0], results_perc[threshold_count:])
    #
    # print "=== above ==="
    # print above_threshold
    # print "=== below ==="
    # print below_threshold
    #
    # print "-----------------"
    # print get_labels_most_confussed_for_labels(results_a, ['fish', 'rabbit', 'bee'])

    # export to file
    # import csv
    # with open("subset_labels.csv", "wb") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(above_threshold)


