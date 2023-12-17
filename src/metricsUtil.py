import numpy as np
from collections import Counter


class util(object):

    @staticmethod
    def true_positive(y_true, y_pred):
        tp = 0
        for (yt, yp) in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                tp += 1
        return tp

    @staticmethod
    def true_negative(y_true, y_pred):
        tn = 0
        for (yt, yp) in zip(y_true, y_pred):
            if yt == 0 and yp == 0:
                tn += 1
        return tn

    @staticmethod
    def false_positive(y_true, y_pred):
        fp = 0
        for (yt, yp) in zip(y_true, y_pred):
            if yt == 0 and yp == 1:
                fp += 1
        return fp

    @staticmethod
    def false_negative(y_true, y_pred):
        fn = 0
        for (yt, yp) in zip(y_true, y_pred):
            if yt == 1 and yp == 0:
                fn += 1
        return fn

    @staticmethod
    def accuracy_score(y_true, y_pred):
        tp = util.true_positive(y_true, y_pred)
        tn = util.true_negative(y_true, y_pred)
        fp = util.false_positive(y_true, y_pred)
        fn = util.false_negative(y_true, y_pred)
        return (tp + tn) / (tp + fn + tn + fp)

    @staticmethod
    def precision_score(y_true, y_pred):
        tp = util.true_positive(y_true, y_pred)
        fp = util.false_positive(y_true, y_pred)
        return tp / (tp + fp)

    @staticmethod
    def recall_score(y_true, y_pred):
        tp = util.true_positive(y_true, y_pred)
        fn = util.false_negative(y_true, y_pred)
        return tp / (tp + fn)

    @staticmethod
    def f1_score(y_true, y_pred):
        r = util.recall_score(y_true, y_pred)
        p = util.precision_score(y_true, y_pred)
        f1 = 0
        if p + r != 0:
            f1 = 2 * p * r / (p + r)
        return f1

    @staticmethod
    def tpr_score(y_true, y_pred):
        return util.recall_score(y_true, y_pred)

    @staticmethod
    def fpr_score(y_true, y_pred):
        fp = util.false_positive(y_true, y_pred)
        tn = util.true_negative(y_true, y_pred)
        return fp / (tn + fp)

    @staticmethod
    def macro_precision(y_true, y_pred):
        classes = np.unique(y_true)
        num_classes = len(classes)
        precision = 0
        for class_ in classes:
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            tp = util.true_positive(temp_true, temp_pred)
            fp = util.false_positive(temp_true, temp_pred)
            temp_precision = tp / (tp + fp)
            precision += temp_precision
        precision /= num_classes
        return precision

    @staticmethod
    def micro_precision(y_true, y_pred):
        classes = np.unique(y_true)
        tp = 0
        fp = 0
        for class_ in classes:
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            tp += util.true_positive(temp_true, temp_pred)
            fp += util.false_positive(temp_true, temp_pred)
        precision = tp / (tp + fp)
        return precision

    @staticmethod
    def weighted_precision(y_true, y_pred):
        classes = np.unique(y_true)
        class_counts = Counter(y_true)
        precision = 0
        for class_ in classes:
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            tp = util.true_positive(temp_true, temp_pred)
            fp = util.false_positive(temp_true, temp_pred)
            temp_precision = tp / (tp + fp)
            precision += temp_precision * class_counts[class_]
        precision /= len(y_true)
        return precision

    @staticmethod
    def macro_recall(y_true, y_pred):
        classes = np.unique(y_true)
        num_classes = len(classes)
        recall = 0
        for class_ in classes:
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            tp = util.true_positive(temp_true, temp_pred)
            fn = util.false_negative(temp_true, temp_pred)
            temp_recall = tp / (tp + fn)
            recall += temp_recall
        recall /= num_classes
        return recall

    @staticmethod
    def micro_recall(y_true, y_pred):
        classes = np.unique(y_true)
        tp = 0
        fn = 0
        for class_ in classes:
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            tp += util.true_positive(temp_true, temp_pred)
            fn += util.false_negative(temp_true, temp_pred)
        recall = tp / (tp + fn)
        return recall

    @staticmethod
    def weighted_recall(y_true, y_pred):
        classes = np.unique(y_true)
        class_counts = Counter(y_true)
        recall = 0
        for class_ in classes:
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            tp = util.true_positive(temp_true, temp_pred)
            fn = util.false_negative(temp_true, temp_pred)
            temp_recall = tp / (tp + fn)
            recall += temp_recall * class_counts[class_]
        recall /= len(y_true)
        return recall

    @staticmethod
    def macro_f1(y_true, y_pred):
        classes = np.unique(y_true)
        f1 = 0
        for class_ in classes:
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            f1 += util.f1_score(temp_true, temp_pred)
        f1 /= len(classes)
        return f1

    @staticmethod
    def macro_f1_1(y_true, y_pred):
        precision = util.macro_precision(y_true, y_pred)
        recall = util.macro_recall(y_true, y_pred)
        if precision + recall != 0:
            return 2 * precision * recall / (precision + recall)
        else:
            return 0

    @staticmethod
    def micro_f1(y_true, y_pred):
        classes = np.unique(y_true)
        tp = 0
        fp = 0
        fn = 0
        for class_ in classes:
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            tp += util.true_positive(temp_true, temp_pred)
            fp += util.false_positive(temp_true, temp_pred)
            fn += util.false_negative(temp_true, temp_pred)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall != 0:
            return 2 * precision * recall / (precision + recall)
        else:
            return 0

    @staticmethod
    def weighted_f1(y_true, y_pred):
        classes = np.unique(y_true)
        class_counts = Counter(y_true)
        f1 = 0
        for class_ in classes:
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            p = util.precision_score(temp_true, temp_pred)
            r = util.recall_score(temp_true, temp_pred)
            if p + r != 0:
                temp_f1 = 2 * p * r / (p + r)
            else:
                temp_f1 = 0
            f1 += class_counts[class_] * temp_f1
        f1 /= len(y_true)
        return f1

    @staticmethod
    def pk(y_true, y_pred, k):
        if k == 0:
            return 0
        y_pred = y_pred[:k]
        pred_set = set(y_pred)
        true_set = set(y_true)
        common_values = pred_set.intersection(true_set)
        return len(common_values) / len(y_pred[:k])

    @staticmethod
    def apk(y_true, y_pred, k):
        pk_values = []
        for i in range(1, k + 1):
            pk_values.append(util.pk(y_true, y_pred, i))
        if len(pk_values) == 0:
            return 0
        return sum(pk_values) / len(pk_values)

    @staticmethod
    def mapk(y_true, y_pred, k):
        apk_values = []
        for i in range(len(y_true)):
            apk_values.append(util.apk(y_true[i], y_pred[i], k=k))
        return sum(apk_values) / len(apk_values)

    @staticmethod
    def log_loss(y_true, y_pred, epsilon=1e-5):
        loss = []
        for (yt, yp) in zip(y_true, y_pred):
            yp = np.clip(yp, epsilon, 1 - epsilon)
            temp_loss = -1 * (yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
            loss.append(temp_loss)
        return np.mean(loss)

    @staticmethod
    def mc_log_loss_col(y_true, y_pred):
        if len(y_true)==0:
            return util.log_loss([1],[0])
        classes = np.unique(y_true)
        num_classes = len(classes)
        ll = 0
        for class_ in classes:
            temp_true = [1]
            temp_pred = [1] if class_ in y_pred else [0]
            ll += util.log_loss(temp_true, temp_pred)
        ll /= num_classes
        return ll

    @staticmethod
    def mc_log_loss(y_true, y_pred):
        ll_values = []
        for i in range(len(y_true)):
            ll_values.append(util.mc_log_loss_col(y_true[i], y_pred[i]))
        return sum(ll_values) / len(ll_values)

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            error += np.abs(yt - yp)
        return error / len(y_true)

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            error += (yt - yp) ** 2
        return error / len(y_true)

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(util.mean_squared_error(y_true, y_pred))

    @staticmethod
    def mean_squared_log_error(y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            error += (np.log(1 + yt) - np.log(1 - yp)) ** 2
        return error

    @staticmethod
    def root_mean_squared_log_error(y_true, y_pred):
        return np.sqrt(util.mean_squared_log_error(y_true, y_pred))

    @staticmethod
    def mean_percentage_error(y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            error += (yt - yp) / yt
        return error / len(y_true)

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            error += np.abs(yt - yp) / yt
        return error / len(y_true)

    @staticmethod
    def r2_score(y_true, y_pred):
        mean_true_value = np.mean(y_true)
        numerator = 0
        denominator = 0
        for yt, yp in zip(y_true, y_pred):
            numerator += (yt - yp) ** 2
            denominator += (yt - mean_true_value) ** 2
        ratio = numerator / denominator
        return 1 - ratio

    @staticmethod
    def mcc(y_true, y_pred):
        tp = util.true_positive(y_true, y_pred)
        tn = util.true_negative(y_true, y_pred)
        fp = util.false_positive(y_true, y_pred)
        fn = util.false_negative(y_true, y_pred)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tn + fn) * (fp + tn) * (tp + fn))
        return mcc
if __name__ == "__main__":
    # Test mean average precision at k or MAP@k
    y_true = [
        [1, 2, 3],
        [0, 2],
        [1],
        [2, 3],
        [1, 0],
        []
    ]
    y_pred = [
        [0, 1, 2],
        [1],
        [0, 2, 3],
        [2, 3, 4, 0],
        [0, 1, 2],
        [0]
    ]
    for i in range(len(y_true)):
        print(f"""mapk@{i}={util.mapk(y_true, y_pred, i)}""")
