import numpy as np


def compute_confusion_matrix(prediction, ground_truth, num_classes):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(num_classes, num_classes),
        range=[(0, num_classes), (0, num_classes)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def iou_score(confusion_matrix, reduce_mean=True):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 1
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return np.mean(ious) if reduce_mean else ious


def f1_score(confusion_matrix, reduce_mean=True):
    f1_scores = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            f1_score = 1
        else:
            f1_score = 2 * float(true_positives) / denom
        f1_scores.append(f1_score)
    return np.mean(f1_scores) if reduce_mean else f1_scores


def precision_recall_fscore_support(confusion_matrix):
    f1_scores, precisions, recalls, supports = [], [], [], []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives

        denom = true_positives + false_positives
        precision = 1 if denom == 0 else float(true_positives) / denom

        denom = true_positives + false_negatives
        recall = 1 if denom == 0 else float(true_positives) / denom

        denom = precision + recall
        f1_score = 1 if denom == 0 else 2 * float(precision * recall) / denom

        f1_scores.append(f1_score)
        precisions.append(precision)
        recalls.append(recall)
        supports.append(int(true_positives+false_negatives))

    return precisions, recalls, f1_scores, supports


def accuracy(confusion_matrix):
    return np.diag(confusion_matrix).sum() / confusion_matrix.sum()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
