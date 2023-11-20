import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef, roc_curve, average_precision_score


def youden_j_stat(fpr, tpr, thresholds):
    j_ordered = sorted(zip(tpr - fpr, thresholds))
    return 1. if j_ordered[-1][1] > 1 else j_ordered[-1][1]


def compute_classifier_metrics(probs, labels, lengths, cdrs, epoch, threshold=None, cv="single"):
    probs = probs.detach()

    matrices = []
    aucs = []
    aupr = []
    mcorrs = []
    jstats = [youden_j_stat(*roc_curve(lbl[:l], p[:l])) for lbl, p, l in zip(labels, probs, lengths)]

    jstat_scores = np.array(jstats)
    jstat = np.mean(jstat_scores)

    if threshold is None:
        threshold = jstat

    for lbl, p, l, cdr in zip(labels, probs, lengths, cdrs):
        # print("lbl", *[1 if x == 1.0 else 0 for x in lbl[:l].tolist()])
        aucs.append(roc_auc_score(lbl[:l], p[:l]))
        aupr.append(average_precision_score(lbl[:l], p[:l], pos_label=1.0))
        l_pred = (p[:l] > threshold).numpy().astype(int)
        # print("pre", *l_pred.tolist())
        # print("cdr", *[1 if x == 1.0 else 0 for x in cdr[:l].tolist()])
        # print(confusion_matrix(lbl[:l], l_pred))
        matrices.append(confusion_matrix(lbl[:l], l_pred, labels=[0, 1]))
        mcorrs.append(matthews_corrcoef(lbl[:l], l_pred))

    matrices = np.stack(matrices)
    mean_conf = np.mean(matrices, axis=0)

    tps = matrices[:, 1, 1]
    fns = matrices[:, 1, 0]
    fps = matrices[:, 0, 1]

    recalls = tps / (tps + fns)
    precisions = tps / (tps + fps)

    rec = np.mean(recalls)

    prec = np.mean(precisions)

    fscores = 2 * precisions * recalls / (precisions + recalls)
    fsc = np.nanmean(fscores)

    auc_scores = np.array(aucs)
    auc = np.mean(auc_scores)

    aupr_scores = np.array(aupr)
    aupr = np.mean(aupr_scores)

    mcorr_scores = np.array(mcorrs)
    mcorr = np.mean(mcorr_scores)

    f = open("results/{}/model_{}.txt".format(cv, epoch), "w")

    f.write(f"Youden's J statistic = {threshold:.3f}. Using it as threshold.\n")

    f.write("Mean confusion matrix and error\n")
    f.write(str(mean_conf) + "\n")

    f.write(f"Recall = {rec:.3f} / 0.669\n")
    f.write(f"Precision = {prec:.3f} / 0.711\n")
    f.write(f"F-score = {fsc:.3f} / 0.689\n")
    f.write(f"ROC AUC = {auc:.3f} / 0.961\n")
    f.write(f"pr = {aupr:.3f} / 0.742\n")
    f.write(f"MCC = {mcorr:.3f} / 0.659\n")
    print("Epoch    {}, Precision   {}, F1-Score    {}".format(epoch, prec, fsc))

    f.close()
    return threshold
