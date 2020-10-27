import numpy as np

from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score


def evaluate(X, Y, W, loader, net):
    metrics = {}

    masses = np.unique(X[:, -1])
    bkg_mask = (X[:, -1] < 0)

    for signal_mass in masses:
        # Skip background class
        if signal_mass < 0:
            continue

        pred = net.evaluate(X, signal_mass)

        sig_mask = (X[:, -1] == signal_mass)

        sig_total = W[sig_mask].sum()
        bkg_total = W[bkg_mask].sum()

        fpr, tpr, thr = roc_curve(Y[sig_mask | bkg_mask], pred[sig_mask | bkg_mask],
                                  sample_weight=W[sig_mask | bkg_mask])
        roc = interp1d(tpr, fpr)

        auc = roc_auc_score(Y[sig_mask | bkg_mask], pred[sig_mask | bkg_mask],
                            sample_weight=W[sig_mask | bkg_mask])

        metrics["AUC_M{:.3f}".format(signal_mass)] = auc
        metrics["ROC95_M{:.3f}".format(signal_mass)] = roc(0.95)
        metrics["ROC80_M{:.3f}".format(signal_mass)] = roc(0.8)

    return metrics
