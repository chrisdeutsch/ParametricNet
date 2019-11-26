from array import array
from math import sqrt, log
import ROOT as R


backgrounds = [
    "Fake",
    "ttbar", "ttbarTF", "ttbarFT", "ttbarFF",
    "stops", "stopt", "stopWt",
    "Zl", "Zcl", "Zcc", "Zbl", "Zbc", "Zbb",
    "Wtt",
    "Zttl", "Zttcl", "Zttcc", "Zttbl", "Zttbc", "Zttbb",
    "ZZ", "WZ", "WW", "VH"
]


def get_roc(sig, bkg):
    h_sig = sig.Clone()
    h_bkg = bkg.Clone()

    # Normalize
    n_bins = h_sig.GetNbinsX()
    h_sig.Scale(1 / h_sig.Integral(0, n_bins + 1))
    h_bkg.Scale(1 / h_bkg.Integral(0, n_bins + 1))

    h_sig_cum = h_sig.GetCumulative(R.kFALSE)
    h_bkg_cum = h_bkg.GetCumulative(R.kFALSE)

    sig_effs = [1.0]
    bkg_rejs = [0.0]
    for bin_idx in range(1, n_bins + 1):
        sig_effs.append(h_sig_cum.GetBinContent(bin_idx))
        bkg_rejs.append(1.0 - h_bkg_cum.GetBinContent(bin_idx))

    sig_effs.append(0)
    bkg_rejs.append(1)

    sig_effs = array("f", sig_effs)
    bkg_rejs = array("f", bkg_rejs)

    # Trapezoidal rule
    deltas = [k - km1 for km1, k in zip(sig_effs[1:], sig_effs[:-1])]
    sums = [k + km1 for k, km1 in zip(bkg_rejs[1:], bkg_rejs[:-1])]
    integral = sum(s * d / 2.0 for s, d in zip(sums, deltas))

    return R.TGraph(len(sig_effs), sig_effs, bkg_rejs), integral


def asimov_significance(sig, bkg):
    assert sig.GetNbinsX() == bkg.GetNbinsX()
    nbins = sig.GetNbinsX()

    h_asimov = R.TH1F("h_asimov", ";;Asimov Significance", nbins, 0, 1)
    h_asimov.SetDirectory(0)
    z0_sumw2 = 0.0

    for idx in range(0, nbins + 2):
        s = sig.GetBinContent(idx)
        b = bkg.GetBinContent(idx)

        if b <= 0:
            continue

        z0 = sqrt(2 * ((s + b) * log(1 + s / b) - s))
        h_asimov.SetBinContent(idx, z0)
        z0_sumw2 += z0**2

    return h_asimov, sqrt(z0_sumw2)
