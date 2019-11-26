import ROOT as R
from math import sqrt


def rebin(sig_initial, bkg_initial):
    nbins = sig_initial.GetNbinsX()
    sig_total = sig_initial.Integral(0, nbins + 1)
    bkg_total = bkg_initial.Integral(0, nbins + 1)

    # Initial histograms
    sig_sumw = [sig_initial.GetBinContent(idx) for idx in range(0, nbins + 2)]
    bkg_sumw = [bkg_initial.GetBinContent(idx) for idx in range(0, nbins + 2)]

    sig_sumw2 = [sig_initial.GetBinError(idx)**2 for idx in range(0, nbins + 2)]
    bkg_sumw2 = [bkg_initial.GetBinError(idx)**2 for idx in range(0, nbins + 2)]

    # Reverse bins so that signal is on the left
    for hist in [sig_sumw, bkg_sumw, sig_sumw2, bkg_sumw2]:
        hist.reverse()

    # Rebinned histogram
    sig_sumw_rebinned, sig_sumw2_rebinned = [], []
    bkg_sumw_rebinned, bkg_sumw2_rebinned = [], []

    # Merged bin counters
    sig_bin_sumw, sig_bin_sumw2 = 0.0, 0.0
    bkg_bin_sumw, bkg_bin_sumw2 = 0.0, 0.0

    for sw, sw2, bw, bw2 in zip(sig_sumw, sig_sumw2, bkg_sumw, bkg_sumw2):
        # Add to merged bin
        sig_bin_sumw += sw
        sig_bin_sumw2 += sw2
        bkg_bin_sumw += bw
        bkg_bin_sumw2 += bw2

        # Maximum relative background uncertainty
        max_bkg_unc = 0.5 * sig_bin_sumw / sig_total + 0.01 if sig_total > 0 else 0.5

        # Minimum number of expected background events
        min_bkg_exp = 5

        # Skip bins without background or negative signal
        if bkg_bin_sumw <= 0 or sig_bin_sumw < 0:
            continue

        # Check if we have a good bin
        rel_bkg_unc = sqrt(bkg_bin_sumw2) / bkg_bin_sumw
        exp_bkg = bkg_bin_sumw

        if rel_bkg_unc < max_bkg_unc and exp_bkg > min_bkg_exp:
            sig_sumw_rebinned.append(sig_bin_sumw)
            sig_sumw2_rebinned.append(sig_bin_sumw2)

            bkg_sumw_rebinned.append(bkg_bin_sumw)
            bkg_sumw2_rebinned.append(bkg_bin_sumw2)

            # Reset bin counters
            sig_bin_sumw, sig_bin_sumw2 = 0.0, 0.0
            bkg_bin_sumw, bkg_bin_sumw2 = 0.0, 0.0

    # Leftover bins are put into the last successfully merged bin
    sig_sumw_rebinned[-1] += sig_bin_sumw
    bkg_sumw_rebinned[-1] += bkg_bin_sumw

    sig_sumw2_rebinned[-1] += sig_bin_sumw2
    bkg_sumw2_rebinned[-1] += bkg_bin_sumw2

    # Reverse again to restore initial ordering
    sig_sumw_rebinned.reverse()
    bkg_sumw_rebinned.reverse()

    sig_sumw2_rebinned.reverse()
    bkg_sumw2_rebinned.reverse()

    # Sanity check
    assert len(sig_sumw_rebinned) == len(bkg_sumw_rebinned)
    assert len(sig_sumw2_rebinned) == len(bkg_sumw2_rebinned)
    assert len(sig_sumw_rebinned) == len(bkg_sumw2_rebinned)

    # Build TH1Fs
    nbins = len(sig_sumw_rebinned)

    h_sig = R.TH1F("h_sig_rebinned", "", nbins, 0, 1)
    h_sig.Sumw2()
    h_sig.SetDirectory(0)
    for i, (sumw, sumw2) in enumerate(zip(sig_sumw_rebinned, sig_sumw2_rebinned), 1):
        h_sig.SetBinContent(i, sumw)
        h_sig.SetBinError(i, sqrt(sumw2))

    h_bkg = R.TH1F("h_bkg_rebinned", "", nbins, 0, 1)
    h_bkg.Sumw2()
    h_bkg.SetDirectory(0)
    for i, (sumw, sumw2) in enumerate(zip(bkg_sumw_rebinned, bkg_sumw2_rebinned), 1):
        h_bkg.SetBinContent(i, sumw)
        h_bkg.SetBinError(i, sqrt(sumw2))

    return h_sig, h_bkg
