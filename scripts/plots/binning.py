#!/usr/bin/env python
import argparse
from array import array
from math import sqrt, log

parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("outfile")
parser.add_argument("--signal-name", default="Hhhbbtautau350")
parser.add_argument("--discriminant", default="PNN350")
parser.add_argument("--mu", default=None, type=float)
parser.add_argument("--formats", nargs="+", default=["eps"])
args = parser.parse_args()

import ROOT as R

R.gROOT.SetBatch(True)
R.gROOT.LoadMacro("$HOME/ATLAS/atlasstyle-00-04-02/AtlasStyle.C")
R.gROOT.LoadMacro("$HOME/ATLAS/atlasstyle-00-04-02/AtlasUtils.C")
R.gROOT.LoadMacro("$HOME/ATLAS/atlasstyle-00-04-02/AtlasLabels.C")
R.SetAtlasStyle()

latex = R.TLatex()
latex.SetNDC()
latex.SetTextSize(0.035)
latex.SetTextFont(42)

from ml_evaluation_tools import backgrounds, asimov_significance
from rebin import rebin


signal_name = args.signal_name
variable_name = args.discriminant
region_pattern = "{}_2tag2pjet_0ptv_LL_OS_{}"


f = R.TFile.Open(args.infile)

h_sig = f.Get(region_pattern.format(signal_name, variable_name))
h_sig.SetDirectory(0)
if args.mu:
    h_sig.Scale(args.mu)

h_bkg = None
for bkg in backgrounds:
    h = f.Get(region_pattern.format(bkg, variable_name))

    if not h:
        print("Omitting: " + bkg)
        continue

    if h_bkg is None:
        h_bkg = h
        h_bkg.SetDirectory(0)
    else:
        h_bkg.Add(h)


h_sig, h_bkg = rebin(h_sig, h_bkg)
nbins = h_sig.GetNbinsX()
#h_sig.Print("all")
#h_bkg.Print("all")

# Calculate per bin Asimov significance
asimov_signif, global_asimov_signif = asimov_significance(h_sig, h_bkg)

# Plotting
y_max = max(h_sig.GetMaximum(), h_bkg.GetMaximum())
y_min = min(h_sig.GetMinimum(), h_bkg.GetMinimum())

h_sig.SetLineColor(R.kRed)
h_sig.GetYaxis().SetRangeUser(y_min / 2.0, 2 * y_max)
h_sig.SetTitle(";Transformed {} Score;Expected Events".format(args.discriminant))
h_bkg.SetLineColor(R.kBlue)

asimov_signif.SetTitle(";Transformed {} Score;Asimov Significance".format(args.discriminant))
asimov_signif.GetYaxis().SetRangeUser(0, 1.2 * asimov_signif.GetMaximum())


leg = R.TLegend(0.45, 0.75, 0.65, 0.88)
leg.AddEntry(h_sig, args.signal_name, "f")
leg.AddEntry(h_bkg, "Total background", "f")
leg.SetBorderSize(0)
leg.SetFillColor(0)
leg.SetTextSize(0.035)
leg.SetTextFont(42)

c = R.TCanvas("c", "", 800, 800)
c.Divide(1, 2)

c.cd(1).SetLogy()
h_sig.Draw("HIST")
h_bkg.Draw("SAME,HIST")
leg.Draw()

c.cd(2)
asimov_signif.Draw("HIST")
R.ATLASLabel(0.25, 0.8, "Internal", R.kBlack)
latex.DrawLatex(0.25, 0.75, "Significance: {:.2f}".format(global_asimov_signif))

for ext in args.formats:
    c.SaveAs(args.outfile + "." + ext)
