#!/usr/bin/env python
import argparse
from array import array

parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("outfile")
parser.add_argument("--signal-name", default="Hhhbbtautau350")
parser.add_argument("--discriminant", default="PNN350")

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


from ml_evaluation_tools import get_roc, backgrounds

signal_name = args.signal_name
variable_name = args.discriminant
region_pattern = "{}_2tag2pjet_0ptv_LL_OS_{}"


f = R.TFile.Open(args.infile)

h_sig = f.Get(region_pattern.format(signal_name, variable_name))
h_sig.SetDirectory(0)

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

roc, auc = get_roc(h_sig, h_bkg)
roc.SetLineWidth(2)
roc.SetLineColor(R.kRed)
roc.SetTitle(";Signal Efficiency;Background Rejection")

c = R.TCanvas("c", "", 800, 600)
roc.Draw("AL")
R.ATLASLabel(0.25, 0.4, "Internal", R.kBlack)
latex.DrawLatex(0.25, 0.35, "Discriminant: " + args.discriminant)
latex.DrawLatex(0.25, 0.3, "Signal: " + args.signal_name)
latex.DrawLatex(0.25, 0.25, "AUC: {:1.5f}".format(auc))

for ext in args.formats:
    c.SaveAs(args.outfile + "." + ext)
