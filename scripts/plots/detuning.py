#!/usr/bin/env python
import argparse
import re
from array import array

parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("outfile")
parser.add_argument("--signal-name", default="Hhhbbtautau350")
parser.add_argument("--discriminant", default="PNN")
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

latex_no_ndc = R.TLatex()
latex_no_ndc.SetTextSize(0.035)
latex_no_ndc.SetTextFont(42)
latex_no_ndc.SetTextAlign(R.kHAlignCenter)


from ml_evaluation_tools import get_roc, backgrounds, asimov_significance
from rebin import rebin


masses = [251, 260, 280, 300, 325, 350, 400, 450,
          500, 550, 600, 700, 800, 900, 1000]

signal_name = args.signal_name
region_pattern = "{}_2tag2pjet_0ptv_LL_OS_{}"

f = R.TFile.Open(args.infile)

mass_list = []
auc_list = []
signif_list = []


for mass_param in sorted([251, 325] + [i for i in range(250, 1001, 2)]):
    variable_name = "{}{}".format(args.discriminant, mass_param)

    h_sig = f.Get(region_pattern.format(signal_name, variable_name))
    h_sig.SetDirectory(0)

    h_bkg = None
    for bkg in backgrounds:
        h = f.Get(region_pattern.format(bkg, variable_name))

        if not h:
            #print("Omitting: " + bkg)
            continue

        if h_bkg is None:
            h_bkg = h
            h_bkg.SetDirectory(0)
        else:
            h_bkg.Add(h)

    roc, auc = get_roc(h_sig, h_bkg)
    asimov_signif, signif = asimov_significance(*rebin(h_sig, h_bkg))

    mass_list.append(mass_param)
    auc_list.append(auc)
    signif_list.append(signif)


g_auc = R.TGraph(len(auc_list), array("f", mass_list), array("f", auc_list))
g_auc.SetLineWidth(2)
g_auc.SetLineColor(R.kRed)
g_auc.SetTitle(";Mass Parameter [GeV];AUC")
g_auc.GetXaxis().SetRangeUser(250, 1000)
g_auc.GetYaxis().SetRangeUser(0.4, 1.0)

g_signif = R.TGraph(len(signif_list), array("f", mass_list), array("f", signif_list))
g_signif.SetLineWidth(2)
g_signif.SetLineColor(R.kRed)
g_signif.SetTitle(";Mass Parameter [GeV];Asimov Significance")
g_signif.GetXaxis().SetRangeUser(250, 1000)
g_signif.GetYaxis().SetRangeUser(0, 1.2 * max(signif_list))


sig_mass = None
match = re.search("^\S+?(\d+)$", signal_name)
if match:
    sig_mass, = match.groups()
    sig_mass = int(sig_mass)

assert match


def frange(x, y, step):
    while x < y:
        yield x
        x+= step

def rfrange(x, y, step):
    while x > y:
        yield x
        x -= step

# Determine width
signif_max = g_signif.Eval(sig_mass)
fraction_of_max = 0.8
signif_threshold = fraction_of_max * signif_max

upper_threshold = None
for mass in frange(sig_mass, 1000, 0.5):
    if g_signif.Eval(mass) < signif_threshold:
        upper_threshold = mass
        break

lower_threshold = None
for mass in rfrange(sig_mass, 250, 0.5):
    if g_signif.Eval(mass) < signif_threshold:
        lower_threshold = mass
        break


line = R.TLine(sig_mass, 0.75, sig_mass, 1.0)
line.SetLineWidth(2)
line.SetLineStyle(R.kDashed)
line.SetLineColor(R.kGray + 1)


# Significance thresholds
line_lower = None
line_upper = None

if lower_threshold:
    line_lower = R.TLine(lower_threshold, 0.0, lower_threshold, signif_max)
    line_lower.SetLineWidth(1)
    line_lower.SetLineStyle(R.kDashed)
    line_lower.SetLineColor(R.kGray + 1)

if upper_threshold:
    line_upper = R.TLine(upper_threshold, 0.0, upper_threshold, signif_max)
    line_upper.SetLineWidth(1)
    line_upper.SetLineStyle(R.kDashed)
    line_upper.SetLineColor(R.kGray + 1)


c = R.TCanvas("c", "", 800, 600)
g_auc.Draw("ALC")
if line:
    line.Draw("SAME")
    latex_no_ndc.DrawLatex(line.GetX1(), 0.7, "#color[921]{Signal mass}")

R.ATLASLabel(0.25, 0.35, "Internal", R.kBlack)
latex.DrawLatex(0.25, 0.3, "Signal: " + args.signal_name)

for ext in args.formats:
    c.SaveAs(args.outfile + "." + ext)


c.Clear()
g_signif.Draw("ALC")

if line_lower:
    line_lower.Draw("SAME")
if line_upper:
    line_upper.Draw("SAME")

R.ATLASLabel(0.25, 0.35, "Internal", R.kBlack)
latex.DrawLatex(0.25, 0.3, "Signal: " + args.signal_name)
latex.DrawLatex(0.25, 0.25, "Significance: {:.2f}".format(signif_max))

latex.DrawLatex(0.6, 0.8, "Width at {:.0%}:".format(fraction_of_max))
if upper_threshold:
    latex.DrawLatex(0.6, 0.75, "+ {:.1f} GeV".format(upper_threshold - sig_mass))
if lower_threshold:
    latex.DrawLatex(0.6, 0.7, "- {:.1f} GeV".format(sig_mass - lower_threshold))

for ext in args.formats:
    c.SaveAs(args.outfile + "_asimov." + ext)
