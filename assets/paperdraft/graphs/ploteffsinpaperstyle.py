import os,sys
import ROOT
from ROOT import *
from math import *
from array import *

basedir = './data/raspodela1/efikasnosti/'
marker = 20
size = 0.5
colour = 1

filestoplot = { '#alpha=0.1':{'dirname':'0','subfiles' : {}},
                #'#alpha=0.28':{'dirname':'1','subfiles' : {}},
                '#alpha=0.47':{'dirname':'2','subfiles' : {}},
                #'#alpha=0.65':{'dirname':'3','subfiles' : {}},
                '#alpha=0.84':{'dirname':'4','subfiles' : {}}}

suffix = 'eff'

# Need a double loop here
for entry in filestoplot :
  i = 0 
  for subfile in ['0','2','4','6','8'] :
    if i == 4 : i = 5
    thisfile = open(basedir+filestoplot[entry]['dirname']+'/'+filestoplot[entry]['dirname']+"_"+subfile+suffix,'r')

    filestoplot[entry]['subfiles'][subfile] = {}
    filestoplot[entry]['subfiles'][subfile]['x'] = array('f',[0])
    filestoplot[entry]['subfiles'][subfile]['y'] = array('f',[0])
    num = 0
    for line in thisfile :
      filestoplot[entry]['subfiles'][subfile]['x'].append(float(line.split()[0]))
      filestoplot[entry]['subfiles'][subfile]['y'].append(float(line.split()[1].strip('\n')))
      num += 1

    filestoplot[entry]['subfiles'][subfile]['graph'] = TGraph(num, filestoplot[entry]['subfiles'][subfile]['x'],
                                                                   filestoplot[entry]['subfiles'][subfile]['y'])
    filestoplot[entry]['subfiles'][subfile]['graph'].SetLineColor(colour+i)
    filestoplot[entry]['subfiles'][subfile]['graph'].SetMarkerColor(colour+i)
    filestoplot[entry]['subfiles'][subfile]['graph'].SetMarkerSize(size)
    filestoplot[entry]['subfiles'][subfile]['graph'].SetMarkerStyle(marker+i)
    filestoplot[entry]['subfiles'][subfile]['graph'].GetXaxis().SetTitle('mass')
    filestoplot[entry]['subfiles'][subfile]['graph'].GetXaxis().SetNdivisions(510)
    filestoplot[entry]['subfiles'][subfile]['graph'].GetYaxis().SetTitle('efficiency')
    filestoplot[entry]['subfiles'][subfile]['graph'].GetXaxis().SetLabelSize(0.075)
    filestoplot[entry]['subfiles'][subfile]['graph'].GetXaxis().SetLabelOffset(0.015)
    filestoplot[entry]['subfiles'][subfile]['graph'].GetXaxis().SetTitleSize(0.10)
    filestoplot[entry]['subfiles'][subfile]['graph'].GetXaxis().SetTitleOffset(0.6)
    filestoplot[entry]['subfiles'][subfile]['graph'].GetYaxis().SetTitleSize(0.10)
    filestoplot[entry]['subfiles'][subfile]['graph'].GetYaxis().SetLabelSize(0.075)
    filestoplot[entry]['subfiles'][subfile]['graph'].GetYaxis().SetTitleOffset(0.65)

    thisfile.close() 
    i+=1

#print filestoplot

c1 = TCanvas("c1","c1",1920,500)
c1.Divide(3)
i=1
for entry in filestoplot :
  c1.cd(i)
  first = True
  for subfile in ['0','2','4','6','8'] :
    if first :
      first = False
      filestoplot[entry]['subfiles'][subfile]['graph'].Draw("APL")
    else :
      filestoplot[entry]['subfiles'][subfile]['graph'].Draw("PL")
    filestoplot[entry]['subfiles'][subfile]['graph'].GetYaxis().SetRangeUser(0.0,1.0)  
    filestoplot[entry]['subfiles'][subfile]['graph'].GetXaxis().SetRangeUser(0.0,1.0)  
  i += 1
#c1.GetPad(0).SetLogy()

c1.cd(1)

leg = TLegend(0.625,0.175,0.925,0.375)
leg.SetFillStyle(1001)
leg.SetFillColor(ROOT.kWhite)
leg.SetMargin(0.35)
leg.SetTextSize(0.04)
for subfile in ['0','2','4','6','8'] :
  leg.AddEntry(filestoplot['#alpha=0.1']['subfiles'][subfile]['graph'],'eff = '+str(int(subfile)+1)+'0\%','pl')
leg.Draw("SAME")

c1.SaveAs('PeakEffs.pdf')
c1.SaveAs('PeakEffs.png')


