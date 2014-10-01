import os,sys
import ROOT
from ROOT import *
from math import *
from array import *

basedir = './data/raspodela1/raspodele/'
marker = 20
size = 0.5
colour = 1

filestoplot = { '#alpha=0.1':{'filename':'0_raspodela'},
                '#alpha=0.28':{'filename':'1_raspodela'},
                '#alpha=0.47':{'filename':'2_raspodela'},
                '#alpha=0.65':{'filename':'3_raspodela'},
                '#alpha=0.84':{'filename':'4_raspodela'}}

for entry in filestoplot :
  thisfile = open(basedir+filestoplot[entry]['filename'],'r')

  filestoplot[entry]['x'] = array('f',[0])
  filestoplot[entry]['y'] = array('f',[0])
  num = 0
  for line in thisfile :
    filestoplot[entry]['x'].append(float(line.split()[0]))
    filestoplot[entry]['y'].append(float(line.split()[1].strip('\n')))
    num += 1

  filestoplot[entry]['graph'] = TGraph(num,filestoplot[entry]['x'],filestoplot[entry]['y'])
  filestoplot[entry]['graph'].SetLineColor(colour)
  filestoplot[entry]['graph'].SetMarkerColor(colour)
  filestoplot[entry]['graph'].SetMarkerSize(size)
  filestoplot[entry]['graph'].SetMarkerStyle(marker)
  filestoplot[entry]['graph'].GetXaxis().SetTitle('mass')
  filestoplot[entry]['graph'].GetXaxis().SetNdivisions(510)
  filestoplot[entry]['graph'].GetYaxis().SetTitle('prediction')
  filestoplot[entry]['graph'].GetXaxis().SetTitleSize(0.08)
  filestoplot[entry]['graph'].GetXaxis().SetTitleOffset(0.6)
  filestoplot[entry]['graph'].GetYaxis().SetTitleSize(0.08)
  filestoplot[entry]['graph'].GetYaxis().SetTitleOffset(0.85)

  thisfile.close() 

#print filestoplot

c1 = TCanvas("c1","c1",1600,800)
c1.Divide(5)
i=1
for entry in filestoplot :
  c1.cd(i)
  filestoplot[entry]['graph'].Draw("AP")
  filestoplot[entry]['graph'].GetYaxis().SetRangeUser(-0.5,10.5)  
  filestoplot[entry]['graph'].GetXaxis().SetRangeUser(0.0,1.0)  
  i += 1
#c1.GetPad(0).SetLogy()

'''
leg = TLegend(0.6,0.2,0.9,0.5)
leg.SetFillStyle(0)
leg.SetFillColor(0)
leg.SetMargin(0.35)
leg.SetTextSize(0.04)
for entry in filestoplot :
  leg.AddEntry(filestoplot[entry]['graph'],entry,'p')
leg.Draw("SAME")
'''

c1.SaveAs('PeakDistributions.pdf')
c1.SaveAs('PeakDistributions.png')

