import os,sys
import ROOT
from ROOT import *
from math import *
from array import *

filestoplot = { 'mse peak' : {'filename' : 'mse1.txt', 'colour' : 1, 'marker' : 20},
                'mse pit' : {'filename' : 'mse2.txt', 'colour' : 2, 'marker' : 21},
                'theil peak' : {'filename' : 'theil1.txt', 'colour' : 3, 'marker' : 22},
                'theil pit' : {'filename' : 'theil2.txt', 'colour' : 4, 'marker' : 23},}

for entry in filestoplot :
  thisfile = open(filestoplot[entry]['filename'],'r')

  filestoplot[entry]['x'] = array('f',[0])
  filestoplot[entry]['y'] = array('f',[0])
  num = 0
  for line in thisfile :
    filestoplot[entry]['x'].append(float(line.split()[0]))
    filestoplot[entry]['y'].append(float(line.split()[1].strip('\n')))
    num += 1

  filestoplot[entry]['graph'] = TGraph(num,filestoplot[entry]['x'],filestoplot[entry]['y'])
  filestoplot[entry]['graph'].SetLineColor(filestoplot[entry]['colour'])
  filestoplot[entry]['graph'].SetMarkerColor(filestoplot[entry]['colour'])
  filestoplot[entry]['graph'].SetMarkerStyle(filestoplot[entry]['marker'])
  filestoplot[entry]['graph'].GetXaxis().SetTitle('#alpha')
  filestoplot[entry]['graph'].GetXaxis().SetNdivisions(510)
  filestoplot[entry]['graph'].GetYaxis().SetTitle('uniformity')
  thisfile.close() 

print filestoplot

c1 = TCanvas("c1","c1",1000,800)
c1.cd()
i=1
for entry in filestoplot :
  if i==1 :
    filestoplot[entry]['graph'].Draw()
    filestoplot[entry]['graph'].GetYaxis().SetRangeUser(0,2)
  else :
    filestoplot[entry]['graph'].Draw('PL')   
  i += 1
c1.GetPad(0).SetLogy()

leg = TLegend(0.6,0.2,0.9,0.5)
leg.SetFillStyle(0)
leg.SetFillColor(0)
leg.SetMargin(0.35)
leg.SetTextSize(0.04)
for entry in filestoplot :
  leg.AddEntry(filestoplot[entry]['graph'],entry,'p')
leg.Draw("SAME")

c1.SaveAs('TheilVsMSE.pdf')
c1.SaveAs('TheilVsMSE.eps')
c1.SaveAs('TheilVsMSE.png')
c1.SaveAs('TheilVsMSE.root')
