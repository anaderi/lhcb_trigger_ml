import os,sys
import ROOT
from ROOT import *
from math import *
from array import *

basedir = './dto3h/Signal/'
marker = 20
size = 0.5
colour = 1

filestoplot = { 'data_sde_sig'   : {'filename':'data_sde_sig'  ,'subclassifiers' : {}},
                'data_theil_sig' : {'filename':'data_theil_sig','subclassifiers' : {}}}

subclassifiercols = {'ada':1,'uGB+knnFL':5,'uBoost':6}

# Need a double loop here
for entry in filestoplot :
  thisfile = open(basedir+filestoplot[entry]['filename'],'r')
  for subfile in ['ada','uGB+knnFL','uBoost'] :

    filestoplot[entry]['subclassifiers'][subfile] = {}
    filestoplot[entry]['subclassifiers'][subfile]['x'] = array('f',[0])
    filestoplot[entry]['subclassifiers'][subfile]['y'] = array('f',[0])
  num = 0
  for line in thisfile :
    if num == 0 : 
      num += 1 
      continue
    for subfile in ['ada','uGB+knnFL','uBoost'] :    
      filestoplot[entry]['subclassifiers'][subfile]['x'].append(float(line.split(',')[0]))
      if entry.find('sde') > -1 :
        filestoplot[entry]['subclassifiers'][subfile]['y'].append(float(line.split(',')[subclassifiercols[subfile]].strip('\n')))
      else :
        filestoplot[entry]['subclassifiers'][subfile]['y'].append(100.*float(line.split(',')[subclassifiercols[subfile]].strip('\n')))
    num += 1

  i = 0   
  for subfile in ['ada','uGB+knnFL','uBoost'] :  
    filestoplot[entry]['subclassifiers'][subfile]['graph'] = TGraph(num, filestoplot[entry]['subclassifiers'][subfile]['x'],
                                                                   filestoplot[entry]['subclassifiers'][subfile]['y'])
    filestoplot[entry]['subclassifiers'][subfile]['graph'].SetLineColor(colour+i)
    filestoplot[entry]['subclassifiers'][subfile]['graph'].SetMarkerColor(colour+i)
    filestoplot[entry]['subclassifiers'][subfile]['graph'].SetMarkerSize(size)
    filestoplot[entry]['subclassifiers'][subfile]['graph'].SetMarkerStyle(marker+i)
    filestoplot[entry]['subclassifiers'][subfile]['graph'].GetXaxis().SetTitle('classifier stage')
    filestoplot[entry]['subclassifiers'][subfile]['graph'].GetXaxis().SetNdivisions(505)
    filestoplot[entry]['subclassifiers'][subfile]['graph'].GetYaxis().SetTitle('global uniformity')
    filestoplot[entry]['subclassifiers'][subfile]['graph'].GetXaxis().SetTitleSize(0.09)
    filestoplot[entry]['subclassifiers'][subfile]['graph'].GetXaxis().SetTitleOffset(0.75)
    filestoplot[entry]['subclassifiers'][subfile]['graph'].GetXaxis().SetLabelSize(0.065)
    filestoplot[entry]['subclassifiers'][subfile]['graph'].GetXaxis().SetLabelOffset(0.015)
    filestoplot[entry]['subclassifiers'][subfile]['graph'].GetYaxis().SetTitleSize(0.075)
    filestoplot[entry]['subclassifiers'][subfile]['graph'].GetYaxis().SetTitleOffset(0.95)
    filestoplot[entry]['subclassifiers'][subfile]['graph'].GetYaxis().SetLabelSize(0.055)
    filestoplot[entry]['subclassifiers'][subfile]['graph'].GetYaxis().SetDecimals()

    i+=1
  thisfile.close() 

#print filestoplot

c1 = TCanvas("c1","c1",1920,800)
c1.Divide(2)
i=1
for entry in filestoplot :
  c1.cd(i)
  first = True
  for subfile in ['ada','uGB+knnFL','uBoost'] :
    if first :
      first = False
      filestoplot[entry]['subclassifiers'][subfile]['graph'].Draw("APL")
    else :
      filestoplot[entry]['subclassifiers'][subfile]['graph'].Draw("PL")
    if entry.find('sde') > -1 :
      filestoplot[entry]['subclassifiers'][subfile]['graph'].GetYaxis().SetRangeUser(0.0,0.3)  
    else :
      filestoplot[entry]['subclassifiers'][subfile]['graph'].GetYaxis().SetRangeUser(0.0,0.08)        
    filestoplot[entry]['subclassifiers'][subfile]['graph'].GetXaxis().SetRangeUser(0.0,200.0)  
  i += 1
#c1.GetPad(0).SetLogy()

c1.cd(1)

leg = TLegend(0.525,0.175,0.925,0.375)
leg.SetFillStyle(1001)
leg.SetFillColor(ROOT.kWhite)
leg.SetMargin(0.35)
leg.SetTextSize(0.04)
for subfile in ['ada','uGB+knnFL','uBoost'] :
  leg.AddEntry(filestoplot['data_sde_sig']['subclassifiers'][subfile]['graph'],subfile,'pl')
leg.Draw("SAME")

c1.SaveAs('D23hSignalEffs.pdf')
c1.SaveAs('D23hSignalEffs.png')



