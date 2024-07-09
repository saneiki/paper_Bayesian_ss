import os
  
import yaml
import SUB as SUB
import pandas as pd

res_dir = 'result/paper_Bayes_scenario_superpose'
fig_dir = 'figures/paper_Bayes_scenario_superpose'
ft = 'pdf'

cv = 1
ROM = True
nmod = 23

w = 'Cols' # 'Cols' or 'Rows'
alp = 'b'
itest = 561

GPU = True

home = os.path.abspath(os.path.dirname(__file__))
finfo = '%s/../../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)
out = '%s/../../%s/waveform' % (home, fig_dir)

ttlst = pd.read_csv(info["fttlst"].format(cv))
tcase = ttlst.loc[ttlst['label']=='test', 'ID']
fil = 'prd_%s_%s.%s' % (itest, tcase.iloc[itest], ft)
sid = list(tcase.index)[itest] + 1

if w=='Cols':
  fig, axs = SUB.MakeAxesWaveTwoCols()
if w=='Rows':
  fig, axs = SUB.MakeAxesWaveTwoRows()

SUB.WaveTest( ax=axs[0], cv=cv, itest=itest, gID=9303, **info)
SUB.WaveTest( ax=axs[1], cv=cv, itest=itest, gID=9303, **info)

SUB.ObsWind(ax=axs[0], obs=30)
SUB.ObsWind(ax=axs[1], obs=120)

SUB.WaveFujita( ax=axs[0], cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=30, 
                gID=9303, GPU=GPU, **info)
SUB.WaveFujita( ax=axs[1], cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=120, 
                gID=9303, GPU=GPU, **info)

SUB.WaveNomura( ax=axs[0], cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=30, 
                gID=9303, GPU=GPU, **info)
SUB.WaveNomura( ax=axs[1], cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=120, 
                gID=9303, GPU=GPU, **info)

SUB.SettingWavePrd(axs=axs, alp=alp, sid=sid, w=w)
SUB.fsave(out, fil)