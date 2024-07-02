import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
import numpy as np
  
import yaml
import SUB as SUB
import pandas as pd

res_dir = 'result/paper_ver2'
fig_dir = 'res_paper_ver2'
ft = 'pdf'

cv = 1
ROM = True
nmod = 23

w = 'Cols' # 'Cols' or 'Rows'
alp = ' '
itest = 561

GPU = True

home = os.path.abspath(os.path.dirname(__file__))
finfo = '%s/../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)
out = os.path.join(home, fig_dir, 'waveform')

ttlst = pd.read_csv(info["fttlst"].format(cv))
tcase = ttlst.loc[ttlst['label']=='test', 'ID']
fil = '%s/prd_%s_%s.%s' % (out, itest, tcase.iloc[itest], ft)
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


res_dir = 'result/paper'
home = os.path.abspath(os.path.dirname(__file__))
finfo = '%s/../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)

SUB.WaveNomura( ax=axs[0], cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=30, 
                gID=9303, GPU=GPU, **info)
SUB.WaveNomura( ax=axs[1], cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=120, 
                gID=9303, GPU=GPU, **info)

SUB.SettingWavePrd(axs=axs, alp=alp, sid=sid, w=w)
SUB.fsave(out, fil)