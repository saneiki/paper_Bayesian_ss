import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
import numpy as np
  
import yaml
import SUB as SUB
import pandas as pd

res_dir = 'result/paper'
fig_dir = 'test'
ft = 'pdf'

cv = 1
ROM = True
nmod = 23

obs = 120
itesta = 42
itestb = 561

nbest = 3
nworst = 3

GPU = True

home = os.path.abspath(os.path.dirname(__file__))
finfo = '%s/../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)
out = os.path.join(home, fig_dir, 'waveform')

ttlst = pd.read_csv(info["fttlst"].format(cv))
tcase = ttlst.loc[ttlst['label']=='test', 'ID']
f = '%s/prd_part_scenarios.%s' % (out, ft)

fig, axs = SUB.MakeAxesWaveTwoCols()

SUB.WaveTest( ax=axs[0], cv=cv, itest=itesta, gID=9303, **info)
SUB.WaveTest( ax=axs[1], cv=cv, itest=itestb, gID=9303, **info)

SUB.ObsWind(ax=axs[0], obs=obs)
SUB.ObsWind(ax=axs[1], obs=obs)

SUB.WavePart( ax=axs[0], nb=nbest, nw=nworst,
                cv=cv, ROM=ROM, nmod=nmod, itest=itesta, obs=120, 
                gID=9303, GPU=GPU, **info)
SUB.WavePart( ax=axs[1], nb=nbest, nw=nworst,
                cv=cv, ROM=ROM, nmod=nmod, itest=itestb, obs=120, 
                gID=9303, GPU=GPU, **info)

SUB.SettingWaveTwoCols( axs=axs, alp=['a','b'], 
                        sid=[list(tcase.index)[itesta]+1, list(tcase.index)[itestb]+1])
SUB.fsave(out, f)
