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

w = 'best' # 'best' or 'worst'
num = 3 # < 7
obs = 120
itesta = 42
itestb = 561

home = os.path.abspath(os.path.dirname(__file__))
finfo = '%s/../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)
out = os.path.join(home, fig_dir, 'waveform')

ttlst = pd.read_csv(info["fttlst"].format(cv))
tcase = ttlst.loc[ttlst['label']=='test', 'ID']
f = '%s/%s_%s_scenarios.%s' % (out, w, num, ft)

fig, axs = SUB.MakeAxesWaveTwoCols()

SUB.WaveTest( ax=axs[0], 
              cv=cv, itest=itesta, gID=9303,
              **info)
SUB.WaveTest( ax=axs[1], 
              cv=cv, itest=itestb, gID=9303,
              **info)

SUB.ObsWind(ax=axs[0], obs=obs)
SUB.ObsWind(ax=axs[1], obs=obs)

SUB.WaveBstWst( ax=axs[0], ver='Fujita', w=w, num=num,
              cv=cv, ROM=ROM, nmod=nmod, itest=itesta, obs=obs, 
              gID=9303, **info)
SUB.WaveBstWst( ax=axs[1], ver='Fujita', w=w, num=num,
              cv=cv, ROM=ROM, nmod=nmod, itest=itestb, obs=obs, 
              gID=9303, **info)

axs[1].legend(ncol=num+2, loc='lower right', 
              bbox_to_anchor=(1.0,1.0), frameon=False, borderaxespad=0.)
SUB.SettingWaveTwoCols( axs=axs, alp=['a','b'], 
                        sid=[list(tcase.index)[itesta] + 1, list(tcase.index)[itestb] + 1])
SUB.fsave(out, f)
