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

itestt = [42,561]
obs = 120
# 
home = os.path.abspath(os.path.dirname(__file__))
finfo = '%s/../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)
outh = os.path.join(home, fig_dir, 'histgram')
outb = os.path.join(home, fig_dir, 'bar')

ttlst = pd.read_csv(info["fttlst"].format(cv))
tcase = ttlst.loc[ttlst['label']=='test', 'ID']

sid = []
for itest in itestt:
  sid.append(list(tcase.index)[itest] + 1)

fig, axs = SUB.MakeAxesWeiDist()

SUB.WeightHistgram(ax=axs[0], 
                   cv=cv, ROM=ROM, nmod=nmod, itest=42, obs=obs, **info)
SUB.WeightHistgram(ax=axs[1],
                   cv=cv, ROM=ROM, nmod=nmod, itest=561, obs=obs, **info)

SUB.WeightBox(ax=axs[2],
              cv=cv, ROM=ROM, nmod=nmod, itest=42, obs=obs, **info)
SUB.WeightBox(ax=axs[3],
              cv=cv, ROM=ROM, nmod=nmod, itest=561, obs=obs, **info)

SUB.SettingHistgram(axs=axs, alp=['a','b'], 
                    sid=[list(tcase.index)[42]+1, list(tcase.index)[561]+1])
SUB.fsave(outh, 'hist.{}'.format(ft))

