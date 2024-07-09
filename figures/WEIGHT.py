import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
import numpy as np
  
import yaml
import SUB as SUB
import pandas as pd

res_dir = 'result/paper_Bayes_scenario_superpose'
fig_dir = 'figures/paper_Bayes_scenario_superpose'
ft = 'pdf'

cv = 1
ROM = True
nmod = 23

alp = 'b'
itest = 561

GPU = True

home = os.path.abspath(os.path.dirname(__file__))
finfo = '%s/../../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)
out = '%s/../../%s/weight' % (home, fig_dir)

ttlst = pd.read_csv(info["fttlst"].format(cv))
tcase = ttlst.loc[ttlst['label']=='test', 'ID']

fil = '%s_%s.%s' % (itest, tcase.iloc[itest], ft)
sid = list(tcase.index)[itest] + 1

fig, axs = SUB.MakeAxesWeight(figs=(3.5,3.5))
SUB.WeiInit(axs=axs, out=out, ft=ft)

fig, axs = SUB.MakeAxesWeightThrCols()
SUB.WeiPost(axs=[axs[0,0],0,axs[1,0],axs[1,1]], out=out, 
              cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=30, 
              **info)
SUB.WeiPost(axs=[axs[0,3],0,axs[1,3],axs[1,4]], out=out, 
              cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=60, 
              **info)
SUB.WeiPost(axs=[axs[0,6],0,axs[1,6],axs[1,7]], out=out, 
              cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=120, 
              **info)

SUB.SettingWeights(axs=axs, alp=alp, sid=sid, obs=[30,60,120])
SUB.fsave(out, fil)