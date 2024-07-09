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

itesta = 42
itestb = 561
obss = [30, 60, 120]

home = os.path.abspath(os.path.dirname(__file__))
finfo = '%s/../../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)
out = '%s/../../%s/taylor' % (home, fig_dir)

ttlst = pd.read_csv(info["fttlst"].format(cv))
tcase = ttlst.loc[ttlst['label']=='test', 'ID']

fig, axs = SUB.MakeAxesTaylor()

SUB.TaylorTest(ax=axs[0])
SUB.TaylorTest(ax=axs[1])

SUB.TaylorFujita( ax=axs[0], 
                  cv=cv, ROM=ROM, nmod=nmod, itest=itesta, obss=obss, 
                  gID=9303, **info)
SUB.TaylorFujita( ax=axs[1], 
                  cv=cv, ROM=ROM, nmod=nmod, itest=itestb, obss=obss, 
                  gID=9303, **info)

SUB.TaylorNomura( ax=axs[0], 
                  cv=cv, ROM=ROM, nmod=nmod, itest=itesta, obss=obss, 
                  gID=9303, **info)
SUB.TaylorNomura( ax=axs[1], 
                  cv=cv, ROM=ROM, nmod=nmod, itest=itestb, obss=obss, 
                  gID=9303, **info)

SUB.TaylorClosest(ax=axs[0], 
                  cv=cv, itest=itesta, 
                  gID=9303, **info)
SUB.TaylorClosest(ax=axs[1], 
                  cv=cv, itest=itestb, 
                  gID=9303, **info)

SUB.SettingTaylor(axs=axs, alp=['a','b'], 
                  sid=[list(tcase.index)[itesta]+1, list(tcase.index)[itestb]+1])

SUB.fsave(out, 'taylor.%s' % ft)