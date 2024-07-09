import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
import numpy as np

import yaml
import SUB as SUB
import pandas as pd

res_dir = 'result/paper_Bayes_scenario_superpose'
fig_dir = 'figures/paper_Bayes_scenario_superpose'
ft = 'pdf'

itest = 561

cv = 1

home = os.path.abspath(os.path.dirname(__file__))
finfo = '%s/../../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)
out = '%s/../../%s/rupture' % (home, fig_dir)

ttlst = pd.read_csv(info["fttlst"].format(cv))
tcase = ttlst.loc[ttlst['label']=='test', 'ID']

f = 'rupture_%s_%s.%s' % (itest, tcase.iloc[itest], ft)
sid = list(tcase.index)[itest] + 1

fig, axs = SUB.MakeAxesRupture()
SUB.SlipDist(ax=axs[0], mw=tcase.iloc[itest][9:11], rnum=tcase.iloc[itest][-5:])
SUB.SettingSlipDist(ax=axs[0])

SUB.DeformDist(ax=axs[1], mw=tcase.iloc[itest][9:11], rnum=tcase.iloc[itest][-5:])
SUB.SettingSlipDist(ax=axs[1])
SUB.fsave(out, f)