import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
import numpy as np
  
import yaml
import SUB as SUB

res_dir = 'result/paper_Bayes_scenario_superpose'
fig_dir = 'figures/paper_Bayes_scenario_superpose'
ft = 'pdf'

cv = 1
ROM = True
nmod = 23

alp = 'b' # 'a' or 'b'

home = os.path.abspath(os.path.dirname(__file__))
finfo = '%s/../../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)
out = '%s/../../%s/max' % (home, fig_dir)

fig, axs = SUB.MakeAxesMax()

if alp=='a':
  SUB.MaxFujita(ax=axs[0], 
                cv=cv, ROM=ROM, nmod=nmod, obs=30, 
                gID=9303, **info)
  SUB.MaxFujita(ax=axs[1], 
                cv=cv, ROM=ROM, nmod=nmod, obs=60, 
                gID=9303, **info)
  SUB.MaxFujita(ax=axs[2], 
                cv=cv, ROM=ROM, nmod=nmod, obs=120, 
                gID=9303, **info)

if alp=='b':
  SUB.MaxNomura(ax=axs[0], 
                cv=cv, ROM=ROM, nmod=nmod, obs=30, 
                gID=9303, **info)
  SUB.MaxNomura(ax=axs[1], 
                cv=cv, ROM=ROM, nmod=nmod, obs=60, 
                gID=9303, **info)
  SUB.MaxNomura(ax=axs[2], 
                cv=cv, ROM=ROM, nmod=nmod, obs=120, 
                gID=9303, **info)

SUB.SettingMax(axs=axs, tit='({})'.format(alp))
SUB.fsave(out, 'max_{}.{}'.format(alp,ft))