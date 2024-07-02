import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
import numpy as np
  
import yaml
import SUB as SUB

res_dir = 'result/paper_ver2'
fig_dir = 'res_paper_ver2'
ft = 'pdf'

cv = 1
ROM = True
nmod = 23

alp = 'a' # 'a' or 'b'

home = os.path.abspath(os.path.dirname(__file__))
finfo = '%s/../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)
out = os.path.join(home, fig_dir, 'max')

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


res_dir = 'result/paper'
home = os.path.abspath(os.path.dirname(__file__))
finfo = '%s/../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)

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