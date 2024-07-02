import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
import numpy as np

import yaml
import SUB as SUB

cv = 1
nmod = 23
res_dir = 'result/paper'
fig_dir = 'test'
ft = 'pdf'

GPU = True

home = os.path.abspath(os.path.dirname(__file__))
finfo = '%s/../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)
out = os.path.join(home, fig_dir, 'pre')

SUB.GaugeFigure(out=out, ft=ft, **info)
SUB.SingularValuesFigure(out=out, ft=ft, cv=cv, **info)
# SUB.ReconstructionErrorFigure(out=out, ft=ft, cv=cv, nmod=nmod, GPU=GPU, **info)