import os
import yaml
import SUB as SUB

cv = 1
nmod = 23
res_dir = 'result/paper_Bayes_scenario_superpose'
fig_dir = 'figures/paper_Bayes_scenario_superpose'
ft = 'pdf'

GPU = True

home = os.path.abspath(os.path.dirname(__file__))
finfo = '%s/../../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)
out = '%s/../../%s/pre' % (home, fig_dir)

SUB.GaugeFigure(out=out, ft=ft, **info)
SUB.SingularValuesFigure(out=out, ft=ft, cv=cv, **info)
# SUB.ReconstructionErrorFigure(out=out, ft=ft, cv=cv, nmod=nmod, GPU=GPU, **info)