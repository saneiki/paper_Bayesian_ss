import os
os.environ['OPENBLAS_NUM_THREADS']  = '1'
import numpy as np

import yaml
import datetime
import PRE
import FORECAST

preproc  = True  # Flag of preprocess execution
forecast = True  # Flag of forecasting execution
home = os.path.abspath(os.path.dirname(__file__))

fcond = os.path.join(home, 'COND.yaml')

with open(fcond, 'r') as f:
  cond = yaml.safe_load(f)
cmn = cond["cmn_param"]
fcst = cond["fcst_param"]

print(datetime.datetime.now())

# =======  Preprocessing  =============
if preproc:
  print('###  Preprocessing  ###')
  pre = PRE.PreProcess( **cmn["dirs"],
                        **cmn["fils"],
                        **cmn["nums"])
  pre.FigGauges()
  pre.TTSplit(cv=cmn["cv"])
  pre.SVD(GPU=cmn["GPU"], cv=cmn["cv"])
  pre.FigSingular(cv=cmn["cv"]) 
  pre.FigReconError(GPU=cmn["GPU"], cv=cmn["cv"]) 
  pre.YamlOut() 
  print(datetime.datetime.now(),'\n')
else:
  print('###  Skip Preprocessing  ###','\n')
# =====================================

res_dir = cond["cmn_param"]["dirs"]["res_dir"]
finfo = '%s/../../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)

# =======  Forecasting  =============
if forecast:
  print('###  Forecasting with arbitrary conditions  ###')

  if fcst["ltest"]=='all':
    ltest = [i for i in range (info['ntst_cv_set'][cmn["cv"]])]
  else:
    ltest = fcst["ltest"]

  prd = FORECAST.Forecasting( ver=fcst["ver"],
                              ROM=fcst["ROM"], 
                              nmod=fcst["nmod"], 
                              **info)
  for itest in ltest:
    if fcst["ver"]=='Fujita':
      prd.Run_wave_based( GPU=cmn["GPU"], 
                          cv=cmn["cv"], 
                          itest=itest, 
                          obs_window=fcst["obs_window"], 
                          cov_out=True)
    elif fcst["ver"]=='Nomura':
      prd.Run_state_based(GPU=cmn["GPU"], 
                          cv=cmn["cv"], 
                          itest=itest, 
                          obs_window=fcst["obs_window"], 
                          cov_out=False)
else:
  print('###  Skip forecasting  ###','\n')
# =====================================