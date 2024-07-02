import os
os.environ['OPENBLAS_NUM_THREADS']  = '1'
import numpy as np

import yaml
import datetime
import PRE
import FORECAST

preproc  = False  # Flag of preprocess execution
forecast = True  # Flag of forecasting execution
forecast_forall = False  # Flag for execution of forecasting with paper conditions
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
finfo = '%s/../%s/INFO.yaml' % (home, res_dir)
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
    prd.Run(  GPU=cmn["GPU"], 
              cv=cmn["cv"], 
              itest=itest, 
              obs_window=fcst["obs_window"], 
              cov_out=True)
    # prd.WaveformFigure( GPU=cmn["GPU"], 
                        # cv=cmn["cv"], 
                        # itest=itest, 
                        # obs_window=fcst["obs_window"], 
                        # gIDlist=[9301,9303,9304,9305,9311])
    # prd.TaylorFigure( cv=cmn["cv"],
                      # itest=itest, 
                      # obs_window=fcst["obs_window"], 
                      # gIDlist=[9301,9303,9304,9305,9311])
    # if prd.ver=='Fujita':
      # prd.WeightsPDFFigure( cv=cmn["cv"], itest=itest, obs_window=fcst["obs_window"])
else:
  print('###  Skip forecasting  ###','\n')
# =====================================