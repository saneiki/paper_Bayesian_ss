import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
import numpy as np
  
import yaml
import SUB as SUB
import pandas as pd

res_dir = 'result/paper'
fig_dir = 'test'
ft = 'pdf'

cv = 1
ROM = True
nmod = 23

# alp = 'b'
itest = 561

GPU = True

home = os.path.abspath(os.path.dirname(__file__))
finfo = '%s/../%s/INFO.yaml' % (home, res_dir)
with open(finfo, 'r') as f:
  info = yaml.safe_load(f)
out = os.path.join(home, fig_dir, 'coefficient')

ttlst = pd.read_csv(info["fttlst"].format(cv))
tcase = ttlst.loc[ttlst['label']=='test', 'ID']
f = 'CoefInvEst.%s' % ft
sid = list(tcase.index)[itest] + 1

fig, axs = SUB.MakeAxesCoefficient()

for i in range(3):
  SUB.CoefTest( ax=axs[i,0], 
                cv=cv, itest=42, cID=i,
                **info)
  SUB.CoefTest( ax=axs[i,1], 
                cv=cv, itest=561, cID=i,
                **info)
  
  # SUB.CoefTest2( ax=axs[i,0], 
  #               cv=cv, itest=42, cID=i,
  #               **info)
  # SUB.CoefTest2( ax=axs[i,1], 
  #               cv=cv, itest=561, cID=i,
  #               **info)
          
  SUB.CoefPseudoInv(ax=axs[i,0], 
                    cv=cv, nmod=nmod, itest=42, 
                    cID=i, **info)
  SUB.CoefPseudoInv(ax=axs[i,1], 
                    cv=cv, nmod=nmod, itest=561, 
                    cID=i, **info)
  
  SUB.CoefKalmanFilter(ax=axs[i,0], 
                    cv=cv, nmod=nmod, itest=42, 
                    cID=i, **info)
  SUB.CoefKalmanFilter(ax=axs[i,1], 
                    cv=cv, nmod=nmod, itest=561, 
                    cID=i, **info)

SUB.SettingCoefficient( axs=axs, alp=['a','b'], 
                        sid=[list(tcase.index)[42]+1, list(tcase.index)[561]+1])
SUB.fsave(out, f)
