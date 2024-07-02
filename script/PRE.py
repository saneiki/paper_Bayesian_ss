import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
import numpy as np

import yaml
import random
import datetime
import cupy as cp
import pandas as pd
import dask.array as da

import subfigure as SF
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib        # Optional 
# matplotlib.use("TkAgg")  # Optional 


class PreProcess():
  """Module for preprocessing
    (data reading and splitting, POD).

  Methods
  ---------
  FigGauges(figs=(6.3), fonts=5)
    Make a figure showing gauge locations with png or kml format.
  TTSplit(cv, seed=7)
    Split all scenarios into training and test scenarios.
  SVD(GPU, cv)
    conduct singular value decomposition for reduced-order modeling.
  FigSingular(cv, thre=[0.8,0.9,0.95,0.99], figs=(6,3), fonts=8)
    Make a figure showing contribution rates of singular values.
  FigReconError(GPU, cv, figs=(6,3), fonts=8)
    Make a figure showing reconstruction errors vs used POD modes.
  YamlOut()
    Output calculation conditions.
  """

  def __init__( self, inp_dir, res_dir, 
                fwave, fgauge, fcase,
                ftrain, ftest, fttlist, fu, fs,
                ncv_set, nsce, ntim, ngag
              ):
    """Parameter initialization"""
    print('  - Initialization for preprocessing')

    home = os.path.abspath(os.path.dirname(__file__))
    self.inp_dir = '%s/../%s' % (home, inp_dir)
    self.res_dir = '%s/../../%s' % (home, res_dir)

    self.res_cv_dir = '%s/CrsVld_{}' % self.res_dir
    self.fwave      = '%s/%s' % (self.inp_dir, fwave)
    self.fgauge     = '%s/%s' % (self.inp_dir, fgauge)
    self.fcase      = '%s/%s' % (self.inp_dir, fcase)
    self.fXtrn      = '%s/%s' % (self.res_cv_dir, ftrain)
    self.fXtst      = '%s/%s' % (self.res_cv_dir, ftest)
    self.fttlst     = '%s/%s' % (self.res_cv_dir, fttlist)
    self.fu         = '%s/%s' % (self.res_cv_dir, fu)
    self.fs         = '%s/%s' % (self.res_cv_dir, fs)

    self.ncv_set = ncv_set
    self.nsce = nsce
    self.ntim = ntim
    self.ngag = ngag

    self.ntst_cv_set = [0] * ncv_set
    self.ntrn_cv_set = [0] * ncv_set


  def FigGauges(self, figs=(6,3), fonts=5):
    """Makes a figure showing used gauge arrangement."""

    os.makedirs(self.res_dir, exist_ok=True)

    gauge = pd.read_csv(self.fgauge)
    donet1  = gauge[gauge['Instruments']=='DONET1']
    donet2  = gauge[gauge['Instruments']=='DONET2']
    nowphas = gauge[gauge['Instruments']=='NOWPHAS']

    if 1:
      gid = list(gauge['ID'])
      dic_d1 = {  "x"    :donet1["Longitude"],
                  "y"    :donet1["Latitude"],
                  "c"    :"green",
                  "label":'DONET1',
                }
      dic_d2 = {  "x"    :donet2["Longitude"],
                  "y"    :donet2["Latitude"],
                  "c"    :"navy",
                  "label":'DONET2',
                }
      dic_ns = {  "x"    :nowphas["Longitude"],
                  "y"    :nowphas["Latitude"],
                  "c"    :"crimson",
                  "label":'NOWPHAS',
                }

      fig = plt.figure(figsize=figs)
      plt.rcParams['font.size'] = fonts
      fig, ax = SF.topograph_projection(fig=fig, fonts=fonts) 

      ax.scatter(**dic_d1, marker='o', s=10, ec='k', lw=0.3, zorder=10)
      ax.scatter(**dic_d2, marker='o', s=10, ec='k', lw=0.3, zorder=10)
      ax.scatter(**dic_ns, marker='o', s=10, ec='k', lw=0.3, zorder=10)
      for i in range(len(gid)):
        ax.text(  x=gauge['Longitude'][i], y=gauge['Latitude'][i],
                  s='ID:{}'.format(gid[i]), size=fonts/2,
                  zorder=15
                )

      ax.set_xlabel("Longitude")
      ax.set_ylabel("Latitude")
      ax.set_title("Configuration of used gauges")

      plt.savefig(  '%s/gauges.png' % self.res_dir, 
                    dpi=300, bbox_inches='tight', pad_inches=0.05
                  )
      plt.clf()
      plt.close()
      print('  - Make figure for gauge arrangement (PNG)')

    if 0:
      import simplekml
      f = '%s/gauges.kml' % self.res_dir
      kml = simplekml.Kml()
      dic_d1 = {  "x" :donet1["Longitude"],
                  "y" :donet1["Latitude"],
                  "id":donet1["ID"],
                  "c" :'ff008000',
                }
      dic_d2 = {  "x" :donet2["Longitude"],
                  "y" :donet2["Latitude"],
                  "id":donet2["ID"],
                  "c" :'ffff0000',
                }
      dic_ns = {  "x" :nowphas["Longitude"],
                  "y" :nowphas["Latitude"],
                  "id":nowphas["ID"],
                  "c" :'ff3c14dc',
                }

      for dic in [dic_d1, dic_d2, dic_ns]:
        for i in range(len(dic["x"])):
          pnt = kml.newpoint( name=dic["id"].iloc[i],
                              coords=[( dic['x'].iloc[i], 
                                        dic['y'].iloc[i])]
                            )
          pnt.style.iconstyle.color = dic["c"]
          pnt.style.iconstyle.scale = 1
          pnt.style.labelstyle.scale = 1

      kml.save(f)
      print('  - Make figure for gauge arrangement (KML)')


  def TTSplit(self, cv, seed=7): 
    """Splits all scenarios into training and testing scenarios"""
    print('  - Split data into training and test set')

    Xmat = np.load(self.fwave)
    cases = pd.read_csv(self.fcase)
    cases['label'] = 0

    random.seed(seed)
    lst_all = [i for i in range(self.nsce)]
    lst_all = random.sample(lst_all, len(lst_all))
    lst_test_all = np.array_split(lst_all, self.ncv_set)
    
    self.ntst_cv_set = [i.shape[0] for i in lst_test_all]
    self.ntrn_cv_set = [self.nsce-i for i in self.ntst_cv_set]

    os.makedirs(self.res_cv_dir.format(cv), exist_ok=True)
    lst_test = sorted(list(np.array_split(lst_all, self.ncv_set)[cv]))
    lst_larn = sorted(list(set(lst_all) - set(lst_test)))

    ntst = len(lst_test)
    ntrn = len(lst_larn)

    Xtrain = np.zeros((ntrn, Xmat.shape[1], Xmat.shape[2]))
    Xtest  = np.zeros((ntst, Xmat.shape[1], Xmat.shape[2]))
    for itrn in range(ntrn):
      Xtrain[itrn, :, :] = Xmat[lst_larn[itrn], :, :]
    for itst in range(ntst):
      Xtest[itst, :, :] = Xmat[lst_test[itst], :, :]

    cases.loc[lst_test, 'label'] = 'test'
    cases.loc[lst_larn, 'label'] = 'larn'

    np.save(self.fXtrn.format(cv), Xtrain)
    np.save(self.fXtst.format(cv), Xtest)
    cases.to_csv(self.fttlst.format(cv), index=None)

    print('    Shape of training data (training scenarios, gauges, time steps):')
    print('      {}'.format(Xtrain.shape))
    print('    Shape of test data (test scenarios, gauges, time steps):')
    print('      {}'.format(Xtest.shape))
    del Xmat


  def SVD(self, GPU, cv):
    """Singular Value Decomposition (SVD)"""
    print('  - sigular value decomposition for reduced-order modeling')

    os.makedirs(self.res_cv_dir.format(cv), exist_ok=True)

    if GPU:
      X = cp.load(self.fXtrn.format(cv))
      ntrn = self.ntrn_cv_set[cv]

      dm = cp.zeros((self.ngag, self.ntim*ntrn))
      for i in range(ntrn):
        dm[:,i*self.ntim:(i+1)*self.ntim] = X[i,:,:]

      U, S, Vh = cp.linalg.svd(dm, full_matrices=False)
      U = -U

      cp.save(self.fu.format(cv), U)
      cp.save(self.fs.format(cv), S)

    else:
      X = np.load(self.fXtrn.format(cv))
      ntrn = self.ntrn_cv_set[cv]

      dm = np.zeros((self.ngag, self.ntim*ntrn))
      for i in range(ntrn):
        dm[:,i*self.ntim:(i+1)*self.ntim] = X[i,:,:]

      dm = da.from_array(dm.T)
      dm = dm.rechunk({0:'auto', 1:-1})
      U, S, Vh = da.linalg.tsqr(dm, compute_svd=True)
      del X
      U, S, Vh = da.compute(U, S, Vh)
      [U, Vh] = [Vh.T, U.T]

      np.save(self.fu.format(cv), U)
      np.save(self.fs.format(cv), S)
    
    print('    Shape of POD mode matrix (modes, modes):')
    print('      {}'.format(U.shape))
    print('    Shape of singular value vector (modes):')
    print('      {}'.format(S.shape))


  def FigSingular(self, cv, thre=[0.8, 0.9, 0.95, 0.99],
                  figs=(6, 3), fonts=8
                  ):
    """Makes a figure showing contributions of each POD mode"""
    print('  - Make figure for contribution rates of singular values')
    sns.set_style('darkgrid')

    d = '%s/figures' % self.res_cv_dir.format(cv) 
    os.makedirs(d, exist_ok=True)

    sv = np.load(self.fs.format(cv))

    s_rate = sv**2 / np.sum(sv**2)
    s_cumsum = np.cumsum(s_rate)
    thmodes = ''
    for ith in thre:
      thmode = str(np.where(s_cumsum>ith)[0][0]+1)
      thmodes += '{} modes are required for {}[%]\n'.format(thmode, int(ith*100))

    fig = plt.figure(figsize=figs)
    plt.rcParams['font.size'] = fonts
    ax = fig.add_subplot(111)

    ax.scatter( np.arange(s_cumsum.size)+1, s_cumsum, 
                marker='o', fc='none', s=10, ec='crimson'
              )
    ax.text(  1.,0., thmodes, 
              ha='right', va='bottom', transform=ax.transAxes
            )

    ax.set_title("Relationship between the number of used modes and cumlative contribution ratio")
    ax.set_xlabel('Number of modes')
    ax.set_ylabel('Cumulative contribution ratio')
    plt.savefig('%s/singular_value.png' % d, 
                dpi=300, bbox_inches='tight', pad_inches=0.05
                )
    plt.clf()
    plt.close()


  def FigReconError(  self, GPU, cv, 
                      figs=(6, 3), fonts=8
                    ):
    """Makes a figure of relationship between the reconstruction errors (RMSE [m]) and the number of spatial modes"""
    print('  - Make figure for reconstruction errors vs used POD modes')

    sns.set_style('darkgrid')
    
    d = '%s/figures' % self.res_cv_dir.format(cv) 
    os.makedirs(d, exist_ok=True)

    if GPU:
      X   = cp.load(self.fXtrn.format(cv))
      Phi = cp.load(self.fu.format(cv))
      sv  = cp.load(self.fs.format(cv))
      ntrn = self.ntrn_cv_set[cv]

      dm = cp.zeros((self.ngag, self.ntim*ntrn))
      for i in range(ntrn):
        dm[:,i*self.ntim:(i+1)*self.ntim] = X[i,:,:]

      vh = cp.linalg.inv(cp.diag(sv)) @ Phi.T @ dm

      error = []
      for imode in range(1, self.ngag+1, 1):
        Xrecon = Phi[:,:imode] @  cp.diag(sv[:imode]) @ vh[:imode,:]
        egpu = cp.sqrt( cp.average((dm - Xrecon)**2) )
        error.append( egpu.get() )

    else:
      X = np.load(self.fXtrn.format(cv))
      Phi = np.load(self.fu.format(cv))
      sv = np.load(self.fs.format(cv))
      ntrn = self.ntrn_cv_set[cv]

      dm = np.zeros((self.ngag, self.ntim*ntrn))
      for i in range(ntrn):
        dm[:,i*self.ntim:(i+1)*self.ntim] = X[i,:,:]

      vh = np.linalg.inv(np.diag(sv)) @ Phi.T @ dm

      error = []
      for imode in range(1, self.ngag+1, 1):
        Xrecon = Phi[:,:imode] @  np.diag(sv[:imode]) @ vh[:imode,:]
        error.append( np.sqrt(np.average((dm - Xrecon)**2)) )

    fig = plt.figure(figsize=figs)
    plt.rcParams['font.size'] = fonts
    ax = fig.add_subplot(111)
    ax.scatter( np.arange(len(error))+1, error, 
                marker='o', fc='none', s=10, ec='crimson'
              )

    ax.set_title("Relationship between the number of used modes and POD reconstruction error")
    ax.set_xlabel('Number of modes')
    ax.set_ylabel('Reconstruction error (RMSE [m])')

    plt.savefig('%s/reconstruction_error.png' % d, 
                  dpi=300, bbox_inches='tight', pad_inches=0.05
                )
    plt.clf()
    plt.close()


  def YamlOut(self):
    os.makedirs(self.res_dir, exist_ok=True)
    info = vars(self)

    with open('%s/INFO.yaml' % self.res_dir, mode="wt", encoding="utf-8") as f:
      yaml.dump(info, f, indent=2)

if __name__ == '__main__':
  GPU = True
  cv = 1

  dirs = {  'inp_dir':'data',
            'res_dir':'result/paper_Bayes_scenario_superpose'
          }
  fils = {  'fwave'  :'wave_seq.npy',
            'fgauge' :'gauge_loc.csv',
            'fcase'  :'case_ID.csv',
            'ftrain' :'Xtrain.npy',
            'ftest'  :'Xtest.npy',
            'fttlist':'ttlist.csv',
            'fu'     :'U.npy',
            'fs'     :'s.npy',
          }
  nums = {  'ncv_set' :4,
            'nsce'    :2342,
            'ntim'    :2160,
            'ngag'    :62
          }

  print(datetime.datetime.now())
  print('###  Preprocessing  ###')

  pre = PreProcess(**dirs, **fils, **nums) 

  pre.FigGauges()
  pre.TTSplit(cv=cv)
  pre.SVD(GPU=GPU, cv=cv)
  pre.FigSingular(cv=cv) 
  pre.FigReconError(GPU=GPU, cv=cv) 
  pre.YamlOut() 
  print(datetime.datetime.now())