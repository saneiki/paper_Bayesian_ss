import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
import numpy as np
import cupy as cp

import copy
import yaml
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as mps
from scipy.stats import multivariate_normal, norm
# import matplotlib        # Optional
# matplotlib.use("TkAgg")  # Optional

class Forecasting():
  """Module for tsunami forecasting with two methods
  - Previous method (Nomura et al. 2022) can be accessed at https://doi.org/10.1029/2021JC018324.
  - Proposed method (###) can be accessed at ###.

  Methods
  ---------
  InfoInit(cv, itest, **kwargs)
    Print the initial information.
  InfoProc(cv, itest, itime, **kwargs)
    Print estimated weight parameters at a representative time step. 
  Loaddata(**kwargs)
    Load data.
  AddNoise(X, loc, scale)
    Add independent Gaussian noise with mean `loc` and std `scale` to the matrix X.
  MakeTrain(cv)
    Generate a matrix gathering POD coefficients of training wave data.
  MakeROMX(cv)
    Generate a reduced-order matrix of an original data matrix
  MakeTest(KALMAN, cv, Xobs, avec, Pmat, Qmat, Rmat, **kwargs)
    Generate a matrix gathering POD coefficients of test wave data, using least square estimaion or Kalman filter.
  InitResult()
    Initialize `self.result'
  InitWeight(cv)
    Initialize weight parameter set.
  BayesianUpdate(GPU, cv, train_now, test_now, cov)
    Conducts Bayesian update/estimation for the weight parameters.
  Output(cv, itest, obs_window, cov_out, **kwargs)
    Output resultant weight parameter set.
  WaveformFigure(GPU, cv, itest, obs_window, gIDlist, figs=(6,3), fonts=8)
    Make a waveform figure for one single test scenario.
  TaylorFigure(cv, itest, obs_window, gIDlist, figs=(4,4), fonts=8)
    Make a Taylor diagram for one single test scenario.
  PrdSigma_N(Xt, Xm, w)
    Calculate a range of prediction reliability for the previous method.
  PrdSigma_F(GPU, cv, cov, idx)
    Calculate a range of prediction reliability for the proposed method.
  Run_wave_based(GPU, cv, itest, obs_window, cov_out)
    Run a tsunami prediction using observed wave data directly.
  Run_state_based(GPU, cv, itest, obs_window, cov_out, KALMAN)
    Run a tsunami prediction using state variables instead of observed waves.
  """

  def __init__( self, ver, ROM, nmod,
                res_cv_dir, fgauge, fXtrn, fXtst, fttlst, fu, fs,
                ntst_cv_set, ntrn_cv_set, ntim, ngag,
                **kwargs
              ):
    """Parameter initialization"""

    self.ver = ver
    self.ROM = ROM
    self.nmod = nmod
    if ROM!=True and nmod<62:
      self.nmod = 62
      print('')
      print('======================================================================================')
      print('Warning: `ROM` was set to `False`, but `nmod` was set to `{}` not `62`'.format(nmod))
      print('Warning: Forced `nmod` to be changed to `62` in order to run the code with `ROM=False`')
      print('======================================================================================')

    self.res_cv_dir = res_cv_dir

    self.fgauge = fgauge
    self.fXtrn = fXtrn
    self.fXtst = fXtst
    self.fttlst = fttlst
    self.fu = fu
    self.fs = fs

    self.ntim = ntim
    self.ngag = ngag
    self.ntst_cv_set = ntst_cv_set
    self.ntrn_cv_set = ntrn_cv_set

    self.weights = [np.array([0]), np.array([0])]
    self.result = []


  def InfoInit(self, cv, itest, **kwargs):
    """Print the initial information.

    Parameters
    -----------
    cv: int
      ID of a cross validation set to be used
    itest: int
      ID of a targeting test scenario

    Returns
    --------
    start: datetime.datetime
      Calculation start time
    """

    try:
      ttlst = kwargs["ttlst"]
    except:
      ttlst = pd.read_csv(self.fttlst.format(cv))

    test_case = ttlst.loc[ttlst['label']=='test', 'ID']
    start = datetime.datetime.now()
    print('\n{}'.format(start))
    print('Calculation starting with {}-version'.format(self.ver))
    if self.ROM:
      print('With ROM via SVD using top {} modes'.format(self.nmod))
    else:
      print('Without ROM via SVD')
    print('Forecasting for scenario No.{} ("{}")'
          .format(itest, test_case.iloc[itest]))
    return start


  def InfoProc(self, cv, itest, itime, **kwargs):
    """Print estimated weight parameters at a representative time step.

    Parameters
    -----------
    cv: int
      ID of a cross validation set to be used
    itest: int
      ID of a targeting test scenario
    itime: int
      Time step to output the weight values
    """

    try:
      ttlst = kwargs["ttlst"]
    except:
      ttlst = pd.read_csv(self.fttlst.format(cv))
    train_case = ttlst.loc[ttlst['label']=='larn', 'ID']
    test_case  = ttlst.loc[ttlst['label']=='test', 'ID']

    print(  '  - Prediction Step: {}={}[min] for {}'
            .format(itime, itime*5./60., test_case.iloc[itest])
          )
    print(  '    Probable scenarios: {}'
            .format(np.array(train_case.iloc[
              np.argsort(self.weights[0])[::-1][:3]
            ]))
          )
    print(  '    weights : {}'
            .format(np.sort(self.weights[0])[::-1][:3])
          )


  def Loaddata(self, **kwargs):
    """Load specific data

    Parameters
    -----------
    obj: str 
      Variable name to read ('Xtest', `Xtrain`, 'Phi', or 'sv')
    cv: int
      Required. ID of a cross validation set to be used
    itest: int
      Required when `obj='Xtest'`. ID of a targeting test scenario

    Return
    -------
    d: numpy.ndarray
      Loaded data.
    """
    try:
      if kwargs['obj']=='Xtrain':
        d = np.load(self.fXtrn.format(kwargs['cv']))
      elif kwargs['obj']=='Xtest':
        d = np.load(self.fXtst.format(kwargs['cv']))[kwargs['itest'], :, :]
      elif kwargs['obj']=='Phi':
        d = np.load(self.fu.format(kwargs['cv']))
      elif kwargs['obj']=='sv':
        d = np.load(self.fs.format(kwargs['cv']))
    except:
      d = False
      print('def Loaddata cannot be executed')
      print('Plsease check `kwargs` contents')
    return d


  def AddNoise(self, X, loc, scale):
    """Add independent Gaussian noise with mean `loc` and std `scale` to the matrix X.

    Parameters
    -----------
    X: numpy.ndarray
      Target matrix given a Gaussian noise
    loc: float
      Mean of the distribution
    scale: float
      Standard deviation of the distribution

    Return
    -------
    X: numpy.ndarray
      Noise-added matrix
    """

    rng = np.random.default_rng(12345)
    X += rng.normal(loc=loc, scale=scale, size=X.shape)
    return X


  def MakeTrain(self, cv):
    """Generate a matrix gathering POD coefficients of training wave data.

    Parameters
    -----------
    cv: int
      ID of a cross validation set to be used

    Return
    -------
    train: numpy.ndarray
      Training coefficient matrix with the shape as (Number of training scenarios, Number of used modes, Number of time steps).
    """

    Xtrain = self.Loaddata(**dict(obj='Xtrain', cv=cv))
    Phi    = self.Loaddata(**dict(obj='Phi', cv=cv))
    ntrn = self.ntrn_cv_set[cv]
    
    train = np.zeros((ntrn, self.nmod, self.ntim))
    for i in range(ntrn):
      mat = np.linalg.pinv(Phi[:,:self.nmod]) @ Xtrain[i, :, :]
      train[i,:,:] = mat
    return train
  
  
  def MakeROMX(self, cv):
    """Generate a reduced-order matrix of an original data matrix

    Parameters
    -----------
    cv: int
      ID of a cross validation set to be used

    Return
    -------
    train: numpy.ndarray
      a reduced-order matrix of an original data matrix
    """
    Xtrain = self.Loaddata(**dict(obj='Xtrain', cv=cv))
    Phi    = self.Loaddata(**dict(obj='Phi', cv=cv))
    ntrn = self.ntrn_cv_set[cv]

    train = np.zeros_like(Xtrain)
    for i in range(ntrn):
      a_mat = np.linalg.inv(Phi) @ Xtrain[i, :, :]
      rom_mat = Phi[:,:self.nmod] @ a_mat[:self.nmod,:]
      train[i,:,:] = rom_mat
    return train


  def MakeTest(self, KALMAN, cv, xobs, avec, Pmat, Qmat, Rmat, **kwargs):
    """Generate a matrix gathering POD coefficients of test wave data, using least square estimaion or Kalman filter.

    Parameters
    -----------
    KALMAN: bool
      Flag to determine whether inverse estimation with Kalman filter or with least square estimation
    cv: int
      ID of a cross validation set to be used
    xobs: numpy.ndarray
      Observed wave heights at a time step, with shape as (Number of gauges)
    avec: numpy.ndarray
      POD coefficient vector at a time step, with shape as (Number of gauges)
    Pmat: numpy.ndarray
      Covariance matrix for Kalman filter estimation
    Qmat: numpy.ndarray
      Covariance matrix for system noise
    Rmat: numpy.ndarray
      Covariance matrix for observation noise

    Return
    -------
    avec: numpy.ndarray
      Mean of estimated POD coefficient vector
    Pmat: numpy.ndarray
      Covariance of Kalman filter estimation
    """

    try:
      Phi = kwargs["Phi"]
    except:
      Phi = self.Loaddata(**dict(obj='Phi', cv=cv))
    Phi = Phi[:,:self.nmod]
    if KALMAN:
      abar = avec
      Pbar = Pmat + Qmat
      Gmat = Pbar @ Phi.T @ np.linalg.inv(Phi@Pbar@Phi.T + Rmat)
      avec = abar +Gmat @ (xobs - Phi@abar)
      Pmat = (np.eye(self.nmod) - Gmat@Phi) @ Pbar
    else:
      avec = np.linalg.pinv(Phi) @ xobs
      Pmat = Pmat
    return avec, Pmat


  def InitResult(self):
    """Initialize `self.result'"""
    self.result = []


  def InitWeight(self, cv):
    """Initialize weight parameter set.
    The weight parameters are treated as discrete variables in the previous method, while as random variables following a PDF in the proposed method.

    Parameters
    -----------
    cv: int
      ID of a cross validation set to be used
    """

    ntrn = self.ntrn_cv_set[cv]
    if self.ver == 'Nomura':
      self.weights = [np.ones(ntrn) / ntrn, np.array([0])]
    elif self.ver == 'Fujita':
      mean_w = np.zeros(ntrn)
      cov_w  = 0.1**2*np.eye(ntrn) 
      self.weights = [mean_w, cov_w]


  def BayesianUpdate(self, GPU, cv, train_now, test_now, cov):
    """Conducts Bayesian update/estimation for the weight parameters.

    Parameters
    -----------
    GPU: bool
      Flag to determine whether the calculation is with or without GPU
    cv: int
      ID of a cross validation set to be used
    train_now: numpy.ndarray
      Vector of training data at a time step, with shape as (Number of scenarios, Number of modes)
    test_now: numpy.ndarray
      Vector of test data at a time step, with shape as (Number of modes)
    cov: numpy.ndarray
      Covariance matrix
    """

    ntrn = self.ntrn_cv_set[cv]
    if self.ver == 'Nomura':
      wei, _ = self.weights
      norm = multivariate_normal( mean=test_now, 
                                  cov=cov
                                )
      for isce in range(ntrn):
        wei[isce] *= norm.pdf( train_now[isce,:] )

      wei += 1.e-50
      wei /= np.sum(wei)
      self.weights = [wei, np.array([0])]

    elif self.ver == 'Fujita':
      train_now = train_now.T
      mean_w, cov_w = self.weights

      if GPU:
        train_now = cp.array(train_now)
        mean_w = cp.array(mean_w)
        cov_w = cp.array(cov_w)
        cov = cp.array(cov)
        test_now = cp.array(test_now)

        G = cov_w @ train_now.T @ cp.linalg.inv(train_now @ cov_w @ train_now.T + cov)
        mean_post = mean_w + G @ (test_now - train_now @ mean_w)
        cov_post = (cp.eye(G.shape[0]) - G @ train_now) @ cov_w

        self.weights = [mean_post.get(), cov_post.get()]

      else:
        G = cov_w @ train_now.T @ np.linalg.inv(train_now @ cov_w @ train_now.T + cov)
        mean_post = mean_w + G @ (test_now - train_now @ mean_w)
        cov_post = (np.eye(G.shape[0]) - G @ train_now) @ cov_w

        self.weights = [mean_post, cov_post]


  def Output(self, cv, itest, obs_window, cov_out, **kwargs):
    """Output resultant weight parameter set.
    The target directory is `/{res_dir}/CrsVld_{cv}/forecast/{nmod}modes/`

    Parameters
    -----------
    cv: int
      ID of a cross validation set to be used
    itest: int
      ID of a targeting test scenario
    obs_window: list(int)
      List of observation widows to output the forecast results.
    cov_out: bool
      Flag to determine whether to output the resultant covariance or not.
    """

    try:
      ttlst = kwargs["ttlst"]
    except:
      ttlst = pd.read_csv(self.fttlst.format(cv))
    train_case = ttlst.loc[ttlst['label']=='larn', 'ID']
    test_case  = ttlst.loc[ttlst['label']=='test', 'ID']

    if self.ROM:
      d = '%s/forecast/%smodes/%s_%s' % ( self.res_cv_dir.format(cv), 
                                          str(self.nmod).zfill(3), str(itest).zfill(3), 
                                          test_case.iloc[itest])
    else:
      d = '%s/forecast/%smodes_withoutROM/%s_%s' % (self.res_cv_dir.format(cv), 
                                                    str(self.nmod).zfill(3), str(itest).zfill(3), 
                                                    test_case.iloc[itest])
    os.makedirs(d, exist_ok=True)

    df = pd.DataFrame(  columns=['Obs. window {} step'.format(i) for i in obs_window], 
                        index=train_case
                      )
    for i in range(len(obs_window)):
      df.iloc[:,i] = self.result[i][0]
    df.to_csv(os.path.join(d, 'wei_{}.csv'.format(self.ver)) )
    
    if self.ver=='Fujita' and cov_out:
      for i in range(len(obs_window)):
        np.save('%s/cov_%swindow.npy' % (d, str(obs_window[i]).zfill(3)), 
                self.result[i][1])


  def WaveformFigure( self, GPU, cv, itest, obs_window, gIDlist,
                      figs=(6,6), fonts=8
                    ):
    """Make a waveform figure for one single test scenario.
    The target directory is `/{res_dir}/CrsVld_{cv}/figures/{nmod}modes/`

    Parameters
    -----------
    GPU: bool
      Flag to determine whether the calculation is with or without GPU
    cv: int
      ID of a cross validation set to be used
    itest: int
      ID of a targeting test scenario
    obs_window: list(int)
      List of observation widows to output the waveform figure
    gIDlist: list(int)
      List of gauge ID
    """

    print('Make a waveform figure for No.{}'.format(itest))
    ttlst = pd.read_csv(self.fttlst.format(cv))
    test_case = ttlst.loc[ttlst['label']=='test', 'ID']
    gauge = pd.read_csv(self.fgauge)
    Xtrn = self.Loaddata(**dict(obj='Xtrain', cv=cv))
    Xtst = self.Loaddata(**dict(obj='Xtest', cv=cv, itest=itest))

    if self.ROM:
      d = '%s/figures/%smodes/%s_%s' % ( self.res_cv_dir.format(cv), 
                                          str(self.nmod).zfill(3), str(itest).zfill(3), 
                                          test_case.iloc[itest])
    else:
      d = '%s/figures/%smodes_withoutROM/%s_%s' % (self.res_cv_dir.format(cv), 
                                                    str(self.nmod).zfill(3), str(itest).zfill(3), 
                                                    test_case.iloc[itest])
    os.makedirs(d, exist_ok=True)

    idx = np.where(gauge['ID'].isin(gIDlist))[0]
    Xtrn = Xtrn[:, idx, :]
    Xtst = Xtst[idx, :]

    for res,obs in zip(self.result, obs_window):
      if self.ver == 'Nomura':
        most = np.argmax(res[0])
        Xprd = Xtrn[most,:,:]
        sigma = None
      elif self.ver=='Fujita': 
        Xprd = np.sum(res[0]) * np.average(Xtrn, axis=0, weights=res[0])
        sigma = self.PrdSigma_F(GPU=GPU, cov=res[1], idx=idx, cv=cv)

      sns.set_style('darkgrid')
      fig = plt.figure(figsize=figs)
      plt.rcParams['font.size'] = fonts
      fig.suptitle( "For {}, ".format(test_case.iloc[itest]) +\
                    "{:.1f}[min], ".format(obs*5./60.) +\
                    "{}-method".format(self.ver)
                  )

      for i in range(len(gIDlist)):
        ax = fig.add_subplot(len(gIDlist),1,i+1)
        ax.axvline(x=obs*5/3600, lw=0.5)
        ax.plot(  np.arange(self.ntim)*5/3600, Xtst[i,:], 
                  c='k', lw=1.0
                )
        ax.plot(  np.arange(self.ntim)*5/3600, Xprd[i,:], 
                  c='crimson', lw=1.0, alpha=0.7, 
                  label='mean'
                )

        if self.ver=='Fujita':
          ax.fill_between(  np.arange(self.ntim)*5/3600, 
                            Xprd[i,:]-sigma[i,:], Xprd[i,:]+sigma[i,:], 
                            color='gray', alpha=0.5, 
                            label=r'$\pm 1 \sigma$'
                          )

        ax.set_title('Gauge ID: {}'.format(gIDlist[i]))
      ax.legend(ncol=3)
      ax.set_xlabel('Time [hours]')
      ax.set_ylabel('Elevation [m]')
      plt.tight_layout()
      plt.savefig('%s/wave_%s_%sstep.png' % (d, self.ver, str(obs).zfill(4)),
                  dpi=100, bbox_inches='tight', pad_inches=0.05)
      plt.clf()
      plt.close()


  def TaylorFigure( self, cv, itest, obs_window, gIDlist, 
                    figs=(4, 4), fonts=8
                  ):
    """Make a Taylor diagram for one single test scenario.
    The target directory is `/{res_dir}/CrsVld_{cv}/figures/{nmod}modes/`

    Parameters
    -----------
    cv: int
      ID of a cross validation set to be used
    itest: int
      ID of a targeting test scenario
    obs_window: list(int)
      List of observation widows to output the waveform figure
    gIDlist: list(int)
      List of gauge ID
    """

    print('Make a Taylor diagram for No.{}'.format(itest))
    ttlst = pd.read_csv(self.fttlst.format(cv))
    test_case = ttlst.loc[ttlst['label']=='test', 'ID']
    gauge = pd.read_csv(self.fgauge)
    Xtrn = self.Loaddata(**dict(obj='Xtrain', cv=cv))
    Xtst = self.Loaddata(**dict(obj='Xtest', cv=cv, itest=itest))

    if self.ROM:
      d = '%s/figures/%smodes/%s_%s' % ( self.res_cv_dir.format(cv), 
                                          str(self.nmod).zfill(3), str(itest).zfill(3), 
                                          test_case.iloc[itest])
    else:
      d = '%s/figures/%smodes_withoutROM/%s_%s' % (self.res_cv_dir.format(cv), 
                                                    str(self.nmod).zfill(3), str(itest).zfill(3), 
                                                    test_case.iloc[itest])
    os.makedirs(d, exist_ok=True)

    idx = np.where(gauge['ID'].isin(gIDlist))[0]
    Xtrn = Xtrn[:, idx, :]
    Xtst = Xtst[idx, :] 

    sns.set_style('darkgrid')
    fig = plt.figure(figsize=figs)
    plt.rcParams['font.size'] = fonts
    ax = fig.add_subplot(111, projection='polar')

    ax.scatter(0, 1, c='k',s=20, label="True")
    marker=[",", "o", "^"]
    color=["crimson", "navy", "green"]
    for j,(res, obs) in enumerate(zip(self.result, obs_window)):
      if self.ver == 'Nomura':
        most = np.argmax(res[0])
        Xprd = Xtrn[most,:,:]
      elif self.ver=='Fujita': 
        Xprd = np.sum(res[0]) * np.average(Xtrn, axis=0, weights=res[0])

      sig_tst = np.std(Xtst, axis=1)
      sig_prd = np.std(Xprd, axis=1)
      R = np.sum((Xprd - np.average(Xprd, axis=1).reshape(-1,1)) \
                 *(Xtst - np.average(Xtst, axis=1).reshape(-1,1)), axis=1
                ) / self.ntim / sig_tst / sig_prd

      ax.scatter( np.arccos(R), sig_prd/sig_tst, 
                  # c=color[j], s=20, marker=marker[j], 
                  ec='k', lw=0.5, 
                  label="obsercation duration {:.1f} [min]".format(obs*5/60)
                )

      for i in range(len(gIDlist)):
        ax.text(  np.arccos(R[i]), sig_prd[i]/sig_tst[i], 
                  "ID: {}".format(gIDlist[i])
                )

    ax.legend(loc=1)
    ax.set_ylabel('Standard Deviation')
    # ax.text(0.7,2.1,'Correlation Coefficient', rotation=-45, ha='center', va='bottom')
    ax.set_xlim(0,np.pi/2.)
    # ax.set_ylim(0,2.)
    theta_ticklabels = [1,0.99,0.95, 0.9,0.8,0.6,0.4,0.2,0]
    theta_ticks = np.arccos(theta_ticklabels)
    ax.set_thetagrids(np.degrees(theta_ticks), labels=theta_ticklabels)

    plt.savefig('%s/taylor_%s.png' % (d, self.ver),
                dpi=100, bbox_inches='tight', pad_inches=0.05)
    plt.clf()
    plt.close()


  def PrdSigma_N(self, Xt, Xm, w):
    """Calculate a range of prediction reliability for the previous method.

    Parameters
    -----------
    Xt: numpy.ndarray
      Matrix of training wave data, with shape as (Number of scenarios, Number of gauges, Number of time steps)
    Xm: numpy.ndarray
      Matrix of predicted wave data, with shape as (Number of gauges, Number of time steps)
    w: numpy.ndarray
      Vector of estimated weight parameter set

    Return
    -------
    Sig: numpy.ndarray
      Standard deviation of prediction results
    """

    Xsqm = np.sum(w) * np.average(Xt**2, axis=0, weights=w)
    Var = Xsqm - Xm**2
    Var = np.where(Var<0, 0, Var)
    Sig = np.sqrt(Var)
    return Sig


  def PrdSigma_F(self, GPU, cv, cov, idx):
    """Calculate a range of prediction reliability for the proposed method.

    Parameters
    -----------
    GPU: bool
      Flag to determine whether the calculation is with or without GPU
    cv: int
      ID of a cross validation set to be used
    cov: numpy.ndarray
      Covariance of weight parameter estimation
    idx: int
      Gauge ID

    Return
    -------
    Sig: nummpy.ndarray
      Standard deviation of prediction results
    """

    if GPU:
      cov = cp.array(cov)
      Phi = cp.array(self.Loaddata(**dict(obj='Phi', cv=cv)))
      Var = cp.zeros((len(idx), self.ntim))
      train = self.MakeTrain(cv=cv)
      train = cp.array(train)
      for i in range(self.ntim):
        cov_wave = (Phi[idx,:self.nmod] @  train[:,:,i].T) @ cov \
          @ (Phi[idx,:self.nmod] @  train[:,:,i].T).T #\
        Var[:,i] = cp.diag(cov_wave)
      Sig = cp.sqrt(Var)
      Sig = Sig.get()

    else:
      Phi = self.Loaddata(**dict(obj='Phi', cv=cv))
      Var = np.zeros((len(idx), self.ntim))
      train = self.MakeTrain(cv=cv)
      for i in range(self.ntim):
        cov_wave = (Phi[idx,:self.nmod] @  train[:,:,i].T) @ cov \
          @ (Phi[idx,:self.nmod] @  train[:,:,i].T).T #\
        Var[:,i] = np.diag(cov_wave)
      Sig = np.sqrt(Var)
    return Sig

  
  def WeightsPDFFigure( self, cv, itest, obs_window, 
                        figs=(5.,5.), fonts=8, widths=[4,1], heights=[1,4]
                      ):
    """Make a probability density distribution for one single test scenario.
    The target directory is `/{res_dir}/CrsVld_{cv}/figures/{nmod}modes/`

    Parameters
    -----------
    cv: int
      ID of a cross validation set to be used
    itest: int
      ID of a targeting test scenario
    obs_window: list(int)
      List of observation widows to output the waveform figure
    """
    
    if self.ver == 'Nomura':
      print("Skip `WeightsPDFFigure`")
      return

    print('Make a figure of PDF for No.{}'.format(itest))
    ttlst = pd.read_csv(self.fttlst.format(cv))
    larn_case = ttlst.loc[ttlst['label']=='larn', 'ID']
    test_case = ttlst.loc[ttlst['label']=='test', 'ID']
    
    if self.ROM:
      d = '%s/figures/%smodes/%s_%s' % ( self.res_cv_dir.format(cv), 
                                          str(self.nmod).zfill(3), str(itest).zfill(3), 
                                          test_case.iloc[itest])
    else:
      d = '%s/figures/%smodes_withoutROM/%s_%s' % (self.res_cv_dir.format(cv), 
                                                    str(self.nmod).zfill(3), str(itest).zfill(3), 
                                                    test_case.iloc[itest])
    os.makedirs(d, exist_ok=True)
    
    for res,obs in zip(self.result, obs_window):
      ID = [np.argmax(res[0]), np.argsort(res[0])[::-1][1]]
      sce_number = [list(larn_case.index)[ID[0]]+1, list(larn_case.index)[ID[1]]+1]
  
      mu_vec = np.array(res[0][ID])
      cov_mat = res[1][ID,:][:,ID]
      var = np.diag(cov_mat)
      std = np.sqrt(var)

      x = np.linspace(-1., 1., 1000)
      y = np.linspace(-1., 1., 1000)
      X, Y = np.meshgrid(x, y)
      XY = np.stack([X.flatten(), Y.flatten()], axis=1)
      joint_dens = multivariate_normal.pdf(
        x=XY, mean=mu_vec, cov=cov_mat
      )
      conditional_dens_a = norm.pdf(x=x, loc=mu_vec[0], scale=std[0])
      conditional_dens_b = norm.pdf(x=x, loc=mu_vec[1], scale=std[1])

      mps.use('default')
      plt.rcParams['font.size'] = fonts
      fig = plt.figure(figsize=figs)
      axs = {}
  
      spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), 
                              width_ratios=widths, height_ratios=heights
                              )
  
      axs[2] = fig.add_subplot(spec[2//len(widths), 2%len(widths)])
      axs[0] = fig.add_subplot(spec[0//len(widths), 0%len(widths)], sharex=axs[2])
      axs[3] = fig.add_subplot(spec[3//len(widths), 3%len(widths)], sharey=axs[2])
      plt.setp(axs[0].get_xticklabels(), visible=False)
      plt.setp(axs[3].get_yticklabels(), visible=False)
      plt.subplots_adjust(hspace=0.1, wspace=0.1)
  
      axs[0].grid(lw=0.8, ls='dotted')
      axs[2].grid(lw=0.8, ls='dotted')
      axs[3].grid(lw=0.8, ls='dotted')
  
      axs[0].plot(x, conditional_dens_a, lw=1., c='k')
      axs[0].set_xlim(0.,0.5)
      axs[0].set_ylabel('PD')
      axs[0].set_title('PDF for weight of #{}'.format(sce_number[0]))#, fontsize=fonts)
  
      axs[3].plot(conditional_dens_b, x, lw=1., c='k')
      axs[3].set_ylim(0.,0.5)
      axs[3].set_xlabel('PD')
      axs[3].set_title( 'PDF for weight of #{}'.format(sce_number[1]), 
                        # fontsize=fonts,
                        rotation=270, va='center', ha='left', x=1.1,y=0.5)

      mapp = axs[2].contourf( X, Y, joint_dens.reshape(X.shape), 20, 
                              cmap='coolwarm')
      cs = axs[2].contour(X, Y, joint_dens.reshape(X.shape), 20, 
                          colors='k', linewidths=0.3)
      axs[2].clabel(cs, cs.levels[::2])
      axs[2].set_xlabel('Weight of #{}'.format(sce_number[0]))
      axs[2].set_ylabel('Weight of #{}'.format(sce_number[1]))
      axs[2].set_xlim(0.,0.5)
      axs[2].set_ylim(0.,0.5)
      
      plt.savefig('%s/PDF_%s_%sstep.png' % (d, self.ver, str(obs).zfill(4)),
                  dpi=100, bbox_inches='tight', pad_inches=0.05)
      plt.clf()
      plt.close()


  def Run_wave_based(self, GPU, cv, itest, obs_window, cov_out):
    """Run a tsunami prediction using observed wave data directly.

    Parameters
    -----------
    GPU: bool
      Flag to determine whether the calculation is with or without GPU
    cv: int
      ID of a cross validation set to be used
    itest: int
      ID of a targeting test scenario
    obs_window: list(int)
      List of observation widows to output the waveform figure
    cov_out: bool
      Flag for outputing covariance of the prediction.
    """

    start = self.InfoInit(cv=cv, itest=itest)
    train = self.MakeROMX(cv)
    self.InitResult()
    
    Xtest = self.Loaddata(**dict(obj='Xtest', cv=cv, itest=itest))
    Xtest = self.AddNoise(X=Xtest, loc=0., scale=0.)
    self.InitWeight(cv)
    
    for itime in range(self.ntim):
      alp_train_now = train[:,:,itime]
      self.BayesianUpdate(  GPU=GPU, 
                            train_now=alp_train_now, 
                            test_now=Xtest[:,itime], 
                            cv=cv,
                            cov=np.eye(62)
                          )
      
      if itime%60==0:
        self.InfoProc(itime=itime, itest=itest, cv=cv)
      if itime in obs_window:
        self.result.append(copy.deepcopy(self.weights))
        if itime == obs_window[-1]:
          self.Output(itest=itest, obs_window=obs_window, cov_out=cov_out, cv=cv)
          break
        continue
    del train, Xtest, alp_train_now, self.weights
    print('Calculation ending at {}'.format(datetime.datetime.now()))
  
  
  def Run_state_based(self, GPU, cv, itest, obs_window, cov_out, KALMAN=True):
    """Run a tsunami prediction using state variables instead of observed waves.
    Therefore, an inverse estimation process is needed to obtain the state variables from the wave data.

    Parameters
    -----------
    GPU: bool
      Flag to determine whether the calculation is with or without GPU
    cv: int
      ID of a cross validation set to be used
    itest: int
      ID of a targeting test scenario
    obs_window: list(int)
      List of observation widows to output the waveform figure
    cov_out: bool
      Flag for outputing covariance of the prediction.
    KALMAN: bool
      Flag for executing a Kalman filter for an inverse estimation
    """

    start = self.InfoInit(cv=cv, itest=itest)
    if self.ROM:
      train = self.MakeTrain(cv=cv)
    else:
      train = self.Loaddata(**dict(obj='Xtrain', cv=cv))
    self.InitResult()

    Xtest = self.Loaddata(**dict(obj='Xtest', cv=cv, itest=itest))
    Xtest = self.AddNoise(X=Xtest, loc=0., scale=0.)
    self.InitWeight(cv)

    sv = self.Loaddata(**dict(obj='sv', cv=cv))
    cov_est = np.diag(np.sqrt(sv[:self.nmod]))
    del sv
    if self.ROM:
      cov_obs_noise = np.eye(self.ngag)
      cov_sys_noise = np.eye(self.nmod) 
      Phi = self.Loaddata(**dict(obj='Phi', cv=cv))
      alp_test_now = np.linalg.pinv(Phi[:, :self.nmod]) @ Xtest[:,0]

    for itime in range(self.ntim):
      alp_train_now = train[:,:,itime]
      if self.ROM:
        alp_test_now, cov_est \
          = self.MakeTest(  KALMAN=KALMAN,
                            xobs=Xtest[:,itime], 
                            avec=alp_test_now, 
                            Pmat=cov_est, 
                            Qmat=cov_sys_noise, 
                            Rmat=cov_obs_noise,
                            cv=cv,
                            Phi=Phi
                          )
      else:
        alp_test_now = Xtest[:,itime]

      self.BayesianUpdate(  GPU=GPU, 
                            train_now=alp_train_now, 
                            test_now=alp_test_now, 
                            cv=cv,
                            cov=cov_est
                          )

      if itime%60==0:
        self.InfoProc(itime=itime, itest=itest, cv=cv)
      if itime in obs_window:
        self.result.append(copy.deepcopy(self.weights))
        if itime == obs_window[-1]:
          self.Output(itest=itest, obs_window=obs_window, cov_out=cov_out, cv=cv)
          break
        continue
    del train, Xtest, alp_train_now, alp_test_now, self.weights, cov_est
    print('Calculation ending at {}'.format(datetime.datetime.now()))


if __name__ == '__main__':
  res_dir = 'result/paper_Bayes_scenario_superpose'
  GPU = True
  obs_window = [30,120]
  cv = 1
  itest = 561
  nmod = 23

    
  home = os.path.abspath(os.path.dirname(__file__))
  finfo = '%s/../../%s/INFO.yaml' % (home, res_dir)

  with open(finfo, 'r') as f:
    info = yaml.safe_load(f)

  F = Forecasting(ver='Fujita', ROM=True, nmod=nmod, **info)
  F.Run_wave_based(GPU=GPU, cv=cv, itest=itest, obs_window=obs_window, cov_out=True)
  F.WaveformFigure( GPU=GPU, cv=cv, itest=itest, obs_window=obs_window, 
                      gIDlist=[9301,9303,9304,9305,9311])
  F.TaylorFigure( itest=itest, obs_window=obs_window, 
                    gIDlist=[9301,9303,9304,9305,9311], cv=cv)
  F.WeightsPDFFigure( cv=cv, itest=itest, obs_window=obs_window)

  N = Forecasting(ver='Nomura', ROM=True, nmod=nmod, **info)
  N.Run_state_based(GPU=GPU, cv=cv, itest=itest, obs_window=obs_window, cov_out=False)
  N.WaveformFigure( GPU=GPU, cv=cv, itest=itest, obs_window=obs_window, 
                      gIDlist=[9311,9301,9303,9304,9305])
  N.TaylorFigure( itest=itest, obs_window=obs_window, 
                    gIDlist=[9311,9301,9303,9304,9305], cv=cv)