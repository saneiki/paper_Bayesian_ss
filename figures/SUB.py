import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
import numpy as np
  
try:
  import cupy as cp
except:
  pass
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from scipy.stats import norm, multivariate_normal
import cartopy.crs as ccrs
import cartopy.feature as cfea
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from clawpack.geoclaw import dtopotools
# import matplotlib        # Optional
# matplotlib.use("TkAgg")  # Optional


def fsave(out, f):
  os.makedirs(out, exist_ok=True)
  plt.savefig(os.path.join(out,f), 
              dpi=300, bbox_inches='tight', pad_inches=0.05)


def topograph_projection( fig, fonts, fontf=None,
                          ta=[131.5, 137.5, 32., 34.5], 
                          xtick=np.arange(132, 138, 1),
                          ytick=np.arange(32,35,1), 
                          landc='gray', lakec='aliceblue', seac='aliceblue'
                        ):
  proj = ccrs.PlateCarree()
  land_h = cfea.NaturalEarthFeature(
    'physical', 'land', '10m', ec='k', lw=0.0, fc=landc)
  lakes_h = cfea.NaturalEarthFeature(
    'physical', 'lakes', '10m', ec='k', lw=0.0, fc=lakec)
  sea_h = cfea.NaturalEarthFeature(
    'physical', 'ocean', '10m', ec='k', lw=0.0, fc=seac)
  
  plt.rcParams['font.size'] = fonts
  if fontf:
    plt.rcParams['font.family'] = fontf

  ax = fig.add_subplot(1,1,1, projection=proj)
  ax.set_extent(ta, proj)
  ax.add_feature(land_h)
  ax.add_feature(lakes_h)
  ax.add_feature(sea_h)
  
  #if set_scale:
  #    self.scale_bar(ax, scale, (0.85,0.05), 3, self.fonts, scale/200)

  grd = ax.gridlines(crs=proj, linestyle=':', draw_labels=False)
  grd.xlocator = tck.FixedLocator(xtick)
  ax.set_xticks(xtick)
  lonfmt = LongitudeFormatter(zero_direction_label=True)
  ax.xaxis.set_major_formatter(lonfmt)
  grd.ylocator = tck.FixedLocator(ytick)
  ax.set_yticks(ytick)
  latfmt = LatitudeFormatter()
  ax.yaxis.set_major_formatter(latfmt)
  return fig, ax

def GaugeFigure(out, ft, fgauge, figs=(6.,10), fonts=6, **kwargs):
  print("  - Graphing synthetic gauge placements")
  os.makedirs(out, exist_ok=True)
  f = os.path.join(out, 'gauge.{}'.format(ft))
  
  gauge = pd.read_csv(fgauge)
  donet1  = gauge[gauge['Instruments']=='DONET1']
  donet2  = gauge[gauge['Instruments']=='DONET2']
  nowphas = gauge[gauge['Instruments']=='NOWPHAS']

  dic_d1 = {  "x"    :donet1["Longitude"],
              "y"    :donet1["Latitude"],
              "c"    :"green",
              "label":'DONET1',
            }

  dic_d2 = {  "x"    :donet2["Longitude"],
              "y"    :donet2["Latitude"],
              "c"    :"blue",
              "label":'DONET2',
            }

  dic_ns = {  "x"    :nowphas["Longitude"],
              "y"    :nowphas["Latitude"],
              "c"    :"red",
              "label":'NOWPHAS',
            }

  plt.rcParams['font.size'] = fonts
  fig = plt.figure(figsize=figs)
  fig, ax = topograph_projection(fig=fig, fonts=fonts) 

  ax.scatter(**dic_d1, marker='o', s=10, ec='k', lw=0.5, zorder=10)
  ax.scatter(**dic_d2, marker='o', s=10, ec='k', lw=0.5, zorder=10)
  ax.scatter(**dic_ns, marker='o', s=10, ec='k', lw=0.5, zorder=10)

  ax.plot([nowphas.iloc[2,1]+.03,134.1], [nowphas.iloc[2,2]-.05,32.2], lw=.5, c='red')
  ax.plot([134.1,134.9], [32.2,32.2], lw=.5, c='red')
  ax.text(x=134.3, y=32.3, s='Gauge A', c='red', ha='left', va='center',fontsize=fonts+1)

  ax.set_xlabel("Longitude")
  ax.set_ylabel("Latitude")
  ax.legend(loc=2)
  plt.savefig(f, dpi=300, bbox_inches='tight', pad_inches=0.05 )
  plt.clf()
  plt.close()


def SingularValuesFigure( out, ft, cv, fs,
                          thre=[0.99], figs=(6.,2.), fonts=6, **kwargs):
  print("  - Graphing contribution rate change of each spatial mode")
  os.makedirs(out, exist_ok=True)
  f = os.path.join(out, 'mode_contribution.{}'.format(ft))
  
  sv = np.load(fs.format(cv))
  s_rate = sv**2 / np.sum(sv**2)
  s_cumsum = np.cumsum(s_rate)

  text = ''
  thmodes = []
  for i,ith in enumerate(thre):
    thmode = str(np.where(s_cumsum>ith)[0][0]+1)
    thmodes.append(int(thmode))
    text += '{} modes required for {}% contribution \n'.format(thmode, int(ith*100))

  sns.set_style('darkgrid')
  plt.rcParams['font.size'] = fonts
  fig = plt.figure(figsize=figs)
  ax = fig.add_subplot(111)

  ax.plot(np.arange(s_cumsum.size)+1, s_cumsum, 
          c='k', lw=.7, 
          marker='o', mfc='w', mec='k', mew=.7, ms=3)
  a = [i-1 for i in thmodes]
  ax.plot(thmodes, s_cumsum[a], 
          c='k', lw=.7, 
          marker='o', mfc='w', mec='crimson', mew=.7, ms=3,
          zorder=10)

  ax.set_xlabel('Number of modes')
  ax.set_ylabel('Cumulative contribution rate')
  ax.set_xticks([0,5,10,15,20,25,30,35,40,45,50,55,60])
  ax.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
  plt.savefig(f, dpi=300, bbox_inches='tight', pad_inches=0.05)
  plt.clf()
  plt.close()


def ReconstructionErrorFigure(out, ft, cv, nmod, GPU, 
                              fXtrn, fu, fs, ngag, ntim, 
                              figs=(3.,1.5), fonts=6, **kwargs):
  print("  - Graphing reconstruction errors vs the number of modes")
  
  os.makedirs(out, exist_ok=True)
  f = os.path.join(out, 'recon_error.{}'.format(ft))

  if GPU:
    X = cp.load(fXtrn.format(cv))
    Phi = cp.load(fu.format(cv))
    sv = cp.load(fs.format(cv))
    ntrn = X.shape[0]
  
    dm = cp.zeros((ngag, ntim*ntrn))
    for i in range(ntrn):
      dm[:,i*ntim:(i+1)*ntim] = X[i,:,:]
    vh = cp.linalg.inv(np.diag(sv)) @ Phi.T @ dm
  
    error = []
    for imode in range(1, ngag+1, 1):
      Xrecon = Phi[:,:imode] @ cp.diag(sv[:imode]) @ vh[:imode,:]
      er = cp.sqrt(cp.average((dm - Xrecon)**2)) 
      error.append( er.get() )
  
  else:
    X = np.load(fXtrn.format(cv))
    Phi = np.load(fu.format(cv))
    sv = np.load(fs.format(cv))
    ntrn = X.shape[0]

    dm = np.zeros((ngag, ntim*ntrn))
    for i in range(ntrn):
      dm[:,i*ntim:(i+1)*ntim] = X[i,:,:]
    vh = np.linalg.inv(np.diag(sv)) @ Phi.T @ dm

    error = []
    for imode in range(1, ngag+1, 1):
      Xrecon = Phi[:,:imode] @  np.diag(sv[:imode]) @ vh[:imode,:]
      error.append( np.sqrt(np.average((dm - Xrecon)**2)) )

  sns.set_style('darkgrid')
  plt.rcParams['font.size'] = fonts
  fig = plt.figure(figsize=figs)
  ax = fig.add_subplot(111)
  ax.plot(np.arange(len(error))+1, error, 
          c='k', lw=0.5,
          marker='o', mfc='w', mec='k', mew=0.5, ms=2)
  ax.plot(nmod, error[nmod-1], 
          c='k', lw=0.5, 
          marker='o', mfc='w', mec='crimson', mew=0.5, ms=2,
          zorder=10)

  ax.set_xlabel('Number of modes')
  ax.set_ylabel('Reconstruction error (RMSE [m])')
  ax.set_xticks([0,5,10,15,20,25,30,35,40,45,50,55,60])
  ax.set_yticks([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35])
  plt.savefig(f, dpi=300, bbox_inches='tight', pad_inches=0.05)
  plt.clf()
  plt.close()


def MakeAxesWeight(figs=(2.2,2.2), fonts=6, widths=[4,1], heights=[1,4]):
  plt.clf()
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

  return fig, axs

def MakeAxesWeightThrCols(figs=(7.,1.8), fonts=6, widths=[4,1,2.,4,1,2.,4,1], heights=[1,4]):
  plt.clf()
  plt.rcParams['font.size'] = fonts
  fig = plt.figure(figsize=figs)
  axs = {}
  
  spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), 
                          width_ratios=widths, height_ratios=heights)
  for ii in range(3):
    axs[1,ii*3]   = fig.add_subplot(spec[1, ii*3])
    axs[0,ii*3]   = fig.add_subplot(spec[0, ii*3],   sharex=axs[1,ii*3])
    axs[1,ii*3+1] = fig.add_subplot(spec[1, ii*3+1], sharey=axs[1,ii*3])

  return fig, axs

def WeiInit(axs, out, ft):
  os.makedirs(out, exist_ok=True)
  f = os.path.join(out, 'Initpdf.{}'.format(ft))
  
  mu_vec = np.zeros(2)
  cov_mat = 0.01*np.eye(2)
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

  axs[0].plot(x, conditional_dens_a, lw=1., c='k')
  axs[0].set_xlim(-0.2,0.2)
  axs[0].set_ylabel('PD')
  axs[0].set_title("PDF of a sampled scenario's weight")
  
  axs[3].plot(conditional_dens_b, x, lw=1., c='k')
  axs[3].set_ylim(-0.2,0.2)
  axs[3].set_xlabel('PD')
  axs[3].set_title( "PDF of a sampled scenario's weight", 
                    rotation=270, va='center', ha='left', x=1.1,y=0.5
                  )
  
  mapp = axs[2].contourf( X, Y, joint_dens.reshape(X.shape), 10, 
                          cmap='coolwarm'
                        )
  cs = axs[2].contour(X, Y, joint_dens.reshape(X.shape), 10, 
                      colors='k', linewidths=0.3
                      )
  axs[2].clabel(cs, cs.levels[::2])
  axs[2].set_xlabel('Weight for a sampled scenario')
  axs[2].set_ylabel('Weight for a sampled scenario')
  axs[2].set_xlim(-0.2,0.2) 
  axs[2].set_ylim(-0.2,0.2) 
  axs[2].set_xticks([-0.2,-0.1,0.0,0.1,0.2])
  axs[2].set_yticks([-0.2,-0.1,0.0,0.1,0.2])
  
  plt.savefig(f, dpi=300, bbox_inches='tight', pad_inches=0.05)

def WeiPost(axs, out, cv, ROM, nmod, itest, obs, res_cv_dir, fttlst, **kwargs):
  os.makedirs(out, exist_ok=True)

  ttlst = pd.read_csv(fttlst.format(cv))
  lcase = ttlst.loc[ttlst['label']=='larn', 'ID']
  tcase = ttlst.loc[ttlst['label']=='test', 'ID']
  
  wei, cov =  InputResults( cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=120, ver='Fujita', cov_out=True, 
                            res_cv_dir=res_cv_dir, fttlst=fttlst)
  ID = [np.argmax(wei), np.argsort(wei)[::-1].iloc[1]]
  wei, cov =  InputResults( cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=obs, ver='Fujita', cov_out=True, 
                            res_cv_dir=res_cv_dir, fttlst=fttlst)
  sce_number = [list(lcase.index)[ID[0]]+1, list(lcase.index)[ID[1]]+1]
  
  mu_vec = np.array(wei[ID])
  cov_mat = cov[ID,:][:,ID]
  var = np.diag(cov_mat)
  std = np.sqrt(var)

  x = np.linspace(0., 1., 1000)
  y = np.linspace(0., 1., 1000)
  X, Y = np.meshgrid(x, y)
  XY = np.stack([X.flatten(), Y.flatten()], axis=1)
  joint_dens = multivariate_normal.pdf(
    x=XY, mean=mu_vec, cov=cov_mat
  )
  conditional_dens_a = norm.pdf(x=x, loc=mu_vec[0], scale=std[0])
  conditional_dens_b = norm.pdf(x=x, loc=mu_vec[1], scale=std[1])

  axs[0].plot(x, conditional_dens_a, lw=1., c='k')
  axs[0].set_xlim(0.,0.4)
  axs[0].set_ylabel('PD')
  axs[0].set_title('PDF for weight of #{}'.format(sce_number[0]))#, fontsize=fonts)

  axs[3].plot(conditional_dens_b, x, lw=1., c='k')
  axs[3].set_ylim(0.,0.4)
  axs[3].set_xlabel('PD')
  axs[3].set_title( 'PDF for weight of #{}'.format(sce_number[1]), 
                    rotation=270, va='center', ha='left', x=1.1,y=0.5)
  
  mapp = axs[2].contourf( X, Y, joint_dens.reshape(X.shape), 10, 
                          cmap='coolwarm')
  cs = axs[2].contour(X, Y, joint_dens.reshape(X.shape), 10, 
                      colors='k', linewidths=0.3)
  axs[2].clabel(cs, cs.levels[::2])
  axs[2].set_xlabel('Weight of #{}'.format(sce_number[0]))
  axs[2].set_ylabel('Weight of #{}'.format(sce_number[1]))
  axs[2].set_xlim(0.,0.4)
  axs[2].set_ylim(0.,0.4)
  axs[2].set_xticks([0.0,0.1,0.2,0.3,0.4])
  axs[2].set_yticks([0.0,0.1,0.2,0.3,0.4])
  
def SettingWeights(axs, alp, sid, obs):
  plt.subplots_adjust(hspace=0.15, wspace=0.15)

  fonts = axs[0,0].title.get_fontsize()
  for ii in range(3):
    # axs[1,ii*3].set_aspect('equal')
    axs[1,ii*3+1].set_xticks([0,5])
    axs[1,ii*3+1].set_xticklabels(["0","5"])
    axs[0,ii*3].text( x=.5, y=1.7, s=r'$T={}$ s'.format(obs[ii]*5), 
                      ha='center', va='bottom', fontsize=fonts,
                      transform=axs[0,ii*3].transAxes)
  
    plt.setp(axs[0,ii*3].get_xticklabels(), visible=False)
    plt.setp(axs[1,ii*3+1].get_yticklabels(), visible=False)
    axs[0,ii*3].grid(lw=0.5, ls='dotted')
    axs[1,ii*3].grid(lw=0.5, ls='dotted')
    axs[1,ii*3+1].grid(lw=0.5, ls='dotted')
  
  plt.suptitle( '({}) Scenario #{}'.format(alp,sid), 
                x=0.1, y=1.1, ha='left', va='center')


def InputResults(cv, ROM, nmod, itest, obs, ver, cov_out, res_cv_dir, fttlst, **kwargs):
  ttlst = pd.read_csv(fttlst.format(cv))
  test_case = ttlst.loc[ttlst['label']=='test', 'ID']
  if ROM:
    d = '%s/forecast/%smodes/%s_%s' % ( res_cv_dir.format(cv), 
                                        str(nmod).zfill(3), 
                                        str(itest).zfill(3), 
                                        test_case.iloc[itest])
  else:
    d = '%s/forecast/%smodes_withoutROM/%s_%s' % ( res_cv_dir.format(cv), 
                                        str(nmod).zfill(3), 
                                        str(itest).zfill(3), 
                                        test_case.iloc[itest])
  
  wei = pd.read_csv(os.path.join(d, 'wei_{}.csv'.format(ver)))
  wei = wei['Obs. window {} step'.format(obs)]
  cov = 0.
  if ver=='Fujita' and cov_out:
    cov = np.load(os.path.join(d, 'cov_{0:03d}window.npy'.format(obs)))
  return wei, cov


def MakeAxesWaveTwoCols(figs=(7.,1.3), fonts=6):
  plt.clf()
  plt.rcParams['font.size'] = fonts
  # sns.set_style('darkgrid')
  fig, axs = plt.subplots(1,2,sharey=True, figsize=figs)
  return fig, axs

def MakeAxesWaveTwoRows(figs=(3.4,2.5), fonts=6):
  plt.clf()
  plt.rcParams['font.size'] = fonts
  # sns.set_style('darkgrid')
  fig, axs = plt.subplots(2,1,sharex=True, figsize=figs)
  return fig, axs


def ObsWind(ax, obs):
  ax.axvline(x=obs*5/3600, lw=0.5, color='green', label='Obs. duration',zorder=5)

def WaveTest(ax, cv, itest, gID, fXtst, fgauge, ntim, **kwargs):
  gag = pd.read_csv(fgauge)
  idx = np.where(gag['ID']==gID)[0]
  Xtest = np.load(fXtst.format(cv))
  Xtst = Xtest[itest, idx, :]
  ax.plot(np.arange(ntim)*5/3600, Xtst[0,:], c='k', lw=.7, label='True')

def WaveFujita(ax, cv, ROM, nmod, itest, obs, gID, GPU, 
            res_cv_dir, fXtrn, fgauge, fttlst, fu, ntim, **kwargs):
  gag = pd.read_csv(fgauge)
  idx = np.where(gag['ID']==gID)[0]
  Xtrain = np.load(fXtrn.format(cv))
  Xtrn = Xtrain[:, idx, :]
  Phi = np.load(fu.format(cv))
  
  wei, cov =  InputResults( cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=obs,
                            ver='Fujita', cov_out=True, 
                            res_cv_dir=res_cv_dir, fttlst=fttlst)
  Xprd = np.sum(wei) * np.average(Xtrn, axis=0, weights=wei)
  sigma = PrdSigma(GPU=GPU, Xtrn=Xtrain, Phi=Phi, cov=cov, nmod=nmod, idx=idx)
  ax.plot( np.arange(ntim)*5/3600, Xprd[0,:], c='crimson', 
                lw=.7, ls='dashed', label='prd')
  ax.fill_between(
                        np.arange(ntim)*5/3600, 
                        Xprd[0,:]-1*sigma[0,:], Xprd[0,:]+1*sigma[0,:], 
                        lw=0., color='pink', label=r'$\pm 1 \sigma$'
                      )

def WaveNomura(ax, cv, ROM, nmod, itest, obs, gID, GPU, 
            res_cv_dir, fXtrn, fgauge, fttlst, fu, ntim, **kwargs):
  gag = pd.read_csv(fgauge)
  idx = np.where(gag['ID']==gID)[0]
  Xtrain = np.load(fXtrn.format(cv))
  Xtrn = Xtrain[:, idx, :]
  Phi = np.load(fu.format(cv))
  
  wei, cov =  InputResults( cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=obs, 
                            ver='Nomura', cov_out=False, 
                            res_cv_dir=res_cv_dir, fttlst=fttlst)
  most = np.argmax(wei)
  Xprd = Xtrn[most,:,:]
  ax.plot( np.arange(ntim)*5/3600, Xprd[0,:], c='blue', 
                lw=.7, ls='dashed', label='prd (N.2022)')


def SettingWavePrd(axs, alp, sid, w):
  if w=='Rows':
    axs[0].legend(ncol=3, loc='lower right', bbox_to_anchor=(1.0,1.0), frameon=False, borderaxespad=0.)
    axs[1].set_xlabel('Time [hours]')
    axs[0].set_ylabel('Elevation [m]')
    axs[1].set_ylabel('Elevation [m]')
    plt.xlim(0,3)
    plt.suptitle( '({}) Scenario #{}'.format(alp,sid), 
                  x=0., y=1, ha='left', va='center')
  if w=='Cols':
    plt.subplots_adjust(wspace=0.05)
    axs[1].legend(ncol=5, loc='lower right', bbox_to_anchor=(1.0,1.0), frameon=False, borderaxespad=0.)
    axs[0].set_xlabel('Time [hours]')
    axs[1].set_xlabel('Time [hours]')
    axs[0].set_ylabel('Elevation [m]')
    axs[0].set_xlim(0,3)
    axs[1].set_xlim(0,3)
    plt.suptitle( '({}) Scenario #{}'.format(alp,sid), 
                  x=.1, y=1, ha='left', va='center')


def PrdSigma(GPU, Xtrn, Phi, cov, nmod, idx):
  if GPU:
    cov = cp.array(cov)
    Var = cp.zeros((len(idx), Xtrn.shape[2]))
    alp_trn = MakeTrain(Xtrn=Xtrn, Phi=Phi, nmod=nmod)
    alp_trn = cp.array(alp_trn)
    Phi = cp.array(Phi)
    for i in range(Xtrn.shape[2]):
      cov_wave = (Phi[idx,:nmod] @  alp_trn[:,:,i].T) @ cov \
        @ (Phi[idx,:nmod] @  alp_trn[:,:,i].T).T #\
      Var[:,i] = cp.diag(cov_wave)
    Sig = cp.sqrt(Var)
    Sig = Sig.get()

  else:
    Var = np.zeros((len(idx), Xtrn.shape[2]))
    alp_trn = MakeTrain(Xtrn=Xtrn, Phi=Phi, nmod=nmod)
    for i in range(Xtrn.shape[2]):
      cov_wave = (Phi[idx,:nmod] @  alp_trn[:,:,i].T) @ cov \
        @ (Phi[idx,:nmod] @  alp_trn[:,:,i].T).T #\
      Var[:,i] = np.diag(cov_wave)
    Sig = np.sqrt(Var)
  return Sig
  

def MakeTrain(Xtrn, Phi, nmod):
  train = np.zeros((Xtrn.shape[0], nmod, Xtrn.shape[2]))
  for i in range(Xtrn.shape[0]):
    mat = Phi.T @ Xtrn[i, :, :]
    train[i,:,:] = mat[:nmod,:]
  return train


def MakeAxesTaylor(figs=(6.,3.), fonts=6):
  plt.clf()
  plt.rcParams['font.size'] = fonts
  sns.set_style('darkgrid')
  fig = plt.figure(figsize=figs)
  axs = [0,1]
  axs[0] = fig.add_subplot(121, projection='polar')
  axs[1] = fig.add_subplot(122, projection='polar')
  return fig, axs

def TaylorTest(ax):
  ax.scatter(0, 1, c='k',s=50, label="True")

def TaylorFujita(ax, cv, ROM, nmod, itest, obss, gID, 
              res_cv_dir, fXtrn, fXtst, fgauge, fttlst, ntim, **kwargs):
  cm = 'Reds'
  cm = plt.get_cmap(cm,5)
  marker=[",", "o", "^", "*"]
  
  gag = pd.read_csv(fgauge)
  idx = np.where(gag['ID']==gID)[0]
  Xtrain = np.load(fXtrn.format(cv))
  Xtest = np.load(fXtst.format(cv))
  Xtrn = Xtrain[:, idx, :]
  Xtst = Xtest[itest, idx, :]
  
  for j,obs in enumerate(obss):
    wei, cov =  InputResults( cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=obs,
                              ver='Fujita', cov_out=False, 
                              res_cv_dir=res_cv_dir, fttlst=fttlst)
    Xprd = np.sum(wei) * np.average(Xtrn, axis=0, weights=wei)
    sig_tst = np.std(Xtst, axis=1)
    sig_prd = np.std(Xprd, axis=1)
    R = np.sum((Xprd - np.average(Xprd, axis=1).reshape(-1,1)) \
               *(Xtst - np.average(Xtst, axis=1).reshape(-1,1)), axis=1
              ) / ntim / sig_tst / sig_prd
    
    ax.scatter( np.arccos(R), sig_prd/sig_tst, color=cm(j+1), 
                s=40, 
                marker=marker[j], alpha=1,
                ec='k', lw=0.3, 
                label=r"prd ($T$ = {}s)".format(obss[j]*5)
              )

def TaylorNomura(ax, cv, ROM, nmod, itest, obss, gID, 
              res_cv_dir, fXtrn, fXtst, fgauge, fttlst, ntim, **kwargs):
  cm = 'Blues'
  cm = plt.get_cmap(cm,5)
  marker=[",", "o", "^", "*"]
  
  gag = pd.read_csv(fgauge)
  idx = np.where(gag['ID']==gID)[0]
  Xtrain = np.load(fXtrn.format(cv))
  Xtest = np.load(fXtst.format(cv))
  Xtrn = Xtrain[:, idx, :]
  Xtst = Xtest[itest, idx, :]
  
  for j,obs in enumerate(obss):
    wei, cov =  InputResults( cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=obs,
                              ver='Nomura', cov_out=False, 
                              res_cv_dir=res_cv_dir, fttlst=fttlst)
    most = np.argmax(wei)
    Xprd = Xtrn[most,:,:]
    sig_tst = np.std(Xtst, axis=1)
    sig_prd = np.std(Xprd, axis=1)
    R = np.sum((Xprd - np.average(Xprd, axis=1).reshape(-1,1)) \
               *(Xtst - np.average(Xtst, axis=1).reshape(-1,1)), axis=1
              ) / ntim / sig_tst / sig_prd
    
    ax.scatter( np.arccos(R), sig_prd/sig_tst, color=cm(j+1), 
                s=40, 
                marker=marker[j], alpha=1,
                ec='k', lw=0.3, 
                label=r"N.2022 ($T$ = {}s)".format(obss[j]*5)
              )

def TaylorClosest(ax, cv, itest, gID, 
                  fXtrn, fXtst, fgauge, ntim, **kwargs):
  gag = pd.read_csv(fgauge)
  idx = np.where(gag['ID']==gID)[0]
  Xtrain = np.load(fXtrn.format(cv))
  Xtest = np.load(fXtst.format(cv))[itest,:,:]
  Xtrn = Xtrain[:, idx, :]
  Xtst = Xtest[idx, :]
  
  error = []
  for Xsub in Xtrain:
    er = np.linalg.norm(Xtest - Xsub, ord='fro')
    error.append(er)
  
  Xprd = Xtrn[np.argmin(error),:,:]
  sig_tst = np.std(Xtst, axis=1)
  sig_prd = np.std(Xprd, axis=1)
  R = np.sum((Xprd - np.average(Xprd, axis=1).reshape(-1,1)) \
             *(Xtst - np.average(Xtst, axis=1).reshape(-1,1)), axis=1
            ) / ntim / sig_tst / sig_prd
  
  ax.scatter( np.arccos(R), sig_prd/sig_tst, c='w', 
              s=40, ec='k', lw=0.3, marker='v', 
              label="Closest scenario")

def SettingTaylor(axs, alp, sid):
  plt.subplots_adjust(wspace=0.3)
  axs[1].legend(ncol=1, loc='center left', bbox_to_anchor=(1.1,.5),
            labelspacing=1.0, borderpad=1.0, frameon=False)#, borderaxespad=0.)
  for ii in range(len(axs)):
    axs[ii].set_ylabel('Normalized standard deviation')
    axs[ii].text( .5,-.1,'Normalized standard deviation', 
                  ha='center', va='top', transform=axs[ii].transAxes)
    axs[ii].text( .82,.82,'Correlation coefficient', rotation=-45, 
                  ha='center', va='center', transform=axs[ii].transAxes)
    axs[ii].set_xlim(0,np.pi/2.)
    axs[ii].set_ylim(0,1.4)
    theta_ticklabels = [1,0.99,0.95, 0.9,0.8,0.6,0.4,0.2,0]
    theta_ticks = np.arccos(theta_ticklabels)
    axs[ii].set_thetagrids(np.degrees(theta_ticks), labels=theta_ticklabels)
    axs[ii].set_title( '({}) Scenario #{}'.format(alp[ii],sid[ii]), 
                        x=0., y=1.1, ha='left', va='bottom')
  

def MakeAxesMax(figs=(6.5,2.1), fonts=6):
  plt.clf()
  plt.rcParams['font.size'] = fonts
  sns.set_style('darkgrid')
  fig, axs = plt.subplots(1,3, sharey=True, figsize=figs)
  plt.subplots_adjust(wspace=0.05)
  return fig, axs

def MaxFujita(ax, cv, ROM, nmod, obs, gID,  
            res_cv_dir, fXtrn, fXtst, fgauge, fttlst, **kwargs):
  gag = pd.read_csv(fgauge)
  idx = np.where(gag['ID']==gID)[0]
  Xtrain = np.load(fXtrn.format(cv))
  Xtest = np.load(fXtst.format(cv))
  Xtrn = Xtrain[:, idx, :]
  Xtst = Xtest[:, idx, :]
  ttlst = pd.read_csv(fttlst.format(cv))
  test_case = ttlst.loc[ttlst['label']=='test', 'ID']

  mw_list = ['81','83','85','87','89','91']
  marker = [',','o','^','8','p','D']
  cm = 'Reds'
  cm = plt.get_cmap(cm,8)

  ax.plot([-1,14], [-1,14], c='k', lw=.5)
  error = 0.
  error_mat = np.zeros(Xtst.shape[0])
  mwl = mw_list

  for itest in range(Xtst.shape[0]):
    mw = test_case.iloc[itest].replace('nagahama_', '')[:2]
    if mw in mwl:
      head = True
      mwl = mwl[1:]
    Xtst_max = np.max(Xtst[itest, :])
    wei, cov =  InputResults( cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=obs,
                              ver='Fujita', cov_out=False, 
                              res_cv_dir=res_cv_dir, fttlst=fttlst)
    Xprd_max = np.max( np.sum(wei) * np.average(Xtrn, axis=0, weights=wei) )

    error_mat[itest] = Xtst_max - Xprd_max
    error += (Xtst_max - Xprd_max)**2
    if head:
      ax.scatter( Xtst_max, Xprd_max, 
                  color=cm(mw_list.index(mw)+1), s=6, 
                  marker=marker[mw_list.index(mw)],  
                  ec='k', lw=.1, zorder=5,
                  label='Mw %s.%s' % (mw[0], mw[1])
                )
      head = False
    else:
      ax.scatter( Xtst_max, Xprd_max, 
                  color=cm(mw_list.index(mw)+1), s=6, 
                  marker=marker[mw_list.index(mw)],  
                  ec='k', lw=0.1, zorder=5
                )

  rmse = np.sqrt(error/Xtst.shape[0])
  ax.set_title(r'prd (Obs. duration $T=${}s)'.format(obs*5))

def MaxNomura(ax, cv, ROM, nmod, obs, gID,  
            res_cv_dir, fXtrn, fXtst, fgauge, fttlst, **kwargs):
  gag = pd.read_csv(fgauge)
  idx = np.where(gag['ID']==gID)[0]
  Xtrain = np.load(fXtrn.format(cv))
  Xtest = np.load(fXtst.format(cv))
  Xtrn = Xtrain[:, idx, :]
  Xtst = Xtest[:, idx, :]
  ttlst = pd.read_csv(fttlst.format(cv))
  test_case = ttlst.loc[ttlst['label']=='test', 'ID']

  mw_list = ['81','83','85','87','89','91']
  marker = [',','o','^','8','p','D']
  cm = 'Blues'
  cm = plt.get_cmap(cm,8)

  ax.plot([-1,14], [-1,14], c='k', lw=.5)
  error = 0.
  error_mat = np.zeros(Xtst.shape[0])
  mwl = mw_list

  for itest in range(Xtst.shape[0]):
    mw = test_case.iloc[itest].replace('nagahama_', '')[:2]
    if mw in mwl:
      head = True
      mwl = mwl[1:]
    Xtst_max = np.max(Xtst[itest, :])
    wei, cov =  InputResults( cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=obs,
                              ver='Nomura', cov_out=False, 
                              res_cv_dir=res_cv_dir, fttlst=fttlst)
    most = np.argmax(wei)
    Xprd_max = np.max(Xtrn[most,:])

    error_mat[itest] = Xtst_max - Xprd_max
    error += (Xtst_max - Xprd_max)**2
    if head:
      ax.scatter( Xtst_max, Xprd_max, 
                  color=cm(mw_list.index(mw)+1), s=6, 
                  marker=marker[mw_list.index(mw)],  
                  ec='k', lw=.1, zorder=5,
                  label='Mw %s.%s' % (mw[0], mw[1])
                )
      head = False
    else:
      ax.scatter( Xtst_max, Xprd_max, 
                  color=cm(mw_list.index(mw)+1), s=6, 
                  marker=marker[mw_list.index(mw)],  
                  ec='k', lw=0.1, zorder=5
                )

  rmse = np.sqrt(error/Xtst.shape[0])
  ax.set_title(r'N.2022 (Obs. duration $T=${}s)'.format(obs*5))

def SettingMax(axs, tit):
  plt.suptitle(tit, x=0.05, y=1.0, ha='left', va='top')
  axs[0].set_ylabel("Predicted maximum wave heights [m]")
  axs[1].set_xlabel("Observed maximum wave heights [m]")
  axs[0].legend( ncol=2, loc='upper left', bbox_to_anchor=(0.,1.0),
                  borderpad=0.3, borderaxespad=0.3, handletextpad=-0.5,
                  labelspacing=0.1, columnspacing=0.5
                )
  for ig in range(3):
    axs[ig].set_xlim(-0.5,13.)
    axs[ig].set_ylim(-0.5,13.)
    axs[ig].set_xticks([0,2,4,6,8,10,12])
    axs[ig].set_yticks([0,2,4,6,8,10,12])


def DeformDist(ax, mw, rnum):
  cmap = mpl.cm.bwr
  home = '%s/../data/rupture_data' % os.path.abspath(os.path.dirname(__file__))
  coast = np.loadtxt(os.path.join(home, 'japan_coast.txt'))
  fault0 = FaultGeo(home=home, mw=mw, rnum=rnum)
  
  ax.plot(coast[:,0],coast[:,1],'g',lw=.5)
  dx = 4/60.  # desired resolution of dtopo arrays
  x,y = fault0.create_dtopo_xy(dx=dx)
  print('Will compute dtopo on rectangle [%.2f, %.2f] x [%.2f, %.2f] with dx = %.4f'\
        % (x.min(),x.max(),y.min(),y.max(),dx))
  tfinal = max( [subfault1.rupture_time + subfault1.rise_time \
                for subfault1 in fault0.subfaults])
  ntimes = 5  # number of times for moving dtopo
  times0 = np.linspace(0.,tfinal,ntimes)
  print('Will create dtopo arrays of shape %i by %i at each of %i times'\
        % (len(x),len(y),ntimes))
  print('Need to apply Okada to %i subfaults at each time' % len(fault0.subfaults))
  print('This will take several minutes...')
  dtopo0 = fault0.create_dtopography(x,y,times=times0,verbose=False);

  X = dtopo0.X; Y = dtopo0.Y; dZ_at_t = dtopo0.dZ_at_t

  dz_max = dtopo0.dZ.max()

  print('Maximum uplift = %.2f m' % dz_max)
  dZ_interval = .3  # contour interval (meters)
  if dz_max > 6:
      dZ_interval = 2.
  if dz_max > 12:
      dZ_interval = 3.

  lev = np.sort(np.hstack([ np.arange(dZ_interval, dz_max, dZ_interval), 
                            -np.arange(dZ_interval, dz_max, dZ_interval)]))
  
  ax.contourf(X, Y, dZ_at_t(tfinal), levels=100,
              vmax=dz_max, vmin=-dz_max, cmap=cmap)
  ax.contour( X, Y, dZ_at_t(tfinal), levels=lev,
              linewidths=.5, colors='black')
  # dtopotools.plot_dZ_colors(X,Y,dZ_at_t(tfinal),axes=ax, dZ_interval=dZ_interval,
                            # cmax_dZ = dz_max, add_colorbar=True);

  dz_max = dZ_at_t(tfinal).max()

  cax,kw = mpl.colorbar.make_axes(ax, shrink=0.6)
  norm = mpl.colors.Normalize(vmin=-dz_max,vmax=dz_max)
  cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
  cb1.set_label("Deformation (m)")

def FaultGeo(home, mw, rnum):
  ruptures = 'nankai_%s_%s.rupt' % (mw,str(rnum).zfill(6))
  print('ruptures = ',ruptures)

  fault_geometry_file = os.path.join(home,'nankai_trough_input', 'nankai.mshout')
  print('Reading fault geometry from %s' % fault_geometry_file)

  fault_geometry = np.loadtxt(fault_geometry_file,skiprows=1)
  fault_geometry[:,[3,6,9,12]] = 1e3*abs(fault_geometry[:,[3,6,9,12]])
  print('Loaded geometry for %i triangular subfaults' % fault_geometry.shape[0])

  rupt_fname = os.path.join(home, ruptures)
  rupture_parameters = np.loadtxt(rupt_fname,skiprows=1)
  
  fault0 = dtopotools.Fault()
  fault0.subfaults = []
  fault0.rupture_type = 'kinematic'
  rake = 90. # assume same rake for all subfaults
  
  J = int(np.floor(fault_geometry.shape[0]))
  for j in range(J):
    subfault0 = dtopotools.SubFault()
    node1 = fault_geometry[j,4:7].tolist()
    node2 = fault_geometry[j,7:10].tolist()
    node3 = fault_geometry[j,10:13].tolist()
    node_list = [node1,node2,node3]
    
    ss_slip = rupture_parameters[j,8]
    ds_slip = rupture_parameters[j,9]
    
    rake = np.rad2deg(np.arctan2(ds_slip, ss_slip))
    
    subfault0.set_corners(node_list,projection_zone='53')
    subfault0.rupture_time = rupture_parameters[j,12]
    subfault0.rise_time = rupture_parameters[j,7]
    subfault0.rake = rake

    slip = np.sqrt(ds_slip ** 2 + ss_slip ** 2)
    subfault0.slip = slip
    fault0.subfaults.append(subfault0)
  return fault0
  

def MakeAxesWeiDist(figs=(7.,3.), fonts=6, widths=[1,1], heights=[4,1]):
  plt.clf()
  plt.rcParams['font.size'] = fonts
  fig = plt.figure(figsize=figs)
  axs = {}
  
  spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), 
                          width_ratios=widths, height_ratios=heights
                          )
  
  axs[2] = fig.add_subplot(spec[2//len(widths), 2%len(widths)])
  axs[0] = fig.add_subplot(spec[0//len(widths), 0%len(widths)], sharex=axs[2])
  axs[3] = fig.add_subplot(spec[3//len(widths), 3%len(widths)])
  axs[1] = fig.add_subplot(spec[0//len(widths), 3%len(widths)], sharex=axs[3])
  return fig, axs

def WeightHistgram( ax, cv, ROM, nmod, itest, obs, 
                    res_cv_dir,  fttlst, **kwargs):
  ttlst = pd.read_csv(fttlst.format(cv))
  lcase = ttlst.loc[ttlst['label']=='larn', 'ID']

  wei, cov =  InputResults( cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=obs,
                            ver='Fujita', cov_out=False, 
                            res_cv_dir=res_cv_dir, fttlst=fttlst)
  idx = np.argsort(wei)[::-1][:3]
  print('\nFor test scenario No.%s' % itest)
  for ii, ilarn in enumerate(idx):
    print('  Scenario with the %s-th largest weight' % (ii+1))
    print('    #', list(lcase.index)[ilarn] + 1, lcase.iloc[ilarn])
    print('    w =', wei[ilarn])

  ax.hist(np.array(wei), bins=50, lw=0, color='crimson',zorder=10)
  ax.set_ylabel('Frequency')

def WeightBox(ax, cv, ROM, nmod, itest, obs, 
              res_cv_dir,  fttlst, **kwargs):
  
  wei, cov =  InputResults( cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=obs,
                            ver='Fujita', cov_out=False, 
                            res_cv_dir=res_cv_dir, fttlst=fttlst)
  ax.boxplot( np.array(wei), vert=False, whis=4., widths=.5,
              boxprops=dict(linewidth=.7),  
              medianprops=dict(color='black', linewidth=.7), 
              whiskerprops=dict(color='black', linewidth=.7), 
              capprops=dict(color='black', linewidth=.7), 
              flierprops=dict(markeredgecolor='black', markeredgewidth=.7, markersize=5.) 
            )
  ax.set_xlabel('Weight mean')

def SettingHistgram(axs, alp, sid):
  plt.setp(axs[0].get_xticklabels(), visible=False)
  plt.setp(axs[1].get_xticklabels(), visible=False)
  plt.setp(axs[2].get_yticklabels(), visible=False)
  plt.setp(axs[3].get_yticklabels(), visible=False)
  plt.subplots_adjust(wspace=0.2)
  
  axs[0].set_title( '({}) Scenario #{}'.format(alp[0],sid[0]), 
                      x=0., y=1., ha='left', va='center')
  axs[1].set_title( '({}) Scenario #{}'.format(alp[1],sid[1]), 
                      x=0., y=1., ha='left', va='center')


def WaveBstWst(ax, ver, w, num, cv, ROM, nmod, itest, obs, gID, 
            res_cv_dir, fXtrn, fgauge, fttlst, ntim, **kwargs):
  cm = 'cool'
  cm = plt.get_cmap(cm,num+2)

  gag = pd.read_csv(fgauge)
  idx = np.where(gag['ID']==gID)[0]
  Xtrain = np.load(fXtrn.format(cv))
  Xtrn = Xtrain[:, idx, :]
  
  lab = ['First','Second','Third','Fourth','Fifth','Sixth','Seventh']
  
  wei, cov =  InputResults( cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=obs,
                            ver=ver, cov_out=False, 
                            res_cv_dir=res_cv_dir, fttlst=fttlst)
  if w=='best':
    Xtop = Xtrn[np.argsort(wei)[::-1][:num], :]
  if w=='worst':
    Xtop = Xtrn[np.argsort(wei)[:num], :]
  for ii in range(3):
    ax.plot( np.arange(ntim)*5/3600, Xtop[ii,0,:], c=cm(ii+1), 
                  lw=.7, ls='dashed', label=lab[ii])

def SettingWaveTwoCols(axs, alp, sid):
  plt.subplots_adjust(wspace=0.05)
  axs[0].set_ylabel('Elevation [m]')

  for ii in range(len(axs)):
    axs[ii].set_xlabel('Time [hours]')
    axs[ii].set_xlim(0,3)
    axs[ii].set_title( '({}) Scenario #{}'.format(alp[ii],sid[ii]), 
                  x=0., y=1.1, ha='left', va='bottom')


def MakeAxesCoefficient(figs=(7.,4.), fonts=7):
  plt.clf()
  plt.rcParams['font.size'] = fonts
  fig, axs = plt.subplots(3,2,sharey=False, figsize=figs)
  plt.subplots_adjust(wspace=.15, hspace=.3)
  return fig, axs

def CoefTest(ax, cv, itest, cID, fXtst, ntim, fu, **kwargs):
  Xtest = np.load(fXtst.format(cv))[itest, :, :]
  Phi = np.load(fu.format(cv))
  Coef = np.linalg.inv(Phi) @ Xtest
  Coef = Coef[cID, :]
  ax.plot(np.arange(ntim)*5/3600, Coef, c='gray', lw=1.2, label='True')

def CoefTest2(ax, cv, itest, cID, fXtrn, fXtst, ntim, **kwargs):
  Xtrain = cp.load(fXtrn.format(cv))[:, :, :]
  Xtest = cp.load(fXtst.format(cv))[itest, :, :]
  X = cp.zeros((Xtrain.shape[0]+1, Xtrain.shape[1], Xtrain.shape[2]))
  X[:-1,:,:] = Xtrain
  X[-1,:,:] = Xtest
  
  ntrn = X.shape[0]
  ngag = X.shape[1]
  ntim = X.shape[2]
  dm = cp.zeros((ngag, ntim*ntrn))
  for i in range(ntrn):
    dm[:,i*ntim:(i+1)*ntim] = X[i,:,:]

  U, S, Vh = cp.linalg.svd(dm, full_matrices=False)
  Coef = cp.diag(S) @ -Vh[:,-1*ntim:]
  Coef = Coef[cID, :].get()
  ax.plot(np.arange(ntim)*5/3600, Coef, c='gray', lw=1.2, label='True')

def CoefPseudoInv(ax, cv, nmod, itest, cID, fXtst, ntim, fu, **kwargs):
  Xtest = np.load(fXtst.format(cv))[itest, :, :]
  Phi = np.load(fu.format(cv))[:,:nmod]
  Coef = np.linalg.pinv(Phi) @ Xtest
  Coef = Coef[cID, :]
  ax.plot(np.arange(ntim)*5/3600, Coef, c='blue', lw=1.2, ls=(3,(2,3)), label='Pseudo-inverse')

def CoefKalmanFilter(ax, cv, nmod, itest, cID, fXtst, ntim, fu, fs, **kwargs):
  Xtest = np.load(fXtst.format(cv))[itest, :, :]
  Phi = np.load(fu.format(cv))[:,:nmod]
  sv = np.load(fs.format(cv))[:nmod]

  Coef = np.zeros((nmod,ntim))
  Sigma = np.zeros_like(Coef)

  avec = np.linalg.pinv(Phi[:, :nmod]) @ Xtest[:,0] 
  Pmat = np.diag(np.sqrt(sv))
  Qmat = np.eye(nmod) 
  Rmat = np.eye(Xtest.shape[0])
  for itime in range(ntim):
    xobs = Xtest[:,itime]
    abar = avec
    Pbar = Pmat + Qmat
    Gmat = Pbar @ Phi.T @ np.linalg.inv(Phi@Pbar@Phi.T + Rmat)
    avec = abar +Gmat @ (xobs - Phi@abar)
    Pmat = (np.eye(nmod) - Gmat@Phi) @ Pbar

    Coef[:,itime] = avec
    Sigma[:,itime] = np.sqrt(np.diag(Pmat))
  
  ax.plot(np.arange(ntim)*5/3600, Coef[cID,:], c='crimson', 
          lw=1.2, ls=(0,(2,3)), label='Kalman filter')
  ax.fill_between(np.arange(ntim)*5/3600, 
                  Coef[cID,:]-1*Sigma[cID,:], Coef[cID,:]+1*Sigma[cID,:], 
                  lw=0., color='pink', label=r'$\pm 1 \sigma$'
                  )

def SettingCoefficient(axs, alp, sid):
  axs[0,1].legend(ncol=4, loc='lower right', bbox_to_anchor=(1.,1.), frameon=False, borderaxespad=0.)
  
  axs[2,0].set_xlabel('Time [hours]')
  axs[2,1].set_xlabel('Time [hours]')
  for i in range(3):
    axs[i,0].set_xlim(0,3)
    axs[i,1].set_xlim(0,3)

  axs[0,0].set_ylabel('1st. coef.')
  axs[1,0].set_ylabel('2nd. coef.')
  axs[2,0].set_ylabel('3rd. coef.')
  
  axs[0,0].set_title( '({}) Scenario #{}'.format(alp[0],sid[0]), 
                      x=0., y=1.2, ha='left', va='center')
  axs[0,1].set_title( '({}) Scenario #{}'.format(alp[1],sid[1]), 
                      x=0., y=1.2, ha='left', va='center')


def WavePart(ax, nb, nw, cv, ROM, nmod, itest, obs, gID, GPU, 
            res_cv_dir, fXtrn, fgauge, fttlst, fu, ntim, **kwargs):
  gag = pd.read_csv(fgauge)
  idx = np.where(gag['ID']==gID)[0]
  Xtrain = np.load(fXtrn.format(cv))
  Xtrn = Xtrain[:, idx, :]
  
  wei, cov =  InputResults( cv=cv, ROM=ROM, nmod=nmod, itest=itest, obs=obs,
                            ver='Fujita', cov_out=True, 
                            res_cv_dir=res_cv_dir, fttlst=fttlst)
  if nb>0:
    wei_top = wei[np.argsort(wei)[::-1][:nb]]
    Xtop = Xtrn[np.argsort(wei)[::-1][:nb], :]
    Xt = np.sum(wei_top) * np.average(Xtop, axis=0, weights=wei_top)
  elif nb==0:
    Xt = np.zeros_like(Xtrn[0, :])
  
  if nw>0:
    wei_bottom = wei[np.argsort(wei)[:nw]]
    Xbottom = Xtrn[np.argsort(wei)[:nw], :]
    Xb = np.sum(wei_bottom) * np.average(Xbottom, axis=0, weights=wei_bottom)
  elif nw==0:
    Xb = np.zeros_like(Xtrn[0, :])

  Xprd = Xt + Xb
  ax.plot(np.arange(ntim)*5/3600, Xprd[0,:], c='crimson', 
          lw=.7, ls='dashed', label='prd')


def MakeAxesRupture(figs=(7.,3.), fonts=7):
  plt.clf()
  plt.rcParams['font.size'] = fonts
  # fig, axs = plt.subplots(2,1,sharex=True, figsize=figs)
  fig, axs = plt.subplots(1,2,sharex=True,  figsize=figs)
  return fig, axs


def SlipDist(ax, mw, rnum):
  cmap = mpl.cm.jet
  home = '%s/../data' % os.path.abspath(os.path.dirname(__file__))
  coast = np.loadtxt(os.path.join(home, 'japan_coast.txt'))
  fault0 = FaultGeo(home=home, mw=mw, rnum=rnum)

  cmin_slip = 0.01  # smaller values will be transparent
  cmax_slip = np.array([s.slip for s in fault0.subfaults]).max()
  
  for s in fault0.subfaults:
    c = s.corners
    c.append(c[0])
    c = np.array(c)
    ax.plot(c[:,0],c[:,1], [.4,.4,.4], linewidth=0.1, zorder=10, c='gray')
  
    x_corners = [ s.corners[2][0],
                  s.corners[0][0],
                  s.corners[1][0],
                  s.corners[2][0]]

    y_corners = [ s.corners[2][1],
                  s.corners[0][1],
                  s.corners[1][1],
                  s.corners[2][1]]
    slip = s.slip
    s = min(1, max(0, (slip-cmin_slip)/(cmax_slip-cmin_slip)))
    c = np.array(cmap(s*.99))  # since 1 does not map properly with jet
    if slip <= cmin_slip:
        c[-1] = 0  # make transparent
    ax.fill(x_corners,y_corners,color=c,edgecolor='none',zorder=5)
  ax.plot(coast[:,0],coast[:,1],'g', lw=.5, zorder=1)

  cax,kw = mpl.colorbar.make_axes(ax, shrink=0.6)
  norm = mpl.colors.Normalize(vmin=cmin_slip,vmax=cmax_slip)
  cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
  cb1.set_label("Slip (m)")
  
  r = mpl.patches.Rectangle(xy=(131.5,32.), width=6., height=2.5,
                            facecolor='none', ec='k', lw=.5, zorder=10)
  ax.add_patch(r)


def DeformDist(ax, mw, rnum):
  cmap = mpl.cm.bwr
  home = '%s/../data' % os.path.abspath(os.path.dirname(__file__))
  coast = np.loadtxt(os.path.join(home, 'japan_coast.txt'))
  fault0 = FaultGeo(home=home, mw=mw, rnum=rnum)
  
  ax.plot(coast[:,0],coast[:,1],'g',lw=.5)
  dx = 4/60.  # desired resolution of dtopo arrays
  x,y = fault0.create_dtopo_xy(dx=dx)
  print('Will compute dtopo on rectangle [%.2f, %.2f] x [%.2f, %.2f] with dx = %.4f'\
        % (x.min(),x.max(),y.min(),y.max(),dx))
  tfinal = max( [subfault1.rupture_time + subfault1.rise_time \
                for subfault1 in fault0.subfaults])
  ntimes = 5  # number of times for moving dtopo
  times0 = np.linspace(0.,tfinal,ntimes)
  print('Will create dtopo arrays of shape %i by %i at each of %i times'\
        % (len(x),len(y),ntimes))
  print('Need to apply Okada to %i subfaults at each time' % len(fault0.subfaults))
  print('This will take several minutes...')
  dtopo0 = fault0.create_dtopography(x,y,times=times0,verbose=False);

  X = dtopo0.X; Y = dtopo0.Y; dZ_at_t = dtopo0.dZ_at_t

  dz_max = dtopo0.dZ.max()

  print('Maximum uplift = %.2f m' % dz_max)
  dZ_interval = .3  # contour interval (meters)
  if dz_max > 6:
      dZ_interval = 2.
  if dz_max > 12:
      dZ_interval = 3.

  lev = np.sort(np.hstack([ np.arange(dZ_interval, dz_max, dZ_interval), 
                            -np.arange(dZ_interval, dz_max, dZ_interval)]))
  
  ax.contourf(X, Y, dZ_at_t(tfinal), levels=100,
              vmax=dz_max, vmin=-dz_max, cmap=cmap)
  ax.contour( X, Y, dZ_at_t(tfinal), levels=lev,
              linewidths=.5, colors='black')
  # dtopotools.plot_dZ_colors(X,Y,dZ_at_t(tfinal),axes=ax, dZ_interval=dZ_interval,
                            # cmax_dZ = dz_max, add_colorbar=True);

  dz_max = dZ_at_t(tfinal).max()

  cax,kw = mpl.colorbar.make_axes(ax, shrink=0.6)
  norm = mpl.colors.Normalize(vmin=-dz_max,vmax=dz_max)
  cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
  cb1.set_label("Deformation (m)")


def FaultGeo(home, mw, rnum):
  ruptures = 'nankai_%s_%s.rupt' % (mw,str(rnum).zfill(6))
  print('ruptures = ',ruptures)

  fault_geometry_file = os.path.join(home,'nankai_trough_input', 'nankai.mshout')
  print('Reading fault geometry from %s' % fault_geometry_file)

  fault_geometry = np.loadtxt(fault_geometry_file,skiprows=1)
  fault_geometry[:,[3,6,9,12]] = 1e3*abs(fault_geometry[:,[3,6,9,12]])
  print('Loaded geometry for %i triangular subfaults' % fault_geometry.shape[0])

  rupt_fname = os.path.join(home, ruptures)
  rupture_parameters = np.loadtxt(rupt_fname,skiprows=1)
  
  fault0 = dtopotools.Fault()
  fault0.subfaults = []
  fault0.rupture_type = 'kinematic'
  rake = 90. # assume same rake for all subfaults
  
  J = int(np.floor(fault_geometry.shape[0]))
  for j in range(J):
    subfault0 = dtopotools.SubFault()
    node1 = fault_geometry[j,4:7].tolist()
    node2 = fault_geometry[j,7:10].tolist()
    node3 = fault_geometry[j,10:13].tolist()
    node_list = [node1,node2,node3]
    
    ss_slip = rupture_parameters[j,8]
    ds_slip = rupture_parameters[j,9]
    
    rake = np.rad2deg(np.arctan2(ds_slip, ss_slip))
    
    subfault0.set_corners(node_list,projection_zone='53')
    subfault0.rupture_time = rupture_parameters[j,12]
    subfault0.rise_time = rupture_parameters[j,7]
    subfault0.rake = rake

    slip = np.sqrt(ds_slip ** 2 + ss_slip ** 2)
    subfault0.slip = slip
    fault0.subfaults.append(subfault0)
  return fault0
  

def SettingSlipDist(ax):
  ax.set_xlim(130,140)
  ax.set_ylim(30,36)
  ax.set_aspect(1./np.cos(32*np.pi/180.))
