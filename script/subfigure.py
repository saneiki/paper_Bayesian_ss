import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import cartopy.crs as ccrs
import cartopy.feature as cfea
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
# import matplotlib        # Optional
# matplotlib.use("TkAgg")  # Optional

def topograph_projection( fig, fonts, fontf=None,
                          ta=[131.5, 137.5, 32., 34.5], 
                          xtick=np.arange(132, 138, 1),
                          ytick=np.arange(32,35,1), 
                          landc='gray', lakec='aliceblue', seac='aliceblue'
                        ):
  """
  Projection of topographic information with cartopy for the figure area.

  parameters
  ------------
  fig : matplotlib.pyplot.figure
      figure field
  fonts : int
      Font size.
  fontf : str
      Font family.
  ta : list[float, float, float, float]
      List of longitude and latitude of the target area. The order is as [smaller longitude, larger longitudes, smallar latitudes, larger latitudes].
  xtick : numpy.array
      Ticks in the x direction to be drawn on the graph.
  ytick : numpy.array
      Ticks in the y direction to be drawn on the graph.
  landc : str
      Color of land regions.
  lakec : str
      Color of lake regions.
  seac : str
      Color of ocean regions.
  
  Returns
  ---------
  fig : matplotlib.pyplot.figure
      A figure object
  ax : matplotlib.axes
      An axes object
  """


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