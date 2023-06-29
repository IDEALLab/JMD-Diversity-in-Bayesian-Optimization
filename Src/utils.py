############################
#function.py
#Date: 12/30/2020

#Comments key:
# - Main description
## - Sub-level within the main description
### - Sub-sub-level
#############################


import math
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
import plotly.graph_objects as go
from scipy.stats import norm
import scipy.stats
from plotly.subplots import make_subplots
from scipy.stats import multivariate_normal
from opensimplex import OpenSimplex
from scipy.spatial.distance import pdist, squareform
import random
from random import choices
from numpy import atleast_1d,atleast_2d
from tqdm.notebook import tqdm
from scipy import interpolate
from bridson import poisson_disc_samples
import ray

def cumoptgap(data):
    """calculates cumulative optimality gap based on the given the data for line plot for optimality gap at each iteration."""
    if isinstance(data[0],(np.float32,int,np.float64,float)):
        cumgap=np.trapz(data)
    else:
        cumgap=np.array([np.trapz(data[a]) for a in range(len(data))])
    return cumgap

def bootstrap(data,trials=1000):
    """
    Calculates 5th,95th percentile and the mean using bootstrap statistics for a NxD array, i.e a matrix with D features and N trials.
    
    Input:
    data: NxD numpy array
    
    Output:
    boot_ci: 3xD array with [mean, 5th percentile, 95th percentile]
    """
    
    ## A few helper functions
    def get_pos_element_without_error(data,pos):
        """
        Acts like the numpy slicing tool, but returns none instead of an error if position is not in the list.
        """
        if pos>len(data)-1:
            return
        else:
            return data[pos]
    def mean_nonsquare_ax0(data):
        """
        returns a mean with axis 0, over a non square array i.e. rows of different lengths.
        """
        filter_none= lambda data: list(filter(lambda x: x is not None,data))
        len_dim_2=[len(row) for row in data]
        return [np.mean(filter_none([get_pos_element_without_error(row,pos) for row in data])) for pos in range(max(len_dim_2))]

    def percentile_nonsquare_ax0(data,percentile):
        """
        returns a percentile with axis 0, over a non square array i.e. rows of different lengths.
        """
        filter_none= lambda data: list(filter(lambda x: x is not None,data))
        len_dim_2=[len(row) for row in data]
        return [np.percentile(filter_none([get_pos_element_without_error(row,pos) for row in data]),percentile) for pos in range(max(len_dim_2))]
    
    def gen_boot_mean(data,N,D):
        if data.ndim>2:
            bootdata=np.array([choices(data[:,a],k=N) for a in range(D)])
            boot_mean=np.mean(bootdata,axis=1)
        elif isinstance(data[0],list) or isinstance(data[0],np.ndarray):
            bootdata=choices(data,k=D)
            boot_mean=mean_nonsquare_ax0(bootdata)
        else:
            bootdata=choices(data,k=D)
            boot_mean=np.mean(bootdata,axis=0)
        return boot_mean

    
    @ray.remote
    def map_(obj, f):
        return f(*obj)
    
    if isinstance(data,list):
        data=np.array(data)

    if data.ndim>2:
        N=data.ndim-2
        D=None
    else:
        N=None
        D=len(data)
    itr=trials
    boot_array=[]
    # boot_array= ray.get([map_.remote([data,N,D], gen_boot_mean) for _ in range(itr)])
    boot_array=[gen_boot_mean(data,N,D) for _ in range(itr)]
    if data.ndim>2:
        boot_ci=np.array([np.mean(boot_array,axis=0),np.percentile(boot_array,np.array(2.5),axis=0),
                          np.percentile(boot_array,np.array(97.5),axis=0)])
    elif isinstance(data[0],list) or isinstance(data[0],np.ndarray):
        boot_ci=np.array([mean_nonsquare_ax0(boot_array),percentile_nonsquare_ax0(boot_array,2.5),
                          percentile_nonsquare_ax0(boot_array,97.5)])
    else:
        boot_ci=np.array([np.mean(boot_array,axis=0),np.percentile(boot_array,np.array(2.5),axis=0),
                          np.percentile(boot_array,np.array(97.5),axis=0)])
        
    return boot_ci

def plot2d(x,y,z):
    """
    Plots a 2d surface in Plotly.
    """
    data=[go.Surface(z=z.T, x=x, y=y)]
    fig = go.Figure(data)    
    return fig

def plot3d(x,y,z,dev):
    """
    Can be used to plot the mean surface and the volume covered by 1 s.d.
    """
    devp=z+dev
    devn=z-dev
    step=(devp.max()-devn.min())/len(devn)
    stepxy=100/len(x)
    J, K, L = np.mgrid[:100:stepxy, :100:stepxy, devn.min():devp.max():step]
    V=L/L
    for k in range(len(L)):
        if devn.min()<L[k,k,k] and devp.max()>L[k,k,k]:
            for i in range(len(L)):
                for j in range(len(L)):
                    if L[i,j,k]>devn[i,j] and L[i,j,k]<devp[i,j]:
                        V[i,j,k]=1
                    else:
                        V[i,j,k]=-10
        else:
            V[k]=V.T[k]*(-10)
    
    V[:,:,0]=V[:,:,0]*(-10)         
    #Plottting the graph with one standard deviation as the interval        
    fig = go.Figure(data=[go.Volume(
    x=K.flatten(), y=J.flatten(), z=L.flatten(),
    value=V.flatten(),
    isomin=0.9,
    isomax=1,
    opacity=0.2,
    surface_count=27,
    caps= dict(x_show=True, y_show=True, z_show=True),
    showscale=False,
    colorscale=[[0.0, "rgb(165,0,38)"],
                [0.166667, "rgb(215,48,39)"],
                [0.33333, "rgb(244,109,67)"],
                [0.5, "rgb(253,174,97)"],
                [0.666667, "rgb(254,224,144)"],
                [0.833333, "rgb(224,243,248)"],
                [1.0, "rgb(171,217,233)"]]
    ),
                     go.Surface(z=z, x=x, y=y)
                     ])
    fig.show()    
    return fig

def wildcatwells_cont(N,smoothness,rug_freq,rug_amp):
    """
    Interpolates to generate a continuous wildcat wells using scipy
    """
    x,y,z=wildcatwells(N,smoothness,rug_freq,rug_amp)
    f = interpolate.interp2d(x,y,z,kind='cubic')
    
    return f

# Text Wrapping
# Defines wrapText which will attach an event to a given mpl.text object, wrapping it within the parent axes object.
# Adapted from : https://stackoverflow.com/questions/4018860/text-box-with-line-wrapping-in-matplotlib
def wrapText(text,fig, margin=25):
    """ Attaches an on-draw event to a given mpl.text object which will
        automatically wrap its string wthin the parent axes object.

        The margin argument controls the gap between the text and axes frame
        in points.
    """
    ax = fig.get_axes()
    margin = margin / 72 * fig.get_dpi()

    def _wrap(event):
        """Wraps text within its parent axes."""
        def _width(s):
            """Gets the length of a string in pixels."""
            text.set_text(s)
            return text.get_window_extent().width

        # Find available space
        clip = fig.get_window_extent()
        x0, y0 = text.get_transform().transform(text.get_position())
        if text.get_horizontalalignment() == 'left':
            width = clip.x1 - x0 - margin
        elif text.get_horizontalalignment() == 'right':
            width = x0 - clip.x0 - margin
        else:
            width = (min(clip.x1 - x0, x0 - clip.x0) - margin) * 2

        # Wrap the text string
        words = [''] + _splitText(text.get_text())[::-1]
        wrapped = []

        line = words.pop()
        while words:
            line = line if line else words.pop()
            lastLine = line

            while _width(line) <= width:
                if words:
                    lastLine = line
                    line += words.pop()
                    # Add in any whitespace since it will not affect redraw width
                    while words and (words[-1].strip() == ''):
                        line += words.pop()
                else:
                    lastLine = line
                    break

            wrapped.append(lastLine)
            line = line[len(lastLine):]
            if not words and line:
                wrapped.append(line)

        text.set_text('\n'.join(wrapped))

        # Draw wrapped string after disabling events to prevent recursion
        handles = fig.canvas.callbacks.callbacks[event.name]
        fig.canvas.callbacks.callbacks[event.name] = {}
        fig.canvas.draw()
        fig.canvas.callbacks.callbacks[event.name] = handles

    fig.canvas.mpl_connect('draw_event', _wrap)

def _splitText(text):
    """ Splits a string into its underlying chucks for wordwrapping.  This
        mostly relies on the textwrap library but has some additional logic to
        avoid splitting latex/mathtext segments.
    """
    import textwrap
    import re
    math_re = re.compile(r'(?<!\\)\$')
    textWrapper = textwrap.TextWrapper()

    if len(math_re.findall(text)) <= 1:
        return textWrapper._split(text)
    else:
        chunks = []
        for n, segment in enumerate(math_re.split(text)):
            if segment and (n % 2):
                # Mathtext
                chunks.append('${}$'.format(segment))
            else:
                chunks += textWrapper._split(segment)
        return chunks

## Taken from : https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta