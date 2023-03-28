import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def matplotlib_config():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['axes.labelsize'] = 30
    matplotlib.rcParams['legend.fontsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 20
    matplotlib.rcParams['ytick.labelsize'] = 20
    matplotlib.rcParams['pcolor.shading']='auto'
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"


def plot_intensity(fig, x1min, x1max, x2min, x2max, intensity_fun,
        fname='outputs/intensity.png'):
    fig = set_fig(fig)
    x1 = np.linspace(x1min, x1max, 1000)
    x2 = np.linspace(x2min, x2max, 1000)
    X1, X2 = np.meshgrid(x1, x2)

    X = np.hstack([X1.flatten().reshape((-1,1)), 
        X2.flatten().reshape((-1,1))])
    
    intensity = intensity_fun(X).reshape(X1.shape)

    plt.pcolormesh(X1, X2, intensity)
    plt.tight_layout()
    plt.xlim([x1min, x1max])
    plt.ylim([x2min, x2max])
    plt.savefig(fname, bbox_inches='tight')
    return fig

def plot_samples(fig, samples, s=1):
    fig = set_fig(fig)
    plt.scatter(samples[:,0], samples[:,1], c='r', marker='+', zorder=10, s=s)
    plt.tight_layout()
    #plt.savefig('outputs/samples.png', bbox_inches='tight')
    return fig

def set_fig(fig):
    if fig is None:
        fig = plt.figure(figsize=[64, 48])
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    return fig

