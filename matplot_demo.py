from __future__ import print_function
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.cm as cm
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Polygon

def demo1():
    # normal distribution center at x=0 and y=5
    x = np.random.randn(100000)
    y = np.random.randn(100000) + 5

    plt.hist2d(x, y, bins=40, norm=LogNorm())
    plt.colorbar()
    plt.show()

def demo2():
    """
    Edward Tufte uses this example from Anscombe to show 4 datasets of x
    and y that have the same mean, standard deviation, and regression
    line, but which are qualitatively different.

    matplotlib fun for a rainy day
    """

    x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
    y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
    y2 = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])
    y3 = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])
    x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8])
    y4 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])


    def fit(x):
        return 3 + 0.5*x


    xfit = np.array([np.amin(x), np.amax(x)])

    plt.subplot(221)
    plt.plot(x, y1, 'ks', xfit, fit(xfit), 'r-', lw=2)
    plt.axis([2, 20, 2, 14])
    plt.setp(plt.gca(), xticklabels=[], yticks=(4, 8, 12), xticks=(0, 10, 20))
    plt.text(3, 12, 'I', fontsize=20)

    plt.subplot(222)
    plt.plot(x, y2, 'ks', xfit, fit(xfit), 'r-', lw=2)
    plt.axis([2, 20, 2, 14])
    plt.setp(plt.gca(), xticklabels=[], yticks=(4, 8, 12), yticklabels=[], xticks=(0, 10, 20))
    plt.text(3, 12, 'II', fontsize=20)

    plt.subplot(223)
    plt.plot(x, y3, 'ks', xfit, fit(xfit), 'r-', lw=2)
    plt.axis([2, 20, 2, 14])
    plt.text(3, 12, 'III', fontsize=20)
    plt.setp(plt.gca(), yticks=(4, 8, 12), xticks=(0, 10, 20))

    plt.subplot(224)

    xfit = np.array([np.amin(x4), np.amax(x4)])
    plt.plot(x4, y4, 'ks', xfit, fit(xfit), 'r-', lw=2)
    plt.axis([2, 20, 2, 14])
    plt.setp(plt.gca(), yticklabels=[], yticks=(4, 8, 12), xticks=(0, 10, 20))
    plt.text(3, 12, 'IV', fontsize=20)

    # verify the stats
    pairs = (x, y1), (x, y2), (x, y3), (x4, y4)
    for x, y in pairs:
        print('mean=%1.2f, std=%1.2f, r=%1.2f' % (np.mean(y), np.std(y), np.corrcoef(x, y)[0][1]))

    plt.show()

def demo3():


    # the random data
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    nullfmt = NullFormatter()  # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    lim = (int(xymax / binwidth) + 1) * binwidth

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.show()

def demo4():
    """
    Demo of the histogram (hist) function with a few features.

    In addition to the basic histogram, this demo shows a few optional features:

        * Setting the number of data bins
        * The ``normed`` flag, which normalizes bin heights so that the integral of
          the histogram is 1. The resulting histogram is a probability density.
        * Setting the face color of the bars
        * Setting the opacity (alpha value).

    """


    # example data
    mu = 100  # mean of distribution
    sigma = 15  # standard deviation of distribution
    x = mu + sigma * np.random.randn(10000)

    num_bins = 50
    # the histogram of the data
    n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()

def demo5():
    """
    Demo of the histogram (hist) function used to plot a cumulative distribution.

    """
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib import mlab

    mu = 200
    sigma = 25
    n_bins = 50
    x = mu + sigma * np.random.randn(10000)

    n, bins, patches = plt.hist(x, n_bins, normed=1,
                                histtype='step', cumulative=True)

    # Add a line showing the expected distribution.
    y = mlab.normpdf(bins, mu, sigma).cumsum()
    y /= y[-1]
    plt.plot(bins, y, 'k--', linewidth=1.5)

    # Overlay a reversed cumulative histogram.
    plt.hist(x, bins=bins, normed=1, histtype='step', cumulative=-1)

    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.title('cumulative step')

    plt.show()

def contours():
    # !/usr/bin/env python
    """
    Illustrate simple contour plotting, contours on an image with
    a colorbar for the contours, and labelled contours.

    See also contour_image.py.
    """
    # import matplotlib
    # import numpy as np
    # import matplotlib.cm as cm
    # import matplotlib.mlab as mlab
    # import matplotlib.pyplot as plt

    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'


    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    # difference of Gaussians
    Z = 10.0 * (Z2 - Z1)

    # Create a simple contour plot with labels using default colors.  The
    # inline argument to clabel will control whether the labels are draw
    # over the line segments of the contour, removing the lines beneath
    # the label
    plt.figure()
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Simplest default with labels')

    # contour labels can be placed manually by providing list of positions
    # (in data coordinate). See ginput_manual_clabel.py for interactive
    # placement.
    plt.figure()
    CS = plt.contour(X, Y, Z)
    manual_locations = [(-1, -1.4), (-0.62, -0.7), (-2, 0.5), (1.7, 1.2), (2.0, 1.4), (2.4, 1.7)]
    plt.clabel(CS, inline=1, fontsize=10, manual=manual_locations)
    plt.title('labels at selected locations')

    # You can force all the contours to be the same color.
    plt.figure()
    CS = plt.contour(X, Y, Z, 6,
                     colors='k',  # negative contours will be dashed by default
                     )
    plt.clabel(CS, fontsize=9, inline=1)
    plt.title('Single color - negative contours dashed')

    # You can set negative contours to be solid instead of dashed:
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    plt.figure()
    CS = plt.contour(X, Y, Z, 6,
                     colors='k',  # negative contours will be dashed by default
                     )
    plt.clabel(CS, fontsize=9, inline=1)
    plt.title('Single color - negative contours solid')

    # And you can manually specify the colors of the contour
    plt.figure()
    CS = plt.contour(X, Y, Z, 6,
                     linewidths=np.arange(.5, 4, .5),
                     colors=('r', 'green', 'blue', (1, 1, 0), '#afeeee', '0.5')
                     )
    plt.clabel(CS, fontsize=9, inline=1)
    plt.title('Crazy lines')

    # Or you can use a colormap to specify the colors; the default
    # colormap will be used for the contour lines
    plt.figure()
    im = plt.imshow(Z, interpolation='bilinear', origin='lower',
                    cmap=cm.gray, extent=(-3, 3, -2, 2))
    levels = np.arange(-1.2, 1.6, 0.2)
    CS = plt.contour(Z, levels,
                     origin='lower',
                     linewidths=2,
                     extent=(-3, 3, -2, 2))

    # Thicken the zero contour.
    zc = CS.collections[6]
    plt.setp(zc, linewidth=4)

    plt.clabel(CS, levels[1::2],  # label every second level
               inline=1,
               fmt='%1.1f',
               fontsize=14)

    # make a colorbar for the contour lines
    CB = plt.colorbar(CS, shrink=0.8, extend='both')

    plt.title('Lines with colorbar')
    # plt.hot()  # Now change the colormap for the contour lines and colorbar
    plt.flag()

    # We can still add a colorbar for the image, too.
    CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)

    # This makes the original colorbar look a bit out of place,
    # so let's improve its position.

    l, b, w, h = plt.gca().get_position().bounds
    ll, bb, ww, hh = CB.ax.get_position().bounds
    CB.ax.set_position([ll, b + 0.1 * h, ww, h * 0.8])

    plt.show()

def integral():
    """
    Plot demonstrating the integral as the area under a curve.

    Although this is a simple example, it demonstrates some important tweaks:

        * A simple line plot with custom color and line width.
        * A shaded region created using a Polygon patch.
        * A text label with mathtext rendering.
        * figtext calls to label the x- and y-axes.
        * Use of axis spines to hide the top and right spines.
        * Custom tick placement and labels.
    """
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Polygon

    def func(x):
        return (x - 3) * (x - 5) * (x - 7) + 85

    a, b = 2, 9  # integral limits
    x = np.linspace(0, 10)
    y = func(x)

    fig, ax = plt.subplots()
    plt.plot(x, y, 'r', linewidth=2)
    plt.ylim(ymin=0)

    # Make the shaded region
    ix = np.linspace(a, b)
    iy = func(ix)
    verts = [(a, 0)] + list(zip(ix, iy)) + [(b, 0)]
    poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
    ax.add_patch(poly)

    plt.text(0.5 * (a + b), 30, r"$\int_a^b f(x)\mathrm{d}x$",
             horizontalalignment='center', fontsize=20)

    plt.figtext(0.9, 0.05, '$x$')
    plt.figtext(0.1, 0.9, '$y$')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xticks((a, b))
    ax.set_xticklabels(('$a$', '$b$'))
    ax.set_yticks([])

    plt.show()

# demo1()
# demo2()
# demo3()
# demo4()
# demo5()
# contours()
integral()
