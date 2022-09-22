"""
@Filename: resampling.py

@Author: Thomas Wagenhäuser, IAU
@Date:   2022-02-15T18:21:15+01:00
@Email:  wagenhaeuser@iau.uni-frankfurt.de

Purpose:
    running_correlation() is desinged to obtain SF6 measurement data with a better
    precision using local SF6-CFC12 correlations. This has been tested and used for
    GhOST-ECD data collected aboard the HALO reseach aircraft.
"""


import numpy as np
import matplotlib.pyplot as plt


# %%#######################################
def _handle_invalid(func):
    """Decorate running_correlation in order to handle invalid values."""

    def inner(x, y, *args, **kwargs):
        x = np.asarray(x)
        y = np.asarray(y)
        w_finite = ~np.isnan(x) & ~np.isnan(y)

        # call function (running_correlation) only with filtered input
        result = func(x[w_finite], y[w_finite], *args, **kwargs)
        if isinstance(result, tuple):
            y_fit_valid = result[0]
        else:
            y_fit_valid = result

        # expand function output to original array dimensions (reintroducing
        #   invalid values)
        y_fit = y * np.nan
        y_fit[w_finite] = y_fit_valid

        if isinstance(result, tuple):
            return (y_fit,) + result[1:]
        else:
            return y_fit

    return inner


@_handle_invalid
def running_correlation(
    x,
    y,
    windowsize=21,
    polyorder=2,
    ploty=False,
    plotxy=False,
    ax_y=None,
    ax_xy=None,
    return_ax_y=False,
    return_ax_xy=False,
):
    """Calculate running polynomial fit for smoothing y with the help of x.

    Handles input arrays containing invalid values (np.nan) as follows:
    1) x and y are compressed to remove all invalid values.
    2) running smoothing is performed: y_fit is calculated
        -> The window skips/ignores invalid values!
        -> Make sure to only process one dataset at once. Larger data gaps may lead
           to unintended behavior. Consider splitting the dataset in that case.
    3) y_fit is assigned to a new array with the original length of y, reintroducing
        the previously dropped invalid values.

    Parameters:
    -----------
    x: 1D np.array of length N
        independent (guiding) input data. Intended for CFC12 data with better precision.
    y: 1D np.array of length N
        dependent (following) input data. Intended for SF6 data with less good precision
        than CFC12.
    windowsize: int (odd number), optional
        provide the (odd!) number of array elements that are used for calculating each
        polynomial fit. The default is 11.
    polyorder: int, optional
        order of the polynomial fit. The default is 1.
    ploty: bool, optional
        plot y and y_fit against index. The default is False.
    plotxy: bool, optional
        plot y and y_fit against x. The default is False.
    ax_y: matplotlib axis, optional
        provide an existing axis to overplot ploty new fit. The default is None.
    ax_xy: matplotlib axis, optional
        provide an existing axis to overplot plot xy new fit. The default is None.
    return_ax_y: bool, optional
        return tuple including y_fit and ax_y (for overplotting later), instead of
        just y_fit. The default is False.
    return_ax_xy: bool, optional
        return tuple including y_fit and ax_xy (for overplotting later), instead of
        just y_fit. The default is False.

    Returns:
    ----------
    y_fit (np.array) or tuple of y_fit and matplotlib axis or axes
        y_fit: np.array
            smoothed array of y
        ax_y: matplotlib axis for ploty, optional
        ax_xy: matplotlib axis for plotxy, optional
    """
    y_fit = y * np.nan
    popt = []
    _jlmax = len(x) - windowsize
    for _j in range(len(x)):
        # center window around index _j,
        # but hold the window for indices on the edges of x
        _jl = int(_j - (windowsize - 1) / 2)
        if _jl < 0:
            _jl = 0
        elif _jl > _jlmax:
            _jl = _jlmax
        _jh = _jl + windowsize

        # select window values
        _x = x[_jl:_jh]
        _y = y[_jl:_jh]

        # calculate fit parameters
        _popt = np.polynomial.polynomial.polyfit(_x, _y, polyorder)
        popt.append(_popt)

        # calculate fitted y values
        y_fit[_j] = np.polynomial.Polynomial(_popt)(x[_j])

    if ploty:
        t = np.arange(len(y))
        if not ax_y:
            fig, ax_y = plt.subplots()
            ax_y.plot(t, y, label="y original")
            axt = ax_y.twinx()
            axt._get_lines.prop_cycler = ax_y._get_lines.prop_cycler
            axt.plot(t, x, label="x")
            ax_y.set_xlabel("data index")
            ax_y.set_ylabel("y concentration")
            axt.set_ylabel("x concentration")
        else:
            fig = ax_y.get_figure()
        ax_y.plot(
            t,
            y_fit,
            label="y_fit, polyorder: {}, windowsize: {}".format(polyorder, windowsize),
        )
        fig.legend()
    if plotxy:
        if not ax_xy:
            fig, ax_xy = plt.subplots()
            ax_xy.scatter(x, y, label="original")
            ax_xy.set_xlabel("x concentration")
            ax_xy.set_ylabel("y concentration")
        ax_xy.scatter(
            x,
            y_fit,
            label="y_fit vs. x, polyorder: {}, windowsize: {}".format(
                polyorder, windowsize
            ),
        )
        ax_xy.legend()

    # handle cases, where plot axes shall be returned in addition to y_fit
    return_val = (y_fit,)
    if return_ax_y or return_ax_xy:
        if return_ax_y:
            return_val += (ax_y,)
        if return_ax_xy:
            return_val += (ax_xy,)
    else:
        return_val = return_val[0]

    return return_val


# %%#################################
# application: calculate more accurate SF6 concentration for mean age calculation
#####################################


# define a decorator function which allows to put x and y data into a decorated function
# that will then use y_fit.


def smoothy(func):
    """Decorate a function to preprocess x and y to obtain y_fit, which is fed into the original function.

    Use running_correlation(x, y, ...) to get y_fit.
    y_fit is the input for the decorated function.
    With this decorator the following keywords will be passed to running_correlation:
    x: 1D np.array of length N
        independent (guiding) input data. Intended for CFC12 data with better precision.
    y: 1D np.array of length N
        dependent (following) input data. Intended for SF6 data with less good precision
        than CFC12.
    windowsize: int (odd number), optional
        provide the (odd!) number of array elements that are used for calculating each
        polynomial fit. The default is 21.
    polyorder: int, optional
        order of the polynomial fit. The default is 2.
    ploty: bool, optional
        plot y and y_fit against index. The default is False.
    plotxy: bool, optional
        plot y and y_fit against x. The default is False.
    """

    def inner(
        x, y, *args, windowsize=21, polyorder=2, ploty=False, plotxy=False, **kwargs
    ):
        y_fit = running_correlation(
            x,
            y,
            windowsize=windowsize,
            polyorder=polyorder,
            ploty=ploty,
            plotxy=plotxy,
            ax_y=None,
            ax_xy=None,
            return_ax_y=False,
            return_ax_xy=False,
        )
        return func(x, y_fit, *args, **kwargs)

    return inner
