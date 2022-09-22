"""
@Filename: Conc2Age_fit_fracts.py

@Author: Thomas Wagenhäuser, IAU
@Date:   2022-02-07T18:44:50+01:00
@Email:  wagenhaeuser@iau.uni-frankfurt.de


Probably the function that you are looking for:
########################
calculate_age_fit_fracts
########################
Calculate mean age assuming an ideal quadratic increasing tracer.

Supports considering extra tropical input into the stratosphere.
    - based on the assumption, that each entry region's reference time series
      can be described by the same quadratic reference time series simply shifted
      by a constant time offset.
    - assumes one age spectrum G with a ratio of moments, that is folded with
      the weighted mean of two entry regions time series.

Details can be found in the functions docstring.

There is also an option to smooth e.g. SF6 data using a local SF6-CFC12 correlation:
################################
calculate_age_fit_fracts_smoothy
################################
"""

import numpy as np
from scipy import integrate


from .calculate_agespectrum_1d import Calculate_AgeSpectrum_1D
from .tools import handle_deseas, check_single_input
from .resampling import smoothy


# %%
def calculate_varint(a_obs, rom, int_perc):
    """
    Calculate integration time to cover a certain percentage of an age spectrum.

    Call _calculate_varint() in a for loop in order to support multiple value input.

    Use a first guess mean age `a_obs` and ratio of moments `rom` to calculate
    an age spectrum from an inverse gaussian distribution. Then find the minimum
    transit time, where at least `int_perc` % of the age spectrum area are covered.

    The function `_calculate_varint` is a shorter and much faster version
    of `calculate_varint_old`.

    Parameters:
    -----------
    a_obs: float or array of floats
        mean age for the inverse gaussian distribution. Must be positive, otherwise a
        value of 5 will be returned.
    rom: float
        ratio of moments for the inverse gaussian distribution
    int_perc: int or float
        percentage of age spectrum to cover (< 100).
    """
    # handle input
    if isinstance(a_obs, float) or isinstance(a_obs, int):
        single_output = True
    else:
        single_output = False
    a_obs = np.asarray(a_obs).flatten()
    rom = np.asarray(rom).flatten()
    if len(rom) == 1:
        # take the same ratio of moments for every observation
        rom = np.ones(a_obs.shape) * rom

    # calculate fit intervals (for loop was equally fast like the vectorized version)
    fitint = []
    for age_i, rom_i in zip(list(a_obs), list(rom)):
        try:
            fitint_i = _calculate_varint(a_obs=age_i, rom=rom_i, int_perc=int_perc)
        except IndexError as IE:
            print("IndexError in calculate_varint:")
            print("age_i:", age_i)
            print("rom_i:", rom_i)
            print("int_perc:", int_perc)
            print("len(fitint):", len(fitint))
            raise IE
        fitint.append(fitint_i)
    fitint = np.asarray(fitint)

    # handle single input values
    if single_output:
        fitint = fitint[0]

    return fitint


def _calculate_varint(a_obs, rom, int_perc):
    """
    Calculate integration time to cover a certain percentage of an age spectrum.

    Use a first guess mean age `a_obs` and ratio of moments `rom` to calculate
    an age spectrum from an inverse gaussian distribution. Then find the minimum
    transit time, where at least `int_perc` % of the age spectrum area are covered.

    This function `_calculate_varint` is a shorter and much faster version
    of `calculate_varint_old`.

    Single value input only!
    Handling multiple inputs at once didn't enhance performance at all. Just do a
    Python for loop. (TW 2022-01-21)

    Parameters:
    -----------
    a_obs: float
        mean age for the inverse gaussian distribution. Must be positive, otherwise a
        value of 5 will be returned.
    rom: float
        ratio of moments for the inverse gaussian distribution
    int_perc: int or float
        percentage of age spectrum to cover (< 100).

    """
    # a_obs: mean age first guess
    # rom: ratio of moments; has no default values
    # int_perc: the percentage of the age spectrum to be covered by the interval

    # check if a_obs is positiv! If its negative, return a value of 5
    if not a_obs > 0:
        return 5

    G = Calculate_AgeSpectrum_1D(a_obs, rom)
    integral = integrate.cumulative_trapezoid(G[:, 1], x=G[:, 0], initial=0)
    where_greater_than_int_perc = integral > (int_perc / 100)
    try:
        t_int = G[where_greater_than_int_perc, 0][0]
    except IndexError as IE:
        print("IndexError in _calculate_varint:")
        print("G.shape:", G.shape)
        print("where_greater_than_int_perc.sum():", where_greater_than_int_perc.sum())
        print("a_obs:", a_obs)
        print("rom:", rom)
        print("integral:", integral)
        raise IE

    return t_int


# %%
def _handle_invalid(func):
    """Decorate fit_referenceTimeSeries_for_many_t_obs in order to handle invalid values."""

    def inner(t_ref, c_ref, t_obs, t_int=30, **kwargs):
        t_obs = np.asarray(t_obs).flatten()
        t_int = np.asarray(t_int).flatten()
        w_finite_t_obs = ~np.isnan(t_obs)
        try:
            if len(t_int) == len(t_obs):
                # if called from fit_referenceTimeSeries_for_many_t_obs_int_prec,
                # then individual t_int values for each t_obs need to get rid of invalid values.
                t_int = t_int[w_finite_t_obs]
        except TypeError as TE:
            print("TypeError in _handle_invalid: {}".format(TE))
            print("type(t_int): {}".format(type(t_int)))
            print("type(t_obs): {}".format(type(t_obs)))
            print("t_int: {}".format(t_int))
            print("t_obs: {}".format(t_obs))
            raise TE

        # call function (fit_referenceTimeSeries_for_many_t_obs) only with filtered input
        params_crude, tt_crude, c_ref_crude = func(
            t_ref, c_ref, t_obs[w_finite_t_obs], t_int, **kwargs
        )

        # expand function output to original array dimensions (reintroducing
        #   invalid values)
        params_crude_all = np.ones((params_crude.shape[0], t_obs.shape[0])) * np.nan
        params_crude_all[:, w_finite_t_obs] = params_crude

        tt_crude_all = tt_crude
        # tt_crude_all = np.ones((tt_crude.shape[0], t_obs.shape[0])) * np.nan
        # tt_crude_all[:, w_finite_t_obs] = tt_crude

        c_ref_crude_all = c_ref_crude
        # c_ref_crude_all = np.ones((c_ref_crude.shape[0], t_obs.shape[0])) * np.nan
        # c_ref_crude_all[:, w_finite_t_obs] = c_ref_crude

        return params_crude_all, tt_crude_all, c_ref_crude_all

    return inner


@_handle_invalid
def fit_referenceTimeSeries_for_many_t_obs(t_ref, c_ref, t_obs, t_int=30, polyorder=2):
    """Apply a polynomial fit to a reference time series for different intervals at once.

    Parameters:
    -----------
    t_ref: np.array of floats
        reference times, monotonic increasing. Designed for fractional years.
    c_ref: np.array of floats
        reference concentrations
    t_obs: np.array of floats or float
        dates of observations. For each value in t_obs, an individual set of
        fit parameters will be calculated.
    t_int: float or np.array of floats, optional
        time span before every t_obs that will be considered for the fit. if t_int is
        a single value, it is used for every t_obs. The default is 30 (years).
    polyorder: int, optional
        order of the polynomial fit. The default is 2. Higher polynoms not tested.

    Returns:
    -----------
    params_crude: np.array
        fit coefficients from np.polynomial.polynomial.polyfit()
    tt_crude: np.array or list of np.arrays
        t_ref relative to t_obs
    c_ref_crude: np.array or list of np.arrays
        for each t_obs the corresponding selected c_ref section
    """
    # handle input values
    t_ref = np.asarray(t_ref).flatten()
    c_ref = np.asarray(c_ref).flatten()
    t_obs = np.asarray(t_obs).reshape(-1, 1)  # for broadcasting
    t_int = np.asarray(t_int).reshape(-1, 1)

    # shift t_ref to tt, so that t_obs = 0 for each t_obs
    tt = t_ref - t_obs  # broadcasting

    # select for each t_obs the shifted time series tt between -1 and t_int
    try:
        wt = np.where(np.logical_and((tt < 1), (tt > -t_int)))
    except ValueError as VE:
        print("ValueError in fit_referenceTimeSeries_for_many_t_obs:")
        print(VE)
        print("tt.shape: {}".format(tt.shape))
        print("t_int.shape: {}".format(t_int.shape))
        raise (VE)
    # wt = np.where((tt < 1) & (tt > -t_int) | np.isnan(tt))  # in case t_obs contains nans... ok, it should not!

    # check, if the number of elements selected for each t_obs is the same
    # count all values for each t_obs
    # np.unique seems to be far faster than pd.groupby!
    wta = np.asarray(wt)
    wta_dim, wta_split, wta_count = np.unique(
        wta[0, :], return_counts=True, return_index=True
    )

    wt_all_same_length = np.all(wta_count == wta_count[0])

    if wt_all_same_length:
        # take the SAME tt for all c_refs in order to be able to fit all at once (5 times faster)
        try:
            tt_crude = tt[wt].reshape(len(t_obs), -1)[0, :]
        except (IndexError, ValueError) as IVE:
            print("IndexError or ValueError in fit_referenceTimeSeries_for_many_t_obs:")
            print("t_ref.shape:", t_ref.shape)
            print("c_ref.shape:", c_ref.shape)
            print("t_obs.shape:", t_obs.shape)
            print("t_int.shape:", t_int.shape)
            print("tt.shape:", tt.shape)
            raise (IVE)

        # copy c_ref for every t_obs, resulting in a 2D-array
        c_ref_crude = c_ref.reshape(1, -1) * np.ones(t_obs.shape)

        # select the right subset of c_ref for each t_obs
        c_ref_crude = c_ref_crude[wt].reshape(len(t_obs), -1)

        # fit all c_refs at once!
        params_crude = np.polynomial.polynomial.polyfit(
            tt_crude, c_ref_crude.T, polyorder
        )

    else:
        # number of values in tt and c_ref is not the same for each t_obs
        # - > get a list of subarrays from wt (for each t_obs the right indices to get shifted time series between -1 and t_int)
        locs = np.split(wta[1, :], wta_split[1:])
        # -> loop through the arrays (slower than doing all at once, but necessary for different tt arrays)
        params_crude = []
        tt_crude = []
        c_ref_crude = []
        for loc_i, wta_dim_i in zip(locs, list(wta_dim)):
            # select exact shifted time series tt_i relative to t_obs i
            tt_i = (tt * np.ones(t_int.shape))[wta_dim_i, loc_i]

            # select nearest c_ref values according to tt_i
            c_ref_i = c_ref[loc_i]

            # fit the reference time series
            params_i = np.polynomial.polynomial.polyfit(tt_i, c_ref_i.T, polyorder)
            params_crude.append(params_i)
            tt_crude.append(tt_i)
            c_ref_crude.append(c_ref_i)

        params_crude = np.asarray(params_crude).T

    return params_crude, tt_crude, c_ref_crude


# %%
def fit_referenceTimeSeries_for_many_t_obs_int_prec(
    t_ref,
    c_ref,
    t_obs,
    c_obs,
    polyorder=2,
    int_perc=98,
    t_int_init=30,
    rom=1.2,
    fract_exTR=0,
    t_shift_exTR=0,
    t_shift_TR=0,
    t_int_min=5,
):
    """Find optimized fit interval and fit polynomial to reference time series.

    1.  Calculate first guess mean age from a quadratic fitted tracer.
    1.  Calculate an age spectrum from an inverse gaussian function using the first
        guess mean age.
    2.  Select the time span that covers a user defined percentage of this age
        spectrums area.
    3.  Apply a polynomial fit to the reference time series.

    Supports considering extra tropical input into the stratosphere.
    Supports processing multiple observations at once.

    Parameters:
    ------------
    t_ref: np.array of floats
        reference times, monotonic increasing. Designed for fractional years.
    c_ref: np.array of floats
        reference concentrations
    t_obs: np.array of floats or float
        dates of observations. For each value in t_obs, an individual set of
        fit parameters will be calculated.
    c_obs: np.array of floats or float
        observed concentration(s). Will be used to calculate a first guess mean age.
    polyorder: int, optional
        order of the polynomial fit. The default is 2. Higher polynoms not tested.
    int_perc: float, optional
        percentage of the first guess age spectrum to be covered. The default is 98.
    t_int_init: float or np.array of floats, optional
        time span before every t_obs that will be considered for the first guess
        mean age. if t_int_init is a single value, it is used for every t_obs.
        The default is 30 (years).
    rom: float or np.array of floats, optional
        ratio of moments for the first guess age spectrum. The default is 1.2.
    fract_exTR: np.array of floats or float, optional
        extra tropical origin fraction, that will be considered to calculate the
        first guess mean age. The default is 0.
    t_shift_exTR: float or np.array of floats, optional
        time shift for c_ref between extra tropical entry region and t_ref.
        The default is 0.
    t_shift_TR: float or np.array of floats, optional
        time shift for c_ref between tropical entry region and t_ref.
        The default is 0.
    t_int_min: float, optional
        minimum time span to consider for polynomial fit. The default is 5.

    Returns:
    -----------
    params_crude: np.array
        fit coefficients from np.polynomial.polynomial.polyfit()
    tt_crude: list of np.arrays
        t_ref relative to t_obs
    c_ref_crude: list of np.arrays
        for each t_obs the corresponding selected c_ref section
    """
    # ###handle input
    t_ref = np.asarray(t_ref).flatten()
    c_ref = np.asarray(c_ref).flatten()
    t_obs = np.asarray(t_obs).flatten()
    c_obs = np.asarray(c_obs).flatten()
    rom = np.asarray(rom).flatten()
    if len(rom) == 1:
        # take the same ratio of moments for every observation
        rom = np.ones(c_obs.shape) * rom
    if len(t_obs) == 1:
        t_obs = np.ones(c_obs.shape) * t_obs

    # ##get individual fit intervals
    # calculate preliminary age
    age_init = calculate_age_fit_fracts(
        t_ref=t_ref,
        c_ref=c_ref,
        t_obs=t_obs,
        c_obs=c_obs,
        rom=rom,
        fract_exTR=fract_exTR,
        t_shift_exTR=t_shift_exTR,
        t_shift_TR=t_shift_TR,
        t_int=t_int_init,
        int_perc=None,
        comment=False,
    )

    # calculate fit intervals using preliminary mean age and an inverse Gaussian distribution
    fitint = calculate_varint(a_obs=age_init, rom=rom, int_perc=int_perc)

    # make sure, that at least t_int_min years of data are fitted
    fitint[fitint < t_int_min] = t_int_min

    # get fit parameters for the individual fit intervals
    try:
        params, tt, c_ref_crude = fit_referenceTimeSeries_for_many_t_obs(
            t_ref, c_ref, t_obs, t_int=fitint, polyorder=polyorder
        )
    except IndexError as IE:
        print("IndexError in fit_referenceTimeSeries_for_many_t_obs_int_prec:")
        print("t_ref.shape:", t_ref.shape)
        print("c_ref.shape:", c_ref.shape)
        print("t_obs.shape:", t_obs.shape)
        print("c_obs.shape:", c_obs.shape)
        print("fitint.shape:", fitint.shape)
        print("rom.shape:", rom.shape)
        raise (IE)
    return params, tt, c_ref_crude


# %%
def calculate_age_lin_fit_fracts(
    t_ref,
    c_ref,
    t_obs,
    c_obs,
    fract_exTR,
    t_shift_exTR,
    t_shift_TR,
    t_int=5,
    int_perc=None,
):
    """Calculate mean age assuming an ideal linear increasing tracer.

    Supports considering extra tropical input into the stratosphere.
        - based on the assumption, that each entry region's reference time series
          can be described by the same linear reference time series simply shifted
          by a constant time offset.
    Supports processing multiple observations at once.

    Parameters:
    ------------
    t_ref: np.array of floats
        reference times, monotonic increasing. Designed for fractional years.
    c_ref: np.array of floats
        reference concentrations
    t_obs: np.array of floats or float
        dates of observations. If it's a single value, it will be used for each
        value in c_obs.
    c_obs: np.array of floats or float
        observed concentration(s).
    fract_exTR: np.array of floats or float, optional
        extra tropical origin fraction. Tropical origin fraction will be calculated
        by subtracting fract_exTR from 1. The default is 0.
    t_shift_exTR: float or np.array of floats, optional
        time shift for c_ref between extra tropical entry region and t_ref.
        The default is 0.
    t_shift_TR: float or np.array of floats, optional
        time shift for c_ref between tropical entry region and t_ref.
        The default is 0.
    t_int: float or np.array of floats, optional
        time span before every t_obs that will be considered for the linear fit.
        If t_int is a single value, it is used for every t_obs.
        If int_perc is not None, t_int will be used to calculate the first guess
        mean age. The default is 5 (years).
    int_perc: float, optional
        percentage of a first guess age spectrum to be covered. If not None,
        will be used to get optimized fit intervals for each observation. The default is None.

    Returns:
    ----------
    mean_age: np.array
    """
    # linear fit, t_ref is shifted so that t_obs = 0
    if int_perc:
        params, tt, c_ref_crude = fit_referenceTimeSeries_for_many_t_obs_int_prec(
            t_ref=t_ref,
            c_ref=c_ref,
            t_obs=t_obs,
            c_obs=c_obs,
            polyorder=1,
            int_perc=int_perc,
            t_int_init=t_int,
            rom=1.2,
            fract_exTR=fract_exTR,
            t_shift_exTR=t_shift_exTR,
            t_shift_TR=t_shift_TR,
        )
    else:
        params, tt, c_ref_crude = fit_referenceTimeSeries_for_many_t_obs(
            t_ref, c_ref, t_obs, t_int=t_int, polyorder=1
        )
    a = params[0, :]
    b = params[1, :]

    # get tropcial origin fraction
    fract_TR = 1 - fract_exTR

    # calculate mean time shift of t_ref, weighted by origin fractions
    t_m = fract_exTR * t_shift_exTR + fract_TR * t_shift_TR

    # calculate mean age, with t = 0
    mean_age = (a - c_obs) / b + t_m

    # check, if input is a single number. Then ouput a single number
    if check_single_input(c_obs):
        mean_age = mean_age[0]

    return mean_age


# %%
def calculate_age_lin_fit(
    t_ref, c_ref, t_obs, c_obs, t_shift_ref=0, t_int=5, int_perc=None
):
    """Calculate mean age assuming an ideal linear increasing tracer.

    Supports processing multiple observations at once.

    Parameters:
    ------------
    t_ref: np.array of floats
        reference times, monotonic increasing. Designed for fractional years.
    c_ref: np.array of floats
        reference concentrations
    t_obs: np.array of floats or float
        dates of observations. If it's a single value, it will be used for each
        value in c_obs.
    c_obs: np.array of floats or float
        observed concentration(s).
    t_shift_ref: float or np.array of floats, optional
        time shift for c_ref between tropical entry region and t_ref.
        The default is 0.
    t_int: float or np.array of floats, optional
        time span before every t_obs that will be considered for the linear fit.
        If t_int is a single value, it is used for every t_obs.
        If int_perc is not None, t_int will be used to calculate the first guess
        mean age. The default is 5 (years).
    int_perc: float, optional
        percentage of a first guess age spectrum to be covered. If not None,
        will be used to get optimized fit intervals for each observation. The default is None.

    Returns:
    ----------
    mean_age: np.array
    """
    mean_age = calculate_age_lin_fit_fracts(
        t_ref,
        c_ref,
        t_obs,
        c_obs,
        fract_exTR=0,
        t_shift_exTR=0,
        t_shift_TR=t_shift_ref,
        t_int=t_int,
        int_perc=int_perc,
    )
    return mean_age


# %%
def calculate_age_fit_fracts(
    t_ref,
    c_ref,
    t_obs,
    c_obs,
    rom,
    fract_exTR,
    t_shift_exTR,
    t_shift_TR,
    t_int=30,
    int_perc=None,
    comment=True,
    deseas=False,
):
    """Calculate mean age assuming an ideal quadratic increasing tracer.

    Supports considering extra tropical input into the stratosphere.
        - based on the assumption, that each entry region's reference time series
          can be described by the same quadratic reference time series simply shifted
          by a constant time offset.
        - assumes one age spectrum G with a ratio of moments, that is folded with
          the weighted mean of two entry regions time series.
    Supports processing multiple observations at once.

    Parameters:
    ------------
    t_ref: np.array of floats
        reference times, monotonic increasing. Designed for fractional years.
    c_ref: np.array of floats
        reference concentrations
    t_obs: np.array of floats or float
        dates of observations. If it's a single value, it will be used for each
        value in c_obs.
    c_obs: np.array of floats or float
        observed concentration(s).
    rom: float or np.array of floats
        ratio of moments for the age spectrum.
    fract_exTR: np.array of floats or float, optional
        extra tropical origin fraction. Tropical origin fraction will be calculated
        by subtracting fract_exTR from 1. The default is 0.
    t_shift_exTR: float or np.array of floats, optional
        time shift for c_ref between extra tropical entry region and t_ref.
        The default is 0.
    t_shift_TR: float or np.array of floats, optional
        time shift for c_ref between tropical entry region and t_ref.
        The default is 0.
    t_int: float or np.array of floats, optional
        time span before every t_obs that will be considered for the quadratic fit.
        If t_int is a single value, it is used for every t_obs.
        If int_perc is not None, t_int will be used to calculate the first guess
        mean age. The default is 30 (years).
    int_perc: float, optional
        percentage of a first guess age spectrum to be covered. If not None,
        will be used to get optimized fit intervals for each observation. The default is None.
    comment: bool, optional
        print additional information and warning messages. The default is True.

    Returns:
    ---------
    mean_age_correct: np.array
        the p-q-solution that is closer to 0 and therefore reasonable.
    """
    # quadratic fit, t_ref is shifted so that t_obs = 0
    c_ref_cut = handle_deseas(t_ref=t_ref, c_ref=c_ref, deseas=deseas)
    if int_perc:
        params, tt, c_ref_crude = fit_referenceTimeSeries_for_many_t_obs_int_prec(
            t_ref=t_ref,
            c_ref=c_ref_cut,
            t_obs=t_obs,
            c_obs=c_obs,
            polyorder=2,
            int_perc=int_perc,
            t_int_init=t_int,
            rom=rom,
            fract_exTR=fract_exTR,
            t_shift_exTR=t_shift_exTR,
            t_shift_TR=t_shift_TR,
        )
    else:
        params, tt, c_ref_crude = fit_referenceTimeSeries_for_many_t_obs(
            t_ref, c_ref_cut, t_obs, t_int=t_int, polyorder=2
        )
    a = params[0, :]
    b = params[1, :]
    c = params[2, :]

    # get tropcial fraction
    fract_TR = 1 - fract_exTR
    t_m = fract_exTR * t_shift_exTR + fract_TR * t_shift_TR

    # p-q solution, with t = 0
    p = -b / c + 2 * rom - 2 * t_m
    q = (
        (a + b * t_m - c_obs) / c
        + fract_exTR * t_shift_exTR ** 2
        + fract_TR * t_shift_TR ** 2
    )

    pqRootTerm = (p / 2) ** 2 - q

    # Check for invalid values under the root
    w_pqRootNegative = pqRootTerm < 0
    if (not np.all(~w_pqRootNegative)) & comment:
        print(
            "calculate_age_fit_fracts(): {} negative value(s) in pqRootTerm detected:".format(
                np.sum(w_pqRootNegative)
            )
        )

        for _pqRootTerm, _t_obs, _c_obs in zip(
            pqRootTerm[w_pqRootNegative],
            t_obs[w_pqRootNegative],
            c_obs[w_pqRootNegative],
        ):
            print(
                "t_obs: {:6.1f}, c_obs: {:6.2f}, pqRootTerm: {:8.1f}\n".format(
                    _t_obs, _c_obs, _pqRootTerm
                )
            )

        print("NaN values will be returned for those observations.\n")

    pqRootTerm[w_pqRootNegative] = np.nan
    mean_age_P = -p / 2 + np.sqrt(pqRootTerm)
    mean_age_N = -p / 2 - np.sqrt(pqRootTerm)
    # select the solution, that is closer to zero
    mean_age_correct = np.where(
        abs(mean_age_P) < abs(mean_age_N), mean_age_P, mean_age_N
    )

    # check, if input is a single number. Then ouput a single number
    if check_single_input(c_obs):
        mean_age_correct = mean_age_correct[0]

    return mean_age_correct


# %%
def calculate_age_fit(
    t_ref,
    c_ref,
    t_obs,
    c_obs,
    rom,
    t_shift_ref=0,
    t_int=30,
    int_perc=None,
    comment=True,
    deseas=False,
):
    """Calculate mean age assuming an ideal quadratic increasing tracer.

    Supports processing multiple observations at once.

    Parameters:
    ------------
    t_ref: np.array of floats
        reference times, monotonic increasing. Designed for fractional years.
    c_ref: np.array of floats
        reference concentrations
    t_obs: np.array of floats or float
        dates of observations. If it's a single value, it will be used for each
        value in c_obs.
    c_obs: np.array of floats or float
        observed concentration(s).
    rom: float or np.array of floats
        ratio of moments for the age spectrum.
    t_shift_ref: float or np.array of floats, optional
        time shift for c_ref between entry region and t_ref.
        The default is 0.
    t_int: float or np.array of floats, optional
        time span before every t_obs that will be considered for the quadratic fit.
        If t_int is a single value, it is used for every t_obs.
        If int_perc is not None, t_int will be used to calculate the first guess
        mean age. The default is 30 (years).
    int_perc: float, optional
        percentage of a first guess age spectrum to be covered. If not None,
        will be used to get optimized fit intervals for each observation. The default is None.
    comment: bool, optional
        print additional information and warning messages. The default is True.

    Returns:
    ---------
    mean_age_correct: np.array
        the p-q-solution that is closer to 0 and therefore reasonable.
    """
    mean_age = calculate_age_fit_fracts(
        t_ref,
        c_ref,
        t_obs,
        c_obs,
        rom,
        fract_exTR=0,
        t_shift_exTR=0,
        t_shift_TR=t_shift_ref,
        t_int=t_int,
        int_perc=int_perc,
        comment=comment,
        deseas=deseas,
    )
    return mean_age


# %%#############################################################
# define decorated functions, that handle simultaneous input of SF6 and CFC12 to obtain
# more precise SF6 data that is used as c_obs
# Details on smoothy can be found within resampling.py
#################################################################
@smoothy
def calculate_age_lin_fit_fracts_smoothy(
    c_obsx, c_obsy, t_ref, c_ref, t_obs, *args, **kwargs
):
    return calculate_age_lin_fit_fracts(t_ref, c_ref, t_obs, c_obsy, *args, **kwargs)


@smoothy
def calculate_age_fit_fracts_smoothy(
    c_obsx, c_obsy, t_ref, c_ref, t_obs, *args, **kwargs
):
    return calculate_age_fit_fracts(t_ref, c_ref, t_obs, c_obsy, *args, **kwargs)


@smoothy
def calculate_age_lin_fit_smoothy(c_obsx, c_obsy, t_ref, c_ref, t_obs, *args, **kwargs):
    return calculate_age_lin_fit(t_ref, c_ref, t_obs, c_obsy, *args, **kwargs)


@smoothy
def calculate_age_fit_smoothy(c_obsx, c_obsy, t_ref, c_ref, t_obs, *args, **kwargs):
    return calculate_age_fit(t_ref, c_ref, t_obs, c_obsy, *args, **kwargs)


@smoothy
def fit_referenceTimeSeries_for_many_t_obs_int_prec_smoothy(
    c_obsx, c_obsy, t_ref, c_ref, t_obs, *args, **kwargs
):
    return fit_referenceTimeSeries_for_many_t_obs_int_prec(
        t_ref, c_ref, t_obs, c_obsy, *args, **kwargs
    )
