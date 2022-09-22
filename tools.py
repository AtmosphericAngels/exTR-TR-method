"""
@Filename: tools.py

@Author: Thomas Wagenhäuser, IAU
@Date:   2022-02-08T10:23:49+01:00
@Email:  wagenhaeuser@iau.uni-frankfurt.de

"""

# from .ccgcrv.ccg_filter import ccgFilter
# %%


def handle_deseas(t_ref, c_ref, deseas=True):
    """Handle deseas keyword externally in order to gain readability.

    NOT IMPLEMENTED IN THIS PUBLIC VERSION
    """
    # if deseas:
    #     c_ref_cut = ccgFilter(t_ref, c_ref).getTrendValue(t_ref)
    # else:
    #     c_ref_cut = c_ref
    # return c_ref_cut
    print("ccgFilter from NOAA not implemented in this public version")
    print(
        "It is available at ftp://ftp.cmdl.noaa.gov/user/thoning/ccgcrv/2021-07-09, last access: 2021-07-09"
    )
    return c_ref


def check_single_input(c_obs):
    """Check if observation input is a single number."""
    if isinstance(c_obs, float) or isinstance(c_obs, int):
        single_output = True
    else:
        single_output = False
    return single_output
