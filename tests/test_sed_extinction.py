import numpy as np
import pytest
import phoebe

from ebsbi.phoebe_wrapper import PhoebeWrapper


def _make_wrapper(ebv):
    """Create PhoebeWrapper for testing with standard binary system"""
    bundle = phoebe.default_binary()

    # Set basic parameters for stable system
    bundle.set_value('teff@primary', 6000)
    bundle.set_value('teff@secondary', 5000)
    bundle.set_value('requiv@primary', 1.0)
    bundle.set_value('requiv@secondary', 0.8)

    return PhoebeWrapper(
        bundle=bundle,
        distance=100.0,
        ebv=ebv,
    )


def test_extinction_reduces_flux():
    """Extinction should reduce flux for all filters"""
    wrapper_no = _make_wrapper(ebv=0.0)
    wrapper_yes = _make_wrapper(ebv=0.2)

    # Compute SEDs at same phase
    res_no = wrapper_no.compute_sed(phases=[0.25])
    res_yes = wrapper_yes.compute_sed(phases=[0.25])

    wl = res_no["wavelengths"]
    f_no = res_no["fluxes"][0]   # [23]
    f_yes = res_yes["fluxes"][0]  # [23]

    # Only check filters that are valid in both
    mask = np.isfinite(wl) & np.isfinite(f_no) & np.isfinite(f_yes)
    wl = wl[mask]
    f_no = f_no[mask]
    f_yes = f_yes[mask]

    # Extinction should not increase flux
    assert np.all(f_yes <= f_no), "Extinction should reduce flux for all filters"
