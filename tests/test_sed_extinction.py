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


def test_extinction_wavelength_dependent():
    """Extinction should be stronger at shorter wavelengths"""
    wrapper_no = _make_wrapper(ebv=0.0)
    wrapper_yes = _make_wrapper(ebv=0.2)

    res_no = wrapper_no.compute_sed(phases=[0.25])
    res_yes = wrapper_yes.compute_sed(phases=[0.25])

    wl = res_no["wavelengths"]
    f_no = res_no["fluxes"][0]
    f_yes = res_yes["fluxes"][0]

    mask = np.isfinite(wl) & np.isfinite(f_no) & np.isfinite(f_yes)
    wl = wl[mask]
    f_no = f_no[mask]
    f_yes = f_yes[mask]

    # Find bluest and reddest filters
    blue_idx = np.argmin(wl)
    red_idx = np.argmax(wl)

    # Compute attenuation ratios
    blue_ratio = f_yes[blue_idx] / f_no[blue_idx]
    red_ratio = f_yes[red_idx] / f_no[red_idx]

    # Blue should be more attenuated (smaller ratio)
    assert blue_ratio < red_ratio, "Blue filters should have stronger extinction"


def test_ebv_zero_is_noop():
    """E(B-V) = 0 should not apply extinction"""
    wrapper = _make_wrapper(ebv=0.0)
    res = wrapper.compute_sed(phases=[0.25])

    # Should complete without error
    assert res["fluxes"].shape[0] == 1
    assert np.any(np.isfinite(res["fluxes"][0]))


def test_ebv_none_is_noop():
    """E(B-V) = None should not apply extinction"""
    bundle = phoebe.default_binary()
    bundle.set_value('teff@primary', 6000)
    bundle.set_value('teff@secondary', 5000)
    bundle.set_value('requiv@primary', 1.0)
    bundle.set_value('requiv@secondary', 0.8)

    wrapper = PhoebeWrapper(
        bundle=bundle,
        distance=100.0,
        ebv=None,
    )
    res = wrapper.compute_sed(phases=[0.25])

    # Should complete without error
    assert res["fluxes"].shape[0] == 1
    assert np.any(np.isfinite(res["fluxes"][0]))


def test_negative_ebv_raises_error():
    """E(B-V) < 0 should raise ValueError"""
    wrapper = _make_wrapper(ebv=-0.1)

    with pytest.raises(ValueError, match="non-negative"):
        wrapper.compute_sed(phases=[0.25])
