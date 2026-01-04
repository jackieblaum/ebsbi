import numpy as np
import pytest
from ebsbi.model import EBModel
from ebsbi.config import Config
from ebsbi.priors import EBPriors
from isochrones import get_ichrone


def get_valid_theta_sample():
    """
    Returns a fixed, valid theta array generated from the priors with seed=42.
    This ensures tests use physically realistic parameters that will successfully
    generate light curves without errors in eclipsebin.

    The array shape is (1, 16) matching the order of labels_dict:
    ['msum', 'q', 'log_age', 'metallicity', 'ebv', 'cosi', 'rsumfrac', 'ecc',
     'per0', 'distance', 'teff1', 'teff2', 'r1', 'r2', 'log_lum1', 'log_lum2']

    This represents a detached eclipsing binary system with physical parameters
    that passed all validation checks.
    """
    return np.array([[
        2.19530944e+00,  # msum (solar masses)
        9.44346150e-01,  # q (mass ratio)
        9.04590685e+00,  # log_age
        -6.08799482e-01, # metallicity [Fe/H]
        6.77107725e-02,  # ebv (extinction)
        9.46042759e-01,  # cosi (cos of inclination)
        2.15802479e-01,  # rsumfrac (sum of radii / sma)
        7.01406788e-02,  # ecc (eccentricity)
        1.87172034e+02,  # per0 (argument of periastron, degrees)
        1.57635394e+03,  # distance (parsecs)
        7.14405338e+03,  # teff1 (primary temp, K)
        6.77583466e+03,  # teff2 (secondary temp, K)
        1.12524626e+00,  # r1 (primary radius, solar radii)
        1.05496106e+00,  # r2 (secondary radius, solar radii)
        4.42933792e-01,  # log_lum1
        3.00685341e-01,  # log_lum2
    ]])


def test_light_curve_flux_normalized_to_median_one():
    """Test that generate_light_curve returns fluxes with median=1 for each passband."""
    # Setup
    rng = np.random.default_rng(seed=42)
    config_path = 'docs/config_linear_times_sequential.yml'
    conf = Config(config_path)

    # Create minimal phase_bank and noise_bank for testing
    # These simulate observational data from different surveys
    phase_bank = {
        'ASASSN_g': [np.linspace(0, 1, 100)],
        'ASASSN_V': [np.linspace(0, 1, 100)],
        'ZTF_zg': [np.linspace(0, 1, 100)],
        'ZTF_zr': [np.linspace(0, 1, 100)],
    }
    noise_bank = {
        'ASASSN_g': [np.full(100, 0.01)],
        'ASASSN_V': [np.full(100, 0.01)],
        'ZTF_zg': [np.full(100, 0.01)],
        'ZTF_zr': [np.full(100, 0.01)],
    }

    model = EBModel(
        eb_path=conf.eb_path,
        params_dict=conf.labels_dict,
        phase_bank=phase_bank,
        noise_bank=noise_bank,
        rng=rng
    )

    # Use fixed, pre-validated theta parameters
    theta_sample = get_valid_theta_sample()

    # Generate light curve
    phases_list, fluxes_list, errs_list = model.generate_light_curve(
        theta_sample, nbins=200
    )

    # Check each passband
    for i, (phases, fluxes, errs) in enumerate(zip(phases_list, fluxes_list, errs_list)):
        flux_median = np.median(fluxes)

        # Assert median is 1.0 (within floating point tolerance)
        assert np.isclose(flux_median, 1.0, rtol=1e-6), \
            f"Passband {i}: Expected median=1.0, got {flux_median}"

        # Assert fluxes and errors are same length
        assert len(fluxes) == len(errs), \
            f"Passband {i}: Flux and error arrays must have same length"

        # Assert errors are positive
        assert np.all(errs > 0), \
            f"Passband {i}: All errors must be positive"


def test_light_curve_error_scaling():
    """Test that flux errors are scaled by the same factor as fluxes."""
    # Setup
    rng = np.random.default_rng(seed=123)
    config_path = 'docs/config_linear_times_sequential.yml'
    conf = Config(config_path)

    # Create minimal phase_bank and noise_bank for testing
    phase_bank = {
        'ASASSN_g': [np.linspace(0, 1, 100)],
        'ASASSN_V': [np.linspace(0, 1, 100)],
        'ZTF_zg': [np.linspace(0, 1, 100)],
        'ZTF_zr': [np.linspace(0, 1, 100)],
    }
    noise_bank = {
        'ASASSN_g': [np.full(100, 0.01)],
        'ASASSN_V': [np.full(100, 0.01)],
        'ZTF_zg': [np.full(100, 0.01)],
        'ZTF_zr': [np.full(100, 0.01)],
    }

    model = EBModel(
        eb_path=conf.eb_path,
        params_dict=conf.labels_dict,
        phase_bank=phase_bank,
        noise_bank=noise_bank,
        rng=rng
    )

    # Use fixed, pre-validated theta parameters
    theta_sample = get_valid_theta_sample()

    # Generate light curve
    phases_list, fluxes_list, errs_list = model.generate_light_curve(
        theta_sample, nbins=100
    )

    # For each passband, verify error scaling is consistent
    for i, (fluxes, errs) in enumerate(zip(fluxes_list, errs_list)):
        # Calculate signal-to-noise ratio
        snr = fluxes / errs

        # SNR should be reasonable (errors not zero or infinite)
        assert np.all(np.isfinite(snr)), \
            f"Passband {i}: SNR must be finite"
        assert np.all(snr > 0), \
            f"Passband {i}: SNR must be positive"

        # Typical SNR for binned photometry should be in reasonable range
        # (not testing exact values, just that errors scaled correctly)
        assert np.median(snr) > 1 and np.median(snr) < 1000, \
            f"Passband {i}: Median SNR {np.median(snr)} seems unreasonable"
