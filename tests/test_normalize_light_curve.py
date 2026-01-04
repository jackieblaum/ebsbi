"""
Test suite for light curve flux normalization.

TDD Phase: RED (Task 1)
-----------------------
These tests are EXPECTED TO FAIL until the normalization implementation is added
in Task 2. This is proper TDD practice:
- Task 1 (RED): Write failing tests that define requirements
- Task 2 (GREEN): Implement code to make tests pass

The tests verify that generated light curves have:
1. Median flux normalized to 1.0 for each passband
2. Flux errors scaled by the same normalization factor
3. All errors remain positive and finite after normalization
"""
import numpy as np
from pathlib import Path
import pytest
from ebsbi.model import EBModel
from ebsbi.config import Config

# Test configuration constants
NUM_PHASE_POINTS = 100
NOISE_LEVEL = 1e-8  # Appropriate noise level for PHOEBE flux scale (~1e-7)
CONFIG_FILE = 'config_linear_times_sequential.yml'


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


@pytest.fixture
def eb_model_setup():
    """
    Pytest fixture that provides common setup for EBModel tests.

    Returns a tuple of (config, phase_bank, noise_bank) that can be used
    to initialize an EBModel instance with consistent test data.

    The config path is resolved relative to the repository root (docs/ directory).
    """
    # Resolve config path relative to test file location
    test_dir = Path(__file__).parent
    repo_root = test_dir.parent
    config_path = repo_root / 'docs' / CONFIG_FILE

    conf = Config(str(config_path))

    # Create minimal phase_bank and noise_bank for testing
    # These simulate observational data from different surveys
    phase_bank = {
        'ASASSN_g': [np.linspace(0, 1, NUM_PHASE_POINTS)],
        'ASASSN_V': [np.linspace(0, 1, NUM_PHASE_POINTS)],
        'ZTF_zg': [np.linspace(0, 1, NUM_PHASE_POINTS)],
        'ZTF_zr': [np.linspace(0, 1, NUM_PHASE_POINTS)],
    }
    noise_bank = {
        'ASASSN_g': [np.full(NUM_PHASE_POINTS, NOISE_LEVEL)],
        'ASASSN_V': [np.full(NUM_PHASE_POINTS, NOISE_LEVEL)],
        'ZTF_zg': [np.full(NUM_PHASE_POINTS, NOISE_LEVEL)],
        'ZTF_zr': [np.full(NUM_PHASE_POINTS, NOISE_LEVEL)],
    }

    return conf, phase_bank, noise_bank


def test_light_curve_flux_normalized_to_median_one(eb_model_setup):
    """Test that generate_light_curve returns fluxes with median=1 for each passband."""
    # Unpack fixture
    conf, phase_bank, noise_bank = eb_model_setup

    # Setup RNG for reproducibility
    rng = np.random.default_rng(seed=42)

    # Create model instance with fixture data
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


def test_light_curve_error_scaling(eb_model_setup):
    """Test that flux errors are scaled by the same factor as fluxes."""
    # Unpack fixture
    conf, phase_bank, noise_bank = eb_model_setup

    # Setup RNG for reproducibility
    rng = np.random.default_rng(seed=123)

    # Create model instance with fixture data
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


def test_nbi_simulator_with_normalized_fluxes(eb_model_setup):
    """Test that nbi_simulator correctly uses normalized light curves."""
    import tempfile
    import os

    # Unpack fixture
    conf, phase_bank, noise_bank = eb_model_setup

    # Setup RNG for reproducibility
    rng = np.random.default_rng(seed=42)

    # Create model instance with fixture data
    model = EBModel(
        eb_path=conf.eb_path,
        params_dict=conf.labels_dict,
        phase_bank=phase_bank,
        noise_bank=noise_bank,
        rng=rng
    )

    # Use fixed, pre-validated theta parameters
    theta_sample = get_valid_theta_sample()
    theta = theta_sample[0]  # Extract 1D array from (1, 16) shape

    # Run simulator
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        temp_path = f.name

    try:
        success = model.nbi_simulator(theta, temp_path)
        assert success, "nbi_simulator should succeed"

        # Load and verify
        x_obj = np.load(temp_path, allow_pickle=True).item()

        # Verify fluxes are normalized
        for i, lc_flux in enumerate(x_obj['lc']):
            flux_median = np.median(lc_flux)
            assert np.isclose(flux_median, 1.0, rtol=1e-6), \
                f"Passband {i}: nbi_simulator flux median should be 1.0, got {flux_median}"

        # Verify errors are present and positive
        for i, lc_err in enumerate(x_obj['lc_err']):
            assert len(lc_err) == len(x_obj['lc'][i]), \
                f"Passband {i}: errors and fluxes must have same length"
            assert np.all(lc_err > 0), \
                f"Passband {i}: all errors must be positive"

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
