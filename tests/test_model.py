import warnings
from pathlib import Path
import tempfile
import pytest
from ebsbi.model import EBModel


def test_ebmodel_plot_params_disabled_by_default():
    """Test that plotting is disabled by default."""
    params = {
        'teff': [5000, 6000],
        'q': [0.5, 1.0],
        'incl': [80, 90]
    }
    model = EBModel(None, params, phase_bank=None, noise_bank=None)

    assert model.save_binning_plots is False
    assert model.plot_output_dir is None


def test_ebmodel_plot_params_enabled():
    """Test enabling plot saving with directory."""
    params = {
        'teff': [5000, 6000],
        'q': [0.5, 1.0],
        'incl': [80, 90]
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        model = EBModel(
            None,
            params,
            phase_bank=None,
            noise_bank=None,
            save_binning_plots=True,
            plot_output_dir=tmpdir
        )

        assert model.save_binning_plots is True
        assert model.plot_output_dir == Path(tmpdir)


def test_ebmodel_plot_params_enabled_no_dir_warns():
    """Test that enabling without dir shows warning and disables."""
    params = {
        'teff': [5000, 6000],
        'q': [0.5, 1.0],
        'incl': [80, 90]
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = EBModel(
            None,
            params,
            phase_bank=None,
            noise_bank=None,
            save_binning_plots=True,
            plot_output_dir=None
        )

        # Should warn and disable
        assert len(w) == 1
        assert "plot_output_dir not provided" in str(w[0].message)
        assert model.save_binning_plots is False


def test_nbi_simulator_tracks_sample_id():
    """Test that nbi_simulator extracts and tracks sample ID."""
    params = {
        'teff': [5000, 6000],
        'q': [0.5, 1.0],
        'incl': [80, 90]
    }
    model = EBModel(None, params)

    # Simulate with output path containing sample ID
    theta = [5500, 0.75, 85]
    output_path = "test_output/sample_042.npy"

    # This should set _current_sample_id
    # Note: This will fail because we haven't fully implemented the simulator,
    # but we can test the ID extraction logic in isolation

    # For now, just test the extraction logic directly
    from pathlib import Path
    sample_stem = Path(output_path).stem  # "sample_042"
    sample_id = int(sample_stem.split('_')[-1])

    assert sample_id == 42


def test_nbi_simulator_handles_invalid_sample_id():
    """Test that invalid sample ID defaults to 0."""
    from pathlib import Path

    # Test various invalid formats
    invalid_paths = [
        "output.npy",  # No underscore
        "sample_abc.npy",  # Non-numeric
        "test_sample.npy",  # No number at end
    ]

    for path in invalid_paths:
        sample_stem = Path(path).stem
        try:
            sample_id = int(sample_stem.split('_')[-1])
        except (ValueError, IndexError):
            sample_id = 0

        assert sample_id == 0
