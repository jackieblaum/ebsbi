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
