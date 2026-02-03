import pytest
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from pathlib import Path
from ebsbi.observational import _copy_axes_content, bin_light_curve


def test_copy_axes_content_basic():
    """Test copying basic plot elements between axes."""
    # Create source figure with plot
    fig_src, ax_src = plt.subplots()
    x = np.linspace(0, 1, 10)
    y = np.sin(2 * np.pi * x)
    ax_src.plot(x, y, 'r-', label='sine')
    ax_src.set_title('Source Title')
    ax_src.set_xlabel('X Label')
    ax_src.set_ylabel('Y Label')
    ax_src.set_xlim(0, 1)
    ax_src.set_ylim(-1.5, 1.5)
    ax_src.legend()

    # Create destination figure
    fig_dst, ax_dst = plt.subplots()

    # Copy content
    _copy_axes_content(ax_src, ax_dst)

    # Verify content copied
    assert ax_dst.get_title() == 'Source Title'
    assert ax_dst.get_xlabel() == 'X Label'
    assert ax_dst.get_ylabel() == 'Y Label'
    assert ax_dst.get_xlim() == (0, 1)
    assert ax_dst.get_ylim() == (-1.5, 1.5)
    assert len(ax_dst.get_lines()) == 1
    assert ax_dst.get_legend() is not None

    plt.close('all')


def test_copy_axes_content_scatter():
    """Test copying scatter plot elements."""
    fig_src, ax_src = plt.subplots()
    x = np.random.rand(20)
    y = np.random.rand(20)
    ax_src.scatter(x, y, c='blue', marker='o')

    fig_dst, ax_dst = plt.subplots()
    _copy_axes_content(ax_src, ax_dst)

    # Should have one collection (scatter plot)
    assert len(ax_dst.collections) == 1

    plt.close('all')


def test_copy_axes_content_empty():
    """Test copying from empty axes."""
    fig_src, ax_src = plt.subplots()
    fig_dst, ax_dst = plt.subplots()

    # Should not raise error
    _copy_axes_content(ax_src, ax_dst)

    plt.close('all')


def test_bin_light_curve_saves_plot_file():
    """Test that bin_light_curve saves plot when filename provided."""
    # Create mock light curve data
    phases = np.linspace(0, 1, 500)
    # Create eclipsing pattern
    fluxes = np.ones_like(phases)
    fluxes[(phases > 0.45) & (phases < 0.55)] = 0.8  # Primary eclipse
    fluxes[(phases > 0.95) | (phases < 0.05)] = 0.9  # Secondary eclipse
    sigmas = np.ones_like(phases) * 0.01

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = Path(tmpdir) / "test_binning.png"

        # Bin with plotting
        ph_b, fl_b, er_b = bin_light_curve(
            phases, fluxes, sigmas,
            nbins=50,
            plot=True,
            plot_filename=plot_path
        )

        # Verify plot file created
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0  # File not empty

        # Verify binning still works (eclipsebin may return slightly fewer bins)
        assert len(ph_b) > 40  # Close to requested 50
        assert len(fl_b) == len(ph_b)
        assert len(er_b) == len(ph_b)


def test_bin_light_curve_no_plot_without_filename():
    """Test that plot=True without filename doesn't crash."""
    phases = np.linspace(0, 1, 500)
    fluxes = np.ones_like(phases)
    fluxes[(phases > 0.45) & (phases < 0.55)] = 0.8
    sigmas = np.ones_like(phases) * 0.01

    # Should not raise error
    ph_b, fl_b, er_b = bin_light_curve(
        phases, fluxes, sigmas,
        nbins=50,
        plot=True,
        plot_filename=None
    )

    assert len(ph_b) > 40  # Close to requested 50
