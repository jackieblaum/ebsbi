import pytest
import matplotlib.pyplot as plt
import numpy as np
from ebsbi.observational import _copy_axes_content


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
