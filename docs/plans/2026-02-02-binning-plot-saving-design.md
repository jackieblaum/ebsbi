# Binning Plot Saving Design

**Date**: 2026-02-02
**Status**: Approved

## Overview

Add functionality to save eclipsebin's diagnostic plots during training sample generation for quality assessment and debugging.

### Goals

- Save light curve binning diagnostic plots automatically during training
- Provide visual feedback on binning quality and eclipse detection
- Support batch generation of many samples with organized plot storage
- Minimal performance impact and clean error handling

### Non-Goals

- SED binning plots (SED doesn't use binning)
- Interactive plot display during generation
- Plot customization or styling options

## Design Decisions

### Scope
- **Light curves only**: Only save plots for light curve binning (not SED)
- **Training time**: Plots generated during `generate_training_samples.py` execution

### Storage Strategy
- **Location**: Subdirectory structure `<output_dir>/binning_plots/`
- **One file per passband**: Combined unbinned + binned side-by-side
- **Naming**: `sample_{id:03d}_{passband}.png` with sanitized passband labels

### Control Mechanism
- **Config-based**: Enable via `EBModel` initialization parameters
- **Explicit opt-in**: Default is `save_binning_plots=False`

### Plot Content
- **eclipsebin's native diagnostics**: Use `binner.plot_unbinned_light_curve()` and `binner.plot_binned_light_curve()`
- **Side-by-side layout**: Unbinned (left) and binned (right) in single file
- **Preserved fidelity**: No modification of eclipsebin's plot content

## API Changes

### EBModel.__init__()

Add two optional parameters:

```python
def __init__(
    self,
    eb_path,
    params_dict,
    phase_bank=None,
    noise_bank=None,
    rng=None,
    save_binning_plots=False,      # NEW
    plot_output_dir=None,           # NEW
):
    """
    Parameters
    ----------
    save_binning_plots : bool, optional
        If True, save diagnostic plots during light curve binning.
        Default: False
    plot_output_dir : str or Path, optional
        Base directory for saving plots. Plots saved to
        <plot_output_dir>/binning_plots/. Required if save_binning_plots=True.
    """
    self.save_binning_plots = save_binning_plots
    self.plot_output_dir = Path(plot_output_dir) if plot_output_dir else None

    # Validation
    if self.save_binning_plots and self.plot_output_dir is None:
        warnings.warn(
            "save_binning_plots=True but plot_output_dir not provided. "
            "Disabling plot saving."
        )
        self.save_binning_plots = False
```

### observational.bin_light_curve()

Add `plot_filename` parameter:

```python
def bin_light_curve(
    phases,
    fluxes,
    sigmas,
    nbins=200,
    fraction_in_eclipse=0.5,
    atol_primary=0.001,
    atol_secondary=0.05,
    plot=False,
    plot_filename=None,    # NEW: Path to save combined plot
):
    """
    Parameters
    ----------
    plot_filename : str or Path, optional
        If provided and plot=True, saves combined unbinned+binned
        plot to this path. If None, plot is displayed but not saved.
    """
```

### generate_training_samples.py

Enable plotting when creating model:

```python
model = EBModel(
    eb_path=None,
    params_dict=params_dict,
    phase_bank=phase_bank,
    noise_bank=noise_bank,
    rng=np.random.default_rng(101),
    save_binning_plots=True,           # Enable plot saving
    plot_output_dir=output_dir,        # "training_samples_test"
)
```

## Implementation Details

### File Naming

**Sanitization function**:
```python
def sanitize_passband_label(label):
    """
    Convert passband labels to filesystem-safe names.

    Examples:
        'TESS:T' -> 'tess_t'
        'Gaia:G' -> 'gaia_g'
        'Pan-STARRS:g' -> 'pan_starrs_g'
    """
    return label.lower().replace(':', '_').replace('-', '_')
```

**Example output**:
```
training_samples_test/
├── binning_plots/
│   ├── sample_000_tess_t.png        # Unbinned + Binned for TESS:T
│   ├── sample_000_gaia_g.png        # Unbinned + Binned for Gaia:G
│   ├── sample_001_tess_t.png
│   └── ...
├── sample_000.npy
├── sample_001.npy
└── ...
```

### Implementation Flow

**Call stack**:
```
generate_training_samples.py
  └─> model.nbi_simulator(theta, output_path)
      └─> model.generate_light_curve(theta)
          └─> [binning loop]
              └─> bin_light_curve(..., plot=True, plot_filename=path)
                  └─> [capture and combine eclipsebin figures]
```

### Changes to model.py:generate_light_curve()

**Sample ID tracking** (in `nbi_simulator()` before calling `generate_light_curve()`):
```python
def nbi_simulator(self, theta, output_path):
    # Extract sample ID from output path
    # "training_samples_test/sample_042.npy" -> 42
    sample_stem = Path(output_path).stem  # "sample_042"
    try:
        self._current_sample_id = int(sample_stem.split('_')[-1])
    except (ValueError, IndexError):
        self._current_sample_id = 0

    # Continue with existing logic...
    theta_dict = {name: val for name, val in zip(self.params_dict.keys(), np.array(theta).T)}
    self.bundle = self.create_phoebe_bundle(theta_dict)
```

**Directory setup and plot path generation** (in `generate_light_curve()`, before binning loop):
```python
# Around line 380, before binning loop
if self.save_binning_plots and self.plot_output_dir:
    binning_plots_dir = self.plot_output_dir / "binning_plots"
    try:
        binning_plots_dir.mkdir(exist_ok=True)
    except OSError as e:
        warnings.warn(f"Failed to create binning_plots directory: {e}")
        self.save_binning_plots = False

# In binning loop (around line 389):
for i, (phases, fluxes, sigmas) in enumerate(zip(lc_phase_unbinned, lc_flux_unbinned, lc_err_unbinned)):
    L = len(phases)
    if L > nbins:
        # Determine plot path if plotting enabled
        plot_path = None
        if self.save_binning_plots:
            passband_label = self._lc_dataset_labels[i]
            sanitized = sanitize_passband_label(passband_label)
            plot_path = binning_plots_dir / f"sample_{self._current_sample_id:03d}_{sanitized}.png"

        # Bin with optional plotting
        ph_b, fl_b, er_b = bin_light_curve(
            phases, fluxes, sigmas,
            nbins=nbins,
            fraction_in_eclipse=0.5,
            atol_primary=0.001,
            atol_secondary=0.05,
            plot=self.save_binning_plots,
            plot_filename=plot_path
        )
        lc_phase.append(ph_b)
        lc_flux.append(fl_b)
        lc_err.append(er_b)
    else:
        # No binning needed, no plot to save
        lc_phase.append(phases)
        lc_flux.append(fluxes)
        lc_err.append(sigmas)
```

### Changes to observational.py:bin_light_curve()

**Capture and combine eclipsebin figures** (after successful binning, around line 885-890):

```python
# After successful binning:
bin_phases, bin_fluxes, bin_sigmas = binner.bin_light_curve(plot=False)

# If plotting requested and filename provided:
if plot and plot_filename:
    try:
        # Create plots using eclipsebin's methods
        binner.plot_unbinned_light_curve()
        fig_unbinned = plt.gcf()

        binner.plot_binned_light_curve(bin_phases, bin_fluxes, bin_sigmas)
        fig_binned = plt.gcf()

        # Create combined figure (1 row, 2 columns)
        combined_fig = plt.figure(figsize=(24, 6))
        gs = combined_fig.add_gridspec(1, 2)

        # Copy axes from captured figures to combined figure
        ax_unbinned = combined_fig.add_subplot(gs[0, 0])
        ax_binned = combined_fig.add_subplot(gs[0, 1])

        # Transfer content from original figures
        _copy_axes_content(fig_unbinned.axes[0], ax_unbinned)
        _copy_axes_content(fig_binned.axes[0], ax_binned)

        # Save combined figure
        combined_fig.savefig(str(plot_filename), dpi=150, bbox_inches='tight')

    except Exception as e:
        warnings.warn(f"Failed to save binning plot {plot_filename}: {e}")
    finally:
        # Always clean up figures to free memory
        plt.close('all')

return bin_phases, bin_fluxes, bin_sigmas
```

**Helper function for copying axes content**:

```python
def _copy_axes_content(src_ax, dst_ax):
    """
    Copy plot elements from source axes to destination axes.

    Copies lines, collections, labels, limits, and styling.
    """
    import copy

    # Copy plot elements
    for line in src_ax.get_lines():
        dst_ax.add_line(copy.copy(line))

    for collection in src_ax.collections:
        dst_ax.add_collection(copy.copy(collection))

    # Copy labels and titles
    dst_ax.set_title(src_ax.get_title(), fontsize=src_ax.title.get_fontsize())
    dst_ax.set_xlabel(src_ax.get_xlabel(), fontsize=src_ax.xaxis.label.get_fontsize())
    dst_ax.set_ylabel(src_ax.get_ylabel(), fontsize=src_ax.yaxis.label.get_fontsize())

    # Copy limits and scales
    dst_ax.set_xlim(src_ax.get_xlim())
    dst_ax.set_ylim(src_ax.get_ylim())
    dst_ax.set_xscale(src_ax.get_xscale())
    dst_ax.set_yscale(src_ax.get_yscale())

    # Copy grid
    dst_ax.grid(src_ax.xaxis._gridOnMajor or src_ax.yaxis._gridOnMajor)

    # Copy legend if present
    legend = src_ax.get_legend()
    if legend:
        dst_ax.legend(loc=legend._loc)
```

## Error Handling

### Directory Creation
```python
try:
    binning_plots_dir.mkdir(exist_ok=True)
except OSError as e:
    warnings.warn(f"Failed to create binning_plots directory: {e}")
    self.save_binning_plots = False  # Disable for this run
```

### Plot Saving
```python
try:
    combined_fig.savefig(str(plot_filename), dpi=150, bbox_inches='tight')
except Exception as e:
    warnings.warn(f"Failed to save binning plot {plot_filename}: {e}")
    # Continue processing - don't fail the sample
finally:
    plt.close('all')  # Always clean up
```

### Validation
```python
# In EBModel.__init__()
if self.save_binning_plots and self.plot_output_dir is None:
    warnings.warn(
        "save_binning_plots=True but plot_output_dir not provided. "
        "Disabling plot saving."
    )
    self.save_binning_plots = False
```

## Edge Cases

### 1. Passband Not Binned
When `L <= nbins` (line 400), no binning occurs:
- Skip plot generation for that passband
- No file created

### 2. Binning Retry Failures
If binning fails after retries (line 892-900):
- No plot saved for failed passband
- Warning logged, processing continues

### 3. Sample ID Extraction
If output path doesn't match expected pattern:
- Default to `sample_id = 0`
- Log warning about unexpected path format

### 4. Missing Dependencies
If matplotlib import fails in observational.py:
- Gracefully skip plotting
- Log warning

### 5. Figure Capture Failure
If `plt.gcf()` doesn't return expected figures:
- Skip plot saving
- Log warning, continue binning

## Memory Management

**Critical for batch processing**:
- Always call `plt.close('all')` in finally block
- Prevents memory accumulation when generating many samples
- eclipsebin creates figures with `figsize=(20, 5)` - can add up quickly

## Testing Considerations

### Unit Tests
- Test `sanitize_passband_label()` with various inputs
- Test `_copy_axes_content()` with mock axes objects
- Test error handling paths (directory creation, file I/O)

### Integration Tests
1. Generate samples with `save_binning_plots=True`
2. Verify directory structure created correctly
3. Verify plot files exist and are valid PNG images
4. Verify correct number of plots (one per binned passband)
5. Test with `save_binning_plots=False` (should not create plots)

### Manual Verification
- Visual inspection of saved plots
- Verify plots match eclipsebin's native output
- Check file sizes (should be reasonable, not corrupted)

## Performance Impact

**Expected overhead per sample**:
- Directory check/creation: ~1ms (one-time)
- Plot generation: ~100-200ms per passband (matplotlib rendering)
- File I/O: ~50ms per file
- **Total**: ~500ms-1s per sample with 3-4 passbands

**Mitigation**:
- Plots saved only when explicitly enabled
- No impact when `save_binning_plots=False` (default)
- Figures immediately closed to free memory

## Future Enhancements

Potential improvements not included in this design:

1. **Parallel plot saving**: Save plots asynchronously to avoid blocking
2. **Plot format options**: Support PDF, SVG in addition to PNG
3. **Customization**: Allow DPI, figure size configuration
4. **Summary plots**: Generate overview plots showing all samples
5. **Selective plotting**: Plot only specific passbands or samples

## References

- `compare_unbinned_vs_binned.py`: Existing pattern for multi-panel plots
- eclipsebin documentation: Plot methods and figure creation
- `observational.py:bin_light_curve()`: Current binning implementation
