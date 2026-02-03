# ebsbi
A package for inferring parameters of eclipsing binary stars using neural simulation-based inference.

## Diagnostic Plot Saving

The training sample generator can optionally save binning diagnostic plots:

```python
model = EBModel(
    ...,
    save_binning_plots=True,
    plot_output_dir="training_samples"
)
```

Plots are saved to `training_samples/binning_plots/` with side-by-side
unbinned and binned views. See [docs/binning_plots.md](docs/binning_plots.md)
for details.
