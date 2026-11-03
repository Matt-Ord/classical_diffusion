---
name: plotting
description: Conventions for plot functions, cached simulations and example scripts in this repo (classical_diffusion/plot.py, the plot_* functions of the package, and examples/*.py). Use whenever adding or changing a plot function, caching a simulation, or composing a figure in an example script.
---

# Plotting conventions

Plot functions draw one thing with sensible defaults. Example scripts compose
figures and own all styling. Simulations are cached on disk, so a figure can be
redrawn without rerunning them.

## Plot functions

- Put the data arguments first, then a keyword-only `ax: Axes | None = None`.
  Return `(fig, ax, artist)` so the caller can style what was drawn.

  ```python
  def plot_isf(
      result: SimulationResult, *, ax: Axes | None = None, delta_k: tuple[float, ...]
  ) -> tuple[Figure, Axes, Line2D, PolyCollection]:
      """Plot the ensemble-averaged ISF over time, with a shaded ±1 SEM band."""
      fig, ax = get_figure(ax)
      ...
      (line,) = ax.plot(times, avg_data)
      fill = ax.fill_between(times, avg_data - sem_data, avg_data + sem_data)
      fill.set_alpha(0.3)
      fill.set_color(line.get_color())
      ax.set_xlabel("Time")
      ax.set_ylabel("ISF")
      return fig, ax, line, fill
  ```

- Always start with `fig, ax = get_figure(ax)` (from `classical_diffusion.plot`).
  It creates a figure when `ax` is None, and otherwise draws into the given axes.
- Use sensible defaults for whatever makes the plot readable on its own: axis
  labels, limits and scales. Keep a band, fit or marker in step with its line,
  e.g. `fill.set_color(line.get_color())`.
- Don't take style parameters (color, label, linewidth, ...), and don't pass
  extra keyword arguments through to matplotlib. Don't set colors inside a plot function. The legend
  label is the caller's to set.
- One plot function draws one dataset or curve. A function that draws several
  (e.g. `plot_isf_with_delta_k`) does it by calling the single plot function
  and styling what it returns. Combining different quantities on one figure is
  the example script's job.

## Cached simulations

Expensive results are cached with `classical_diffusion.util.cached`. A path
function sits next to each cached function and takes the same arguments.

```python
def _solve_effective_mass_path(
    system: PeriodicSystem1D, barrier_energy_ratio: np.ndarray, n_samples: np.ndarray
) -> Path:
    return Path(
        f"effective_mass_{hash_array((n_samples,))}_{hash_array((barrier_energy_ratio,))}_{hash(system)}.npz"
    )


@cached(_solve_effective_mass_path)
def _get_simulated_effective_mass(
    system: PeriodicSystem1D, barrier_energy_ratio: np.ndarray, n_samples: np.ndarray
) -> np.ndarray:
    ...
```

- Build the file name from every argument that changes the result. Use
  `hash(system)` and `hash(time_span)` for the (hashable) dataclasses,
  `hash_array((a, b))` for arrays, and `hash_key(_key)` for a PRNG key.
- Keep the path relative. It is resolved under the directory set by
  `cache_base_path`, which example scripts set to `examples/data`.
- Stack `@timed` under `@cached` to time a run (`solve_many`). Wrap a loop of
  many short runs in `disabled_timing()`, and use `.call_uncached(...)` to
  skip the cache for runs that aren't worth storing.
- Cache the simulation, not the figure. Plot functions take results, so a
  figure can be restyled without rerunning anything.

## Example scripts (`examples/*.py`)

- Write one private `_plot_<figure>() -> None` per figure. It gets its data
  (through cached functions), draws, and saves.
- Create the figure with `fig, ax = get_fancy_figure()`, or
  `get_two_panel_figure()` / `get_three_panel_figure()` for several panels.
- Call plot functions with `ax=ax` and style the returned artists with their
  setters:

  ```python
  _, ax, line_0, _ = plot_isf(result=full_result, ax=ax, delta_k=delta_k)
  line_0.set_label("full simulation")
  _, ax, line_1, _ = plot_isf(result=ballistic_result, ax=ax, delta_k=delta_k)
  line_1.set_label("ballistic simulation")
  ```

  For a curve drawn directly with matplotlib, unpack the artist and style it in
  the same way:

  ```python
  (exact_line,) = ax.plot(barrier_energy_ratio, exact)
  exact_line.set_label("exact")
  exact_line.set_color(CAM_BLUE.dark)
  ```

- Colors come from the Cambridge palette in `classical_diffusion.plot`
  (`CAM_BLUE`, `CAM_CHERRY`, `CAM_CREST`, `CAM_PURPLE`, `CAM_INDIGO`,
  `CAM_GREEN`, each with `.dark`, `.base` and `.warm` shades, and the
  `CAM_SLATE_*` greys).
- Set the limits, the legend (`ax.legend(handles=[...])`) and
  `fig.savefig("examples/<folder>/<script>.<variant>.pdf")` in the script.
- Call each figure function under `if __name__ == "__main__":`, inside
  `with cache_base_path(Path("examples/data")):`. Put a short comment on what
  each figure shows.

  ```python
  if __name__ == "__main__":
      # The ISF of a full Langevin and a ballistic trajectory, and of the
      # elastic and inelastic parts of the ballistic trajectory
      with cache_base_path(Path("examples/data")):
          _plot_periodic_isf_1d()
          _plot_periodic_isf_2d()
  ```
