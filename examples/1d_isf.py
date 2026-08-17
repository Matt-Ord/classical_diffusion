def _plot_1d_periodic_isf_overdamped() -> None:
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=0.1, temperature=0.5, m=1.0, delta_x=5, barrier_energy=0.5
    )

    fig, ax = get_fancy_figure()
    delta_k = (0.5 * 2 * np.pi / system.delta_x,)

    result = solve_overdamped_ensemble(
        system,
        TimeSpan(t_end=40 / system.gamma, n_steps=4000),
        (np.full((80, 1), 0.0), np.full((80, 1), 0.0)),
        _key=key,
    )
    _, _, line, _ = plot_isf(
        result=result, ax=ax, delta_k=delta_k, pairwise=True, measure="real"
    )
    line.set_label("overdamped")

    times = np.linspace(0, 1 / system.gamma, 4000)
    expected = np.exp(-(system.kbt / system.gamma) * (delta_k[0] ** 2) * times)
    (line_1,) = ax.plot(times, expected, label="flat surface", linestyle=":")

    ax.set_xlim(0, 0.4 / system.gamma)
    ax.set_ylim(0, 1)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.legend(handles=[line, line_1])
    fig.savefig("./examples/1d_system.isf.overdamped.pdf", dpi=300, bbox_inches="tight")
