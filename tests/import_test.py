def test_import() -> None:
    try:
        import classical_diffusion  # ruff:ignore[import-outside-top-level]
    except ImportError:
        classical_diffusion = None  # ty:ignore[invalid-assignment]

    assert classical_diffusion is not None, (
        "classical_diffusion module should not be None"
    )
