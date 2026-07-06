def test_import() -> None:
    try:
        import classical_diffusion  # noqa: PLC0415
    except ImportError:
        classical_diffusion = None  # ty:ignore[invalid-assignment]

    assert classical_diffusion is not None, (
        "classical_diffusion module should not be None"
    )
