"""Command-line entrypoint for `python -m agent_alpha`."""

from .runner import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
