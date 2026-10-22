"""Default entrypoint for `python -m octolyzer` (launches the OCTolyzer GUI)."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        from octolyzer.main import main as cli_main

        return cli_main(arguments)

    from gui.app import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
