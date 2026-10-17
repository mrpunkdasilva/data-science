"""Entry point that launches the Streamlit dashboard.

Installed as the ``capstone-dashboard`` console script, it delegates to
``streamlit run`` on ``app/dashboard.py``. The dashboard can also be started
directly with ``streamlit run src/app/dashboard.py`` or ``make dashboard``.
"""

# mypy: ignore-errors

import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli


def main() -> None:
    """Launch the Streamlit dashboard."""
    dashboard = Path(__file__).with_name("dashboard.py")
    sys.argv = ["streamlit", "run", str(dashboard)]
    sys.exit(streamlit_cli.main())


if __name__ == "__main__":
    main()
