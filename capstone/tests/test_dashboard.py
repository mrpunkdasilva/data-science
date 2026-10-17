"""Smoke tests for the Streamlit dashboard."""

from pathlib import Path
from typing import Any

import pytest

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "app" / "dashboard.py"


def _run_dashboard() -> Any:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    app_test.run()
    return app_test


def test_dashboard_renders_without_error():
    """The dashboard module runs without raising any exception."""
    app_test = _run_dashboard()
    assert not app_test.exception


def test_dashboard_renders_tabs_when_export_available():
    """When the enriched export exists, all three tabs are rendered."""
    from config import settings

    if not Path(settings.EXPORT_PATH).exists():
        pytest.skip("enriched export not available; run `make pipeline` first")

    app_test = _run_dashboard()
    assert not app_test.exception
    assert [tab.label for tab in app_test.tabs] == [
        "Search",
        "Cluster map",
        "Quality & anomalies",
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
