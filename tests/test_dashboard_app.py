from unittest.mock import patch


def test_dashboard_pid_and_log_paths_live_under_synapt_root(tmp_path):
    from synapt.dashboard.app import _dashboard_log_path, _dashboard_pid_path

    with patch("synapt.dashboard.app.project_data_dir", return_value=tmp_path / ".synapt" / "recall"):
        assert _dashboard_pid_path() == tmp_path / ".synapt" / "dashboard.pid"
        assert _dashboard_log_path() == tmp_path / ".synapt" / "dashboard.log"


def test_read_pid_returns_none_for_missing_or_invalid(tmp_path):
    from synapt.dashboard.app import _read_pid

    missing = tmp_path / "missing.pid"
    invalid = tmp_path / "invalid.pid"
    invalid.write_text("abc\n")

    assert _read_pid(missing) is None
    assert _read_pid(invalid) is None


def test_background_command_uses_foreground_child_mode():
    from synapt.dashboard.app import _background_command

    cmd = _background_command("127.0.0.1", 9000)
    assert cmd[1:4] == ["-m", "synapt.cli", "dashboard"]
    assert "--foreground" in cmd
    assert "--no-open" in cmd
    assert "9000" in cmd


def test_stop_dashboard_cleans_stale_pidfile(tmp_path):
    from synapt.dashboard.app import _stop_dashboard

    pid_path = tmp_path / ".synapt" / "dashboard.pid"
    pid_path.parent.mkdir(parents=True)
    pid_path.write_text("12345\n")

    with patch("synapt.dashboard.app._dashboard_pid_path", return_value=pid_path), \
         patch("synapt.dashboard.app._pid_is_running", return_value=False):
        assert _stop_dashboard() is False
        assert not pid_path.exists()


def test_synapt_help_lists_dashboard(capsys):
    from synapt.cli import main

    with patch("synapt.cli._discover_commands", return_value={}):
        import sys

        sys.argv = ["synapt", "--help"]
        main()

    captured = capsys.readouterr()
    assert "dashboard" in captured.out
