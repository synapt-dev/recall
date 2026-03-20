import sys
from unittest.mock import patch


def test_synapt_init_dispatches_to_recall_setup():
    from synapt.cli import main

    captured_argv: list[str] = []

    def fake_recall_main():
        captured_argv.extend(sys.argv)

    with patch("synapt.cli._discover_commands", return_value={}), \
         patch("synapt.recall.cli.main", side_effect=fake_recall_main):
        sys.argv = ["synapt", "init", "--no-hook", "--no-embeddings"]
        main()

    assert captured_argv == [
        "synapt recall",
        "setup",
        "--no-hook",
        "--no-embeddings",
    ]


def test_synapt_help_lists_init(capsys):
    from synapt.cli import main

    with patch("synapt.cli._discover_commands", return_value={}):
        sys.argv = ["synapt", "--help"]
        main()

    captured = capsys.readouterr()
    assert "init" in captured.out
