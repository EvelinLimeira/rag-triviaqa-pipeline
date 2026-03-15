"""Unit tests for the CLI entry point in main.py.

Tests verify that each subcommand dispatches to the correct handler and
that missing subcommands print usage and exit with code 1.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

import main


class TestSubcommandRouting:
    """Verify each CLI subcommand dispatches to the correct handler."""

    @patch("main.cmd_download")
    def test_download_routes_to_cmd_download(self, mock_cmd: MagicMock) -> None:
        """Validates: Requirements 19.1"""
        with patch("sys.argv", ["prog", "download"]):
            main.main()
        mock_cmd.assert_called_once()

    @patch("main.cmd_index")
    def test_index_routes_to_cmd_index(self, mock_cmd: MagicMock) -> None:
        """Validates: Requirements 19.1"""
        with patch("sys.argv", ["prog", "index"]):
            main.main()
        mock_cmd.assert_called_once()

    @patch("main.cmd_evaluate")
    def test_evaluate_routes_to_cmd_evaluate(self, mock_cmd: MagicMock) -> None:
        """Validates: Requirements 19.1"""
        with patch("sys.argv", ["prog", "evaluate"]):
            main.main()
        mock_cmd.assert_called_once_with(584)

    @patch("main.cmd_evaluate_full_pool")
    def test_evaluate_full_pool_routes_to_cmd_evaluate_full_pool(
        self, mock_cmd: MagicMock
    ) -> None:
        """Validates: Requirements 19.1"""
        with patch("sys.argv", ["prog", "evaluate-full-pool"]):
            main.main()
        mock_cmd.assert_called_once_with(584)

    @patch("main.cmd_evaluate")
    def test_evaluate_with_custom_sample_size(self, mock_cmd: MagicMock) -> None:
        """Validates: Requirements 19.1"""
        with patch("sys.argv", ["prog", "evaluate", "--sample-size", "10"]):
            main.main()
        mock_cmd.assert_called_once_with(10)

    @patch("main.cmd_evaluate_full_pool")
    def test_evaluate_full_pool_with_custom_sample_size(
        self, mock_cmd: MagicMock
    ) -> None:
        """Validates: Requirements 19.1"""
        with patch("sys.argv", ["prog", "evaluate-full-pool", "--sample-size", "25"]):
            main.main()
        mock_cmd.assert_called_once_with(25)


class TestMissingSubcommand:
    """Verify that missing subcommand prints usage and exits with code 1."""

    def test_no_subcommand_exits_with_code_1(self) -> None:
        """Validates: Requirements 19.5"""
        with patch("sys.argv", ["prog"]):
            with pytest.raises(SystemExit) as exc_info:
                main.main()
            assert exc_info.value.code == 1

    def test_no_subcommand_prints_usage_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Validates: Requirements 19.5"""
        with patch("sys.argv", ["prog"]):
            with pytest.raises(SystemExit):
                main.main()
        captured = capsys.readouterr()
        assert "usage:" in captured.err.lower()
