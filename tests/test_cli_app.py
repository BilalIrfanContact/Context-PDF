import io
import unittest
from unittest.mock import patch

import askmydoc_cli.app
from askmydoc_cli.app import AskMyDocShell, EntryState, run_cli


class FakeTtyStream(io.StringIO):
    def isatty(self) -> bool:
        return True


class AskMyDocCliTestCase(unittest.TestCase):
    def test_run_cli_starts_at_entry_screen_and_exits_cleanly(self):
        output = io.StringIO()

        exit_code = run_cli(
            ["askmydoc"],
            input_stream=io.StringIO("/help\n/exit\n"),
            output_stream=output,
        )

        rendered = output.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("AskMyDoc CLI", rendered)
        self.assertIn("Current state: entry", rendered)
        self.assertIn("Starter commands:", rendered)
        self.assertIn("> /docs", rendered)
        self.assertIn("/ls", rendered)
        self.assertIn("Type a command directly.", rendered)
        self.assertIn("/open <document-name> - Enter the scaffolded document state by name.", rendered)
        self.assertIn("Goodbye.", rendered)

    def test_shell_handles_document_state_lifecycle(self):
        output = io.StringIO()
        shell = AskMyDocShell(
            input_stream=io.StringIO("/open notes.md\nsummarize this\n/close\n/exit\n"),
            output_stream=output,
        )

        exit_code = shell.run()
        rendered = output.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("Opened document: notes.md", rendered)
        self.assertIn("Current state: document", rendered)
        self.assertIn("document:notes.md> ", rendered)
        self.assertIn("[notes.md] Chat is not wired yet. Received: summarize this", rendered)
        self.assertIn("Closed document: notes.md", rendered)
        self.assertEqual(shell.state, EntryState())

    def test_help_includes_document_state_context(self):
        output = io.StringIO()

        exit_code = run_cli(
            ["askmydoc"],
            input_stream=io.StringIO("/open notes.md\n/help\n/exit\n"),
            output_stream=output,
        )

        rendered = output.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("Current state: document", rendered)
        self.assertIn("Active document: notes.md", rendered)
        self.assertIn("/close - Return to the entry screen.", rendered)
        self.assertIn("<text> - Send a placeholder message to the active document.", rendered)

    def test_arrow_keys_select_entry_commands(self):
        shell = AskMyDocShell(
            input_stream=io.StringIO("\x1b[B\n"),
            output_stream=io.StringIO(),
        )
        shell._interactive_terminal = True

        selected_command = shell._read_entry_command()

        self.assertEqual(selected_command, "/ls")

    def test_entry_feedback_persists_on_interactive_redraw(self):
        output = io.StringIO()
        shell = AskMyDocShell(
            input_stream=io.StringIO(),
            output_stream=output,
        )
        shell._interactive_terminal = True

        shell.handle_line("/help")
        shell._render_current_view()

        rendered = output.getvalue()

        self.assertIn("Feedback:", rendered)
        self.assertIn("Current state: entry", rendered)
        self.assertIn("/help - Show commands for the current CLI state.", rendered)
        self.assertIn("Starter commands:", rendered)

    def test_shell_closes_cleanly_on_eof(self):
        output = io.StringIO()

        exit_code = run_cli(
            ["askmydoc"],
            input_stream=io.StringIO(""),
            output_stream=output,
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("Session closed.", output.getvalue())

    def test_run_cli_falls_back_without_termios(self):
        output = FakeTtyStream()

        with patch.object(askmydoc_cli.app, "termios", None):
            exit_code = run_cli(
                ["askmydoc"],
                input_stream=FakeTtyStream("/help\n/exit\n"),
                output_stream=output,
            )

        rendered = output.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("AskMyDoc CLI", rendered)
        self.assertIn("Type a command directly.", rendered)
        self.assertIn("Current state: entry", rendered)
        self.assertIn("Available commands:", rendered)
        self.assertIn("Goodbye.", rendered)


if __name__ == "__main__":
    unittest.main()
