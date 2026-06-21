import io
import unittest

from askmydoc_cli.app import AskMyDocShell, EntryState, run_cli


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
        self.assertIn("Use Up/Down and Enter to choose, or type a command directly.", rendered)
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
        output = io.StringIO()

        exit_code = run_cli(
            ["askmydoc"],
            input_stream=io.StringIO("\x1b[B\n/exit\n"),
            output_stream=output,
        )

        rendered = output.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("Local file browsing is not wired yet.", rendered)
        self.assertIn("> /ls", rendered)

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


if __name__ == "__main__":
    unittest.main()
