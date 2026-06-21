import io
import unittest

from askmydoc_cli.app import AskMyDocShell, EntryState, run_cli


class AskMyDocCliTestCase(unittest.TestCase):
    def test_run_cli_launches_shell_and_exits_cleanly(self):
        output = io.StringIO()

        exit_code = run_cli(
            ["askmydoc"],
            input_stream=io.StringIO("/help\n/exit\n"),
            output_stream=output,
        )

        rendered = output.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("AskMyDoc CLI", rendered)
        self.assertIn("Interactive shell scaffold. Type /help to get started.", rendered)
        self.assertIn("Entry commands:", rendered)
        self.assertIn("/open <document-name>  Enter a scaffolded document session", rendered)
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
        self.assertIn("document:notes.md> ", rendered)
        self.assertIn("[notes.md] Chat is not wired yet. Received: summarize this", rendered)
        self.assertIn("Closed document: notes.md", rendered)
        self.assertEqual(shell.state, EntryState())

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
