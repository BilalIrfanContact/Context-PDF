from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
import termios
from typing import TextIO


@dataclass(frozen=True)
class EntryState:
    name: str = "entry"


@dataclass(frozen=True)
class DocumentState:
    document_name: str
    name: str = "document"


ENTRY_HELP_ITEMS = (
    ("/docs", "Browse uploaded documents when the document list is wired."),
    ("/ls", "Browse supported local files from the current directory."),
    ("/login", "Authenticate with AskMyDoc when the auth flow is connected."),
    ("/open <document-name>", "Enter the scaffolded document state by name."),
    ("/help", "Show commands for the current CLI state."),
    ("/exit", "Close the AskMyDoc shell."),
)

ENTRY_STARTER_COMMANDS = ("/docs", "/ls", "/login", "/help", "/exit")

DOCUMENT_HELP_ITEMS = (
    ("/docs", "Switch documents when uploaded document browsing is wired."),
    ("/ls", "Upload a local document and switch to it when wired."),
    ("/delete", "Delete the active document when delete support is wired."),
    ("/logout", "Clear authentication and return to the entry screen when wired."),
    ("/close", "Return to the entry screen."),
    ("/help", "Show commands for the current CLI state."),
    ("/exit", "Close the AskMyDoc shell."),
    ("<text>", "Send a placeholder message to the active document."),
)


class AskMyDocShell:
    def __init__(
        self,
        *,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout
        self.state: EntryState | DocumentState = EntryState()
        self._running = True
        self._entry_selection_index = 0
        self._interactive_terminal = False
        self._entry_feedback: list[str] = []

    def run(self) -> int:
        self._interactive_terminal = (
            hasattr(self.input_stream, "isatty")
            and hasattr(self.input_stream, "fileno")
            and hasattr(self.output_stream, "isatty")
            and self.input_stream.isatty()
            and self.output_stream.isatty()
        )

        with self._interactive_input_mode():
            self._render_current_view()

            while self._running:
                line = self._read_command()
                if line is None:
                    if self._interactive_terminal:
                        self._write_line("")
                    self._write_line("Session closed.")
                    break

                self.handle_line(line)
                if self._running:
                    self._render_current_view()

        return 0

    def handle_line(self, line: str) -> None:
        line = line.strip()
        if line == "":
            return

        if line == "/help":
            self._show_help()
            return

        if line == "/exit":
            self._write_line("Goodbye.")
            self._running = False
            return

        if isinstance(self.state, EntryState):
            self._handle_entry_line(line)
            return

        self._handle_document_line(line)

    def open_document(self, document_name: str) -> None:
        self.state = DocumentState(document_name=document_name)
        self._write_line(f"Opened document: {document_name}")
        self._write_line("Document chat scaffolding is ready for future workflows.")

    def close_document(self) -> None:
        if not isinstance(self.state, DocumentState):
            return

        closed_document = self.state.document_name
        self.state = EntryState()
        self._write_entry_feedback_line(f"Closed document: {closed_document}")

    def _handle_entry_line(self, line: str) -> None:
        if line == "/docs":
            self._write_entry_feedback_line("Uploaded document browsing is not wired yet.")
            return

        if line == "/ls":
            self._write_entry_feedback_line("Local file browsing is not wired yet.")
            return

        if line == "/login":
            self._write_entry_feedback_line("CLI login is not wired yet.")
            return

        if line == "/open" or line.startswith("/open "):
            document_name = line.removeprefix("/open").strip()
            if document_name == "":
                self._write_entry_feedback_line("Usage: /open <document-name>")
                return

            self.open_document(document_name)
            return

        if line.startswith("/"):
            self._write_entry_feedback_line(f"Unknown command: {line}")
            self._write_entry_feedback_line("Use /help to see the available shell commands.")
            return

        self._write_entry_feedback_line(
            "Type a slash command or use the starter menu before opening a document."
        )

    def _handle_document_line(self, line: str) -> None:
        if line == "/docs":
            self._write_line("Document switching is not wired yet.")
            return

        if line == "/ls":
            self._write_line("Document upload from the CLI is not wired yet.")
            return

        if line == "/delete":
            self._write_line("Document deletion from the CLI is not wired yet.")
            return

        if line == "/logout":
            self._write_line("CLI logout is not wired yet.")
            return

        if line == "/close":
            self.close_document()
            return

        if line.startswith("/"):
            self._write_line(f"Unknown command: {line}")
            self._write_line("Use /help to see the available shell commands.")
            return

        assert isinstance(self.state, DocumentState)
        self._write_line(
            f"[{self.state.document_name}] Chat is not wired yet. Received: {line}"
        )

    def _prompt(self) -> str:
        if isinstance(self.state, DocumentState):
            return f"document:{self.state.document_name}> "
        return "entry> "

    def _show_help(self) -> None:
        if isinstance(self.state, DocumentState):
            self._write_line("Current state: document")
            self._write_line(f"Active document: {self.state.document_name}")
            self._write_line("Available commands:")
            for command, description in DOCUMENT_HELP_ITEMS:
                self._write_line(f"{command} - {description}")
            return

        self._entry_feedback = []
        self._write_entry_feedback_line("Current state: entry")
        self._write_entry_feedback_line("Available commands:")
        for command, description in ENTRY_HELP_ITEMS:
            self._write_entry_feedback_line(f"{command} - {description}")

    def _render_current_view(self) -> None:
        if isinstance(self.state, EntryState):
            self._render_entry_screen()
            return

        assert isinstance(self.state, DocumentState)
        self._write_line("Current state: document")
        self._write_line(f"Active document: {self.state.document_name}")
        self._write(self._prompt())

    def _render_entry_screen(self, typed_buffer: str = "") -> None:
        if self._interactive_terminal:
            self._clear_screen()

        self._write_line("AskMyDoc CLI")
        self._write_line("Current state: entry")
        if self._entry_feedback:
            self._write_line("Feedback:")
            for line in self._entry_feedback:
                self._write_line(line)
        self._write_line("Starter commands:")

        for index, command in enumerate(ENTRY_STARTER_COMMANDS):
            indicator = ">" if index == self._entry_selection_index else " "
            description = self._description_for_entry_command(command)
            self._write_line(f"{indicator} {command:<7} {description}")

        self._write_line("Use Up/Down and Enter to choose, or type a command directly.")
        self._write(self._prompt() + typed_buffer)

    def _description_for_entry_command(self, command: str) -> str:
        descriptions = dict(ENTRY_HELP_ITEMS)
        return descriptions.get(command, "Starter command")

    def _write_entry_feedback_line(self, value: str) -> None:
        if not self._interactive_terminal:
            self._write_line(value)
            return

        self._entry_feedback.append(value)
        self._entry_feedback = self._entry_feedback[-12:]

    def _read_command(self) -> str | None:
        if isinstance(self.state, EntryState):
            return self._read_entry_command()
        return self._read_text_command()

    def _read_entry_command(self) -> str | None:
        buffer = ""

        while True:
            character = self._read_character()
            if character == "":
                return None if buffer == "" else buffer

            if character == "\x04":
                return None if buffer == "" else buffer

            if character in ("\r", "\n"):
                if self._interactive_terminal:
                    self._write_line("")
                stripped = buffer.strip()
                if stripped != "":
                    return stripped
                return ENTRY_STARTER_COMMANDS[self._entry_selection_index]

            if character in ("\x7f", "\b"):
                if buffer != "":
                    buffer = buffer[:-1]
                    if self._interactive_terminal:
                        self._write("\b \b")
                continue

            if character == "\x1b" and buffer == "":
                sequence = self._read_escape_sequence()
                if sequence == "[A":
                    self._entry_selection_index = (
                        self._entry_selection_index - 1
                    ) % len(ENTRY_STARTER_COMMANDS)
                    self._render_entry_screen(buffer)
                elif sequence == "[B":
                    self._entry_selection_index = (
                        self._entry_selection_index + 1
                    ) % len(ENTRY_STARTER_COMMANDS)
                    self._render_entry_screen(buffer)
                continue

            if character.isprintable():
                buffer += character
                if self._interactive_terminal:
                    self._write(character)

    def _read_text_command(self) -> str | None:
        buffer = ""

        while True:
            character = self._read_character()
            if character == "":
                return None if buffer == "" else buffer

            if character == "\x04":
                return None if buffer == "" else buffer

            if character in ("\r", "\n"):
                if self._interactive_terminal:
                    self._write_line("")
                return buffer

            if character in ("\x7f", "\b"):
                if buffer != "":
                    buffer = buffer[:-1]
                    if self._interactive_terminal:
                        self._write("\b \b")
                continue

            if character.isprintable():
                buffer += character
                if self._interactive_terminal:
                    self._write(character)

    def _read_character(self) -> str:
        return self.input_stream.read(1)

    def _read_escape_sequence(self) -> str:
        first = self._read_character()
        if first == "":
            return ""
        second = self._read_character()
        if second == "":
            return first
        return first + second

    @contextlib.contextmanager
    def _interactive_input_mode(self):
        if not self._interactive_terminal:
            yield
            return

        file_descriptor = self.input_stream.fileno()
        original_attributes = termios.tcgetattr(file_descriptor)
        updated_attributes = termios.tcgetattr(file_descriptor)
        updated_attributes[3] &= ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, updated_attributes)
        try:
            yield
        finally:
            termios.tcsetattr(file_descriptor, termios.TCSADRAIN, original_attributes)

    def _clear_screen(self) -> None:
        self._write("\x1b[2J\x1b[H")

    def _write(self, value: str) -> None:
        self.output_stream.write(value)
        self.output_stream.flush()

    def _write_line(self, value: str) -> None:
        self._write(f"{value}\n")


def run_cli(
    argv: list[str] | None = None,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> int:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) > 1:
        target = output_stream or sys.stderr
        target.write("AskMyDoc CLI does not accept startup arguments yet.\n")
        target.flush()
        return 2

    return AskMyDocShell(
        input_stream=input_stream,
        output_stream=output_stream,
    ).run()
