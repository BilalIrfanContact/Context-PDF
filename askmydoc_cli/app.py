from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO


@dataclass(frozen=True)
class EntryState:
    name: str = "entry"


@dataclass(frozen=True)
class DocumentState:
    document_name: str
    name: str = "document"


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

    def run(self) -> int:
        self._write_line("AskMyDoc CLI")
        self._write_line("Interactive shell scaffold. Type /help to get started.")

        while self._running:
            self._write(self._prompt())
            raw_line = self.input_stream.readline()

            if raw_line == "":
                self._write_line("")
                self._write_line("Session closed.")
                break

            self.handle_line(raw_line.strip())

        return 0

    def handle_line(self, line: str) -> None:
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
        self._write_line(f"Closed document: {closed_document}")

    def _handle_entry_line(self, line: str) -> None:
        if line == "/open" or line.startswith("/open "):
            document_name = line.removeprefix("/open").strip()
            if document_name == "":
                self._write_line("Usage: /open <document-name>")
                return

            self.open_document(document_name)
            return

        if line.startswith("/"):
            self._write_line(f"Unknown command: {line}")
            self._write_line("Use /help to see the available shell commands.")
            return

        self._write_line("Open a document before sending messages. Use /open <document-name>.")

    def _handle_document_line(self, line: str) -> None:
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
        return "askmydoc> "

    def _show_help(self) -> None:
        if isinstance(self.state, DocumentState):
            self._write_line("Document commands:")
            self._write_line("/help  Show this help")
            self._write_line("/close Return to the entry screen")
            self._write_line("/exit  Close the AskMyDoc shell")
            self._write_line("<text>  Send a placeholder message to the active document")
            return

        self._write_line("Entry commands:")
        self._write_line("/help  Show this help")
        self._write_line("/open <document-name>  Enter a scaffolded document session")
        self._write_line("/exit  Close the AskMyDoc shell")

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
