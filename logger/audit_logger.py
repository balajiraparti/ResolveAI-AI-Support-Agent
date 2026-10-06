"""
logger/audit_logger.py
Append-only, in-process audit trail for the ResolveAI support workflow.

Each graph node emits one AuditEvent via _audit.emit(). The singleton
_audit is shared across all requests in the process. The FastAPI layer
exposes GET /audit to read the log.

Persistence: every event is also appended to  logger/logs/audit.txt
so the trail survives a server restart inspection (read-only reference).
"""

import uuid
import json
import datetime
import threading
from dataclasses import dataclass, asdict
from pathlib import Path


# ── Log-file location ─────────────────────────────────────────────────────────
_LOG_DIR  = Path(__file__).resolve().parent / "logs"
_LOG_FILE = _LOG_DIR / "audit.txt"


def _ensure_log_dir() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Event schema ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AuditEvent:
    event_id:   str    # AUD_<10 hex chars>
    ts:         str    # ISO-8601 UTC timestamp
    session_id: str    # multi-turn session
    thread_id:  str    # single LangGraph run
    node:       str    # actor / node name  (e.g. DECISION_NODE)
    event_type: str    # semantic label     (e.g. RETRIEVAL_DECISION)
    status:     str    # SUCCESS | ESCALATED | WARNING | PENDING | FAILURE
    message:    str    # human-readable one-liner (success summary or error text)
    details:    dict   # arbitrary JSON-serialisable payload


# ── Logger ────────────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Thread-safe append-only audit store.

    In-memory list for fast reads + file append for persistence.
    File writes use a threading.Lock so concurrent requests don't interleave
    partial lines in the log file.
    """

    def __init__(self, log_file: Path = _LOG_FILE):
        self._log:  list[AuditEvent] = []
        self._file: Path = log_file
        self._lock: threading.Lock = threading.Lock()
        _ensure_log_dir()

    # ── write ─────────────────────────────────────────────────────────────────

    def emit(
        self,
        node:       str,
        event_type: str,
        status:     str  = "SUCCESS",
        message:    str  = "",
        session_id: str  = "",
        thread_id:  str  = "",
        **details,
    ) -> AuditEvent:
        """Append one event to in-memory log AND persist to audit.txt."""
        ev = AuditEvent(
            event_id   = "AUD_" + uuid.uuid4().hex[:10].upper(),
            ts         = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            session_id = session_id or "",
            thread_id  = thread_id  or "",
            node       = node,
            event_type = event_type,
            status     = status,
            message    = message or "",
            details    = dict(details),
        )
        self._log.append(ev)

        # Console echo
        icon = {"SUCCESS": "✓", "FAILURE": "✗", "WARNING": "⚠", "ESCALATED": "↑", "PENDING": "…"}.get(status, "?")
        print(
            f"[AUDIT] {icon} {ev.ts} | {ev.node:<22} | {ev.event_type:<28} | {ev.status}"
            + (f" | {message[:80]}" if message else "")
        )

        self._write_to_file(ev)
        return ev

    def _write_to_file(self, ev: AuditEvent) -> None:
        """Append a single JSON line to the audit log file (thread-safe)."""
        line = json.dumps(asdict(ev), ensure_ascii=False) + "\n"
        with self._lock:
            try:
                with self._file.open("a", encoding="utf-8") as fh:
                    fh.write(line)
            except OSError as exc:
                print(f"[AUDIT] WARNING: could not write to {self._file}: {exc}")

    # ── read ──────────────────────────────────────────────────────────────────

    def get_all(self) -> list[dict]:
        return [asdict(e) for e in self._log]

    def get_by_session(self, session_id: str) -> list[dict]:
        return [asdict(e) for e in self._log if e.session_id == session_id]

    def get_by_thread(self, thread_id: str) -> list[dict]:
        return [asdict(e) for e in self._log if e.thread_id == thread_id]

    def count(self) -> int:
        return len(self._log)

    def clear(self) -> None:
        """Admin-only: wipe the in-memory log (file is NOT truncated — immutable)."""
        self._log.clear()

    @property
    def log_file(self) -> Path:
        return self._file

    def load_from_file(self) -> int:
        """Re-hydrate in-memory log from persisted audit.txt after a server restart."""
        if not self._file.exists():
            return 0
        loaded = 0
        with self._file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    ev   = AuditEvent(**data)
                    self._log.append(ev)
                    loaded += 1
                except Exception:
                    pass
        return loaded


# ── Module singleton ──────────────────────────────────────────────────────────
_audit = AuditLogger()
