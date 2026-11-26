"""Disk-backed two-layer Token Ledger for deterministic 100% retokenization, partial KV reuse, and compaction.

Design (Anton's token-history persistence + non-destructive archive):
  1. `ledger.jsonl` (append-only archive):
     Preserves every committed turn and every context-compaction boundary across epochs.
     Old epochs are never erased, keeping full auditability and rollback history on disk.
  2. `active_window.json` (atomically rewritten minimal active slice):
     Stores ONLY the exact `token_ids`, per-turn boundaries, and canonical message signatures
     required for 100% bit-exact retokenization of the currently active context window.
     On context compaction, the closed epoch is archived in `ledger.jsonl` and
     `active_window.json` is atomically rewritten with the new compact epoch
     (`[system] + [summary] + [retained live tail turns]`) sized to fit the active context.

Zero BPE boundary drift:
  When a client trims whitespace (`.strip()`), drops `<think>` blocks, restarts the server,
  or edits a mid-conversation turn, the ledger matches canonical message signatures and
  supplies the exact original `token_ids` instead of re-tokenizing old text from scratch.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _safe_slug(name: str, fallback: str = "default") -> str:
    cleaned = _SAFE_NAME_RE.sub("_", str(name or "").strip()).strip("._-")
    return cleaned[:96] if cleaned else fallback


def compute_tokenizer_fingerprint(tokenizer: Any) -> str:
    """Stable short SHA-256 fingerprint binding a Token Ledger to one exact tokenizer."""
    h = hashlib.sha256()
    vocab_sz = getattr(tokenizer, "vocab_size", None)
    try:
        tot_len = len(tokenizer)
    except Exception:
        tot_len = vocab_sz
    h.update(f"v={vocab_sz}|len={tot_len}|".encode("utf-8"))
    h.update(f"eos={getattr(tokenizer, 'eos_token_id', None)}|bos={getattr(tokenizer, 'bos_token_id', None)}|".encode("utf-8"))
    tmpl = getattr(tokenizer, "chat_template", None)
    if tmpl:
        h.update(str(tmpl).encode("utf-8", errors="replace"))
    # Probe tokenization of boundary-sensitive text (leading space, newline, colon)
    try:
        probe_ids = tokenizer("Assistant: The quick\n\nUser: 123", add_special_tokens=False).input_ids
        h.update(f"|probe={probe_ids}".encode("utf-8"))
    except Exception:
        pass
    return h.hexdigest()[:24]


def canonical_message_sig(msg: dict[str, Any]) -> str:
    """Whitespace- and `<think>`-tolerant canonical signature for one chat message.

    Matches identical conversational content even when a client strips trailing/leading
    spaces (e.g. Gemma-2 trailing spaces before `<end_of_turn>`, BLOOM leading space)
    or omits prior `<think>...</think>` blocks.
    """
    role = str(msg.get("role") or "user").strip().lower()
    raw_content = msg.get("content", "")
    if isinstance(raw_content, list):
        raw_content = "".join(
            str(p.get("text", "")) for p in raw_content if isinstance(p, dict)
        )
    text = str(raw_content or "")
    if role == "assistant":
        text = _THINK_BLOCK_RE.sub("", text)
    norm_text = text.strip()

    tc_canon: list[tuple[str, str]] = []
    raw_tcs = msg.get("tool_calls")
    if isinstance(raw_tcs, list):
        for tc in raw_tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            fname = str(fn.get("name") or "")
            fargs = fn.get("arguments", "")
            if isinstance(fargs, str):
                try:
                    parsed = json.loads(fargs) if fargs.strip() else {}
                    fargs_s = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
                except Exception:
                    fargs_s = fargs.strip()
            else:
                fargs_s = json.dumps(fargs, sort_keys=True, ensure_ascii=False)
            tc_canon.append((fname, fargs_s))

    tool_name = str(msg.get("name") or "") if role == "tool" else ""
    payload = json.dumps(
        {"r": role, "c": norm_text, "tc": tc_canon, "n": tool_name},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def canonical_message_sigs(messages: list[dict[str, Any]]) -> list[str]:
    return [canonical_message_sig(m) for m in messages]


@dataclass(frozen=True)
class LedgerMatch:
    """Result of matching an incoming message list against the disk-backed Token Ledger."""

    kind: str                       # "exact_continuation" | "partial_turn_prefix" | "none"
    prefix_token_ids: list[int]     # Exact historical token IDs up to the matched turn boundary
    tail_messages: list[dict[str, Any]]  # New messages to append after `prefix_token_ids`
    matched_turn_index: int         # Matched turn index (-1 if none)
    prev_ended_with_eos: bool       # Whether `prefix_token_ids` ended on an EOS token
    matched_message_count: int      # How many leading messages matched canonically
    epoch: int                      # Ledger epoch that supplied the match


class DiskTokenLedger:
    """Two-layer persistent token journal (`ledger.jsonl` archive + `active_window.json` slice)."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        storage_dir: str | Path | None,
        model_id: str,
        tokenizer_fingerprint: str,
        session_id: str = "default",
        enabled: bool = True,
        max_active_tokens: int | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.model_id = str(model_id or "model")
        self.tokenizer_fingerprint = str(tokenizer_fingerprint or "")
        self.session_id = _safe_slug(session_id, "default")
        self.max_active_tokens = (
            int(max_active_tokens) if (max_active_tokens is not None and int(max_active_tokens) > 0) else None
        )

        self.base_dir: Path | None = None
        self.active_path: Path | None = None
        self.archive_path: Path | None = None
        if self.enabled and storage_dir is not None:
            self.base_dir = Path(storage_dir) / _safe_slug(self.model_id, "model") / self.session_id
            self.active_path = self.base_dir / "active_window.json"
            self.archive_path = self.base_dir / "ledger.jsonl"

        self.epoch: int = 0
        self.tools_sig: str = ""
        self.chatml_think_format: bool = False
        self.active_tokens: list[int] = []
        self.committed_len: int = 0
        self.turns: list[dict[str, Any]] = []
        self.last_recovery_kind: str = "fresh"
        self.last_recovered_tokens: int = 0
        self.total_compactions: int = 0

        if self.enabled and self.active_path is not None and self.active_path.is_file():
            self.load_active_window()

    # ------------------------------------------------------------------ persistence

    def _ensure_dir(self) -> None:
        if self.base_dir is not None:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def _append_archive(self, record: dict[str, Any]) -> None:
        if not self.enabled or self.archive_path is None:
            return
        try:
            self._ensure_dir()
            rec = {"ts": round(time.time(), 3), **record}
            with self.archive_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _write_active_window_atomic(self) -> None:
        if not self.enabled or self.active_path is None:
            return
        try:
            self._ensure_dir()
            payload = {
                "schema_version": self.SCHEMA_VERSION,
                "model_id": self.model_id,
                "tokenizer_fingerprint": self.tokenizer_fingerprint,
                "session_id": self.session_id,
                "epoch": self.epoch,
                "tools_sig": self.tools_sig,
                "chatml_think_format": self.chatml_think_format,
                "committed_len": self.committed_len,
                "active_tokens": self.active_tokens,
                "turns": self.turns,
                "updated_at": round(time.time(), 3),
            }
            tmp_path = self.active_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp_path, self.active_path)
        except Exception:
            pass

    def load_active_window(self) -> bool:
        """Load `active_window.json` if it exists and matches this exact tokenizer fingerprint."""
        if not self.enabled or self.active_path is None or not self.active_path.is_file():
            return False
        try:
            data = json.loads(self.active_path.read_text(encoding="utf-8"))
            if int(data.get("schema_version", 0)) != self.SCHEMA_VERSION:
                return False
            if (
                self.tokenizer_fingerprint
                and data.get("tokenizer_fingerprint")
                and str(data.get("tokenizer_fingerprint")) != self.tokenizer_fingerprint
            ):
                # Different tokenizer -> refuse to inject incompatible token IDs!
                return False
            toks = [int(t) for t in (data.get("active_tokens") or [])]
            turns = list(data.get("turns") or [])
            self.epoch = int(data.get("epoch", 0))
            self.tools_sig = str(data.get("tools_sig", ""))
            self.chatml_think_format = bool(data.get("chatml_think_format", False))
            self.active_tokens = toks
            self.committed_len = max(0, min(len(toks), int(data.get("committed_len", len(toks)))))
            self.turns = turns
            return bool(toks and turns)
        except Exception:
            return False

    # ------------------------------------------------------------------ matching & recording

    def match_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools_sig: str = "",
    ) -> LedgerMatch:
        """Match incoming conversation against the active epoch's turn boundaries.

        Checks:
          1. Exact continuation of the latest recorded turn (most common multi-turn case).
          2. Partial prefix match against an earlier turn boundary (edit/rewind in middle of chat).
        """
        if not self.enabled or not self.turns or not self.active_tokens:
            return LedgerMatch("none", [], list(messages), -1, False, 0, self.epoch)
        if str(tools_sig or "") != self.tools_sig:
            return LedgerMatch("none", [], list(messages), -1, False, 0, self.epoch)

        in_sigs = canonical_message_sigs(messages)
        if not in_sigs:
            return LedgerMatch("none", [], list(messages), -1, False, 0, self.epoch)

        # Scan from latest turn backwards so the longest valid turn-boundary prefix wins.
        last_idx = len(self.turns) - 1
        for idx in range(last_idx, -1, -1):
            t = self.turns[idx]
            t_sigs = list(t.get("canonical_sigs") or [])
            m_cnt = len(t_sigs)
            if m_cnt <= 0 or len(in_sigs) <= m_cnt:
                continue
            if in_sigs[:m_cnt] != t_sigs:
                continue
            tail = messages[m_cnt:]
            if not tail or not all(str(m.get("role") or "") in ("user", "tool") for m in tail):
                continue
            tok_end = int(t.get("token_end", 0))
            if tok_end <= 0 or tok_end > len(self.active_tokens):
                continue
            kind = "exact_continuation" if idx == last_idx else "partial_turn_prefix"
            return LedgerMatch(
                kind=kind,
                prefix_token_ids=list(self.active_tokens[:tok_end]),
                tail_messages=list(tail),
                matched_turn_index=idx,
                prev_ended_with_eos=bool(t.get("eos_ended", False)),
                matched_message_count=m_cnt,
                epoch=self.epoch,
            )

        return LedgerMatch("none", [], list(messages), -1, False, 0, self.epoch)

    def record_turn(
        self,
        *,
        full_messages: list[dict[str, Any]],
        all_token_ids: list[int],
        prompt_token_len: int,
        committed_len: int,
        eos_ended: bool,
        tools_sig: str = "",
        chatml_think_format: bool = False,
        recovery_kind: str = "fresh",
        matched_turn_index: int = -1,
    ) -> None:
        """Persist one completed turn to both `ledger.jsonl` (archive) and `active_window.json`."""
        if not self.enabled:
            return
        ids_clean = [int(t) for t in all_token_ids]
        sigs = canonical_message_sigs(full_messages)
        t_sig = str(tools_sig or "")

        if recovery_kind == "exact_continuation" and t_sig == self.tools_sig and self.turns:
            token_start = int(self.turns[-1].get("token_end", 0))
            turn_index = len(self.turns)
        elif (
            recovery_kind == "partial_turn_prefix"
            and t_sig == self.tools_sig
            and 0 <= matched_turn_index < len(self.turns)
        ):
            # Rewind active window turns cleanly to the matched branch point; archive keeps the old branch!
            self._append_archive(
                {
                    "event": "branch_rewind",
                    "epoch": self.epoch,
                    "from_turns": len(self.turns),
                    "to_turn_index": matched_turn_index,
                }
            )
            self.turns = list(self.turns[: matched_turn_index + 1])
            token_start = int(self.turns[-1].get("token_end", 0))
            turn_index = len(self.turns)
        else:
            # New conversation branch / fresh epoch
            if self.turns:
                self._append_archive(
                    {
                        "event": "epoch_closed",
                        "epoch": self.epoch,
                        "total_turns": len(self.turns),
                        "total_tokens": len(self.active_tokens),
                        "reason": recovery_kind,
                    }
                )
                self.epoch += 1
            self.turns = []
            token_start = 0
            turn_index = 0

        turn_rec = {
            "epoch": self.epoch,
            "turn_index": turn_index,
            "token_start": token_start,
            "prompt_end": int(prompt_token_len),
            "token_end": len(ids_clean),
            "eos_ended": bool(eos_ended),
            "canonical_sigs": sigs,
            "messages": [
                {
                    "role": str(m.get("role") or ""),
                    "content": str(m.get("content") or ""),
                    **({"tool_calls": m["tool_calls"]} if m.get("tool_calls") else {}),
                    **({"name": m["name"]} if m.get("name") else {}),
                }
                for m in full_messages
            ],
        }
        self.turns.append(turn_rec)
        self.active_tokens = ids_clean
        self.committed_len = max(0, min(len(ids_clean), int(committed_len)))
        self.tools_sig = t_sig
        self.chatml_think_format = bool(chatml_think_format)

        self._append_archive(
            {
                "event": "turn",
                "epoch": self.epoch,
                "turn_index": turn_index,
                "recovery_kind": recovery_kind,
                "token_start": token_start,
                "prompt_end": int(prompt_token_len),
                "token_end": len(ids_clean),
                "new_tokens": ids_clean[token_start:],
                "eos_ended": bool(eos_ended),
                "message_count": len(full_messages),
            }
        )
        self._write_active_window_atomic()

    def compact_active_window(
        self,
        *,
        compacted_messages: list[dict[str, Any]],
        compacted_token_ids: list[int],
        eos_ended: bool = True,
        tools_sig: str | None = None,
        reason: str = "context_compaction",
        max_active_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Archive the current epoch in `ledger.jsonl` and atomically rewrite `active_window.json`
        with ONLY the compact token sequence required for 100% exact retokenization of the new epoch.
        """
        old_epoch = self.epoch
        old_tokens = len(self.active_tokens)
        old_turns = len(self.turns)

        ids_new = [int(t) for t in compacted_token_ids]
        cap = max_active_tokens if max_active_tokens is not None else self.max_active_tokens
        trimmed = False
        if cap is not None and cap > 0 and len(ids_new) > cap:
            # Preserve exact tail within the hard active-window token cap if requested.
            ids_new = ids_new[-cap:]
            trimmed = True

        self._append_archive(
            {
                "event": "compaction",
                "closed_epoch": old_epoch,
                "new_epoch": old_epoch + 1,
                "reason": str(reason),
                "old_tokens": old_tokens,
                "old_turns": old_turns,
                "new_tokens": len(ids_new),
                "trimmed_to_cap": trimmed,
            }
        )

        self.epoch = old_epoch + 1
        self.total_compactions += 1
        if tools_sig is not None:
            self.tools_sig = str(tools_sig)
        self.active_tokens = ids_new
        self.committed_len = 0  # Needs fresh prefill of the new compact epoch
        self.turns = [
            {
                "epoch": self.epoch,
                "turn_index": 0,
                "token_start": 0,
                "prompt_end": len(ids_new),
                "token_end": len(ids_new),
                "eos_ended": bool(eos_ended),
                "canonical_sigs": canonical_message_sigs(compacted_messages),
                "messages": [
                    {
                        "role": str(m.get("role") or ""),
                        "content": str(m.get("content") or ""),
                        **({"tool_calls": m["tool_calls"]} if m.get("tool_calls") else {}),
                        **({"name": m["name"]} if m.get("name") else {}),
                    }
                    for m in compacted_messages
                ],
            }
        ]
        self.last_recovery_kind = "compacted_epoch"
        self.last_recovered_tokens = len(ids_new)
        self._write_active_window_atomic()

        return {
            "closed_epoch": old_epoch,
            "new_epoch": self.epoch,
            "old_tokens": old_tokens,
            "new_tokens": len(ids_new),
            "trimmed_to_cap": trimmed,
            "active_path": str(self.active_path) if self.active_path else None,
            "archive_path": str(self.archive_path) if self.archive_path else None,
        }

    def clear_active_window(self, *, reason: str = "session_reset") -> None:
        """Reset active window while preserving the append-only archive (`ledger.jsonl`)."""
        if self.turns or self.active_tokens:
            self._append_archive(
                {
                    "event": "active_cleared",
                    "epoch": self.epoch,
                    "old_tokens": len(self.active_tokens),
                    "old_turns": len(self.turns),
                    "reason": reason,
                }
            )
            self.epoch += 1
        self.active_tokens = []
        self.committed_len = 0
        self.turns = []
        self.tools_sig = ""
        self.last_recovery_kind = "fresh"
        self.last_recovered_tokens = 0
        self._write_active_window_atomic()

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "epoch": self.epoch,
            "active_tokens": len(self.active_tokens),
            "committed_len": self.committed_len,
            "active_turns": len(self.turns),
            "total_compactions": self.total_compactions,
            "last_recovery_kind": self.last_recovery_kind,
            "last_recovered_tokens": self.last_recovered_tokens,
            "active_path": str(self.active_path) if self.active_path else None,
            "archive_path": str(self.archive_path) if self.archive_path else None,
        }
