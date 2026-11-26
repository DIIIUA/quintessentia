"""Tests for disk-backed Token Ledger, whitespace/fallback KV prefix continuity, partial KV truncation, and compaction."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from quintessentia_cli.token_ledger import (
    DiskTokenLedger,
    canonical_message_sig,
    compute_tokenizer_fingerprint,
)
from quintessentia_cli.tool_calling import render_incremental_turn_suffix
from quintessentia_cli.unified_runtime import QuintessentiaKVPrefixSession


class _TinyTokenizer:
    """Minimal deterministic tokenizer supporting optional `chat_template`."""

    def __init__(self, *, chat_template: str | None = None, vocab_size: int = 1000) -> None:
        self.chat_template = chat_template
        self.vocab_size = vocab_size
        self.eos_token = "<eos>"
        self.eos_token_id = 2
        self.bos_token_id = 1

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, text: str, add_special_tokens: bool = True, return_tensors: str | None = None):
        # Stable byte-level token IDs shifted above special tokens so whitespace differences are observable!
        ids = [10 + b for b in text.encode("utf-8")]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))
        return SimpleNamespace(input_ids=ids)


def test_token_ledger_persists_archive_and_recovers_exact_tokens_after_restart(tmp_path: Path):
    tok = _TinyTokenizer(chat_template=None)
    fp = compute_tokenizer_fingerprint(tok)

    ledger1 = DiskTokenLedger(
        storage_dir=tmp_path,
        model_id="bloom-560m",
        tokenizer_fingerprint=fp,
        session_id="s1",
    )
    m1 = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": " The capital of France is Paris.  "},
    ]
    t1_ids = [1, 100, 101, 102, 200, 201, 202]
    ledger1.record_turn(
        full_messages=m1,
        all_token_ids=t1_ids,
        prompt_token_len=4,
        committed_len=len(t1_ids),
        eos_ended=False,
    )
    assert ledger1.active_path is not None and ledger1.active_path.is_file()
    assert ledger1.archive_path is not None and ledger1.archive_path.is_file()

    # Simulate full server restart (brand new ledger instance loading from disk)
    ledger2 = DiskTokenLedger(
        storage_dir=tmp_path,
        model_id="bloom-560m",
        tokenizer_fingerprint=fp,
        session_id="s1",
    )
    assert ledger2.active_tokens == t1_ids
    assert len(ledger2.turns) == 1

    # Client sends trimmed assistant text (`.strip()`, like OpenWebUI/Cursor/Gemma-2/BLOOM!) -> exact 100% match!
    m2_incoming = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Name one landmark."},
    ]
    match = ledger2.match_messages(m2_incoming)
    assert match.kind == "exact_continuation"
    assert match.prefix_token_ids == t1_ids
    assert match.tail_messages == [{"role": "user", "content": "Name one landmark."}]

    # Different tokenizer fingerprint MUST NOT load incompatible token IDs!
    ledger_wrong = DiskTokenLedger(
        storage_dir=tmp_path,
        model_id="bloom-560m",
        tokenizer_fingerprint="deadbeef_different_tok",
        session_id="s1",
    )
    assert ledger_wrong.active_tokens == []
    assert ledger_wrong.match_messages(m2_incoming).kind == "none"


def test_canonical_signature_ignores_trailing_spaces_and_think_blocks():
    a = {"role": "assistant", "content": "<think>internal chain</think>\n\nThe capital of France is Paris.  "}
    b = {"role": "assistant", "content": "The capital of France is Paris."}
    assert canonical_message_sig(a) == canonical_message_sig(b)


def test_render_incremental_turn_suffix_works_without_chat_template_for_base_models():
    """Fixes BLOOM / GPT-2 / Pythia / Mamba KV prefix loss on turn 2+."""
    tok = _TinyTokenizer(chat_template=None)
    suf = render_incremental_turn_suffix(
        tok,
        [{"role": "user", "content": "Name one famous landmark."}],
        enable_thinking=False,
        prev_ended_with_eos=False,
    )
    assert suf == "\n\nUser: Name one famous landmark.\n\nAssistant:"


def test_partial_turn_rewind_and_pure_attention_kv_truncation(tmp_path: Path):
    """Editing mid-conversation rewinds cleanly to the last matching turn boundary and truncates pure-attention KV."""
    tok = _TinyTokenizer(chat_template=None)
    fp = compute_tokenizer_fingerprint(tok)
    ledger = DiskTokenLedger(storage_dir=tmp_path, model_id="llama-1b", tokenizer_fingerprint=fp)

    m1 = [
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Reply 1"},
    ]
    ids1 = [1, 11, 12, 13, 14]
    ledger.record_turn(full_messages=m1, all_token_ids=ids1, prompt_token_len=3, committed_len=5, eos_ended=True)

    m2 = m1 + [
        {"role": "user", "content": "Turn 2"},
        {"role": "assistant", "content": "Reply 2"},
    ]
    ids2 = ids1 + [21, 22, 23, 24]
    ledger.record_turn(
        full_messages=m2,
        all_token_ids=ids2,
        prompt_token_len=7,
        committed_len=9,
        eos_ended=True,
        recovery_kind="exact_continuation",
    )

    # Client edits Turn 2 -> matches Turn 1 boundary as `partial_turn_prefix`!
    m_edited = m1 + [{"role": "user", "content": "Edited Turn 2"}]
    match = ledger.match_messages(m_edited)
    assert match.kind == "partial_turn_prefix"
    assert match.matched_turn_index == 0
    assert match.prefix_token_ids == ids1

    # Pure-attention parked KV session truncates smoothly to `len(ids1) == 5` instead of dropping to 0!
    sess = QuintessentiaKVPrefixSession()
    sess.fed = list(ids2)
    sess.parked_backbone = {
        0: {
            "keys": torch.randn((1, 4, len(ids2), 8)),
            "values": torch.randn((1, 4, len(ids2), 8)),
        }
    }
    sess.parked_bytes = sess.parked_backbone[0]["keys"].nbytes * 2
    assert sess.can_truncate_parked_prefix() is True
    reused = sess.kv_prefix_reuse(ids1 + [99, 100], allow_truncate=True)
    assert reused == len(ids1)
    assert sess.fed == ids1
    assert sess.parked_backbone[0]["keys"].shape == (1, 4, len(ids1), 8)

    # Recurrent hybrid layer (`linear_states`) MUST NOT be truncated (safety invariant!)
    sess_rec = QuintessentiaKVPrefixSession()
    sess_rec.fed = list(ids2)
    sess_rec.parked_backbone = {
        0: {"keys": torch.randn((1, 4, len(ids2), 8)), "values": torch.randn((1, 4, len(ids2), 8))},
        1: {"linear_states": {0: {"has_previous_state": True}}},
    }
    assert sess_rec.can_truncate_parked_prefix() is False
    assert sess_rec.kv_prefix_reuse(ids1 + [99, 100], allow_truncate=True) == 0


def test_compaction_archives_old_epoch_and_rewrites_active_window_cleanly(tmp_path: Path):
    """Context compaction preserves old epochs in `ledger.jsonl` while atomically rewriting `active_window.json`
    with ONLY the exact compact token sequence needed for 100% deterministic retokenization."""
    tok = _TinyTokenizer(chat_template=None)
    fp = compute_tokenizer_fingerprint(tok)
    ledger = DiskTokenLedger(
        storage_dir=tmp_path,
        model_id="qwythos-9b",
        tokenizer_fingerprint=fp,
        max_active_tokens=16,
    )
    m_old = [
        {"role": "user", "content": "Long history turn"},
        {"role": "assistant", "content": "Long answer"},
    ]
    ledger.record_turn(
        full_messages=m_old,
        all_token_ids=list(range(100, 140)),  # 40 tokens
        prompt_token_len=20,
        committed_len=40,
        eos_ended=True,
    )
    assert ledger.epoch == 0
    assert len(ledger.active_tokens) == 40

    compacted_msgs = [
        {"role": "user", "content": "[Context Summary]\nKey facts."},
        {"role": "assistant", "content": "Understood."},
    ]
    compacted_ids = [1, 50, 51, 52, 53, 54, 55, 56]
    rep = ledger.compact_active_window(
        compacted_messages=compacted_msgs,
        compacted_token_ids=compacted_ids,
        eos_ended=True,
        reason="test_compaction",
    )
    assert rep["closed_epoch"] == 0 and rep["new_epoch"] == 1
    assert rep["old_tokens"] == 40 and rep["new_tokens"] == len(compacted_ids)
    assert ledger.active_tokens == compacted_ids

    # Disk `active_window.json` holds ONLY the new compact epoch
    on_disk = json.loads(ledger.active_path.read_text(encoding="utf-8"))
    assert on_disk["epoch"] == 1
    assert on_disk["active_tokens"] == compacted_ids

    # Disk `ledger.jsonl` STILL retains the full old turn AND the compaction event (never erased!)
    archive_events = [json.loads(line) for line in ledger.archive_path.read_text(encoding="utf-8").splitlines()]
    event_types = [e["event"] for e in archive_events]
    assert "turn" in event_types and "compaction" in event_types
    assert archive_events[0]["token_end"] == 40

    # Follow-up turn matches the compacted epoch directly from disk
    reloaded = DiskTokenLedger(storage_dir=tmp_path, model_id="qwythos-9b", tokenizer_fingerprint=fp)
    next_msgs = compacted_msgs + [{"role": "user", "content": "Continue from summary"}]
    m_next = reloaded.match_messages(next_msgs)
    assert m_next.kind == "exact_continuation"
    assert m_next.epoch == 1
    assert m_next.prefix_token_ids == compacted_ids
