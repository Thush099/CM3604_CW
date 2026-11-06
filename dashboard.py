#!/usr/bin/env python3
"""
CM3604 Deep Learning Coursework — Local Progress & Testing Dashboard
====================================================================
100% standalone developer UI using only Python 3.10 standard library.
Does NOT modify `factchecking_main.py` and is NOT imported by `factcheck.py`,
so `factcheck.py` remains completely clean for CampusMoodle submission.

Usage:
    source .venv/bin/activate
    python dashboard.py
    # Then open http://127.0.0.1:8765 in your browser
"""

import argparse
import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OFFICIAL_MAIN_SHA256 = "5c40a1ec1a41cbdb8a9655377fe46cefcf2feceba57026a12d2a340eec5783b8"
ALLOWED_THIRD_PARTY = {
    "numpy",
    "torch",
    "torchvision",
    "tqdm",
    "spacy",
    "nltk",
    "transformers",
    "sentencepiece",
    "google",  # protobuf
    "rouge_score",
}

ROOT_DIR = Path(__file__).resolve().parent
LABELS_PATH = ROOT_DIR / "data" / "dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = ROOT_DIR / "data" / "passages_bm25_ChatGPT_humfacts.jsonl"

_DATA_CACHE: Optional[List[Any]] = None
_DEBERTA_CACHE: Dict[str, Any] = {}
_JOB_LOCK = threading.Lock()
_CURRENT_JOB: Dict[str, Any] = {
    "running": False,
    "mode": None,
    "done": 0,
    "total": 0,
    "correct_so_far": 0,
    "started_at": None,
    "elapsed_sec": 0.0,
    "error": None,
    "result": None,
}
_HISTORY: Dict[str, Any] = {}


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_factcheck_imports(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"ok": False, "missing_file": True, "disallowed": [], "imports": []}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return {"ok": False, "syntax_error": f"Line {exc.lineno}: {exc.msg}", "disallowed": [], "imports": []}

    stdlib = getattr(sys, "stdlib_module_names", set())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])

    disallowed = sorted(
        m for m in found if m not in stdlib and m not in ALLOWED_THIRD_PARTY and m != "__future__"
    )
    return {"ok": len(disallowed) == 0, "disallowed": disallowed, "imports": sorted(found)}


def _git_info() -> Dict[str, Any]:
    def _run(cmd: List[str]) -> str:
        try:
            res = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, timeout=3)
            return res.stdout.strip() if res.returncode == 0 else ""
        except Exception:
            return ""

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    remote = _run(["git", "remote", "get-url", "origin"])
    commits_raw = _run(["git", "log", "-n", "5", "--pretty=format:%h|%an|%ar|%s"])
    commits = []
    for line in commits_raw.splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            commits.append({"hash": parts[0], "author": parts[1], "when": parts[2], "subject": parts[3]})
    status_lines = [ln for ln in _run(["git", "status", "--porcelain"]).splitlines() if ln.strip()]
    return {
        "is_repo": bool(branch),
        "branch": branch or "not initialized",
        "remote": remote or "none",
        "dirty_count": len(status_lines),
        "commits": commits,
    }


def _load_examples() -> List[Dict[str, Any]]:
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE
    if not LABELS_PATH.exists() or not PASSAGES_PATH.exists():
        return []

    fact_to_passages: Dict[str, List[dict]] = {}
    with PASSAGES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            fact_to_passages[row["sent"]] = row["passages"]

    items: List[Dict[str, Any]] = []
    with LABELS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            bio = json.loads(line)
            topic = bio.get("topic", "")
            if bio.get("annotations"):
                for sent in bio["annotations"]:
                    facts = sent.get("human-atomic-facts") or []
                    for fact in facts:
                        text = fact["text"]
                        if text in fact_to_passages:
                            raw_label = fact["label"]
                            bin_label = "NS" if raw_label == "IR" else raw_label
                            items.append(
                                {
                                    "idx": len(items),
                                    "topic": topic or (fact_to_passages[text][0]["title"] if fact_to_passages[text] else ""),
                                    "fact": text,
                                    "raw_label": raw_label,
                                    "gold": bin_label,
                                    "passages": fact_to_passages[text],
                                }
                            )
    _DATA_CACHE = items
    return items


def _load_fresh_factcheck():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    if "factcheck" in sys.modules:
        return importlib.reload(sys.modules["factcheck"])
    return importlib.import_module("factcheck")


def _get_deberta():
    if "model" not in _DEBERTA_CACHE:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        _DEBERTA_CACHE["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
        _DEBERTA_CACHE["model"] = AutoModelForSequenceClassification.from_pretrained(model_name)
    return _DEBERTA_CACHE["model"], _DEBERTA_CACHE["tokenizer"]


def _compute_metrics(confusion: List[List[int]]) -> Dict[str, Any]:
    # confusion indexed as [gold_idx][pred_idx] for ["S", "NS"]
    tp = confusion[0][0]
    fn = confusion[0][1]
    fp = confusion[1][0]
    tn = confusion[1][1]
    total = tp + fn + fp + tn
    correct = tp + tn
    acc = (correct / total) if total else 0.0

    def _prf(c_tp: int, c_fp: int, c_fn: int) -> Tuple[float, float, float]:
        prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return round(prec, 4), round(rec, 4), round(f1, 4)

    p_s, r_s, f1_s = _prf(tp, fp, fn)
    p_ns, r_ns, f1_ns = _prf(tn, fn, fp)
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100.0, 2),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "s_metrics": {"precision": p_s, "recall": r_s, "f1": f1_s},
        "ns_metrics": {"precision": p_ns, "recall": r_ns, "f1": f1_ns},
    }


def get_system_status() -> Dict[str, Any]:
    main_sha = _sha256_file(ROOT_DIR / "factchecking_main.py")
    examples = _load_examples()
    gitignore_ok = (ROOT_DIR / ".gitignore").exists()
    gitnore_typo = (ROOT_DIR / ".gitnore").exists()
    sha_typo = (ROOT_DIR / ".factchecking_").exists()
    imports_check = _check_factcheck_imports(ROOT_DIR / "factcheck.py")

    # Quick probe of whether Part 1 / Part 2 are still starter stubs
    part1_state = "unknown"
    part2_state = "unknown"
    if imports_check.get("ok") and not imports_check.get("syntax_error"):
        try:
            fc = _load_fresh_factcheck()
            try:
                checker = fc.WordRecallThresholdFactChecker()
                _ = checker.predict("Test fact.", [{"title": "Test", "text": "Test fact."}])
                part1_state = "implemented"
            except Exception as exc:
                msg = str(exc).lower()
                part1_state = "stub" if ("implement" in msg or isinstance(exc, NotImplementedError)) else f"error: {exc}"

            p2_src = Path(ROOT_DIR / "factcheck.py").read_text(encoding="utf-8")
            if 'raise Exception("Not implemented")' in p2_src or "TODO (Part 2): Implement EntailmentFactChecker" in p2_src:
                part2_state = "stub"
            else:
                part2_state = "implemented"
        except Exception as exc:
            part1_state = f"error: {exc}"

    return {
        "workspace": str(ROOT_DIR),
        "python_version": sys.version.split()[0],
        "main_driver_untouched": main_sha == OFFICIAL_MAIN_SHA256,
        "main_driver_sha256": main_sha,
        "dataset_loaded": len(examples) == 221,
        "num_examples": len(examples),
        "gitignore_ok": gitignore_ok,
        "typos_detected": {
            "gitnore": gitnore_typo,
            "factchecking_prefix": sha_typo,
        },
        "imports_check": imports_check,
        "part1_code_state": part1_state,
        "part2_code_state": part2_state,
        "git": _git_info(),
        "history": _HISTORY,
    }


def _run_evaluation_worker(mode: str, limit: int = 0):
    started = time.time()
    try:
        examples = _load_examples()
        if not examples:
            raise RuntimeError("Dataset files missing in data/ folder.")

        if limit and 0 < limit < len(examples):
            # Balanced deterministic slice across dataset
            step = max(1, len(examples) // limit)
            subset = examples[::step][:limit]
        else:
            subset = examples

        with _JOB_LOCK:
            _CURRENT_JOB.update(
                {
                    "running": True,
                    "mode": mode,
                    "done": 0,
                    "total": len(subset),
                    "correct_so_far": 0,
                    "started_at": started,
                    "elapsed_sec": 0.0,
                    "error": None,
                    "result": None,
                }
            )

        fc = _load_fresh_factcheck()
        if mode == "always_entail":
            checker = fc.AlwaysEntailedFactChecker()
        elif mode == "random":
            checker = fc.RandomGuessFactChecker()
        elif mode == "word_overlap":
            checker = fc.WordRecallThresholdFactChecker()
        elif mode == "entailment":
            model, tokenizer = _get_deberta()
            ent_model = fc.EntailmentModel(model, tokenizer, False)
            checker = fc.EntailmentFactChecker(ent_model)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        confusion = [[0, 0], [0, 0]]
        errors: List[Dict[str, Any]] = []

        for pos, item in enumerate(subset, start=1):
            pred = checker.predict(item["fact"], item["passages"])
            if pred not in ("S", "NS"):
                raise ValueError(f"predict() must return 'S' or 'NS', got {pred!r}")

            score = None
            if hasattr(checker, "score_example"):
                try:
                    score = round(float(checker.score_example(item["fact"], item["passages"])), 4)
                except Exception:
                    score = None

            g_idx = 0 if item["gold"] == "S" else 1
            p_idx = 0 if pred == "S" else 1
            confusion[g_idx][p_idx] += 1

            if g_idx != p_idx:
                errors.append(
                    {
                        "idx": item["idx"],
                        "topic": item["topic"],
                        "fact": item["fact"],
                        "raw_label": item["raw_label"],
                        "gold": item["gold"],
                        "pred": pred,
                        "error_type": "FP" if pred == "S" else "FN",
                        "score": score,
                        "passages": item["passages"],
                    }
                )

            with _JOB_LOCK:
                _CURRENT_JOB["done"] = pos
                _CURRENT_JOB["correct_so_far"] = confusion[0][0] + confusion[1][1]
                _CURRENT_JOB["elapsed_sec"] = round(time.time() - started, 2)

        elapsed = round(time.time() - started, 2)
        metrics = _compute_metrics(confusion)
        target = 75.0 if mode == "word_overlap" else (83.0 if mode == "entailment" else 0.0)
        result_payload = {
            "mode": mode,
            "is_subset": len(subset) < len(examples),
            "elapsed_sec": elapsed,
            "target_pct": target,
            "passed_target": metrics["accuracy_pct"] >= target if target > 0 else True,
            "metrics": metrics,
            "fp_count": sum(1 for e in errors if e["error_type"] == "FP"),
            "fn_count": sum(1 for e in errors if e["error_type"] == "FN"),
            "errors": errors,
            "timestamp": time.strftime("%H:%M:%S"),
        }
        key = f"{mode}_quick" if len(subset) < len(examples) else mode
        _HISTORY[key] = result_payload

        with _JOB_LOCK:
            _CURRENT_JOB["running"] = False
            _CURRENT_JOB["elapsed_sec"] = elapsed
            _CURRENT_JOB["result"] = result_payload

    except Exception as exc:
        msg = str(exc)
        is_stub = isinstance(exc, NotImplementedError) or "implement" in msg.lower()
        with _JOB_LOCK:
            _CURRENT_JOB["running"] = False
            _CURRENT_JOB["elapsed_sec"] = round(time.time() - started, 2)
            _CURRENT_JOB["error"] = {
                "is_stub": is_stub,
                "message": msg,
                "traceback": traceback.format_exc(),
            }


def inspect_single_fact(idx: int, run_entailment: bool = False) -> Dict[str, Any]:
    examples = _load_examples()
    if idx < 0 or idx >= len(examples):
        raise ValueError("Fact index out of range")
    item = examples[idx]
    fc = _load_fresh_factcheck()

    out: Dict[str, Any] = {
        "idx": item["idx"],
        "topic": item["topic"],
        "fact": item["fact"],
        "raw_label": item["raw_label"],
        "gold": item["gold"],
        "passages": item["passages"],
        "word_overlap": None,
        "entailment": None,
    }

    try:
        w_checker = fc.WordRecallThresholdFactChecker()
        w_pred = w_checker.predict(item["fact"], item["passages"])
        w_score = w_checker.score_example(item["fact"], item["passages"]) if hasattr(w_checker, "score_example") else None
        out["word_overlap"] = {
            "status": "ok",
            "pred": w_pred,
            "correct": w_pred == item["gold"],
            "score": round(float(w_score), 4) if w_score is not None else None,
        }
    except Exception as exc:
        out["word_overlap"] = {"status": "stub_or_error", "message": str(exc)}

    if run_entailment:
        try:
            t0 = time.time()
            model, tokenizer = _get_deberta()
            ent_model = fc.EntailmentModel(model, tokenizer, False)
            e_checker = fc.EntailmentFactChecker(ent_model)
            e_pred = e_checker.predict(item["fact"], item["passages"])
            e_score = (
                e_checker.score_example(item["fact"], item["passages"], early_stop=False)
                if hasattr(e_checker, "score_example")
                else None
            )
            out["entailment"] = {
                "status": "ok",
                "pred": e_pred,
                "correct": e_pred == item["gold"],
                "score": round(float(e_score), 4) if e_score is not None else None,
                "elapsed_sec": round(time.time() - t0, 2),
            }
        except Exception as exc:
            out["entailment"] = {"status": "stub_or_error", "message": str(exc)}

    return out


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>CM3604 Fact-Checking Progress & Test Studio</title>
<style>
  :root {
    --bg: #0f172a;
    --panel: #1e293b;
    --panel-alt: #0b1220;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --green: #22c55e;
    --green-bg: rgba(34, 197, 94, 0.14);
    --amber: #f59e0b;
    --amber-bg: rgba(245, 158, 11, 0.15);
    --red: #ef4444;
    --red-bg: rgba(239, 68, 68, 0.15);
    --blue: #38bdf8;
    --blue-bg: rgba(56, 189, 248, 0.14);
    --purple: #a78bfa;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }
  .wrap { max-width: 1220px; margin: 0 auto; padding: 24px 20px 60px; }
  header {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    gap: 12px; padding-bottom: 18px; border-bottom: 1px solid var(--border); margin-bottom: 20px;
  }
  h1 { margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.02em; }
  .subtitle { color: var(--muted); font-size: 13px; margin-top: 2px; }
  .btn {
    background: #2563eb; color: #fff; border: 1px solid #3b82f6;
    padding: 8px 14px; border-radius: 8px; font-size: 13px; font-weight: 600;
    cursor: pointer; transition: 0.15s;
  }
  .btn:hover:not(:disabled) { background: #1d4ed8; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-secondary { background: var(--panel-alt); border-color: var(--border); color: var(--text); }
  .btn-secondary:hover:not(:disabled) { background: #334155; }
  .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 16px; margin-bottom: 20px; }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px;
  }
  .card h2 { margin: 0 0 6px; font-size: 16px; display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 9px; border-radius: 999px; font-size: 12px; font-weight: 600;
  }
  .pill-ok { background: var(--green-bg); color: var(--green); }
  .pill-warn { background: var(--amber-bg); color: var(--amber); }
  .pill-bad { background: var(--red-bg); color: var(--red); }
  .pill-info { background: var(--blue-bg); color: var(--blue); }
  .stat-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .stat-val { font-size: 18px; font-weight: 700; margin-top: 4px; }
  .meter {
    height: 10px; background: var(--panel-alt); border-radius: 999px;
    overflow: hidden; position: relative; margin: 12px 0 8px; border: 1px solid var(--border);
  }
  .meter-fill { height: 100%; background: linear-gradient(90deg, #3b82f6, #22c55e); transition: width 0.3s ease; }
  .meter-target {
    position: absolute; top: 0; bottom: 0; width: 2px; background: #facc15; z-index: 2;
  }
  .metrics-box {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;
    background: var(--panel-alt); padding: 10px; border-radius: 8px; margin-top: 12px; font-size: 12px;
  }
  .metrics-box div strong { display: block; font-size: 15px; color: var(--text); }
  .banner {
    background: var(--panel-alt); border: 1px solid var(--blue);
    border-radius: 10px; padding: 12px 16px; margin-bottom: 18px; display: none;
  }
  .banner.active { display: block; }
  .alert-warn {
    background: var(--amber-bg); border: 1px solid var(--amber); color: #fde68a;
    padding: 10px 14px; border-radius: 8px; margin-bottom: 16px; font-size: 13px; display: none;
  }
  select, input[type="text"] {
    background: var(--panel-alt); color: var(--text); border: 1px solid var(--border);
    padding: 8px 10px; border-radius: 8px; font-size: 13px; width: 100%;
  }
  .passage-box {
    background: var(--panel-alt); border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 12px; margin-top: 8px; font-size: 12.5px; color: #cbd5e1; max-height: 210px; overflow-y: auto;
  }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
  th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; }
  code { background: rgba(148,163,184,0.15); padding: 2px 5px; border-radius: 4px; font-size: 12px; }
  .commit-list { list-style: none; padding: 0; margin: 8px 0 0; font-size: 12.5px; }
  .commit-list li { padding: 5px 0; border-bottom: 1px dashed var(--border); display: flex; justify-content: space-between; gap: 10px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>CM3604 Fact-Checking Progress & Test Studio</h1>
      <div class="subtitle" id="ws-path">Loading workspace...</div>
    </div>
    <div style="display:flex; gap:8px; align-items:center;">
      <button class="btn btn-secondary" onclick="refreshStatus()">↻ Refresh Checks</button>
      <button class="btn btn-secondary" onclick="startRun('always_entail', 0)">Run Majority Baseline</button>
    </div>
  </header>

  <div id="typo-alert" class="alert-warn"></div>

  <div id="job-banner" class="banner">
    <div style="display:flex; justify-content:space-between; font-size:13px; font-weight:600;">
      <span id="job-title">Running evaluation...</span>
      <span id="job-stats">0 / 221</span>
    </div>
    <div class="meter"><div id="job-fill" class="meter-fill" style="width:0%"></div></div>
  </div>

  <!-- Submission Readiness Checks -->
  <div class="grid-4">
    <div class="card">
      <div class="stat-label">Driver Integrity (CW.pdf p.3)</div>
      <div class="stat-val" id="chk-driver">Checking...</div>
      <div class="subtitle" style="margin-top:4px;"><code>factchecking_main.py</code> SHA-256</div>
    </div>
    <div class="card">
      <div class="stat-label">Dataset & Passages</div>
      <div class="stat-val" id="chk-data">Checking...</div>
      <div class="subtitle" style="margin-top:4px;">221 facts across 31 biographies</div>
    </div>
    <div class="card">
      <div class="stat-label">Allowed Imports Only</div>
      <div class="stat-val" id="chk-imports">Checking...</div>
      <div class="subtitle" id="chk-imports-sub" style="margin-top:4px;">Python 3.10 + requirements.txt</div>
    </div>
    <div class="card">
      <div class="stat-label">Git Branch & Hygiene</div>
      <div class="stat-val" id="chk-git">Checking...</div>
      <div class="subtitle" id="chk-git-sub" style="margin-top:4px;">.gitignore check</div>
    </div>
  </div>

  <!-- Part 1 & Part 2 Live Runners -->
  <div class="grid-2">
    <!-- Part 1 Card -->
    <div class="card">
      <h2>
        <span>Part 1: Word Overlap (40 Marks)</span>
        <span id="p1-badge" class="pill pill-info">Not Tested Yet</span>
      </h2>
      <div class="subtitle">Full-credit threshold: <strong>≥ 75.0%</strong> (<code>≥ 166 / 221</code>) via <code>WordRecallThresholdFactChecker</code></div>
      <div class="meter">
        <div class="meter-target" style="left:75%;" title="75% Full Credit Target"></div>
        <div id="p1-bar" class="meter-fill" style="width:0%"></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:12px; color:var(--muted);">
        <span id="p1-acc-label">Accuracy: —</span>
        <span>Target: 75.0%</span>
      </div>
      <div id="p1-metrics" class="metrics-box" style="display:none;"></div>
      <div style="margin-top:14px; display:flex; gap:8px; flex-wrap:wrap;">
        <button class="btn" id="btn-p1" onclick="startRun('word_overlap', 0)">▶ Run Part 1 (221 Facts)</button>
      </div>
    </div>

    <!-- Part 2 Card -->
    <div class="card">
      <h2>
        <span>Part 2: Textual Entailment (40 Marks)</span>
        <span id="p2-badge" class="pill pill-info">Not Tested Yet</span>
      </h2>
      <div class="subtitle">Full-credit threshold: <strong>≥ 83.0%</strong> (<code>≥ 184 / 221</code>) via DeBERTa-v3 NLI + Pruning</div>
      <div class="meter">
        <div class="meter-target" style="left:83%;" title="83% Full Credit Target"></div>
        <div id="p2-bar" class="meter-fill" style="width:0%"></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:12px; color:var(--muted);">
        <span id="p2-acc-label">Accuracy: —</span>
        <span>Target: 83.0%</span>
      </div>
      <div id="p2-metrics" class="metrics-box" style="display:none;"></div>
      <div style="margin-top:14px; display:flex; gap:8px; flex-wrap:wrap;">
        <button class="btn btn-secondary" id="btn-p2-quick" onclick="startRun('entailment', 20)">⚡ Quick Sample (20 Facts)</button>
        <button class="btn" id="btn-p2-full" onclick="startRun('entailment', 0)">▶ Full Official Run (221 Facts)</button>
      </div>
    </div>
  </div>

  <!-- Single Fact Playground + Git Log -->
  <div class="grid-2">
    <div class="card">
      <h2><span>Single-Fact Inspector & Debugger</span></h2>
      <div class="subtitle" style="margin-bottom:10px;">Pick any of the 221 facts to see its Wikipedia passages and test your current <code>factcheck.py</code> instantly:</div>
      <div style="display:flex; gap:8px; margin-bottom:10px;">
        <select id="fact-select" onchange="loadFact(false)"></select>
        <button class="btn btn-secondary" onclick="loadFact(false)">Test Part 1</button>
        <button class="btn" onclick="loadFact(true)">Test Part 1 + DeBERTa</button>
      </div>
      <div id="fact-detail" style="font-size:13px;"></div>
    </div>

    <div class="card">
      <h2><span>Recent Git Commits & Team Progress</span></h2>
      <div class="subtitle" id="git-remote-label">Remote: —</div>
      <ul class="commit-list" id="commit-list"><li>Loading commits...</li></ul>
      <div style="margin-top:14px; padding-top:10px; border-top:1px solid var(--border); font-size:12.5px; color:var(--muted);">
        <strong>Coursework Checklist:</strong>
        <div id="stage-checklist" style="margin-top:6px; display:grid; gap:4px;"></div>
      </div>
    </div>
  </div>

  <!-- Part 3 Error Analysis Table -->
  <div class="card">
    <h2>
      <span>Part 3: Error Analysis Explorer (20 Marks — False Positives & False Negatives)</span>
      <div style="display:flex; gap:6px;">
        <button class="btn btn-secondary" style="padding:4px 10px; font-size:12px;" onclick="filterErrors('ALL')">All</button>
        <button class="btn btn-secondary" style="padding:4px 10px; font-size:12px;" onclick="filterErrors('FP')">False Positives (FP)</button>
        <button class="btn btn-secondary" style="padding:4px 10px; font-size:12px;" onclick="filterErrors('FN')">False Negatives (FN)</button>
      </div>
    </h2>
    <div class="subtitle" id="err-summary">Run Part 1 or Part 2 above to inspect misclassified facts alongside their BM25 Wikipedia passages.</div>
    <div style="overflow-x:auto;">
      <table id="err-table" style="display:none;">
        <thead>
          <tr>
            <th style="width:60px;">#</th>
            <th style="width:90px;">Type</th>
            <th style="width:150px;">Biography</th>
            <th>Atomic Fact & Retrieved Passages</th>
            <th style="width:90px;">Score</th>
          </tr>
        </thead>
        <tbody id="err-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
let latestErrors = [];
let errorFilter = 'ALL';
let pollTimer = null;

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function refreshStatus() {
  const res = await fetch('/api/status');
  const st = await res.json();
  document.getElementById('ws-path').textContent = `Workspace: ${st.workspace} • Python ${st.python_version}`;

  document.getElementById('chk-driver').innerHTML = st.main_driver_untouched
    ? '<span class="pill pill-ok">✓ Unmodified</span>'
    : '<span class="pill pill-bad">✗ Modified / Missing</span>';

  document.getElementById('chk-data').innerHTML = st.dataset_loaded
    ? `<span class="pill pill-ok">✓ ${st.num_examples} Facts Ready</span>`
    : '<span class="pill pill-bad">✗ Missing JSONL</span>';

  if (st.imports_check.syntax_error) {
    document.getElementById('chk-imports').innerHTML = '<span class="pill pill-bad">✗ Syntax Error</span>';
    document.getElementById('chk-imports-sub').textContent = st.imports_check.syntax_error;
  } else if (st.imports_check.ok) {
    document.getElementById('chk-imports').innerHTML = '<span class="pill pill-ok">✓ Clean Standalone</span>';
    document.getElementById('chk-imports-sub').textContent = st.imports_check.imports.join(', ');
  } else {
    document.getElementById('chk-imports').innerHTML = '<span class="pill pill-bad">✗ Extra Package</span>';
    document.getElementById('chk-imports-sub').textContent = 'Disallowed: ' + st.imports_check.disallowed.join(', ');
  }

  document.getElementById('chk-git').innerHTML = `<span class="pill ${st.gitignore_ok ? 'pill-ok' : 'pill-warn'}">${esc(st.git.branch)}</span>`;
  document.getElementById('chk-git-sub').textContent = st.gitignore_ok
    ? `.gitignore OK • ${st.git.dirty_count} uncommitted file(s)`
    : 'Missing .gitignore!';

  const alerts = [];
  if (st.typos_detected.gitnore) alerts.push('Found <code>.gitnore</code> typo — rename to <code>.gitignore</code>!');
  if (st.typos_detected.factchecking_prefix) alerts.push('Found <code>.factchecking_</code> typo — rename to <code>.factchecking_main.sha256</code>!');
  const alertBox = document.getElementById('typo-alert');
  if (alerts.length) {
    alertBox.style.display = 'block';
    alertBox.innerHTML = '⚠️ <strong>Filename Fix Needed:</strong> ' + alerts.join(' &nbsp;|&nbsp; ');
  } else {
    alertBox.style.display = 'none';
  }

  document.getElementById('git-remote-label').textContent = `Remote: ${st.git.remote}`;
  const cl = document.getElementById('commit-list');
  cl.innerHTML = st.git.commits.length
    ? st.git.commits.map(c => `<li><span><code>${esc(c.hash)}</code> ${esc(c.subject)}</span><span style="color:var(--muted);white-space:nowrap;">${esc(c.author)} • ${esc(c.when)}</span></li>`).join('')
    : '<li style="color:var(--muted);">No commits yet on this branch.</li>';

  // Restore cached run cards if available
  if (st.history.word_overlap) renderRunResult('p1', st.history.word_overlap);
  else if (st.part1_code_state === 'stub') {
    document.getElementById('p1-badge').className = 'pill pill-warn';
    document.getElementById('p1-badge').textContent = '⏳ Starter Stub';
  }

  const p2Hist = st.history.entailment || st.history.entailment_quick;
  if (p2Hist) renderRunResult('p2', p2Hist);
  else if (st.part2_code_state === 'stub') {
    document.getElementById('p2-badge').className = 'pill pill-warn';
    document.getElementById('p2-badge').textContent = '⏳ Starter Stub';
  }

  const p1Done = st.history.word_overlap && st.history.word_overlap.passed_target;
  const p2Done = st.history.entailment && st.history.entailment.passed_target;
  document.getElementById('stage-checklist').innerHTML = [
    [st.main_driver_untouched && st.dataset_loaded && st.gitignore_ok, 'Step 0: Clean workspace, .gitignore & untouched driver'],
    [st.part1_code_state === 'implemented', 'Step 1A–1C: WordRecallThresholdFactChecker implemented'],
    [p1Done, 'Step 1D: Part 1 verified ≥ 75% accuracy'],
    [st.part2_code_state === 'implemented', 'Step 2A–2C: EntailmentModel + EntailmentFactChecker implemented'],
    [p2Done, 'Step 2D: Part 2 verified ≥ 83% accuracy on all 221 facts'],
  ].map(([ok, txt]) => `<div>${ok ? '✅' : '⬜'} ${esc(txt)}</div>`).join('');
}

function renderRunResult(prefix, res) {
  const m = res.metrics;
  const badge = document.getElementById(`${prefix}-badge`);
  const tag = res.is_subset ? ` (${m.total}-fact sample)` : '';
  if (res.passed_target) {
    badge.className = 'pill pill-ok';
    badge.textContent = `✅ PASS: ${m.accuracy_pct}%${tag}`;
  } else {
    badge.className = 'pill pill-warn';
    badge.textContent = `🔧 ${m.accuracy_pct}% < ${res.target_pct}%${tag}`;
  }
  document.getElementById(`${prefix}-bar`).style.width = `${Math.min(100, m.accuracy_pct)}%`;
  document.getElementById(`${prefix}-acc-label`).textContent =
    `Accuracy: ${m.correct} / ${m.total} = ${m.accuracy_pct}% (${res.elapsed_sec}s)`;

  const box = document.getElementById(`${prefix}-metrics`);
  box.style.display = 'grid';
  box.innerHTML = `
    <div>TP (S→S)<strong>${m.confusion.tp}</strong></div>
    <div>TN (NS→NS)<strong>${m.confusion.tn}</strong></div>
    <div>FP (NS→S)<strong>${m.confusion.fp}</strong></div>
    <div>FN (S→NS)<strong>${m.confusion.fn}</strong></div>
    <div>F1 ("S")<strong>${m.s_metrics.f1}</strong></div>
    <div>F1 ("NS")<strong>${m.ns_metrics.f1}</strong></div>
    <div>Prec ("S")<strong>${m.s_metrics.precision}</strong></div>
    <div>Rec ("S")<strong>${m.s_metrics.recall}</strong></div>
  `;

  latestErrors = res.errors || [];
  document.getElementById('err-summary').innerHTML =
    `Showing errors from <strong>${esc(res.mode)}</strong> (${res.timestamp}): ` +
    `<strong>${res.fp_count} False Positives</strong> + <strong>${res.fn_count} False Negatives</strong> = ${latestErrors.length} total mistakes.`;
  renderErrors();
}

function filterErrors(kind) {
  errorFilter = kind;
  renderErrors();
}

function renderErrors() {
  const tbl = document.getElementById('err-table');
  const tbody = document.getElementById('err-tbody');
  const rows = latestErrors.filter(e => errorFilter === 'ALL' || e.error_type === errorFilter);
  if (!rows.length) {
    tbl.style.display = 'none';
    return;
  }
  tbl.style.display = 'table';
  tbody.innerHTML = rows.map(e => `
    <tr>
      <td><code>#${e.idx}</code></td>
      <td><span class="pill ${e.error_type === 'FP' ? 'pill-bad' : 'pill-warn'}">${e.error_type}</span><div style="font-size:11.5px;color:var(--muted);margin-top:3px;">Gold: ${e.raw_label} → Pred: ${e.pred}</div></td>
      <td><strong>${esc(e.topic)}</strong></td>
      <td>
        <div><strong>${esc(e.fact)}</strong></div>
        <details style="margin-top:5px;">
          <summary style="cursor:pointer;color:var(--blue);font-size:12px;">Inspect ${e.passages.length} retrieved Wikipedia passage(s)</summary>
          <div class="passage-box">${e.passages.map((p, i) => `<div style="margin-bottom:8px;"><strong>[Passage ${i+1} — ${esc(p.title)}]</strong><br/>${esc(p.text)}</div>`).join('')}</div>
        </details>
      </td>
      <td><code>${e.score !== null ? e.score : '—'}</code></td>
    </tr>
  `).join('');
}

async function startRun(mode, limit) {
  const res = await fetch(`/api/run?mode=${encodeURIComponent(mode)}&limit=${limit}`, { method: 'POST' });
  const data = await res.json();
  if (!data.started && data.reason) {
    alert(data.reason);
    return;
  }
  document.getElementById('job-banner').classList.add('active');
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollJob, 600);
  pollJob();
}

async function pollJob() {
  const res = await fetch('/api/job');
  const job = await res.json();
  const pct = job.total ? Math.round((job.done / job.total) * 100) : 0;
  const liveAcc = job.done ? ((job.correct_so_far / job.done) * 100).toFixed(1) : '0.0';
  document.getElementById('job-title').textContent = job.running
    ? `Running ${job.mode}... (live accuracy: ${liveAcc}%)`
    : `Finished ${job.mode}`;
  document.getElementById('job-stats').textContent = `${job.done} / ${job.total} (${job.elapsed_sec}s)`;
  document.getElementById('job-fill').style.width = `${pct}%`;

  if (!job.running) {
    clearInterval(pollTimer);
    pollTimer = null;
    setTimeout(() => document.getElementById('job-banner').classList.remove('active'), 1200);
    if (job.error) {
      alert(job.error.is_stub
        ? `Not implemented yet in factcheck.py:\n\n${job.error.message}`
        : `Error while running ${job.mode}:\n\n${job.error.message}`);
    } else if (job.result) {
      if (job.result.mode === 'word_overlap') renderRunResult('p1', job.result);
      else if (job.result.mode === 'entailment') renderRunResult('p2', job.result);
      else alert(`${job.result.mode} accuracy: ${job.result.metrics.correct}/${job.result.metrics.total} = ${job.result.metrics.accuracy_pct}%`);
    }
    refreshStatus();
  }
}

async function initFactsDropdown() {
  const res = await fetch('/api/facts');
  const items = await res.json();
  const sel = document.getElementById('fact-select');
  sel.innerHTML = items.map(f =>
    `<option value="${f.idx}">#${f.idx} [${f.raw_label}] ${esc(f.topic)} — ${esc(f.fact.slice(0, 68))}</option>`
  ).join('');
  if (items.length) loadFact(false);
}

async function loadFact(runEnt) {
  const idx = document.getElementById('fact-select').value || 0;
  const box = document.getElementById('fact-detail');
  box.innerHTML = '<span style="color:var(--muted);">Testing fact...</span>';
  const res = await fetch(`/api/inspect?idx=${idx}&ent=${runEnt ? 1 : 0}`);
  const d = await res.json();

  const fmtPred = (label, obj) => {
    if (!obj) return `<span class="pill pill-info">${label}: click 'Test Part 1 + DeBERTa'</span>`;
    if (obj.status !== 'ok') return `<span class="pill pill-warn">${label}: ${esc(obj.message)}</span>`;
    return `<span class="pill ${obj.correct ? 'pill-ok' : 'pill-bad'}">${label}: Pred ${obj.pred} ${obj.correct ? '✓' : '✗'} ${obj.score !== null ? `(score=${obj.score})` : ''}</span>`;
  };

  box.innerHTML = `
    <div style="margin-bottom:8px;"><strong>Claim:</strong> ${esc(d.fact)}</div>
    <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px;">
      <span class="pill pill-info">Gold: ${d.gold} (raw: ${d.raw_label})</span>
      ${fmtPred('Part 1', d.word_overlap)}
      ${fmtPred('Part 2', d.entailment)}
    </div>
    <div class="passage-box">
      ${d.passages.map((p, i) => `<div style="margin-bottom:8px;"><strong>[Passage ${i+1} — ${esc(p.title)}]</strong><br/>${esc(p.text)}</div>`).join('')}
    </div>
  `;
}

refreshStatus();
initFactsDropdown();
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return  # keep terminal output clean

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/":
            raw = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif parsed.path == "/api/status":
            self._send_json(get_system_status())
        elif parsed.path == "/api/job":
            with _JOB_LOCK:
                self._send_json(dict(_CURRENT_JOB))
        elif parsed.path == "/api/facts":
            items = [
                {"idx": x["idx"], "topic": x["topic"], "fact": x["fact"], "raw_label": x["raw_label"]}
                for x in _load_examples()
            ]
            self._send_json(items)
        elif parsed.path == "/api/inspect":
            idx = int(qs.get("idx", ["0"])[0])
            run_ent = qs.get("ent", ["0"])[0] == "1"
            self._send_json(inspect_single_fact(idx, run_entailment=run_ent))
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/api/run":
            mode = qs.get("mode", ["word_overlap"])[0]
            limit = int(qs.get("limit", ["0"])[0])
            with _JOB_LOCK:
                if _CURRENT_JOB["running"]:
                    self._send_json({"started": False, "reason": "A test run is already in progress."})
                    return
            t = threading.Thread(target=_run_evaluation_worker, args=(mode, limit), daemon=True)
            t.start()
            self._send_json({"started": True, "mode": mode, "limit": limit})
        else:
            self._send_json({"error": "Not found"}, status=404)


def main() -> None:
    parser = argparse.ArgumentParser(description="CM3604 Local Progress & Testing Dashboard")
    parser.add_argument("--port", type=int, default=8765, help="Local port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    parser.add_argument("--self-test", action="store_true", help="Run a one-shot health & Part 1 self-test and exit")
    args = parser.parse_args()

    if args.self_test:
        st = get_system_status()
        print("Status check OK:", json.dumps({k: st[k] for k in ("main_driver_untouched", "dataset_loaded", "part1_code_state", "part2_code_state")}))
        _run_evaluation_worker("word_overlap", limit=0)
        print("Part 1 self-test accuracy:", _CURRENT_JOB["result"]["metrics"]["accuracy_pct"], "%")
        return

    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\n🚀 CM3604 Progress & Test Studio running at: {url}")
    print("   Edit factcheck.py anytime — every click automatically reloads your latest code!")
    print("   Press Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped dashboard.")


if __name__ == "__main__":
    main()
