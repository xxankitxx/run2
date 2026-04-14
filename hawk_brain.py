#!/usr/bin/env python3
"""
HAWK BRAIN v17.5 — Realistic data calibrated, tighter IMS/SV gates
"""
import os, sys, time, logging, logging.handlers, json
import multiprocessing as mp
from typing import Optional
from datetime import datetime, timezone, timedelta

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

IST = timezone(timedelta(hours=5, minutes=30))

# =============================================================================
# LOGGING
# =============================================================================
def _setup_logging() -> logging.Logger:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "hawk_brain.log")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=2, encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(fmt); ch.setLevel(logging.WARNING)
    log = logging.getLogger("hawk.brain")
    log.setLevel(logging.DEBUG); log.addHandler(fh); log.addHandler(ch)
    return log

# =============================================================================
# BRAIN PROCESS ENTRY POINT
# =============================================================================
def run_brain(snap_queue: mp.Queue, decision_queue: mp.Queue, result_queue: mp.Queue):
    log = _setup_logging()
    log.info("HawkBrain v17 started (pid=%d)", os.getpid())
    try:
        from hawk_engine import NiftyEngine
        engine = NiftyEngine()
    except ImportError as e:
        log.critical("HawkBrain: cannot import hawk_engine.py — %s", e)
        sys.exit(1)

    learning: dict = _load_learning()
    last_learning_save = time.monotonic()
    last_reset_date: Optional[str] = None
    session_reset_done = False
    tick = 0; last_hb = time.monotonic(); last_entry_signals: dict = {}

    while True:
        try:
            now_ist = datetime.now(IST); today_str = now_ist.strftime("%Y-%m-%d")
            if today_str != last_reset_date:
                if now_ist.hour == 9 and now_ist.minute >= 15:
                    engine.reset_session(); last_reset_date = today_str; session_reset_done = True
                    tick = 0; last_entry_signals = {}
                elif now_ist.hour > 9:
                    engine.reset_session(); last_reset_date = today_str; session_reset_done = True
                    tick = 0; last_entry_signals = {}

            pre_market = (now_ist.hour < 9 or (now_ist.hour == 9 and now_ist.minute < 15))

            try:
                while True:
                    res = result_queue.get_nowait()
                    if res is None:
                        _save_learning(learning); return
                    _record_result(res, learning, last_entry_signals)
                    _save_learning(learning)
                    exit_reason = res.get("exit_reason", "SL HIT")
                    exit_dir = res.get("direction", "NEUTRAL")
                    engine.reset_after_exit(exit_reason=exit_reason, direction=exit_dir)
            except Exception:
                pass

            try:
                snap = snap_queue.get(timeout=1.0)
            except Exception:
                if time.monotonic() - last_hb > 30: last_hb = time.monotonic()
                continue

            if snap is None:
                _save_learning(learning); return

            if pre_market and not session_reset_done:
                pre_decision = {
                    "entry_allowed": False, "entry_direction": "NEUTRAL", "entry_conviction": 0.0,
                    "blocked_reason": f"PRE-MARKET", "spot": snap.get("spot", 0.0),
                    "futures": snap.get("futures", 0.0), "atm": snap.get("atm", 0.0),
                    "ce_ltp": snap.get("ce_ltp", 0.0), "pe_ltp": snap.get("pe_ltp", 0.0),
                    "pcr": snap.get("pcr", 1.0), "is_clean": snap.get("is_clean", False),
                    "votes_for": 0, "vote_detail": [], "smart_money": "IDLE",
                    "suggested_sl": 0.0, "suggested_tgt": 0.0, "engine_summary": "",
                    "reversal_warnings": [], "ts": snap.get("ts", time.monotonic()),
                    "vix_regime": "MID", "probability": 0.0, "entry_thesis": {},
                }
                try: decision_queue.put_nowait(pre_decision)
                except Exception: pass
                tick += 1; continue

            snap_obj = _SnapshotAdapter(snap)
            total_options_oi = snap.get("total_ce_oi", 0.0) + snap.get("total_pe_oi", 0.0)
            signal_weights = _compute_signal_weights(learning)
            vix_value = snap.get("vix", 0.0)

            result = engine.update(snap_obj, snap.get("futures", 0.0),
                                   futures_oi=total_options_oi,
                                   signal_weights=signal_weights,
                                   vix_value=vix_value)

            _apply_learning_modifier(result, learning)
            decision = _build_decision(snap, result)
            # v22: Pass fut_vel_disp EVERY tick (not just entries) for FDC exit intelligence
            decision["fut_vel_disp"] = round(engine.fut_vel.displacement, 3) if hasattr(engine, 'fut_vel') else 0.0
            # v25-LIVE: Pass ALL gate diagnostics so dashboard shows everything at once
            decision["gate_diagnostics"] = engine.get_gate_diagnostics()

            if result.entry_allowed:
                last_entry_signals = {}
                for detail in result.vote_detail:
                    try:
                        name = detail.split(":")[0]
                        direction = detail.split(":")[1].split("(")[0]
                        last_entry_signals[name] = direction
                    except Exception: pass

            try: decision_queue.put_nowait(decision)
            except Exception: pass

            tick += 1
            if time.monotonic() - last_learning_save > 300:
                _save_learning(learning); last_learning_save = time.monotonic()

            if time.monotonic() - last_hb > 60: last_hb = time.monotonic()

        except Exception as e:
            time.sleep(0.1)


# =============================================================================
# SNAPSHOT ADAPTER
# =============================================================================
class _SnapshotAdapter:
    __slots__ = ("_d",)
    def __init__(self, d: dict): self._d = d
    @property
    def spot(self) -> float: return self._d.get("spot", 0.0)
    @property
    def atm_strike(self) -> float: return self._d.get("atm", 0.0)
    @property
    def atm_ce_ltp(self) -> float: v = self._d.get("ce_ltp", 0.0); return v if v > 0 else float("nan")
    @property
    def atm_pe_ltp(self) -> float: v = self._d.get("pe_ltp", 0.0); return v if v > 0 else float("nan")
    @property
    def total_call_oi(self) -> float: return self._d.get("total_ce_oi", 0.0)
    @property
    def total_put_oi(self) -> float: return self._d.get("total_pe_oi", 0.0)
    @property
    def pcr(self) -> float: return self._d.get("pcr", 1.0)
    @property
    def is_clean(self) -> bool: return self._d.get("is_clean", False)
    @property
    def call_strikes(self) -> '_StrikesProxy': return _StrikesProxy(self._d.get("strikes", {}), "call")
    @property
    def put_strikes(self) -> '_StrikesProxy': return _StrikesProxy(self._d.get("strikes", {}), "put")

class _StrikesProxy:
    def __init__(self, strikes: dict, right: str):
        self._s = {}
        for key, v in strikes.items():
            if isinstance(key, (tuple, list)) and len(key) == 2:
                s, r = key
                if r == right: self._s[float(s)] = v
    def __contains__(self, item): return float(item) in self._s
    def __iter__(self): return iter(self._s)
    def __bool__(self): return len(self._s) > 0
    def items(self): return [(s, _StrikeDataProxy(v)) for s, v in self._s.items()]
    def get(self, key, default=None):
        v = self._s.get(float(key))
        return _StrikeDataProxy(v) if v is not None else default

class _StrikeDataProxy:
    def __init__(self, d: dict):
        self.oi = d.get("oi", 0.0); self.ltp = d.get("ltp", 0.0); self.vol = d.get("vol", 0.0)
        self.bid = d.get("bid", 0.0); self.ask = d.get("ask", 0.0)

# =============================================================================
# DECISION BUILDER
# =============================================================================
def _build_decision(snap: dict, result) -> dict:
    return {
        "entry_allowed": result.entry_allowed, "entry_direction": result.direction,
        "entry_conviction": result.score, "blocked_reason": result.blocked_reason,
        "spot": snap.get("spot", 0.0), "futures": snap.get("futures", 0.0),
        "atm": snap.get("atm", 0.0), "ce_ltp": snap.get("ce_ltp", 0.0),
        "pe_ltp": snap.get("pe_ltp", 0.0), "pcr": snap.get("pcr", 1.0),
        "is_clean": snap.get("is_clean", False), "votes_for": result.votes_for,
        "vote_detail": result.vote_detail, "smart_money": result.smart_money_bias,
        "suggested_sl": result.suggested_sl_pts, "suggested_tgt": result.suggested_tgt_pts,
        "engine_summary": "", "reversal_warnings": result.reversal_warnings if hasattr(result, 'reversal_warnings') else [],
        "ts": snap.get("ts", time.monotonic()),
        "vix_regime": result.vix_regime if hasattr(result, 'vix_regime') else "MID",
        "probability": result.probability if hasattr(result, 'probability') else 0.0,
        "imm_momentum": result.imm_momentum if hasattr(result, 'imm_momentum') else 0.0,
        "entry_thesis": result.entry_thesis if hasattr(result, 'entry_thesis') else {},
    }

# =============================================================================
# LEARNING (unchanged from v16)
# =============================================================================
_LEARNING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hawk_learning.json")

def _load_learning() -> dict:
    try:
        if os.path.exists(_LEARNING_FILE):
            with open(_LEARNING_FILE) as f: data = json.load(f)
            data.setdefault("time_slots", {}); data.setdefault("total_trades", 0); data.setdefault("signal_accuracy", {})
            return data
    except Exception: pass
    return {"time_slots": {}, "total_trades": 0, "signal_accuracy": {}}

def _save_learning(learning: dict):
    try:
        with open(_LEARNING_FILE, "w") as f: json.dump(learning, f, indent=2)
    except Exception: pass

def _record_result(result: dict, learning: dict, last_entry_signals: dict):
    try:
        pnl = result.get("pnl_pts", 0.0) or 0.0
        hour = result.get("entry_hour", 0); minute = result.get("entry_minute", 0)
        slot = f"{hour:02d}:{(minute // 15) * 15:02d}"
        is_win = pnl >= 0
        slots = learning.setdefault("time_slots", {})
        s = slots.setdefault(slot, {"wins": 0, "losses": 0, "total_pnl": 0.0})
        if is_win: s["wins"] += 1
        else: s["losses"] += 1
        s["total_pnl"] += pnl; learning["total_trades"] = learning.get("total_trades", 0) + 1
        sig_acc = learning.setdefault("signal_accuracy", {})
        direction = result.get("direction", "?")
        for sig_name, sig_dir in last_entry_signals.items():
            sa = sig_acc.setdefault(sig_name, {"wins": 0, "losses": 0, "correct": 0, "total": 0})
            sa["total"] += 1
            if is_win: sa["wins"] += 1
            else: sa["losses"] += 1
            if sig_dir == direction and is_win: sa["correct"] += 1
    except Exception: pass

def _compute_signal_weights(learning: dict) -> dict:
    weights = {}; sig_acc = learning.get("signal_accuracy", {})
    for sig_name, s in sig_acc.items():
        t = s.get("total", 0)
        if t < 5: weights[sig_name] = 1.0; continue
        acc = s.get("correct", 0) / t
        if acc >= 0.60: weights[sig_name] = 1.5
        elif acc >= 0.45: weights[sig_name] = 1.2
        elif acc >= 0.30: weights[sig_name] = 1.0
        elif acc >= 0.10: weights[sig_name] = 0.5
        else: weights[sig_name] = 0.3
    return weights

def _apply_learning_modifier(result, learning: dict):
    if not result.entry_allowed: return
    try:
        STALE_PENALTY = 0.02
        if result.vote_detail:
            for detail in result.vote_detail:
                try:
                    parts = detail.split(":")
                    direction = parts[1].split("(")[0]
                    if direction == result.direction and "[stale]" in detail:
                        result.score = max(result.score - STALE_PENALTY, 0.0)
                except Exception: pass

        now = datetime.now(IST)
        hour = now.hour; minute = (now.minute // 15) * 15; slot = f"{hour:02d}:{minute:02d}"
        penalty = 0.0; boost = 0.0; reasons = []
        slots = learning.get("time_slots", {})
        s = slots.get(slot)
        if s is not None:
            total = s["wins"] + s["losses"]
            if total >= 12:
                wr = s["wins"] / total
                if wr < 0.35: penalty += 0.05; reasons.append(f"TimeSlot {slot} WR={wr:.0%}")
                elif wr < 0.45: penalty += 0.03; reasons.append(f"TimeSlot {slot} WR={wr:.0%}")

        sig_acc = learning.get("signal_accuracy", {})
        if sig_acc and result.vote_detail:
            accuracy_scores = []
            for detail in result.vote_detail:
                try:
                    name = detail.split(":")[0]; direction = detail.split(":")[1].split("(")[0]
                    if direction == result.direction:
                        sa = sig_acc.get(name)
                        if sa and sa["total"] >= 12:
                            acc = sa["correct"] / sa["total"]; accuracy_scores.append(acc)
                except Exception: pass
            if len(accuracy_scores) >= 2:
                avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
                if avg_accuracy < 0.35: penalty += 0.06; reasons.append(f"Hist_Signal_Acc={avg_accuracy:.0%}")
                elif avg_accuracy < 0.40: penalty += 0.04; reasons.append(f"Hist_Signal_Acc={avg_accuracy:.0%}")
                elif avg_accuracy > 0.60: boost += 0.04

        old_score = result.score
        result.score = min(max(result.score - penalty + boost, 0.0), 0.99)
        # v17.2: conviction floors match engine
        conv_floor = 0.60
        vix_regime = getattr(result, 'vix_regime', 'MID')
        if vix_regime == "HIGH": conv_floor = 0.63
        elif vix_regime == "EXTREME": conv_floor = 0.70
        if result.score < conv_floor:
            result.entry_allowed = False
            result.blocked_reason = f"Learning Filter: Score dropped {old_score:.2f} -> {result.score:.2f} due to: {', '.join(reasons)}"
    except Exception: pass

if __name__ == "__main__":
    pass