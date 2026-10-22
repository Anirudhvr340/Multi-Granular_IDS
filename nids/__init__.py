"""
NIDS — Multi-Granular Network Intrusion Detection System
=========================================================
Four-model hierarchical evidence-fusion pipeline.

Models:
  - Flow XGBoost       (flow-level, 17 features, 15 classes)
  - Time-Window XGBoost (window-level, 268 selected features, 11 trained classes)
  - Temporal LSTM      (sequence-level, 64 features, seq=60, 14 classes)
  - Temporal TCN       (sequence-level, 64 features, seq=60, 14 classes)
  - Packet Model       *** MISSING — stub returns unavailable ***
  - Session Model      *** MISSING — stub returns unavailable ***

Pipeline stages:
  1. Model Adapters → Common Evidence Schema
  2. Attack Gate     → IS_ATTACK binary decision
  3. Rule Engine     → Deterministic fast-path rules
  4. Context Engine  → Stateful memory across windows
  5. Policy Agent    → Gemini-backed reasoning
  6. Action Executor → ALLOW / FLAG / BLOCK artifact
"""

__version__ = "1.0.0"
__all__ = [
    "adapters",
    "evidence_schema",
    "attack_gate",
    "rule_engine",
    "context_engine",
    "policy_agent",
    "action_executor",
    "pipeline",
]

