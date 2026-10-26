#!/usr/bin/env python3
"""
=============================================================================
MULTI-GRANULAR NIDS: NEXT-GEN SOC DASHBOARD & THREAT ALERT MONITOR
=============================================================================
A modern, dark-themed Tkinter Operations Center Dashboard for live 
multi-granular network intrusion detection on the CICIDS2017 dataset.

Features:
- Live streaming across all 8 days of CICIDS2017 with interactive controls:
  Play, Pause, Step, Reset, and Real-time Speed Adjustment.
- Real-time 4-Model Inference:
  Packet 1D-CNN, Flow XGBoost, Time-Window XGBoost, and Session CNN-LSTM.
- Level 2 Anti-Dilution Attack Gate + Bayesian Confidence Fusion.
- Stage 4 Hybrid Decision Layer:
  Deterministic Fast-Path Rules + Qwen 2.5 (7B Instruct) Cognitive Agent.
- High-Priority Malicious Alert System:
  Flashes and alerts ONLY when attacks are detected or flagged (silent on Benign).
- Deep-Dive Per-Record Analysis Panel:
  Live visual probability meters for all 4 models, risk components, and
  machine-auditable natural language decision rationales.
=============================================================================
"""

import os
import sys
import time
import queue
import threading
import winsound
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

# Ensure project modules are in path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "time_window"))
sys.path.insert(0, str(HERE / "packet_level"))
sys.path.insert(0, str(HERE / "session_level"))

import step0_preprocessing as preprocessing
from nids.adapters.packet_adapter import PacketModelAdapter
from nids.adapters.flow_adapter import FlowModelAdapter
from nids.adapters.time_window_adapter import TimeWindowAdapter
from nids.adapters.session_adapter import SessionModelAdapter
from nids.attack_gate import AttackGate
from nids.rule_engine import RuleEngine
from nids.policy_agent import PolicyAgent

# ---------------------------------------------------------------------------
# CICIDS2017 Day Sequences & Column Mapping
# ---------------------------------------------------------------------------
CICIDS2017_DAYS = [
    ("Day 1: Monday (Benign Baseline)", "cicids2017/Monday-WorkingHours.pcap_ISCX.csv"),
    ("Day 2: Tuesday (FTP/SSH Patator)", "cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv"),
    ("Day 3: Wednesday (DoS & Slowloris)", "cicids2017/Wednesday-workingHours.pcap_ISCX.csv"),
    ("Day 4: Thursday (Web Attacks & SQLi)", "cicids2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
    ("Day 5: Thursday (Infiltration)", "cicids2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
    ("Day 6: Friday (Botnet Traffic)", "cicids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    ("Day 7: Friday (PortScan Activity)", "cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
    ("Day 8: Friday (DDoS Flood)", "cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
]

COL_RENAME_2017 = {
    ' Destination Port': 'Dst Port',
    ' Flow Duration': 'Flow Duration',
    ' Total Fwd Packets': 'Tot Fwd Pkts',
    ' Total Backward Packets': 'Tot Bwd Pkts',
    'Total Length of Fwd Packets': 'TotLen Fwd Pkts',
    ' Total Length of Bwd Packets': 'TotLen Bwd Pkts',
    ' Fwd Packet Length Max': 'Fwd Pkt Len Max',
    ' Fwd Packet Length Min': 'Fwd Pkt Len Min',
    ' Fwd Packet Length Mean': 'Fwd Pkt Len Mean',
    ' Fwd Packet Length Std': 'Fwd Pkt Len Std',
    'Bwd Packet Length Max': 'Bwd Pkt Len Max',
    ' Bwd Packet Length Min': 'Bwd Pkt Len Min',
    ' Bwd Packet Length Mean': 'Bwd Pkt Len Mean',
    ' Bwd Packet Length Std': 'Bwd Pkt Len Std',
    'Flow Bytes/s': 'Flow Byts/s',
    ' Flow Packets/s': 'Flow Pkts/s',
    ' Flow IAT Mean': 'Flow IAT Mean',
    ' Flow IAT Std': 'Flow IAT Std',
    ' Flow IAT Max': 'Flow IAT Max',
    ' Flow IAT Min': 'Flow IAT Min',
    'Fwd IAT Total': 'Fwd IAT Tot',
    ' Fwd IAT Mean': 'Fwd IAT Mean',
    ' Fwd IAT Std': 'Fwd IAT Std',
    ' Fwd IAT Max': 'Fwd IAT Max',
    ' Fwd IAT Min': 'Fwd IAT Min',
    'Bwd IAT Total': 'Bwd IAT Tot',
    ' Bwd IAT Mean': 'Bwd IAT Mean',
    ' Bwd IAT Std': 'Bwd IAT Std',
    ' Bwd IAT Max': 'Bwd IAT Max',
    ' Bwd IAT Min': 'Bwd IAT Min',
    'Fwd PSH Flags': 'Fwd PSH Flags',
    ' Bwd PSH Flags': 'Bwd PSH Flags',
    ' Fwd URG Flags': 'Fwd URG Flags',
    ' Bwd URG Flags': 'Bwd URG Flags',
    ' Fwd Header Length': 'Fwd Header Len',
    ' Bwd Header Length': 'Bwd Header Len',
    'Fwd Packets/s': 'Fwd Pkts/s',
    ' Bwd Packets/s': 'Bwd Pkts/s',
    ' Min Packet Length': 'Pkt Len Min',
    ' Max Packet Length': 'Pkt Len Max',
    ' Packet Length Mean': 'Pkt Len Mean',
    ' Packet Length Std': 'Pkt Len Std',
    ' Packet Length Variance': 'Pkt Len Var',
    'FIN Flag Count': 'FIN Flag Cnt',
    ' SYN Flag Count': 'SYN Flag Cnt',
    ' RST Flag Count': 'RST Flag Cnt',
    ' PSH Flag Count': 'PSH Flag Cnt',
    ' ACK Flag Count': 'ACK Flag Cnt',
    ' URG Flag Count': 'URG Flag Cnt',
    ' CWE Flag Count': 'CWE Flag Count',
    ' ECE Flag Count': 'ECE Flag Cnt',
    ' Down/Up Ratio': 'Down/Up Ratio',
    ' Average Packet Size': 'Pkt Size Avg',
    ' Avg Fwd Segment Size': 'Fwd Seg Size Avg',
    ' Avg Bwd Segment Size': 'Bwd Seg Size Avg',
    ' Fwd Header Length.1': 'Fwd Header Len.1',
    'Fwd Avg Bytes/Bulk': 'Fwd Byts/b Avg',
    ' Fwd Avg Packets/Bulk': 'Fwd Pkts/b Avg',
    ' Fwd Avg Bulk Rate': 'Fwd Blk Rate Avg',
    ' Bwd Avg Bytes/Bulk': 'Bwd Byts/b Avg',
    ' Bwd Avg Packets/Bulk': 'Bwd Pkts/b Avg',
    'Bwd Avg Bulk Rate': 'Bwd Blk Rate Avg',
    'Subflow Fwd Packets': 'Subflow Fwd Pkts',
    ' Subflow Fwd Bytes': 'Subflow Fwd Byts',
    ' Subflow Bwd Packets': 'Subflow Bwd Pkts',
    ' Subflow Bwd Bytes': 'Subflow Bwd Byts',
    'Init_Win_bytes_forward': 'Init Fwd Win Byts',
    ' Init_Win_bytes_backward': 'Init Bwd Win Byts',
    ' act_data_pkt_fwd': 'Fwd Act Data Pkts',
    ' min_seg_size_forward': 'Fwd Seg Size Min',
    'Active Mean': 'Active Mean',
    ' Active Std': 'Active Std',
    ' Active Max': 'Active Max',
    ' Active Min': 'Active Min',
    'Idle Mean': 'Idle Mean',
    ' Idle Std': 'Idle Std',
    ' Idle Max': 'Idle Max',
    ' Idle Min': 'Idle Min',
    ' Label': 'Label',
}

# ---------------------------------------------------------------------------
# Visual Dark SOC Color Palette
# ---------------------------------------------------------------------------
THEME = {
    "bg_dark":      "#0b0f17",
    "card_bg":      "#131b26",
    "card_alt":     "#1a2433",
    "border":       "#263447",
    "text_main":    "#f1f5f9",
    "text_muted":   "#94a3b8",
    "text_dim":     "#64748b",
    
    # Action Badges
    "block_red":    "#ef4444",
    "block_bg":     "#3b1219",
    "flag_amber":   "#f59e0b",
    "flag_bg":      "#3d2406",
    "allow_green":  "#10b981",
    "allow_bg":     "#092e20",
    
    # Accent & Telemetry
    "accent_cyan":  "#06b6d4",
    "accent_blue":  "#3b82f6",
    "meter_bg":     "#1e293b",
}


class SOCDashboardApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Multi-Granular NIDS — Real-Time Operations Center (CICIDS2017)")
        self.root.geometry("1480x920")
        self.root.minsize(1280, 800)
        self.root.configure(bg=THEME["bg_dark"])

        # Threading and Queues
        self.data_queue = queue.Queue()
        self.worker_thread = None
        self.is_running = False
        self.is_paused = False
        self.step_event = threading.Event()
        self.stop_requested = False

        # Dynamic Context Tracking
        self.consecutive_attack_windows = 0
        self.last_attack_time = 0.0
        self.stream_delay = 0.45

        # Statistics Counters
        self.total_windows = 0
        self.total_flows = 0
        self.blocked_count = 0
        self.flagged_count = 0
        self.allowed_count = 0
        self.sound_enabled = tk.BooleanVar(value=True)

        # Storage for table selection
        self.history_records = []
        self.selected_record = None

        # Threat Alert Flash State
        self.flash_state = False
        self.active_alert = None

        # Initialize ML Adapters
        self._init_models()

        # Build UI
        self._build_styles()
        self._build_layout()

        # Start Queue Polling loop
        self.root.after(50, self._process_queue)
        self.root.after(600, self._pulse_alert_banner)

    # -----------------------------------------------------------------------
    def _init_models(self):
        try:
            self.pkt_adapter = PacketModelAdapter()
            self.flow_adapter = FlowModelAdapter()
            self.tw_adapter = TimeWindowAdapter()
            self.sess_adapter = SessionModelAdapter()
            self.attack_gate = AttackGate(threshold=0.35)
            self.rule_engine = RuleEngine()
            self.policy_agent = PolicyAgent(use_gemini=True)
            self.model_status_text = "4 Models Active + Qwen 2.5 Policy Layer Ready"
        except Exception as e:
            self.model_status_text = f"Initialization Error: {e}"

    # -----------------------------------------------------------------------
    def _build_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Treeview styling
        style.configure(
            "SOC.Treeview",
            background=THEME["card_bg"],
            foreground=THEME["text_main"],
            fieldbackground=THEME["card_bg"],
            borderwidth=0,
            rowheight=26,
            font=("Segoe UI", 9),
        )
        style.configure(
            "SOC.Treeview.Heading",
            background=THEME["card_alt"],
            foreground=THEME["accent_cyan"],
            relief="flat",
            font=("Segoe UI", 9, "bold"),
        )
        style.map("SOC.Treeview", background=[("selected", THEME["accent_blue"])])

        # Scrollbar styling
        style.configure(
            "Vertical.TScrollbar",
            background=THEME["card_alt"],
            bordercolor=THEME["border"],
            arrowcolor=THEME["text_muted"],
            troughcolor=THEME["bg_dark"],
        )

    # -----------------------------------------------------------------------
    def _build_layout(self):
        # 1. Top Header Bar
        header = tk.Frame(self.root, bg=THEME["card_bg"], height=64, bd=0, highlightthickness=1, highlightbackground=THEME["border"])
        header.pack(fill="x", side="top", padx=12, pady=(10, 6))

        title_box = tk.Frame(header, bg=THEME["card_bg"])
        title_box.pack(side="left", padx=16, pady=8)

        lbl_title = tk.Label(
            title_box,
            text="MULTI-GRANULAR NIDS  |  SECURITY OPERATIONS CENTER",
            font=("Segoe UI", 13, "bold"),
            fg=THEME["text_main"],
            bg=THEME["card_bg"],
        )
        lbl_title.pack(anchor="w")

        lbl_sub = tk.Label(
            title_box,
            text=f"Live 4-Model Hierarchical Arbitration  •  Target: CICIDS2017 Dataset  •  {self.model_status_text}",
            font=("Segoe UI", 9),
            fg=THEME["accent_cyan"],
            bg=THEME["card_bg"],
        )
        lbl_sub.pack(anchor="w")

        # Sound Mute Toggle
        sound_chk = tk.Checkbutton(
            header,
            text="🔊 Alert Sound Enabled",
            variable=self.sound_enabled,
            bg=THEME["card_bg"],
            fg=THEME["text_muted"],
            selectcolor=THEME["card_alt"],
            activebackground=THEME["card_bg"],
            activeforeground=THEME["text_main"],
            font=("Segoe UI", 9),
        )
        sound_chk.pack(side="right", padx=16)

        # 2. Executive KPI Cards Row
        kpi_frame = tk.Frame(self.root, bg=THEME["bg_dark"])
        kpi_frame.pack(fill="x", padx=12, pady=4)

        self.card_total = self._create_kpi_card(kpi_frame, "TOTAL PROCESSED", "0 Windows (0 Flows)", THEME["accent_blue"])
        self.card_threat = self._create_kpi_card(kpi_frame, "DEFENSE STATUS", "NORMAL", THEME["allow_green"])
        self.card_blocked = self._create_kpi_card(kpi_frame, "ATTACKS BLOCKED", "0", THEME["block_red"])
        self.card_flagged = self._create_kpi_card(kpi_frame, "ATTACKS FLAGGED", "0", THEME["flag_amber"])
        self.card_allowed = self._create_kpi_card(kpi_frame, "BENIGN ALLOWED", "0", THEME["allow_green"])
        self.card_risk = self._create_kpi_card(kpi_frame, "AVG FUSED RISK", "0.000", THEME["accent_cyan"])

        for card in [self.card_total, self.card_threat, self.card_blocked, self.card_flagged, self.card_allowed, self.card_risk]:
            card.pack(side="left", fill="both", expand=True, padx=4)

        # 3. Dedicated Malicious Threat Alert Banner (Silent on Benign)
        self.alert_banner = tk.Frame(self.root, bg=THEME["card_alt"], bd=1, relief="solid", highlightthickness=1, highlightbackground=THEME["border"])
        self.alert_banner.pack(fill="x", padx=16, pady=6)

        self.alert_icon = tk.Label(self.alert_banner, text="🛡️", font=("Segoe UI", 16), bg=THEME["card_alt"], fg=THEME["allow_green"])
        self.alert_icon.pack(side="left", padx=12, pady=8)

        self.alert_headline = tk.Label(
            self.alert_banner,
            text="ALL CLEAR  —  Monitoring real-time traffic across 4 model granularities. No active security alerts.",
            font=("Segoe UI", 10, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["card_alt"],
        )
        self.alert_headline.pack(side="left", fill="x", expand=True, padx=4)

        self.alert_badge = tk.Label(
            self.alert_banner,
            text="STATUS: SECURE",
            font=("Segoe UI", 9, "bold"),
            bg=THEME["allow_bg"],
            fg=THEME["allow_green"],
            padx=10,
            pady=4,
        )
        self.alert_badge.pack(side="right", padx=16)

        # 4. Stream Playback Control Strip
        ctrl_strip = tk.Frame(self.root, bg=THEME["card_bg"], bd=0, highlightthickness=1, highlightbackground=THEME["border"])
        ctrl_strip.pack(fill="x", padx=12, pady=4)

        # Control Buttons
        self.btn_play = tk.Button(
            ctrl_strip,
            text="▶ Play Stream",
            command=self._start_stream,
            font=("Segoe UI", 9, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            padx=14,
            pady=4,
            cursor="hand2",
        )
        self.btn_play.pack(side="left", padx=(12, 6), pady=8)

        self.btn_pause = tk.Button(
            ctrl_strip,
            text="⏸ Pause",
            command=self._pause_stream,
            font=("Segoe UI", 9),
            bg=THEME["card_alt"],
            fg=THEME["text_main"],
            relief="flat",
            padx=10,
            pady=4,
            state="disabled",
            cursor="hand2",
        )
        self.btn_pause.pack(side="left", padx=4)

        self.btn_step = tk.Button(
            ctrl_strip,
            text="⏭ Step Window",
            command=self._step_stream,
            font=("Segoe UI", 9),
            bg=THEME["card_alt"],
            fg=THEME["text_main"],
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
        )
        self.btn_step.pack(side="left", padx=4)

        self.btn_reset = tk.Button(
            ctrl_strip,
            text="↺ Reset",
            command=self._reset_stream,
            font=("Segoe UI", 9),
            bg=THEME["card_alt"],
            fg=THEME["text_muted"],
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
        )
        self.btn_reset.pack(side="left", padx=4)

        # Jump-to-day dropdown
        tk.Label(ctrl_strip, text="Jump to Day:", font=("Segoe UI", 9), fg=THEME["text_muted"], bg=THEME["card_bg"]).pack(side="left", padx=(18, 4))
        self.selected_day_idx = tk.IntVar(value=0)
        day_names = [d[0] for d in CICIDS2017_DAYS]
        self.day_combo = ttk.Combobox(ctrl_strip, values=day_names, state="readonly", width=30)
        self.day_combo.current(0)
        self.day_combo.pack(side="left", padx=4)
        self.day_combo.bind("<<ComboboxSelected>>", self._on_day_selected)

        # Speed slider
        tk.Label(ctrl_strip, text="Speed:", font=("Segoe UI", 9), fg=THEME["text_muted"], bg=THEME["card_bg"]).pack(side="left", padx=(18, 4))
        self.speed_slider = tk.Scale(
            ctrl_strip,
            from_=0.1,
            to=1.5,
            resolution=0.1,
            orient="horizontal",
            length=110,
            bg=THEME["card_bg"],
            fg=THEME["accent_cyan"],
            highlightthickness=0,
            troughcolor=THEME["card_alt"],
            command=self._on_speed_change,
        )
        self.speed_slider.set(0.4)
        self.speed_slider.pack(side="left", padx=4)

        # Current Day / Progress Indicator
        self.lbl_progress = tk.Label(
            ctrl_strip,
            text="Sequence: Day 1/8 (Monday-WorkingHours)",
            font=("Segoe UI", 9, "bold"),
            fg=THEME["accent_cyan"],
            bg=THEME["card_bg"],
        )
        self.lbl_progress.pack(side="right", padx=16)

        # 5. Split Main View (Tables on Left, Deep-Dive on Right)
        main_split = tk.Frame(self.root, bg=THEME["bg_dark"])
        main_split.pack(fill="both", expand=True, padx=12, pady=(4, 10))

        # LEFT PANE: Notebook (Live Feed & Malicious Alerts Only)
        left_pane = tk.Frame(main_split, bg=THEME["card_bg"], bd=0, highlightthickness=1, highlightbackground=THEME["border"])
        left_pane.pack(side="left", fill="both", expand=True, padx=(0, 6))

        notebook = ttk.Notebook(left_pane)
        notebook.pack(fill="both", expand=True, padx=4, pady=4)

        # Tab 1: All Live Traffic
        tab_all = tk.Frame(notebook, bg=THEME["card_bg"])
        notebook.add(tab_all, text="  Live Traffic Stream  ")

        # Tab 2: Actionable Alerts Only
        tab_alerts = tk.Frame(notebook, bg=THEME["card_bg"])
        notebook.add(tab_alerts, text="  🚨 Threat Alerts Feed  ")

        self.tree_all = self._build_treeview(tab_all)
        self.tree_alerts = self._build_treeview(tab_alerts)

        # RIGHT PANE: Deep-Dive Record Inspector
        right_pane = tk.Frame(main_split, bg=THEME["card_bg"], width=460, bd=0, highlightthickness=1, highlightbackground=THEME["border"])
        right_pane.pack(side="right", fill="both", expand=False, padx=(6, 0))
        right_pane.pack_propagate(False)

        self._build_inspector_panel(right_pane)

    # -----------------------------------------------------------------------
    def _create_kpi_card(self, parent, title: str, init_val: str, color: str):
        card = tk.Frame(parent, bg=THEME["card_bg"], bd=0, highlightthickness=1, highlightbackground=THEME["border"])
        
        lbl_t = tk.Label(card, text=title, font=("Segoe UI", 8, "bold"), fg=THEME["text_dim"], bg=THEME["card_bg"])
        lbl_t.pack(anchor="w", padx=12, pady=(8, 2))

        lbl_v = tk.Label(card, text=init_val, font=("Segoe UI", 14, "bold"), fg=color, bg=THEME["card_bg"])
        lbl_v.pack(anchor="w", padx=12, pady=(0, 8))
        
        card.val_label = lbl_v
        return card

    # -----------------------------------------------------------------------
    def _build_treeview(self, parent):
        frame = tk.Frame(parent, bg=THEME["card_bg"])
        frame.pack(fill="both", expand=True)

        cols = ("time", "source", "ground_truth", "p_att", "agree", "action", "severity", "family", "engine")
        tree = ttk.Treeview(frame, columns=cols, show="headings", style="SOC.Treeview", selectmode="browse")

        tree.heading("time", text="Timestamp")
        tree.heading("source", text="Day / File")
        tree.heading("ground_truth", text="Ground Truth")
        tree.heading("p_att", text="Fused P(att)")
        tree.heading("agree", text="Agree")
        tree.heading("action", text="Action")
        tree.heading("severity", text="Severity")
        tree.heading("family", text="Detected Family")
        tree.heading("engine", text="Decision Engine")

        tree.column("time", width=80, anchor="center")
        tree.column("source", width=110, anchor="w")
        tree.column("ground_truth", width=120, anchor="w")
        tree.column("p_att", width=75, anchor="center")
        tree.column("agree", width=55, anchor="center")
        tree.column("action", width=70, anchor="center")
        tree.column("severity", width=75, anchor="center")
        tree.column("family", width=110, anchor="w")
        tree.column("engine", width=95, anchor="center")

        sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview, style="Vertical.TScrollbar")
        tree.configure(yscrollcommand=sb.set)

        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Color row tags
        tree.tag_configure("BLOCK", foreground="#f87171", background="#2a1215")
        tree.tag_configure("FLAG", foreground="#fbbf24", background="#291e0a")
        tree.tag_configure("ALLOW", foreground="#94a3b8", background=THEME["card_bg"])

        tree.bind("<<TreeviewSelect>>", self._on_row_selected)
        return tree

    # -----------------------------------------------------------------------
    def _build_inspector_panel(self, parent):
        header_lbl = tk.Label(
            parent,
            text="RECORD ANALYSIS & EXPLAINABILITY",
            font=("Segoe UI", 10, "bold"),
            fg=THEME["accent_cyan"],
            bg=THEME["card_alt"],
            pady=8,
        )
        header_lbl.pack(fill="x")

        # Scrollable container for deep inspection
        canvas = tk.Canvas(parent, bg=THEME["card_bg"], bd=0, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview, style="Vertical.TScrollbar")
        self.inspector_content = tk.Frame(canvas, bg=THEME["card_bg"])

        self.inspector_content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inspector_content, anchor="nw", width=440)
        canvas.configure(yscrollcommand=sb.set)

        canvas.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        sb.pack(side="right", fill="y")

        # Section 1: Selected Overview
        sec1 = self._inspector_section(self.inspector_content, "Selected Window Context")
        self.insp_lbl_id = tk.Label(sec1, text="Window ID: None selected", font=("Segoe UI", 9, "bold"), fg=THEME["text_main"], bg=THEME["card_bg"])
        self.insp_lbl_id.pack(anchor="w")
        self.insp_lbl_gt = tk.Label(sec1, text="Ground Truth: —", font=("Segoe UI", 9), fg=THEME["text_muted"], bg=THEME["card_bg"])
        self.insp_lbl_gt.pack(anchor="w")
        self.insp_lbl_action = tk.Label(sec1, text="Final Verdict: —", font=("Segoe UI", 10, "bold"), fg=THEME["text_muted"], bg=THEME["card_bg"])
        self.insp_lbl_action.pack(anchor="w", pady=(2, 0))

        # Section 2: 4-Model Probability Meters
        sec2 = self._inspector_section(self.inspector_content, "4-Model Multi-Granular Evidence")
        self.meter_pkt = self._create_model_meter(sec2, "Packet 1D-CNN (L3/L4)")
        self.meter_flow = self._create_model_meter(sec2, "Flow XGBoost (Statistical)")
        self.meter_tw = self._create_model_meter(sec2, "Time-Window (Macro 30s)")
        self.meter_sess = self._create_model_meter(sec2, "Session CNN-LSTM (Sequential)")

        # Section 3: Dynamic Risk Components
        sec3 = self._inspector_section(self.inspector_content, "Dynamic Risk Score Breakdown")
        self.lbl_risk_total = tk.Label(sec3, text="Risk Score R = 0.000", font=("Segoe UI", 10, "bold"), fg=THEME["accent_cyan"], bg=THEME["card_bg"])
        self.lbl_risk_total.pack(anchor="w", pady=(0, 4))

        self.lbl_risk_details = tk.Label(
            sec3,
            text="Attack Prob: 0.00  |  Agreement: 0.00\nPersistence: 0.00  |  Escalation: 0.00",
            font=("Segoe UI", 8),
            fg=THEME["text_muted"],
            bg=THEME["card_bg"],
            justify="left",
        )
        self.lbl_risk_details.pack(anchor="w")

        # Section 4: Machine Explainability Reason
        sec4 = self._inspector_section(self.inspector_content, "Decision Layer Audit Rationale")
        self.txt_reason = tk.Text(
            sec4,
            height=7,
            wrap="word",
            bg=THEME["card_alt"],
            fg=THEME["text_main"],
            bd=0,
            font=("Segoe UI", 9),
            padx=8,
            pady=8,
        )
        self.txt_reason.insert("1.0", "Select any traffic window in the live stream to inspect the exact model probability breakdown and Qwen 2.5 policy rationale.")
        self.txt_reason.configure(state="disabled")
        self.txt_reason.pack(fill="x", pady=4)

    # -----------------------------------------------------------------------
    def _inspector_section(self, parent, title: str):
        box = tk.Frame(parent, bg=THEME["card_bg"], bd=0, highlightthickness=1, highlightbackground=THEME["border"], padx=10, pady=8)
        box.pack(fill="x", pady=4)
        tk.Label(box, text=title.upper(), font=("Segoe UI", 8, "bold"), fg=THEME["accent_cyan"], bg=THEME["card_bg"]).pack(anchor="w", pady=(0, 4))
        return box

    def _create_model_meter(self, parent, name: str):
        box = tk.Frame(parent, bg=THEME["card_bg"], pady=3)
        box.pack(fill="x")

        hdr = tk.Frame(box, bg=THEME["card_bg"])
        hdr.pack(fill="x")

        lbl_name = tk.Label(hdr, text=name, font=("Segoe UI", 8, "bold"), fg=THEME["text_muted"], bg=THEME["card_bg"])
        lbl_name.pack(side="left")

        lbl_prob = tk.Label(hdr, text="P=0.00 | —", font=("Segoe UI", 8), fg=THEME["text_dim"], bg=THEME["card_bg"])
        lbl_prob.pack(side="right")

        canvas = tk.Canvas(box, height=8, bg=THEME["meter_bg"], bd=0, highlightthickness=0)
        canvas.pack(fill="x", pady=(2, 0))

        bar = canvas.create_rectangle(0, 0, 0, 8, fill=THEME["accent_cyan"], width=0)

        box.lbl_prob = lbl_prob
        box.canvas = canvas
        box.bar = bar
        return box

    def _update_meter(self, meter, prob: float, pred: str):
        clamped = max(0.0, min(1.0, prob))
        color = THEME["block_red"] if clamped >= 0.70 else (THEME["flag_amber"] if clamped >= 0.35 else THEME["allow_green"])
        meter.lbl_prob.config(text=f"P(att)={prob:.2f}  [{pred}]", fg=color)
        
        meter.canvas.update_idletasks()
        w = meter.canvas.winfo_width()
        meter.canvas.coords(meter.bar, 0, 0, int(w * clamped), 8)
        meter.canvas.itemconfig(meter.bar, fill=color)

    # -----------------------------------------------------------------------
    def _on_speed_change(self, val):
        self.stream_delay = float(val)

    def _on_day_selected(self, event):
        idx = self.day_combo.current()
        self.selected_day_idx.set(idx)
        name, _ = CICIDS2017_DAYS[idx]
        self.lbl_progress.config(text=f"Sequence: Day {idx+1}/8 ({name})")
        if self.is_running:
            self._reset_stream()
            self.selected_day_idx.set(idx)
            self.day_combo.current(idx)
            self.lbl_progress.config(text=f"Ready: Day {idx+1}/8 ({name})")

    # -----------------------------------------------------------------------
    def _start_stream(self):
        if self.is_running and self.is_paused:
            self.is_paused = False
            self.btn_pause.config(state="normal", text="⏸ Pause")
            self.btn_play.config(state="disabled")
            return

        if not self.is_running:
            self.is_running = True
            self.is_paused = False
            self.stop_requested = False
            self.btn_play.config(state="disabled")
            self.btn_pause.config(state="normal", text="⏸ Pause")
            
            # Launch Background Ingestion Thread
            self.worker_thread = threading.Thread(target=self._stream_worker, daemon=True)
            self.worker_thread.start()

    def _pause_stream(self):
        if self.is_running:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.btn_pause.config(text="▶ Resume")
                self.btn_play.config(state="normal")
            else:
                self.btn_pause.config(text="⏸ Pause")
                self.btn_play.config(state="disabled")

    def _step_stream(self):
        if not self.is_running:
            self._start_stream()
            self.is_paused = True
            self.btn_pause.config(text="▶ Resume")
            self.btn_play.config(state="normal")
        self.step_event.set()

    def _reset_stream(self):
        self.stop_requested = True
        self.is_running = False
        self.is_paused = False
        self.consecutive_attack_windows = 0
        self.btn_play.config(state="normal")
        self.btn_pause.config(state="disabled", text="⏸ Pause")

        # Drain any residual queued items
        try:
            while not self.data_queue.empty():
                self.data_queue.get_nowait()
        except Exception:
            pass

        # Clear UI records
        for tree in [self.tree_all, self.tree_alerts]:
            for row in tree.get_children():
                tree.delete(row)

        self.history_records.clear()
        self.total_windows = 0
        self.total_flows = 0
        self.blocked_count = 0
        self.flagged_count = 0
        self.allowed_count = 0
        self._update_kpi_display()

        self.alert_headline.config(text="ALL CLEAR  —  Monitoring real-time traffic across 4 model granularities. No active security alerts.")
        self.alert_badge.config(text="STATUS: SECURE", bg=THEME["allow_bg"], fg=THEME["allow_green"])
        self.alert_icon.config(text="🛡️", fg=THEME["allow_green"])
        self.active_alert = None

    # -----------------------------------------------------------------------
    def _stream_worker(self):
        """Worker thread that streams windows sequentially across all 8 days of CICIDS2017."""
        start_day_idx = self.selected_day_idx.get()
        days_to_process = CICIDS2017_DAYS[start_day_idx:]

        for day_num, (day_label, rel_path) in enumerate(days_to_process, start=start_day_idx + 1):
            if self.stop_requested:
                break

            self.data_queue.put(("STATUS_UPDATE", f"Ingesting Day {day_num}/8: {Path(rel_path).name}"))
            abs_path = HERE / rel_path
            if not abs_path.exists():
                continue

            try:
                # Read chunks of CSV with proper encoding
                chunk_reader = pd.read_csv(abs_path, chunksize=8000, low_memory=False, encoding="cp1252")
                for chunk in chunk_reader:
                    if self.stop_requested:
                        break

                    # Rename columns to standard 2018 schema
                    chunk = chunk.rename(columns=COL_RENAME_2017)
                    if "Label" not in chunk.columns:
                        continue

                    # Clean labels
                    chunk["Label"] = chunk["Label"].astype(str).str.strip().replace("", "-")
                    
                    # Split into evaluation windows of ~60 flows
                    window_size = 60
                    for w_idx in range(0, len(chunk), window_size):
                        if self.stop_requested:
                            break

                        # Handle Play / Pause / Step
                        while self.is_paused and not self.stop_requested:
                            if self.step_event.is_set():
                                self.step_event.clear()
                                break
                            time.sleep(0.05)

                        w_slice = chunk.iloc[w_idx : w_idx + window_size].copy()
                        if len(w_slice) < 5:
                            continue

                        # Execute 4 Models & Decision Layer
                        record = self._process_window(w_slice, day_label, Path(rel_path).name)
                        self.data_queue.put(("WINDOW_RESULT", record))

                        time.sleep(self.stream_delay)

            except Exception as e:
                self.data_queue.put(("STATUS_UPDATE", f"Error in {Path(rel_path).name}: {e}"))

        self.data_queue.put(("STREAM_FINISHED", None))

    # -----------------------------------------------------------------------
    def _process_window(self, df_slice: pd.DataFrame, day_label: str, filename: str) -> dict:
        """Runs the 4 models + fusion + policy agent on a traffic window."""
        # 1. Schema & Feature Harmonization for CICIDS2017
        for w_col in ['Init Fwd Win Byts', 'Init Bwd Win Byts']:
            if w_col in df_slice.columns:
                df_slice[w_col] = pd.to_numeric(df_slice[w_col], errors='coerce').fillna(0).clip(lower=0)

        # 2017 Total bytes vs payload normalization
        if 'TotLen Fwd Pkts' in df_slice.columns and 'Fwd Header Len' in df_slice.columns:
            df_slice['TotLen Fwd Pkts'] = df_slice['TotLen Fwd Pkts'] + df_slice['Fwd Header Len']
        if 'TotLen Bwd Pkts' in df_slice.columns and 'Bwd Header Len' in df_slice.columns:
            df_slice['TotLen Bwd Pkts'] = df_slice['TotLen Bwd Pkts'] + df_slice['Bwd Header Len']

        durations = pd.to_numeric(df_slice.get('Flow Duration', 1000), errors='coerce').fillna(1000).clip(lower=50) / 1e6
        cum_time = np.cumsum(durations.values)
        t_base = pd.Timestamp("2018-02-15 09:00:00")
        df_slice["Timestamp"] = [t_base + pd.Timedelta(seconds=float(s)) for s in cum_time]

        if "Protocol" not in df_slice.columns:
            df_slice["Protocol"] = 6.0

        t_start = df_slice["Timestamp"].iloc[0]
        gt_mode = df_slice["Label"].mode()[0] if not df_slice.empty else "Benign"

        # 2. Packet Model
        try:
            pkt_ev = self.pkt_adapter.predict(df_slice)
        except Exception:
            pkt_ev = None

        # 3. Flow Model
        try:
            flow_ev = self.flow_adapter.predict(df_slice)
        except Exception:
            flow_ev = None

        # 4. Time Window Model
        try:
            self.tw_adapter.reset_history()
            agg = preprocessing.window_aggregate(df_slice, window_start=t_start)
            tw_ev = self.tw_adapter.predict(agg)
        except Exception:
            tw_ev = None

        # 5. Session Model
        try:
            sess_ev = self.sess_adapter.predict(df_slice)
        except Exception:
            sess_ev = None

        # Multi-granular Byzantine consensus guardrail (Sommer & Paxson 2010):
        # If at least 3 models confirm Benign traffic (attack_prob <= 0.35, not is_attack),
        # attenuate single-model out-of-distribution noise (e.g. Flow or Session false alarms)
        benign_models = [ev for ev in [flow_ev, pkt_ev, tw_ev, sess_ev] if ev and not ev.is_attack and ev.attack_probability <= 0.35]
        if len(benign_models) >= 3:
            for ev in [flow_ev, pkt_ev, tw_ev, sess_ev]:
                if ev and ev.is_attack and ev.attack_probability > 0.35:
                    ev.attack_probability = min(ev.attack_probability, 0.25)
                    ev.prediction = "Benign"
                    ev.is_attack = False

        # 6. Fusion & Decision Layer
        integrated = self.attack_gate.evaluate(pkt_ev, flow_ev, tw_ev, sess_ev, window_id="W_LIVE")
        
        # Dynamic context state
        c_att = self.consecutive_attack_windows
        esc = "CRITICAL" if c_att >= 3 else ("MEDIUM" if c_att >= 1 else "NONE")
        ctx = {"consecutive_attack_windows": c_att, "escalation_level": esc}
        
        rule_res = self.rule_engine.evaluate(integrated, context=ctx)
        decision = self.policy_agent.decide(integrated, context_summary=ctx, rule_result=rule_res)

        # Update dynamic context for next window:
        # A true attack streak requires corroborated high-threat attacks to escalate
        if decision.action == "BLOCK" or (decision.action == "FLAG" and integrated.models_indicating_attack >= 2 and integrated.fused_attack_probability >= 0.50):
            self.consecutive_attack_windows = min(self.consecutive_attack_windows + 1, 10)
        else:
            self.consecutive_attack_windows = 0

        return {
            "window_id": f"W_{int(time.time()*1000)%1000000:06d}",
            "time_str": time.strftime("%H:%M:%S"),
            "day_label": day_label,
            "filename": filename,
            "flows_count": len(df_slice),
            "ground_truth": gt_mode,
            
            "pkt_ev": {"prob": pkt_ev.attack_probability if pkt_ev else 0.0, "pred": pkt_ev.prediction if pkt_ev else "—"},
            "flow_ev": {"prob": flow_ev.attack_probability if flow_ev else 0.0, "pred": flow_ev.prediction if flow_ev else "—"},
            "tw_ev": {"prob": tw_ev.attack_probability if tw_ev else 0.0, "pred": tw_ev.prediction if tw_ev else "—"},
            "sess_ev": {"prob": sess_ev.attack_probability if sess_ev else 0.0, "pred": sess_ev.prediction if sess_ev else "—"},
            
            "fused_prob": integrated.fused_attack_probability,
            "agreement": integrated.model_agreement_score,
            "fused_family": integrated.fused_attack_family or "None",
            "models_attacking": integrated.models_indicating_attack,
            
            "action": decision.action,
            "severity": decision.severity,
            "attack_family": decision.attack_family or "None",
            "source": decision.source,
            "rule_name": decision.rule_triggered,
            "risk_score": decision.risk_score,
            "risk_components": decision.risk_components,
            "reason": decision.reason,
        }

    # -----------------------------------------------------------------------
    def _process_queue(self):
        """Processes records arriving from the background streaming thread."""
        try:
            while True:
                msg_type, payload = self.data_queue.get_nowait()

                if msg_type == "WINDOW_RESULT":
                    self._handle_new_record(payload)

                elif msg_type == "STATUS_UPDATE":
                    self.lbl_progress.config(text=str(payload))

                elif msg_type == "STREAM_FINISHED":
                    self.is_running = False
                    self.btn_play.config(state="normal")
                    self.btn_pause.config(state="disabled", text="⏸ Pause")
                    messagebox.showinfo("Stream Complete", "Completed continuous traffic replay across all selected CICIDS2017 files.")

        except queue.Empty:
            pass

        self.root.after(40, self._process_queue)

    # -----------------------------------------------------------------------
    def _handle_new_record(self, r: dict):
        self.history_records.append(r)
        self.total_windows += 1
        self.total_flows += r["flows_count"]

        action = r["action"]
        if action == "BLOCK":
            self.blocked_count += 1
        elif action == "FLAG":
            self.flagged_count += 1
        else:
            self.allowed_count += 1

        self._update_kpi_display()

        # Insert into Live Feed Table
        row_vals = (
            r["time_str"],
            r["filename"][:22],
            r["ground_truth"][:16],
            f"{r['fused_prob']:.3f}",
            f"{int(r['agreement']*100)}%",
            r["action"],
            r["severity"],
            r["attack_family"],
            f"{r['source']}",
        )
        self.tree_all.insert("", 0, values=row_vals, tags=(action,))

        # Alert System: ONLY for Malicious (FLAG and BLOCK) — Silent on Benign
        if action in ("BLOCK", "FLAG"):
            self.tree_alerts.insert("", 0, values=row_vals, tags=(action,))
            self._trigger_threat_alert(r)
        else:
            # Benign traffic resets alert to calm state
            if self.active_alert is None or (time.time() - self.active_alert.get("time", 0) > 4.0):
                self._clear_threat_alert()

        # Automatically select the latest record in the inspector
        self._populate_inspector(r)

    # -----------------------------------------------------------------------
    def _trigger_threat_alert(self, r: dict):
        """Triggers prominent visual and auditory security alert."""
        self.active_alert = {"record": r, "time": time.time()}
        action = r["action"]
        family = r["attack_family"]
        source = r["source"]

        if action == "BLOCK":
            bg_col = THEME["block_red"]
            badge_text = f"CRITICAL BREACH: BLOCKED [{family}]"
            icon = "🚨"
            msg = f"MALICIOUS ATTACK TERMINATED  —  {family} blocked via {source} fast-path (Risk={r['risk_score']:.3f})"
        else:
            bg_col = THEME["flag_amber"]
            badge_text = f"ANOMALOUS THREAT: FLAGGED [{family}]"
            icon = "⚠️"
            msg = f"SECURITY ALERT: QUARANTINED  —  {family} flagged for SOC analyst review by {source} (P_att={r['fused_prob']:.3f})"

        self.alert_banner.config(bg=bg_col, highlightbackground=bg_col)
        self.alert_icon.config(text=icon, fg="#ffffff", bg=bg_col)
        self.alert_headline.config(text=msg, fg="#ffffff", bg=bg_col)
        self.alert_badge.config(text=badge_text, bg="#ffffff", fg=bg_col)

        # Auditory alert (Windows beep)
        if self.sound_enabled.get():
            try:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION if action == "FLAG" else winsound.MB_ICONHAND)
            except Exception:
                pass

    def _clear_threat_alert(self):
        self.alert_banner.config(bg=THEME["card_alt"], highlightbackground=THEME["border"])
        self.alert_icon.config(text="🛡️", fg=THEME["allow_green"], bg=THEME["card_alt"])
        self.alert_headline.config(text="ALL CLEAR  —  Monitoring real-time traffic across 4 model granularities. No active security alerts.", fg=THEME["text_muted"], bg=THEME["card_alt"])
        self.alert_badge.config(text="STATUS: SECURE", bg=THEME["allow_bg"], fg=THEME["allow_green"])

    def _pulse_alert_banner(self):
        """Creates a subtle pulsing effect when an attack is actively firing."""
        if self.active_alert and (time.time() - self.active_alert["time"] < 3.0):
            self.flash_state = not self.flash_state
            if self.active_alert["record"]["action"] == "BLOCK":
                col = "#b91c1c" if self.flash_state else THEME["block_red"]
            else:
                col = "#b45309" if self.flash_state else THEME["flag_amber"]
            self.alert_banner.config(bg=col)
            self.alert_headline.config(bg=col)
            self.alert_icon.config(bg=col)

        self.root.after(500, self._pulse_alert_banner)

    # -----------------------------------------------------------------------
    def _update_kpi_display(self):
        self.card_total.val_label.config(text=f"{self.total_windows:,} W ({self.total_flows:,} F)")
        self.card_blocked.val_label.config(text=f"{self.blocked_count:,}")
        self.card_flagged.val_label.config(text=f"{self.flagged_count:,}")
        self.card_allowed.val_label.config(text=f"{self.allowed_count:,}")

        # Threat Level Indicator
        if self.blocked_count > 0:
            self.card_threat.val_label.config(text="CRITICAL", fg=THEME["block_red"])
        elif self.flagged_count > 0:
            self.card_threat.val_label.config(text="ELEVATED", fg=THEME["flag_amber"])
        else:
            self.card_threat.val_label.config(text="NORMAL", fg=THEME["allow_green"])

        # Running average risk
        if self.history_records:
            avg_r = sum(rec["risk_score"] for rec in self.history_records) / len(self.history_records)
            self.card_risk.val_label.config(text=f"{avg_r:.3f}")

    # -----------------------------------------------------------------------
    def _on_row_selected(self, event):
        tree = event.widget
        sel = tree.selection()
        if not sel:
            return
        item_vals = tree.item(sel[0], "values")
        if not item_vals:
            return
        
        # Match by timestamp / index
        time_str = item_vals[0]
        for rec in reversed(self.history_records):
            if rec["time_str"] == time_str:
                self._populate_inspector(rec)
                break

    def _populate_inspector(self, r: dict):
        self.selected_record = r
        self.insp_lbl_id.config(text=f"Window: {r['window_id']}  ({r['time_str']})")
        self.insp_lbl_gt.config(text=f"Ground Truth: {r['ground_truth']}  •  File: {r['filename']}")
        
        act_col = THEME["block_red"] if r["action"] == "BLOCK" else (THEME["flag_amber"] if r["action"] == "FLAG" else THEME["allow_green"])
        self.insp_lbl_action.config(
            text=f"Action: {r['action']}  [{r['severity']}]  •  Detected: {r['attack_family']}  ({r['source']})",
            fg=act_col,
        )

        # Update 4-Model Probability Meters
        self._update_meter(self.meter_pkt, r["pkt_ev"]["prob"], r["pkt_ev"]["pred"])
        self._update_meter(self.meter_flow, r["flow_ev"]["prob"], r["flow_ev"]["pred"])
        self._update_meter(self.meter_tw, r["tw_ev"]["prob"], r["tw_ev"]["pred"])
        self._update_meter(self.meter_sess, r["sess_ev"]["prob"], r["sess_ev"]["pred"])

        # Update Risk Details
        self.lbl_risk_total.config(text=f"Risk Score R = {r['risk_score']:.3f}  (Agreement: {int(r['agreement']*100)}%)")
        comp = r.get("risk_components", {})
        self.lbl_risk_details.config(
            text=f"P(attack): {comp.get('attack_probability', r['fused_prob']):.3f}  |  Agreement: {comp.get('model_agreement', r['agreement']):.2f}\n"
                 f"Persistence: {comp.get('persistence', 0.0):.2f}  |  Escalation: {comp.get('escalation', 0.0):.2f}"
        )

        # Update Natural Language Explanation
        self.txt_reason.configure(state="normal")
        self.txt_reason.delete("1.0", "end")
        self.txt_reason.insert("1.0", str(r.get("reason", "No detailed audit rationale available.")))
        self.txt_reason.configure(state="disabled")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = SOCDashboardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

