"""
ScratchLM Desktop Application GUI Interface.

Built with Python Tkinter/ttk using dark mode styling, thread-safe streaming,
integrated model training, real-time loss plotting, evaluation, and checkpoint management.

Provides a full graphical runner and training environment for ScratchLM language models.
"""

import sys
import os
import json
import time
import queue
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    tk = None
    ttk = None

import torch

from scratchlm.inference import InferenceEngine, GenerationResult, ModelInfo
from scratchlm.training import Trainer, CheckpointManager
from scratchlm.evaluation import EvaluationSuite
from scratchlm.data import load_data, load_text_file, load_text_directory
from scratchlm.model.config import get_config_preset, TransformerConfig, TrainingConfig, TokenizerConfig
from scratchlm.utils.paths import paths
from scratchlm.gui.chart import LossChartCanvas


class ScratchLMApp:
    """
    Main Tkinter Desktop Application for ScratchLM:
    - Text Generation / Inference
    - Real-Time Training Pipeline with Loss Graph
    - Model Evaluation Suite
    - Checkpoint Management
    """

    def __init__(self, root: Optional[Any] = None):
        self.engine = InferenceEngine(device="cpu")
        self.is_generating = False
        self.is_training = False
        self.is_evaluating = False
        self.stop_requested = False
        self.stream_queue: queue.Queue = queue.Queue()

        self.last_trained_checkpoint: Optional[str] = None
        self.last_trained_tokenizer: Optional[str] = None

        if not HAS_TKINTER:
            print("Notice: Tkinter module not found in environment.")
            self.root = None
            return

        if root is None:
            try:
                self.root = tk.Tk()
            except Exception as e:
                print(f"Notice: Could not initialize Tkinter root window ({e}). Running in headless mode.")
                self.root = None
                return
        else:
            self.root = root

        self.root.title("ScratchLM - Desktop Runner & Trainer (Phase 1 English)")
        self.root.geometry("1040x840")
        self.root.minsize(880, 680)

        # Config file path
        self.config_path = Path.home() / ".scratchlm_app_config.json"

        # Custom Dark Palette Styling
        self.bg_color = "#1E1E2E"
        self.card_bg = "#2B2B3D"
        self.fg_color = "#D9E0EE"
        self.accent_color = "#89B4FA"
        self.success_color = "#A6E3A1"
        self.stop_color = "#F38BA8"
        self.input_bg = "#181825"
        self.prompt_fg = "#A6E3A1"

        self._configure_styles()
        self._build_ui()
        self._load_app_config()
        self._auto_discover_models()
        self._start_queue_poller()

    def _configure_styles(self):
        if not HAS_TKINTER or self.root is None:
            return
        self.root.configure(bg=self.bg_color)
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background=self.bg_color, foreground=self.fg_color, font=("Segoe UI", 10))
        style.configure("TLabel", background=self.bg_color, foreground=self.fg_color)
        style.configure("Card.TFrame", background=self.card_bg, relief="flat", borderwidth=1)
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground=self.accent_color)
        style.configure("Title.TLabel", font=("Segoe UI", 11, "bold"), foreground=self.fg_color)
        style.configure("Muted.TLabel", font=("Segoe UI", 9), foreground="#A6ADC8")
        style.configure("Status.TLabel", font=("Segoe UI", 9, "italic"), foreground="#89DCEB")

        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), background="#89B4FA", foreground="#11111B")
        style.map("Primary.TButton", background=[("active", "#B4BEFE"), ("disabled", "#45475A")])

        style.configure("Stop.TButton", font=("Segoe UI", 10, "bold"), background="#F38BA8", foreground="#11111B")
        style.map("Stop.TButton", background=[("active", "#F5E0DC"), ("disabled", "#45475A")])

        style.configure("TButton", background="#313244", foreground=self.fg_color)
        style.map("TButton", background=[("active", "#45475A")])

        style.configure("TNotebook", background=self.bg_color, borderwidth=0)
        style.configure("TNotebook.Tab", background="#313244", foreground=self.fg_color, padding=[14, 8], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#89B4FA"), ("active", "#45475A")], foreground=[("selected", "#11111B")])

    def _build_ui(self):
        if not HAS_TKINTER or self.root is None:
            return

        main_container = ttk.Frame(self.root, padding=12)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Header Frame with Title & Quick Navigation Buttons
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(header_frame, text="ScratchLM", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(header_frame, text="  |  Desktop Application & Trainer", style="Muted.TLabel").pack(side=tk.LEFT, pady=(3, 0))

        nav_box = ttk.Frame(header_frame)
        nav_box.pack(side=tk.RIGHT)

        ttk.Button(nav_box, text="Generate", width=10, command=lambda: self.notebook.select(self.tab_generate)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_box, text="Training", style="Primary.TButton", width=10, command=lambda: self.notebook.select(self.tab_train)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_box, text="Evaluate", width=10, command=lambda: self.notebook.select(self.tab_evaluate)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_box, text="Checkpoints", width=12, command=lambda: self.notebook.select(self.tab_checkpoints)).pack(side=tk.LEFT, padx=2)

        # Tab Notebook
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tabs
        self.tab_generate = ttk.Frame(self.notebook, padding=8)
        self.tab_train = ttk.Frame(self.notebook, padding=8)
        self.tab_evaluate = ttk.Frame(self.notebook, padding=8)
        self.tab_checkpoints = ttk.Frame(self.notebook, padding=8)
        self.tab_settings = ttk.Frame(self.notebook, padding=8)

        self.notebook.add(self.tab_generate, text=" Generate ")
        self.notebook.add(self.tab_train, text=" Training ")
        self.notebook.add(self.tab_evaluate, text=" Evaluate ")
        self.notebook.add(self.tab_checkpoints, text=" Checkpoints ")
        self.notebook.add(self.tab_settings, text=" Settings ")

        self._build_generate_tab()
        self._build_train_tab()
        self._build_evaluate_tab()
        self._build_checkpoints_tab()
        self._build_settings_tab()

    # =========================================================================
    # TAB 1: GENERATE / INFERENCE
    # =========================================================================

    def _build_generate_tab(self):
        parent = self.tab_generate

        model_card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        model_card.pack(fill=tk.X, pady=(0, 6))

        m_header = ttk.Frame(model_card, style="Card.TFrame")
        m_header.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(m_header, text="Model Checkpoint & Tokenizer", style="Title.TLabel").pack(side=tk.LEFT)

        chk_row = ttk.Frame(model_card, style="Card.TFrame")
        chk_row.pack(fill=tk.X, pady=2)
        ttk.Label(chk_row, text="Checkpoint:", width=12).pack(side=tk.LEFT)

        self.chk_combobox = ttk.Combobox(chk_row, font=("Segoe UI", 9))
        self.chk_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(chk_row, text="Browse...", width=10, command=self._browse_checkpoint).pack(side=tk.LEFT)

        tok_row = ttk.Frame(model_card, style="Card.TFrame")
        tok_row.pack(fill=tk.X, pady=2)
        ttk.Label(tok_row, text="Tokenizer:", width=12).pack(side=tk.LEFT)

        self.tok_combobox = ttk.Combobox(tok_row, font=("Segoe UI", 9))
        self.tok_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(tok_row, text="Browse...", width=10, command=self._browse_tokenizer).pack(side=tk.LEFT)

        action_row = ttk.Frame(model_card, style="Card.TFrame")
        action_row.pack(fill=tk.X, pady=(6, 0))

        self.btn_load_model = ttk.Button(action_row, text="Load Selected Checkpoint", command=self._load_model_threaded)
        self.btn_load_model.pack(side=tk.LEFT)

        self.lbl_model_summary = ttk.Label(action_row, text="No model loaded", style="Status.TLabel")
        self.lbl_model_summary.pack(side=tk.LEFT, padx=12)

        prompt_card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        prompt_card.pack(fill=tk.X, pady=4)

        p_header = ttk.Frame(prompt_card, style="Card.TFrame")
        p_header.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(p_header, text="Prompt Input", style="Title.TLabel").pack(side=tk.LEFT)

        ttk.Label(p_header, text="Examples:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(16, 4))
        self.example_combobox = ttk.Combobox(p_header, state="readonly", width=38, font=("Segoe UI", 9))
        self.example_combobox['values'] = [
            "The Earth revolves around",
            "The solar system consists of",
            "The most important thing about learning is",
            "Water evaporates when",
            "Once upon a time in a small woodland village",
            "In nature, how infinitely complex and close-fitting",
        ]
        self.example_combobox.set("Select example prompt...")
        self.example_combobox.bind("<<ComboboxSelected>>", self._on_select_example)
        self.example_combobox.pack(side=tk.LEFT)

        self.txt_prompt = scrolledtext.ScrolledText(
            prompt_card, height=3, font=("Consolas", 10),
            bg=self.input_bg, fg=self.fg_color, insertbackground=self.fg_color,
            relief="flat", bd=4
        )
        self.txt_prompt.pack(fill=tk.X, expand=True, pady=4)
        self.txt_prompt.insert(tk.END, "The Earth revolves around")

        ctrl_card = ttk.Frame(parent, style="Card.TFrame", padding=8)
        ctrl_card.pack(fill=tk.X, pady=4)

        p_sub = ttk.Frame(ctrl_card, style="Card.TFrame")
        p_sub.pack(fill=tk.X)

        ttk.Label(p_sub, text="Max Tokens:").pack(side=tk.LEFT, padx=(0, 2))
        self.spn_max_tokens = ttk.Spinbox(p_sub, from_=10, to=1000, width=6)
        self.spn_max_tokens.set(100)
        self.spn_max_tokens.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(p_sub, text="Temperature:").pack(side=tk.LEFT, padx=(0, 2))
        self.spn_temp = ttk.Spinbox(p_sub, from_=0.0, to=2.0, increment=0.1, width=5)
        self.spn_temp.set(0.7)
        self.spn_temp.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(p_sub, text="Top-K:").pack(side=tk.LEFT, padx=(0, 2))
        self.spn_top_k = ttk.Spinbox(p_sub, from_=0, to=500, width=5)
        self.spn_top_k.set(50)
        self.spn_top_k.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(p_sub, text="Top-P:").pack(side=tk.LEFT, padx=(0, 2))
        self.spn_top_p = ttk.Spinbox(p_sub, from_=0.0, to=1.0, increment=0.05, width=5)
        self.spn_top_p.set(0.9)
        self.spn_top_p.pack(side=tk.LEFT, padx=(0, 12))

        self.var_greedy = tk.BooleanVar(value=False)
        self.chk_greedy = ttk.Checkbutton(p_sub, text="Greedy", variable=self.var_greedy)
        self.chk_greedy.pack(side=tk.LEFT, padx=6)

        btn_row = ttk.Frame(ctrl_card, style="Card.TFrame")
        btn_row.pack(fill=tk.X, pady=(6, 0))

        self.btn_generate = ttk.Button(btn_row, text="▶ Generate Continuation", style="Primary.TButton", command=self._start_generation_threaded)
        self.btn_generate.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_stop = ttk.Button(btn_row, text="⏹ Stop", style="Stop.TButton", command=self._stop_generation, state="disabled")
        self.btn_stop.pack(side=tk.LEFT)

        self.lbl_gen_stats = ttk.Label(btn_row, text="Ready", style="Muted.TLabel")
        self.lbl_gen_stats.pack(side=tk.RIGHT, padx=6)

        output_card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        output_card.pack(fill=tk.BOTH, expand=True, pady=4)

        o_header = ttk.Frame(output_card, style="Card.TFrame")
        o_header.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(o_header, text="Generated Text", style="Title.TLabel").pack(side=tk.LEFT)

        ttk.Button(o_header, text="Copy Text", width=10, command=self._copy_output).pack(side=tk.RIGHT, padx=2)
        ttk.Button(o_header, text="Clear", width=8, command=self._clear_output).pack(side=tk.RIGHT, padx=2)

        self.txt_output = scrolledtext.ScrolledText(
            output_card, font=("Consolas", 10),
            bg=self.input_bg, fg=self.fg_color, insertbackground=self.fg_color,
            relief="flat", bd=4
        )
        self.txt_output.pack(fill=tk.BOTH, expand=True, pady=4)

        self.txt_output.tag_config("prompt", foreground=self.prompt_fg, font=("Consolas", 10, "bold"))
        self.txt_output.tag_config("continuation", foreground=self.fg_color, font=("Consolas", 10))

    # =========================================================================
    # TAB 2: TRAINING SECTION
    # =========================================================================

    def _build_train_tab(self):
        parent = self.tab_train

        # Split into Left Config Frame and Right Dashboard Frame
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(paned, padding=4)
        right_frame = ttk.Frame(paned, padding=4)
        paned.add(left_frame, weight=1)
        paned.add(right_frame, weight=1)

        # --- LEFT: Config Panel ---
        cfg_card = ttk.Frame(left_frame, style="Card.TFrame", padding=10)
        cfg_card.pack(fill=tk.BOTH, expand=True)

        ttk.Label(cfg_card, text="Training Hyperparameters", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 6))

        # Preset Row
        r0 = ttk.Frame(cfg_card, style="Card.TFrame")
        r0.pack(fill=tk.X, pady=3)
        ttk.Label(r0, text="Model Preset:", width=14).pack(side=tk.LEFT)
        self.train_preset_combo = ttk.Combobox(r0, values=["tiny", "small", "medium"], state="readonly", width=12)
        self.train_preset_combo.set("tiny")
        self.train_preset_combo.pack(side=tk.LEFT, padx=4)

        # Datasets
        r1 = ttk.Frame(cfg_card, style="Card.TFrame")
        r1.pack(fill=tk.X, pady=3)
        ttk.Label(r1, text="Train Data:", width=14).pack(side=tk.LEFT)
        self.txt_train_data = ttk.Entry(r1)
        default_train_path = paths.data_processed / "corpus" / "train.txt"
        if default_train_path.exists():
            self.txt_train_data.insert(0, str(default_train_path))
        elif (paths.data_processed / "corpus" / "train.jsonl").exists():
            self.txt_train_data.insert(0, str(paths.data_processed / "corpus" / "train.jsonl"))
        self.txt_train_data.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(r1, text="Browse", width=8, command=self._browse_train_data).pack(side=tk.LEFT)

        r2 = ttk.Frame(cfg_card, style="Card.TFrame")
        r2.pack(fill=tk.X, pady=3)
        ttk.Label(r2, text="Val Data:", width=14).pack(side=tk.LEFT)
        self.txt_val_data = ttk.Entry(r2)
        default_val_path = paths.data_processed / "corpus" / "val.txt"
        if default_val_path.exists():
            self.txt_val_data.insert(0, str(default_val_path))
        elif (paths.data_processed / "corpus" / "val.jsonl").exists():
            self.txt_val_data.insert(0, str(paths.data_processed / "corpus" / "val.jsonl"))
        self.txt_val_data.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(r2, text="Browse", width=8, command=self._browse_val_data).pack(side=tk.LEFT)

        # Epochs & Batch Size
        r3 = ttk.Frame(cfg_card, style="Card.TFrame")
        r3.pack(fill=tk.X, pady=3)
        ttk.Label(r3, text="Epochs:", width=14).pack(side=tk.LEFT)
        self.spn_epochs = ttk.Spinbox(r3, from_=1, to=100, width=6)
        self.spn_epochs.set(10)
        self.spn_epochs.pack(side=tk.LEFT, padx=4)

        ttk.Label(r3, text="Batch Size:", width=10).pack(side=tk.LEFT, padx=(8, 0))
        self.spn_batch_size = ttk.Spinbox(r3, from_=1, to=64, width=6)
        self.spn_batch_size.set(4)
        self.spn_batch_size.pack(side=tk.LEFT, padx=4)

        # Learning Rate & Device
        r4 = ttk.Frame(cfg_card, style="Card.TFrame")
        r4.pack(fill=tk.X, pady=3)
        ttk.Label(r4, text="Learning Rate:", width=14).pack(side=tk.LEFT)
        self.txt_lr = ttk.Entry(r4, width=10)
        self.txt_lr.insert(0, "0.0001")
        self.txt_lr.pack(side=tk.LEFT, padx=4)

        ttk.Label(r4, text="Device:", width=10).pack(side=tk.LEFT, padx=(8, 0))
        avail_devices = ["cpu"]
        if torch.cuda.is_available():
            avail_devices.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            avail_devices.append("mps")
        self.combo_device = ttk.Combobox(r4, values=avail_devices, state="readonly", width=8)
        self.combo_device.set("cpu")
        self.combo_device.pack(side=tk.LEFT, padx=4)

        # Experiment ID & Version
        r5 = ttk.Frame(cfg_card, style="Card.TFrame")
        r5.pack(fill=tk.X, pady=3)
        ttk.Label(r5, text="Experiment ID:", width=14).pack(side=tk.LEFT)
        self.txt_exp_id = ttk.Entry(r5, width=14)
        self.txt_exp_id.insert(0, "exp_phase1_english")
        self.txt_exp_id.pack(side=tk.LEFT, padx=4)

        ttk.Label(r5, text="Version:", width=8).pack(side=tk.LEFT, padx=(4, 0))
        self.txt_version = ttk.Entry(r5, width=8)
        self.txt_version.insert(0, "v0.1")
        self.txt_version.pack(side=tk.LEFT, padx=4)

        # Train Tokenizer Toggle
        r6 = ttk.Frame(cfg_card, style="Card.TFrame")
        r6.pack(fill=tk.X, pady=3)
        self.var_train_tok = tk.BooleanVar(value=True)
        self.chk_train_tok = ttk.Checkbutton(r6, text="Train New Tokenizer from Data", variable=self.var_train_tok)
        self.chk_train_tok.pack(side=tk.LEFT)

        # Advanced Accordion / Settings
        self.btn_adv_toggle = ttk.Button(cfg_card, text="▼ Advanced / Research Settings", command=self._toggle_advanced_train_settings)
        self.btn_adv_toggle.pack(anchor=tk.W, pady=(8, 2))

        self.adv_frame = ttk.Frame(cfg_card, style="Card.TFrame", padding=6)
        self.adv_visible = False

        adv_r1 = ttk.Frame(self.adv_frame, style="Card.TFrame")
        adv_r1.pack(fill=tk.X, pady=2)
        ttk.Label(adv_r1, text="Context Length:", width=16).pack(side=tk.LEFT)
        self.txt_context_len = ttk.Entry(adv_r1, width=8)
        self.txt_context_len.insert(0, "128")
        self.txt_context_len.pack(side=tk.LEFT)

        ttk.Label(adv_r1, text="Grad Accum:", width=12).pack(side=tk.LEFT, padx=(8, 0))
        self.txt_grad_accum = ttk.Entry(adv_r1, width=6)
        self.txt_grad_accum.insert(0, "1")
        self.txt_grad_accum.pack(side=tk.LEFT)

        adv_r2 = ttk.Frame(self.adv_frame, style="Card.TFrame")
        adv_r2.pack(fill=tk.X, pady=2)
        ttk.Label(adv_r2, text="Warmup Steps:", width=16).pack(side=tk.LEFT)
        self.txt_warmup = ttk.Entry(adv_r2, width=8)
        self.txt_warmup.insert(0, "100")
        self.txt_warmup.pack(side=tk.LEFT)

        ttk.Label(adv_r2, text="Weight Decay:", width=12).pack(side=tk.LEFT, padx=(8, 0))
        self.txt_weight_decay = ttk.Entry(adv_r2, width=6)
        self.txt_weight_decay.insert(0, "0.01")
        self.txt_weight_decay.pack(side=tk.LEFT)

        # Action Buttons
        act_frame = ttk.Frame(cfg_card, style="Card.TFrame")
        act_frame.pack(fill=tk.X, pady=(12, 0))

        self.btn_start_train = ttk.Button(act_frame, text="▶ Start Training", style="Primary.TButton", command=self._confirm_and_start_training)
        self.btn_start_train.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_stop_train = ttk.Button(act_frame, text="⏹ Stop Training", style="Stop.TButton", command=self._stop_training, state="disabled")
        self.btn_stop_train.pack(side=tk.LEFT)

        # --- RIGHT: Dashboard, Chart & Log ---
        dash_card = ttk.Frame(right_frame, style="Card.TFrame", padding=10)
        dash_card.pack(fill=tk.BOTH, expand=True)

        ttk.Label(dash_card, text="Live Training Dashboard", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 4))

        # Progress Bar
        self.train_progress = ttk.Progressbar(dash_card, mode="determinate")
        self.train_progress.pack(fill=tk.X, pady=4)

        # Live Metrics Grid Card
        m_grid = ttk.Frame(dash_card, style="Card.TFrame", padding=6)
        m_grid.pack(fill=tk.X, pady=4)

        self.lbl_train_status = ttk.Label(m_grid, text="Status: Ready to train", style="Status.TLabel")
        self.lbl_train_status.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.lbl_epoch_info = ttk.Label(m_grid, text="Epoch: 0/0  |  Batch: 0/0")
        self.lbl_epoch_info.grid(row=1, column=0, sticky=tk.W, pady=2)

        self.lbl_loss_info = ttk.Label(m_grid, text="Train Loss: --  |  Val Loss: --")
        self.lbl_loss_info.grid(row=1, column=1, sticky=tk.W, pady=2)

        self.lbl_ppl_info = ttk.Label(m_grid, text="Train PPL: --  |  Val PPL: --")
        self.lbl_ppl_info.grid(row=2, column=0, sticky=tk.W, pady=2)

        self.lbl_speed_info = ttk.Label(m_grid, text="Speed: -- tok/s  |  ETA: --")
        self.lbl_speed_info.grid(row=2, column=1, sticky=tk.W, pady=2)

        # Loss Graph
        ttk.Label(dash_card, text="Loss Curves", style="Title.TLabel").pack(anchor=tk.W, pady=(6, 2))
        self.loss_chart = LossChartCanvas(dash_card, height=180)
        self.loss_chart.pack(fill=tk.BOTH, expand=True, pady=4)

        # Training Log
        log_header = ttk.Frame(dash_card, style="Card.TFrame")
        log_header.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(log_header, text="Training Event Log", style="Title.TLabel").pack(side=tk.LEFT)

        ttk.Button(log_header, text="Copy Log", width=10, command=self._copy_train_log).pack(side=tk.RIGHT, padx=2)
        ttk.Button(log_header, text="Save Log...", width=10, command=self._save_train_log).pack(side=tk.RIGHT, padx=2)

        self.txt_train_log = scrolledtext.ScrolledText(
            dash_card, height=6, font=("Consolas", 9),
            bg=self.input_bg, fg=self.fg_color, insertbackground=self.fg_color,
            relief="flat", bd=3
        )
        self.txt_train_log.pack(fill=tk.BOTH, expand=True, pady=2)

    def _toggle_advanced_train_settings(self):
        if not HAS_TKINTER or self.root is None:
            return
        if self.adv_visible:
            self.adv_frame.pack_forget()
            self.btn_adv_toggle.config(text="▼ Advanced / Research Settings")
            self.adv_visible = False
        else:
            self.adv_frame.pack(fill=tk.X, pady=4)
            self.btn_adv_toggle.config(text="▲ Hide Advanced / Research Settings")
            self.adv_visible = True

    def _browse_train_data(self):
        if not HAS_TKINTER or self.root is None:
            return
        path = filedialog.askopenfilename(title="Select Training Data File", filetypes=[("Text/JSONL", "*.txt *.jsonl"), ("All Files", "*.*")])
        if path:
            self.txt_train_data.delete(0, tk.END)
            self.txt_train_data.insert(0, path)

    def _browse_val_data(self):
        if not HAS_TKINTER or self.root is None:
            return
        path = filedialog.askopenfilename(title="Select Validation Data File", filetypes=[("Text/JSONL", "*.txt *.jsonl"), ("All Files", "*.*")])
        if path:
            self.txt_val_data.delete(0, tk.END)
            self.txt_val_data.insert(0, path)

    def _confirm_and_start_training(self):
        if not HAS_TKINTER or self.root is None:
            return

        train_path = self.txt_train_data.get().strip()
        if not train_path or not Path(train_path).exists():
            messagebox.showerror("Invalid Dataset", f"Training dataset path does not exist:\n{train_path}")
            return

        val_path = self.txt_val_data.get().strip()
        if val_path and not Path(val_path).exists():
            messagebox.showerror("Invalid Validation Dataset", f"Validation dataset path does not exist:\n{val_path}")
            return

        preset = self.train_preset_combo.get()
        epochs = self.spn_epochs.get()
        batch_size = self.spn_batch_size.get()
        lr = self.txt_lr.get().strip()
        device = self.combo_device.get()
        exp_id = self.txt_exp_id.get().strip()
        version = self.txt_version.get().strip()

        # Confirmation Dialog
        msg = (
            f"You are about to train ScratchLM:\n\n"
            f"• Model Preset:   {preset.upper()}\n"
            f"• Experiment ID:  {exp_id}\n"
            f"• Version:        {version}\n"
            f"• Training Data:  {Path(train_path).name}\n"
            f"• Val Data:       {Path(val_path).name if val_path else 'None'}\n"
            f"• Epochs:         {epochs}\n"
            f"• Batch Size:     {batch_size}\n"
            f"• Learning Rate: {lr}\n"
            f"• Device:        {device}\n\n"
            f"Start ScratchLM Training Run?"
        )

        if messagebox.askyesno("Confirm Training Launch", msg):
            self._start_training_threaded(
                train_path=train_path,
                val_path=val_path if val_path else None,
                preset=preset,
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                device=device,
                exp_id=exp_id,
                version=version,
                train_tokenizer=self.var_train_tok.get(),
            )

    def _start_training_threaded(
        self,
        train_path: str,
        val_path: Optional[str],
        preset: str,
        epochs: int,
        batch_size: int,
        lr: float,
        device: str,
        exp_id: str,
        version: str,
        train_tokenizer: bool,
    ):
        self.is_training = True
        self.stop_requested = False

        self.btn_start_train.config(state="disabled")
        self.btn_stop_train.config(state="normal")
        self.lbl_train_status.config(text="Status: Initializing training environment...", style="Status.TLabel")

        self.txt_train_log.delete("1.0", tk.END)
        self.loss_chart.clear_chart()
        self.train_progress['value'] = 0

        def _cancel_check() -> bool:
            return self.stop_requested

        def _log_cb(msg: str):
            self.stream_queue.put(("TRAIN_LOG", msg))

        def _step_cb(stats: Dict[str, Any]):
            self.stream_queue.put(("TRAIN_STEP", stats))

        def _epoch_cb(metrics: Dict[str, Any]):
            self.stream_queue.put(("TRAIN_EPOCH", metrics))

        def _worker():
            try:
                _log_cb(f"Loading training data from {train_path}...")
                train_texts = load_data([train_path], min_length=10, max_length=10000, deduplicate=True)
                _log_cb(f"Loaded {len(train_texts):,} training text segments.")

                val_texts = None
                if val_path:
                    _log_cb(f"Loading validation data from {val_path}...")
                    val_texts = load_data([val_path], min_length=10, max_length=10000, deduplicate=False)
                    _log_cb(f"Loaded {len(val_texts):,} validation text segments.")

                m_cfg, t_cfg, tok_cfg = get_config_preset(preset)
                m_cfg.version = version
                t_cfg.model_version = version
                t_cfg.experiment_id = exp_id
                t_cfg.num_epochs = epochs
                t_cfg.batch_size = batch_size
                t_cfg.learning_rate = lr
                t_cfg.device = device

                trainer = Trainer(
                    model_config=m_cfg,
                    training_config=t_cfg,
                    tokenizer_config=tok_cfg,
                    model_version=version,
                    experiment_id=exp_id,
                    device=device,
                )

                _log_cb(f"Model initialized: Preset={preset} | Params={trainer.model_config.n_params:,}")

                res = trainer.train(
                    train_texts=train_texts,
                    val_texts=val_texts,
                    train_tokenizer=train_tokenizer,
                    step_callback=_step_cb,
                    epoch_callback=_epoch_cb,
                    cancel_callback=_cancel_check,
                    log_callback=_log_cb,
                )

                chk_path = str(paths.checkpoints / version / exp_id / f"{version}_final.pt")
                if not Path(chk_path).exists():
                    chk_path = str(paths.checkpoints / version / exp_id / "best.pt")

                self.stream_queue.put(("TRAIN_COMPLETE", {
                    "result": res,
                    "experiment_id": exp_id,
                    "version": version,
                    "preset": preset,
                    "epochs": epochs,
                    "checkpoint_path": chk_path,
                    "tokenizer_path": str(paths.tokenizers / f"{version}_bpe_{tok_cfg.vocab_size}"),
                }))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.stream_queue.put(("TRAIN_ERROR", str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _stop_training(self):
        if self.is_training:
            self.stop_requested = True
            self.lbl_train_status.config(text="Status: Requesting safe stop at next batch...", style="Muted.TLabel")

    def _copy_train_log(self):
        if not HAS_TKINTER or self.root is None:
            return
        txt = self.txt_train_log.get("1.0", tk.END).strip()
        if txt:
            self.root.clipboard_clear()
            self.root.clipboard_append(txt)
            messagebox.showinfo("Copied", "Training log copied to clipboard!")

    def _save_train_log(self):
        if not HAS_TKINTER or self.root is None:
            return
        txt = self.txt_train_log.get("1.0", tk.END).strip()
        if not txt:
            return
        file_path = filedialog.asksaveasfilename(title="Save Training Log", defaultextension=".txt", filetypes=[("Text File", "*.txt"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(txt)
            messagebox.showinfo("Saved", f"Log saved to {file_path}")

    # =========================================================================
    # TAB 3: EVALUATION SECTION
    # =========================================================================

    def _build_evaluate_tab(self):
        parent = self.tab_evaluate

        eval_card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        eval_card.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(eval_card, text="Model Evaluation Suite", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 6))

        r1 = ttk.Frame(eval_card, style="Card.TFrame")
        r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="Checkpoint:", width=12).pack(side=tk.LEFT)
        self.eval_chk_combo = ttk.Combobox(r1, font=("Segoe UI", 9))
        self.eval_chk_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(r1, text="Browse...", width=10, command=self._browse_eval_checkpoint).pack(side=tk.LEFT)

        r2 = ttk.Frame(eval_card, style="Card.TFrame")
        r2.pack(fill=tk.X, pady=2)
        ttk.Label(r2, text="Eval Data:", width=12).pack(side=tk.LEFT)
        self.eval_data_entry = ttk.Entry(r2)
        default_val_path = paths.data_processed / "corpus" / "val.txt"
        if default_val_path.exists():
            self.eval_data_entry.insert(0, str(default_val_path))
        self.eval_data_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(r2, text="Browse...", width=10, command=self._browse_eval_data).pack(side=tk.LEFT)

        act_r = ttk.Frame(eval_card, style="Card.TFrame")
        act_r.pack(fill=tk.X, pady=(6, 0))

        self.btn_run_eval = ttk.Button(act_r, text="▶ Run Full Model Evaluation", style="Primary.TButton", command=self._start_eval_threaded)
        self.btn_run_eval.pack(side=tk.LEFT)

        self.lbl_eval_status = ttk.Label(act_r, text="Ready", style="Status.TLabel")
        self.lbl_eval_status.pack(side=tk.LEFT, padx=12)

        out_card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        out_card.pack(fill=tk.BOTH, expand=True, pady=4)

        ttk.Label(out_card, text="Evaluation Results & Task Benchmark", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 4))

        self.txt_eval_out = scrolledtext.ScrolledText(
            out_card, font=("Consolas", 10),
            bg=self.input_bg, fg=self.fg_color, insertbackground=self.fg_color,
            relief="flat", bd=4
        )
        self.txt_eval_out.pack(fill=tk.BOTH, expand=True, pady=4)

    def _browse_eval_checkpoint(self):
        if not HAS_TKINTER or self.root is None:
            return
        path = filedialog.askopenfilename(title="Select Checkpoint for Evaluation", filetypes=[("PyTorch Checkpoint", "*.pt"), ("All Files", "*.*")])
        if path:
            self.eval_chk_combo.set(path)

    def _browse_eval_data(self):
        if not HAS_TKINTER or self.root is None:
            return
        path = filedialog.askopenfilename(title="Select Evaluation Dataset", filetypes=[("Text/JSONL", "*.txt *.jsonl"), ("All Files", "*.*")])
        if path:
            self.eval_data_entry.delete(0, tk.END)
            self.eval_data_entry.insert(0, path)

    def _start_eval_threaded(self):
        if not HAS_TKINTER or self.root is None:
            return

        chk_path = self.eval_chk_combo.get().strip()
        if not chk_path or not Path(chk_path).exists():
            messagebox.showerror("Missing Checkpoint", "Please select an existing ScratchLM checkpoint file.")
            return

        data_path = self.eval_data_entry.get().strip()

        self.btn_run_eval.config(state="disabled")
        self.lbl_eval_status.config(text="Running evaluation suite...", style="Status.TLabel")
        self.txt_eval_out.delete("1.0", tk.END)

        def _worker():
            try:
                chk = torch.load(chk_path, map_location="cpu")
                cfg_dict = chk.get("config", {})
                config = TransformerConfig.from_dict(cfg_dict)

                from scratchlm.model import ScratchLM
                model = ScratchLM(config)
                model.load_state_dict(chk.get("state_dict", chk.get("model_state_dict", {})))

                # Find tokenizer
                tok_dir = Path(chk_path).parent / "tokenizer"
                if not tok_dir.exists():
                    tok_dir = paths.tokenizers / f"{config.version}_bpe_{config.vocab_size}"

                from scratchlm.tokenizer import load_tokenizer
                tokenizer = load_tokenizer(tok_dir) if tok_dir.exists() else None

                suite = EvaluationSuite(model=model, tokenizer=tokenizer, device="cpu")

                report_lines = [
                    f"=======================================================================",
                    f"ScratchLM Evaluation Suite Benchmark",
                    f"=======================================================================",
                    f"Checkpoint:   {Path(chk_path).name}",
                    f"Model Preset: {config.name} ({config.version})",
                    f"Parameters:   {model.get_num_params():,} total parameters",
                    f"Vocab Size:   {config.vocab_size:,} subwords",
                    f"-----------------------------------------------------------------------"
                ]

                if data_path and Path(data_path).exists() and tokenizer:
                    texts = load_data([data_path], min_length=10, max_length=10000, deduplicate=False)
                    tokenized = [tokenizer.encode(t, add_special_tokens=False) for t in texts]

                    from scratchlm.data import TokenizedDataset
                    dataset = TokenizedDataset(tokenized_texts=tokenized, context_length=config.context_length, tokenizer=tokenizer)

                    metrics = suite.evaluate_dataset(dataset, split="eval", batch_size=4)
                    report_lines.append("\nDataset Evaluation Metrics:")
                    for k, v in metrics.items():
                        report_lines.append(f"  • {k:20s}: {v:.4f}")

                if tokenizer:
                    report_lines.append("\nTask Benchmarks:")
                    task_res, samples = suite.evaluate_tasks(num_samples=5)
                    for t_name, t_metrics in task_res.items():
                        report_lines.append(f"\n  Task: {t_name}")
                        for m_name, val in t_metrics.items():
                            report_lines.append(f"    - {m_name}: {val:.4f}")

                    if samples:
                        report_lines.append("\nGeneration Samples:")
                        for idx, s in enumerate(samples[:3], 1):
                            report_lines.append(f"  {idx}. Prompt: {s.get('prompt', '')}")
                            report_lines.append(f"     Output: {s.get('output', '')}\n")

                out_str = "\n".join(report_lines)
                self.stream_queue.put(("EVAL_COMPLETE", out_str))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.stream_queue.put(("EVAL_ERROR", str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    # =========================================================================
    # TAB 4: CHECKPOINTS SECTION
    # =========================================================================

    def _build_checkpoints_tab(self):
        parent = self.tab_checkpoints

        top_card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        top_card.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(top_card, text="Checkpoints Registry & Management", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Button(top_card, text="🔄 Refresh Checkpoints", command=self._refresh_checkpoints_list).pack(side=tk.RIGHT)

        tree_frame = ttk.Frame(parent, style="Card.TFrame", padding=6)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        columns = ("file", "exp_id", "version", "step", "epoch", "loss", "val_loss", "size", "timestamp")
        self.chk_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)

        self.chk_tree.heading("file", text="Checkpoint File")
        self.chk_tree.heading("exp_id", text="Experiment")
        self.chk_tree.heading("version", text="Version")
        self.chk_tree.heading("step", text="Step")
        self.chk_tree.heading("epoch", text="Epoch")
        self.chk_tree.heading("loss", text="Train Loss")
        self.chk_tree.heading("val_loss", text="Val Loss")
        self.chk_tree.heading("size", text="Size (MB)")
        self.chk_tree.heading("timestamp", text="Timestamp")

        self.chk_tree.column("file", width=220)
        self.chk_tree.column("exp_id", width=120)
        self.chk_tree.column("version", width=70)
        self.chk_tree.column("step", width=60)
        self.chk_tree.column("epoch", width=60)
        self.chk_tree.column("loss", width=80)
        self.chk_tree.column("val_loss", width=80)
        self.chk_tree.column("size", width=80)
        self.chk_tree.column("timestamp", width=140)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.chk_tree.yview)
        self.chk_tree.configure(yscroll=scrollbar.set)

        self.chk_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        act_card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        act_card.pack(fill=tk.X, pady=4)

        ttk.Button(act_card, text="🚀 Load in Generator", style="Primary.TButton", command=self._action_load_checkpoint_in_gen).pack(side=tk.LEFT, padx=4)
        ttk.Button(act_card, text="📊 Evaluate Checkpoint", command=self._action_eval_selected_checkpoint).pack(side=tk.LEFT, padx=4)
        ttk.Button(act_card, text="📁 Open Checkpoints Folder", command=self._open_checkpoints_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(act_card, text="🗑️ Delete Checkpoint", style="Stop.TButton", command=self._delete_selected_checkpoint).pack(side=tk.RIGHT, padx=4)

        self._refresh_checkpoints_list()

    def _refresh_checkpoints_list(self):
        if not HAS_TKINTER or self.root is None:
            return
        for item in self.chk_tree.get_children():
            self.chk_tree.delete(item)

        checkpoints = self.engine.find_local_checkpoints()
        for c in checkpoints:
            p = Path(c["path"])
            size_mb = f"{p.stat().st_size / (1024*1024):.1f}" if p.exists() else "--"

            meta = c.get("meta", {})
            self.chk_tree.insert("", tk.END, iid=c["path"], values=(
                p.name,
                c.get("experiment_id", "exp001"),
                c.get("version", "v0.1"),
                meta.get("step", "--"),
                meta.get("epoch", "--"),
                f"{meta.get('loss', 0.0):.4f}" if meta.get("loss") else "--",
                f"{meta.get('val_loss', 0.0):.4f}" if meta.get("val_loss") else "--",
                size_mb,
                meta.get("timestamp", "--"),
            ))

    def _action_load_checkpoint_in_gen(self):
        selected = self.chk_tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a checkpoint from the registry.")
            return

        chk_path = selected[0]
        self.chk_combobox.set(chk_path)

        toks = self.engine.find_local_tokenizers()
        if toks:
            self.tok_combobox.set(toks[0]["path"])

        self.notebook.select(self.tab_generate)
        self._load_model_threaded()

    def _action_eval_selected_checkpoint(self):
        selected = self.chk_tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a checkpoint from the registry.")
            return
        self.eval_chk_combo.set(selected[0])
        self.notebook.select(self.tab_evaluate)

    def _open_checkpoints_folder(self):
        chk_dir = str(paths.checkpoints)
        try:
            if sys.platform == "win32":
                os.startfile(chk_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", chk_dir])
            else:
                subprocess.Popen(["xdg-open", chk_dir])
        except Exception as e:
            messagebox.showinfo("Checkpoints Directory", f"Path: {chk_dir}\nError opening folder: {e}")

    def _delete_selected_checkpoint(self):
        selected = self.chk_tree.selection()
        if not selected:
            return
        chk_path = Path(selected[0])
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to permanently delete:\n{chk_path.name}?"):
            try:
                if chk_path.exists():
                    chk_path.unlink()
                json_path = chk_path.with_suffix(".json")
                if json_path.exists():
                    json_path.unlink()
                self._refresh_checkpoints_list()
                self._auto_discover_models()
            except Exception as e:
                messagebox.showerror("Delete Error", f"Could not delete checkpoint file:\n{e}")

    # =========================================================================
    # TAB 5: SETTINGS & SYSTEM INFO
    # =========================================================================

    def _build_settings_tab(self):
        parent = self.tab_settings

        sys_card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        sys_card.pack(fill=tk.BOTH, expand=True)

        ttk.Label(sys_card, text="System Hardware & PyTorch Diagnostics", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 8))

        info_lines = [
            f"PyTorch Version:  {torch.__version__}",
            f"CUDA Available:   {torch.cuda.is_available()}",
            f"CUDA Devices:     {torch.cuda.device_count() if torch.cuda.is_available() else 0}",
            f"MPS Available:    {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}",
            f"CPU Cores:        {os.cpu_count() or 'Unknown'}",
            f"Checkpoints Dir:  {paths.checkpoints}",
            f"Tokenizers Dir:   {paths.tokenizers}",
            f"Processed Data:   {paths.data_processed}",
        ]

        if torch.cuda.is_available():
            info_lines.append(f"GPU Device Name:  {torch.cuda.get_device_name(0)}")

        for line in info_lines:
            ttk.Label(sys_card, text=line, font=("Consolas", 10)).pack(anchor=tk.W, pady=3)

    # =========================================================================
    # HELPER & QUEUE POLLER METHODS
    # =========================================================================

    def _on_select_example(self, event=None):
        if not HAS_TKINTER or self.root is None:
            return
        selected = self.example_combobox.get()
        if selected and not selected.startswith("Select"):
            self.txt_prompt.delete("1.0", tk.END)
            self.txt_prompt.insert(tk.END, selected)

    def _browse_checkpoint(self):
        if not HAS_TKINTER or self.root is None:
            return
        initial_dir = str(paths.checkpoints) if paths.checkpoints.exists() else str(Path.cwd())
        file_path = filedialog.askopenfilename(title="Select ScratchLM Checkpoint", initialdir=initial_dir, filetypes=[("PyTorch Checkpoint", "*.pt"), ("All Files", "*.*")])
        if file_path:
            self.chk_combobox.set(file_path)

    def _browse_tokenizer(self):
        if not HAS_TKINTER or self.root is None:
            return
        initial_dir = str(paths.tokenizers) if paths.tokenizers.exists() else str(Path.cwd())
        dir_path = filedialog.askdirectory(title="Select Tokenizer Directory", initialdir=initial_dir)
        if dir_path:
            self.tok_combobox.set(dir_path)

    def _auto_discover_models(self):
        if not HAS_TKINTER or self.root is None:
            return
        checkpoints = self.engine.find_local_checkpoints()
        chk_paths = [c["path"] for c in checkpoints]
        self.chk_combobox['values'] = chk_paths
        self.eval_chk_combo['values'] = chk_paths

        tokenizers = self.engine.find_local_tokenizers()
        tok_paths = [t["path"] for t in tokenizers]
        self.tok_combobox['values'] = tok_paths

        if chk_paths and not self.chk_combobox.get():
            self.chk_combobox.set(chk_paths[0])
            self.eval_chk_combo.set(chk_paths[0])

        if tok_paths and not self.tok_combobox.get():
            self.tok_combobox.set(tok_paths[0])

    def _load_model_threaded(self):
        if not HAS_TKINTER or self.root is None:
            return
        chk_path = self.chk_combobox.get().strip()
        tok_path = self.tok_combobox.get().strip() or None

        if not chk_path:
            messagebox.showwarning("Missing Checkpoint", "Please select or browse for a ScratchLM checkpoint file (.pt).")
            return

        self.btn_load_model.config(state="disabled")
        self.lbl_model_summary.config(text="Loading model into memory...")

        def _worker():
            try:
                info = self.engine.load_model_and_tokenizer(chk_path, tok_path)
                self.stream_queue.put(("LOAD_SUCCESS", info))
            except Exception as e:
                self.stream_queue.put(("LOAD_ERROR", str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _start_generation_threaded(self):
        if not HAS_TKINTER or self.root is None:
            return

        if self.engine.model is None:
            chk_path = self.chk_combobox.get().strip()
            if chk_path:
                try:
                    self.engine.load_model_and_tokenizer(chk_path, self.tok_combobox.get().strip() or None)
                except Exception as e:
                    messagebox.showerror("Model Load Error", f"Failed to load checkpoint before generating:\n{e}")
                    return
            else:
                messagebox.showwarning("No Model Loaded", "Please select and load a ScratchLM checkpoint first.")
                return

        prompt = self.txt_prompt.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Empty Prompt", "Please enter a prompt to generate continuation.")
            return

        try:
            max_len = int(self.spn_max_tokens.get())
            temp = float(self.spn_temp.get())
            top_k = int(self.spn_top_k.get()) if int(self.spn_top_k.get()) > 0 else None
            top_p = float(self.spn_top_p.get()) if float(self.spn_top_p.get()) > 0 else None
            greedy = self.var_greedy.get()
        except ValueError as e:
            messagebox.showerror("Invalid Parameter", f"Please check numerical generation controls:\n{e}")
            return

        self.is_generating = True
        self.stop_requested = False
        self.btn_generate.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.lbl_gen_stats.config(text="Generating...")

        self.txt_output.delete("1.0", tk.END)
        self.txt_output.insert(tk.END, prompt, "prompt")
        self.txt_output.insert(tk.END, " ", "continuation")
        self.txt_output.see(tk.END)

        def _cancel_check() -> bool:
            return self.stop_requested

        def _token_cb(token_str: str):
            self.stream_queue.put(("TOKEN", token_str))

        def _worker():
            try:
                res = self.engine.generate(
                    prompt=prompt,
                    max_length=max_len,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    greedy=greedy,
                    token_callback=_token_cb,
                    cancel_callback=_cancel_check,
                )
                self.stream_queue.put(("GEN_COMPLETE", res))
            except Exception as e:
                self.stream_queue.put(("GEN_ERROR", str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _stop_generation(self):
        if self.is_generating:
            self.stop_requested = True
            self.lbl_gen_stats.config(text="Stopping generation...")

    def _start_queue_poller(self):
        if not HAS_TKINTER or self.root is None:
            return

        def _poll():
            try:
                while True:
                    msg_type, payload = self.stream_queue.get_nowait()

                    if msg_type == "TOKEN":
                        self.txt_output.insert(tk.END, payload, "continuation")
                        self.txt_output.see(tk.END)

                    elif msg_type == "LOAD_SUCCESS":
                        info: ModelInfo = payload
                        self.btn_load_model.config(state="normal")
                        summary = f"✓ Loaded: {info.version} ({info.preset_name}) | {info.num_params:,} params | Vocab: {info.vocab_size:,} | Device: {info.device}"
                        self.lbl_model_summary.config(text=summary, style="Status.TLabel")
                        self._save_app_config()

                    elif msg_type == "LOAD_ERROR":
                        self.btn_load_model.config(state="normal")
                        self.lbl_model_summary.config(text="⚠️ Failed to load model", style="Muted.TLabel")
                        messagebox.showerror("Checkpoint Load Error", f"Could not load ScratchLM checkpoint:\n{payload}")

                    elif msg_type == "GEN_COMPLETE":
                        res: GenerationResult = payload
                        self.is_generating = False
                        self.btn_generate.config(state="normal")
                        self.btn_stop.config(state="disabled")

                        status_text = f"Done: {res.tokens_generated} tokens in {res.elapsed_seconds}s ({res.tokens_per_second} tok/s)"
                        if res.stopped_early:
                            status_text += " [Stopped]"
                        self.lbl_gen_stats.config(text=status_text)

                    elif msg_type == "GEN_ERROR":
                        self.is_generating = False
                        self.btn_generate.config(state="normal")
                        self.btn_stop.config(state="disabled")
                        self.lbl_gen_stats.config(text="Generation Error")
                        messagebox.showerror("Generation Failure", f"An error occurred during text generation:\n{payload}")

                    elif msg_type == "TRAIN_LOG":
                        self.txt_train_log.insert(tk.END, payload + "\n")
                        self.txt_train_log.see(tk.END)

                    elif msg_type == "TRAIN_STEP":
                        stats = payload
                        ep = stats.get("epoch", 1)
                        total_ep = stats.get("total_epochs", 1)
                        batch = stats.get("batch_idx", 0)
                        tot_batches = stats.get("total_batches", 1)

                        total_steps_est = total_ep * tot_batches
                        current_global_step = (ep - 1) * tot_batches + batch
                        pct = (current_global_step / max(1, total_steps_est)) * 100.0
                        self.train_progress['value'] = pct

                        self.lbl_train_status.config(text=f"Status: Training Epoch {ep}/{total_ep} ({pct:.1f}%)", style="Status.TLabel")
                        self.lbl_epoch_info.config(text=f"Epoch: {ep}/{total_ep}  |  Batch: {batch}/{tot_batches}")
                        self.lbl_loss_info.config(text=f"Train Loss: {stats.get('avg_loss', 0.0):.4f}")

                        t_ppl = float(torch.exp(torch.tensor(stats.get('avg_loss', 0.0))).item()) if stats.get('avg_loss') else 0.0
                        self.lbl_ppl_info.config(text=f"Train PPL: {t_ppl:.2f}")

                        tok_s = stats.get("tokens_per_sec", 0.0)
                        el = stats.get("elapsed_seconds", 0.0)
                        self.lbl_speed_info.config(text=f"Speed: {tok_s:.1f} tok/s  |  Elapsed: {el:.1f}s")

                        self.loss_chart.add_point(step=current_global_step, train_loss=stats.get("batch_loss"))

                    elif msg_type == "TRAIN_EPOCH":
                        metrics = payload
                        ep = metrics.get("epoch", 1)
                        t_loss = metrics.get("train_loss", 0.0)
                        v_loss = metrics.get("val_loss")

                        v_str = f"{v_loss:.4f}" if v_loss is not None else "--"
                        self.lbl_loss_info.config(text=f"Train Loss: {t_loss:.4f}  |  Val Loss: {v_str}")

                        v_ppl = float(torch.exp(torch.tensor(v_loss)).item()) if v_loss else 0.0
                        v_ppl_str = f"{v_ppl:.2f}" if v_loss else "--"
                        t_ppl = float(torch.exp(torch.tensor(t_loss)).item()) if t_loss else 0.0
                        self.lbl_ppl_info.config(text=f"Train PPL: {t_ppl:.2f}  |  Val PPL: {v_ppl_str}")

                        if v_loss is not None:
                            self.loss_chart.add_point(step=ep * 100, val_loss=v_loss)

                    elif msg_type == "TRAIN_COMPLETE":
                        self.is_training = False
                        self.btn_start_train.config(state="normal")
                        self.btn_stop_train.config(state="disabled")
                        self.train_progress['value'] = 100

                        info = payload
                        chk_path = info.get("checkpoint_path", "")
                        tok_path = info.get("tokenizer_path", "")

                        self.last_trained_checkpoint = chk_path
                        self.last_trained_tokenizer = tok_path

                        self.lbl_train_status.config(text="Status: Training Complete!", style="Status.TLabel")
                        self._auto_discover_models()
                        self._refresh_checkpoints_list()

                        # Completion Modal
                        summary_msg = (
                            f"Training Complete!\n\n"
                            f"Experiment: {info.get('experiment_id')}\n"
                            f"Model Preset: {info.get('preset')}\n"
                            f"Epochs Completed: {info.get('epochs')}\n"
                            f"Checkpoint Saved:\n{chk_path}\n\n"
                            f"Would you like to load this checkpoint into the Generator now?"
                        )
                        if messagebox.askyesno("Training Finished", summary_msg):
                            self.chk_combobox.set(chk_path)
                            if tok_path and Path(tok_path).exists():
                                self.tok_combobox.set(tok_path)
                            self.notebook.select(self.tab_generate)
                            self._load_model_threaded()

                    elif msg_type == "TRAIN_ERROR":
                        self.is_training = False
                        self.btn_start_train.config(state="normal")
                        self.btn_stop_train.config(state="disabled")
                        self.lbl_train_status.config(text="Status: Training Failed", style="Muted.TLabel")
                        messagebox.showerror("Training Error", f"Training run failed:\n{payload}")

                    elif msg_type == "EVAL_COMPLETE":
                        self.btn_run_eval.config(state="normal")
                        self.lbl_eval_status.config(text="Evaluation Complete", style="Status.TLabel")
                        self.txt_eval_out.insert(tk.END, payload)

                    elif msg_type == "EVAL_ERROR":
                        self.btn_run_eval.config(state="normal")
                        self.lbl_eval_status.config(text="Evaluation Failed", style="Muted.TLabel")
                        messagebox.showerror("Evaluation Error", f"Evaluation failed:\n{payload}")

            except queue.Empty:
                pass

            self.root.after(50, _poll)

        self.root.after(50, _poll)

    def _copy_output(self):
        if not HAS_TKINTER or self.root is None:
            return
        full_txt = self.txt_output.get("1.0", tk.END).strip()
        if full_txt:
            self.root.clipboard_clear()
            self.root.clipboard_append(full_txt)
            self.lbl_gen_stats.config(text="Copied to clipboard!")

    def _clear_output(self):
        if not HAS_TKINTER or self.root is None:
            return
        self.txt_output.delete("1.0", tk.END)

    def _load_app_config(self):
        if not HAS_TKINTER or self.root is None or not self.config_path.exists():
            return
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                if cfg.get("last_checkpoint"):
                    self.chk_combobox.set(cfg["last_checkpoint"])
                if cfg.get("last_tokenizer"):
                    self.tok_combobox.set(cfg["last_tokenizer"])
        except Exception:
            pass

    def _save_app_config(self):
        if not HAS_TKINTER or self.root is None:
            return
        try:
            cfg = {
                "last_checkpoint": self.chk_combobox.get(),
                "last_tokenizer": self.tok_combobox.get(),
            }
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    def run(self):
        if not HAS_TKINTER or self.root is None:
            print("Cannot launch GUI loop: Tkinter is unavailable in this environment.")
            return
        self.root.mainloop()


def main():
    """Main GUI launcher entry point."""
    if not HAS_TKINTER:
        print("Error: Tkinter is required for the ScratchLM Desktop Application GUI.")
        sys.exit(1)
    app = ScratchLMApp()
    app.run()


if __name__ == "__main__":
    main()
