"""
ScratchLM Desktop Application GUI Interface.

Built with Python Tkinter/ttk using dark mode styling and thread-safe streaming.
Provides a native graphical runner for locally trained ScratchLM Tiny models.
"""

import sys
import os
import json
import time
import queue
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    tk = None
    ttk = None

from scratchlm.inference import InferenceEngine, GenerationResult, ModelInfo
from scratchlm.utils.paths import paths


class ScratchLMApp:
    """
    Main Tkinter Desktop Application for ScratchLM Local Inference.
    """

    def __init__(self, root: Optional[Any] = None):
        # Shared Inference Engine
        self.engine = InferenceEngine(device="cpu")
        self.is_generating = False
        self.stop_requested = False
        self.stream_queue: queue.Queue = queue.Queue()

        if not HAS_TKINTER:
            print("Notice: Tkinter module not found in environment (Windows Python installations include Tkinter by default).")
            self.root = None
            return

        # Custom Dark Palette Styling
        self.bg_color = "#1E1E2E"
        self.card_bg = "#2B2B3D"
        self.fg_color = "#D9E0EE"
        self.accent_color = "#89B4FA"
        self.stop_color = "#F38BA8"
        self.input_bg = "#181825"
        self.prompt_fg = "#A6E3A1"

        self._configure_styles()
        self._build_ui()
        self._load_app_config()
        self._auto_discover_models()
        self._start_queue_poller()

    def _configure_styles(self):
        if not HAS_TKINTER:
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

    def _build_ui(self):
        if not HAS_TKINTER:
            return

        main_container = ttk.Frame(self.root, padding=12)
        main_container.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(header_frame, text="ScratchLM", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(header_frame, text="  |  Local Model Runner (Phase 1 English)", style="Muted.TLabel").pack(side=tk.LEFT, pady=(3, 0))

        model_card = ttk.Frame(main_container, style="Card.TFrame", padding=10)
        model_card.pack(fill=tk.X, pady=6)

        m_header = ttk.Frame(model_card, style="Card.TFrame")
        m_header.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(m_header, text="Model Checkpoint & Tokenizer", style="Title.TLabel").pack(side=tk.LEFT)

        chk_row = ttk.Frame(model_card, style="Card.TFrame")
        chk_row.pack(fill=tk.X, pady=2)
        ttk.Label(chk_row, text="Checkpoint:", width=12).pack(side=tk.LEFT)

        self.chk_combobox = ttk.Combobox(chk_row, font=("Segoe UI", 9))
        self.chk_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, px=4)
        ttk.Button(chk_row, text="Browse...", width=10, command=self._browse_checkpoint).pack(side=tk.LEFT)

        tok_row = ttk.Frame(model_card, style="Card.TFrame")
        tok_row.pack(fill=tk.X, pady=2)
        ttk.Label(tok_row, text="Tokenizer:", width=12).pack(side=tk.LEFT)

        self.tok_combobox = ttk.Combobox(tok_row, font=("Segoe UI", 9))
        self.tok_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, px=4)
        ttk.Button(tok_row, text="Browse...", width=10, command=self._browse_tokenizer).pack(side=tk.LEFT)

        action_row = ttk.Frame(model_card, style="Card.TFrame")
        action_row.pack(fill=tk.X, pady=(6, 0))

        self.btn_load_model = ttk.Button(action_row, text="Load Selected Checkpoint", command=self._load_model_threaded)
        self.btn_load_model.pack(side=tk.LEFT)

        self.lbl_model_summary = ttk.Label(action_row, text="No model loaded", style="Status.TLabel")
        self.lbl_model_summary.pack(side=tk.LEFT, padx=12)

        prompt_card = ttk.Frame(main_container, style="Card.TFrame", padding=10)
        prompt_card.pack(fill=tk.BOTH, expand=False, pady=6)

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
            prompt_card, height=4, font=("Consolas", 10),
            bg=self.input_bg, fg=self.fg_color, insertbackground=self.fg_color,
            relief="flat", bd=4
        )
        self.txt_prompt.pack(fill=tk.BOTH, expand=True, pady=4)
        self.txt_prompt.insert(tk.END, "The Earth revolves around")

        ctrl_card = ttk.Frame(main_container, style="Card.TFrame", padding=8)
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

        output_card = ttk.Frame(main_container, style="Card.TFrame", padding=10)
        output_card.pack(fill=tk.BOTH, expand=True, pady=6)

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

        self.debug_frame = ttk.Frame(main_container, style="Card.TFrame", padding=6)
        self.debug_frame.pack(fill=tk.X, pady=(4, 0))

        self.btn_toggle_debug = ttk.Button(self.debug_frame, text="▼ Show Model Technical Info", command=self._toggle_debug_info)
        self.btn_toggle_debug.pack(anchor=tk.W)

        self.debug_details_lbl = ttk.Label(self.debug_frame, text="", style="Muted.TLabel", justify=tk.LEFT)
        self.debug_visible = False

    def _on_select_example(self, event=None):
        if not HAS_TKINTER:
            return
        selected = self.example_combobox.get()
        if selected and not selected.startswith("Select"):
            self.txt_prompt.delete("1.0", tk.END)
            self.txt_prompt.insert(tk.END, selected)

    def _browse_checkpoint(self):
        if not HAS_TKINTER:
            return
        initial_dir = str(paths.checkpoints) if paths.checkpoints.exists() else str(Path.cwd())
        file_path = filedialog.askopenfilename(
            title="Select ScratchLM Checkpoint",
            initialdir=initial_dir,
            filetypes=[("PyTorch Checkpoint", "*.pt"), ("All Files", "*.*")]
        )
        if file_path:
            self.chk_combobox.set(file_path)

    def _browse_tokenizer(self):
        if not HAS_TKINTER:
            return
        initial_dir = str(paths.tokenizers) if paths.tokenizers.exists() else str(Path.cwd())
        dir_path = filedialog.askdirectory(
            title="Select Tokenizer Directory",
            initialdir=initial_dir,
        )
        if dir_path:
            self.tok_combobox.set(dir_path)

    def _auto_discover_models(self):
        if not HAS_TKINTER:
            return
        checkpoints = self.engine.find_local_checkpoints()
        chk_paths = [c["path"] for c in checkpoints]
        self.chk_combobox['values'] = chk_paths

        tokenizers = self.engine.find_local_tokenizers()
        tok_paths = [t["path"] for t in tokenizers]
        self.tok_combobox['values'] = tok_paths

        if chk_paths and not self.chk_combobox.get():
            self.chk_combobox.set(chk_paths[0])

        if tok_paths and not self.tok_combobox.get():
            self.tok_combobox.set(tok_paths[0])

    def _load_model_threaded(self):
        if not HAS_TKINTER:
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
        if not HAS_TKINTER:
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
        if not HAS_TKINTER:
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
                        self._update_debug_text(info)
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

            except queue.Empty:
                pass

            self.root.after(50, _poll)

        self.root.after(50, _poll)

    def _update_debug_text(self, info: ModelInfo):
        if not HAS_TKINTER:
            return
        details = (
            f"Checkpoint: {info.checkpoint_path}\n"
            f"Tokenizer:  {info.tokenizer_path}\n"
            f"Preset:     {info.preset_name} | Version: {info.version} | Exp ID: {info.experiment_id}\n"
            f"Params:     Total = {info.num_params:,} | Trainable = {info.num_trainable_params:,}\n"
            f"Arch:       Layers = {info.n_layers} | d_model = {info.d_model} | Heads = {info.n_heads} | Context = {info.context_length}\n"
            f"Device:     {info.device}"
        )
        self.debug_details_lbl.config(text=details)

    def _toggle_debug_info(self):
        if not HAS_TKINTER:
            return
        if self.debug_visible:
            self.debug_details_lbl.pack_forget()
            self.btn_toggle_debug.config(text="▼ Show Model Technical Info")
            self.debug_visible = False
        else:
            self.debug_details_lbl.pack(anchor=tk.W, pady=(4, 0))
            self.btn_toggle_debug.config(text="▲ Hide Model Technical Info")
            self.debug_visible = True

    def _copy_output(self):
        if not HAS_TKINTER:
            return
        full_txt = self.txt_output.get("1.0", tk.END).strip()
        if full_txt:
            self.root.clipboard_clear()
            self.root.clipboard_append(full_txt)
            self.lbl_gen_stats.config(text="Copied to clipboard!")

    def _clear_output(self):
        if not HAS_TKINTER:
            return
        self.txt_output.delete("1.0", tk.END)

    def _load_app_config(self):
        if not HAS_TKINTER or not self.config_path.exists():
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
        if not HAS_TKINTER:
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
