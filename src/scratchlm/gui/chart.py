"""
Native Tkinter Loss Chart Component for ScratchLM Desktop App.

Renders real-time training and validation loss curves on a Tkinter Canvas
with auto-scaling, gridlines, legends, and smooth dynamic updates.
"""

from typing import Optional, List, Tuple, Any

try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    tk = object  # Fallback dummy class
    ttk = None


class LossChartCanvas(tk.Canvas if HAS_TKINTER else object):
    """
    Custom Tkinter Canvas component that plots loss curves dynamically.
    """

    def __init__(
        self,
        master: Optional[Any] = None,
        width: int = 500,
        height: int = 220,
        bg: str = "#181825",
        fg: str = "#D9E0EE",
        grid_color: str = "#313244",
        train_color: str = "#89B4FA",  # Catppuccin Blue
        val_color: str = "#A6E3A1",    # Catppuccin Green
        **kwargs
    ):
        if not HAS_TKINTER:
            return

        super().__init__(master, width=width, height=height, bg=bg, highlightthickness=0, **kwargs)
        self.chart_bg = bg
        self.fg_color = fg
        self.grid_color = grid_color
        self.train_color = train_color
        self.val_color = val_color

        self.train_history: List[Tuple[float, float]] = []  # (x_step, loss)
        self.val_history: List[Tuple[float, float]] = []    # (x_step, loss)

        self.padding_left = 45
        self.padding_right = 15
        self.padding_top = 25
        self.padding_bottom = 30

        self.bind("<Configure>", lambda e: self.redraw())

    def clear_chart(self):
        """Reset loss histories and clear canvas."""
        if not HAS_TKINTER:
            return
        self.train_history.clear()
        self.val_history.clear()
        self.redraw()

    def add_point(self, step: float, train_loss: Optional[float] = None, val_loss: Optional[float] = None):
        """Add a step point to train or val history and trigger redraw."""
        if not HAS_TKINTER:
            return
        if train_loss is not None:
            self.train_history.append((float(step), float(train_loss)))
        if val_loss is not None:
            self.val_history.append((float(step), float(val_loss)))
        self.redraw()

    def set_data(self, train_history: List[Tuple[float, float]], val_history: Optional[List[Tuple[float, float]]] = None):
        """Set full dataset and redraw."""
        if not HAS_TKINTER:
            return
        self.train_history = list(train_history)
        self.val_history = list(val_history) if val_history else []
        self.redraw()

    def redraw(self):
        """Redraw axes, grid, legends, and loss lines."""
        if not HAS_TKINTER:
            return

        self.delete("all")

        w = self.winfo_width()
        h = self.winfo_height()

        if w <= 10 or h <= 10:
            return

        pl, pr = self.padding_left, self.padding_right
        pt, pb = self.padding_top, self.padding_bottom

        plot_w = w - pl - pr
        plot_h = h - pt - pb

        # Draw empty chart state if no data
        if not self.train_history and not self.val_history:
            self.create_text(
                w / 2, h / 2,
                text="Loss chart will update when training begins...",
                fill="#A6ADC8", font=("Segoe UI", 9, "italic")
            )
            return

        # Determine bounds
        all_x = [pt[0] for pt in self.train_history + self.val_history]
        all_y = [pt[1] for pt in self.train_history + self.val_history]

        min_x = min(all_x) if all_x else 0
        max_x = max(all_x) if all_x else 1
        if min_x == max_x:
            max_x = min_x + 1

        min_y = max(0.0, min(all_y) * 0.9) if all_y else 0.0
        max_y = max(all_y) * 1.1 if all_y else 10.0
        if min_y == max_y:
            max_y = min_y + 1.0

        # Draw Grid & Y-Axis Labels
        num_y_grid = 4
        for i in range(num_y_grid + 1):
            y_val = min_y + (max_y - min_y) * (i / num_y_grid)
            y_canvas = pt + plot_h - (i / num_y_grid) * plot_h

            self.create_line(pl, y_canvas, w - pr, y_canvas, fill=self.grid_color, dash=(2, 4))
            self.create_text(pl - 6, y_canvas, text=f"{y_val:.2f}", fill="#A6ADC8", font=("Segoe UI", 8), anchor="e")

        # Draw X-Axis Title & Bounding Axes
        self.create_line(pl, pt, pl, h - pb, fill=self.grid_color)
        self.create_line(pl, h - pb, w - pr, h - pb, fill=self.grid_color)

        self.create_text(pl + plot_w / 2, h - 8, text="Training Epochs / Steps", fill="#A6ADC8", font=("Segoe UI", 8))

        # Helper to convert (x, y) data to canvas (cx, cy)
        def to_canvas(x, y):
            cx = pl + ((x - min_x) / (max_x - min_x)) * plot_w
            cy = pt + plot_h - ((y - min_y) / (max_y - min_y)) * plot_h
            return cx, cy

        # Draw Training Loss Line
        if len(self.train_history) >= 2:
            coords = []
            for x, y in self.train_history:
                cx, cy = to_canvas(x, y)
                coords.extend([cx, cy])
            self.create_line(coords, fill=self.train_color, width=2, smooth=True)
            for x, y in self.train_history:
                cx, cy = to_canvas(x, y)
                self.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=self.train_color, outline="#181825")
        elif len(self.train_history) == 1:
            cx, cy = to_canvas(*self.train_history[0])
            self.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=self.train_color)

        # Draw Validation Loss Line
        if len(self.val_history) >= 2:
            coords = []
            for x, y in self.val_history:
                cx, cy = to_canvas(x, y)
                coords.extend([cx, cy])
            self.create_line(coords, fill=self.val_color, width=2, smooth=True)
            for x, y in self.val_history:
                cx, cy = to_canvas(x, y)
                self.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=self.val_color, outline="#181825")
        elif len(self.val_history) == 1:
            cx, cy = to_canvas(*self.val_history[0])
            self.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=self.val_color)

        # Legend
        leg_x = w - pr - 120
        leg_y = pt - 14

        self.create_line(leg_x, leg_y, leg_x + 12, leg_y, fill=self.train_color, width=2)
        self.create_text(leg_x + 16, leg_y, text="Train Loss", fill=self.fg_color, font=("Segoe UI", 8, "bold"), anchor="w")

        if self.val_history:
            self.create_line(leg_x + 65, leg_y, leg_x + 77, leg_y, fill=self.val_color, width=2)
            self.create_text(leg_x + 81, leg_y, text="Val Loss", fill=self.fg_color, font=("Segoe UI", 8, "bold"), anchor="w")
