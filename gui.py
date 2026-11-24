import os
import shutil
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
from types import SimpleNamespace

from PIL import Image, ImageTk

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # تا مدل پیدا شود
import Numberr  # noqa: E402

# ---- رنگ‌ها و فونت ----
BG = "#f3f5fa"
CARD = "#ffffff"
PRIMARY = "#2563eb"
PRIMARY_HOVER = "#1d4ed8"
TEXT = "#111827"
MUTED = "#6b7280"
BORDER = "#e2e6ef"
FONT = "Tahoma"

# سطح دقت: عدد = حداقل اطمینان برای قبول کردن یک رقم
PRESETS = {"سخت‌گیرانه": 0.75, "متوسط": 0.5, "آسان‌گیر": 0.3}


class FlatButton(tk.Label):
    """دکمهٔ تخت و تمیز با حالت hover."""

    def __init__(self, master, text, command, primary=False):
        self.primary = primary
        self.cmd = command
        self.enabled = True
        self.normal_bg = PRIMARY if primary else CARD
        self.hover_bg = PRIMARY_HOVER if primary else "#eef2ff"
        self.fg = "white" if primary else TEXT
        super().__init__(master, text=text, font=(FONT, 11, "bold" if primary else "normal"),
                         bg=self.normal_bg, fg=self.fg, padx=18, pady=9, cursor="hand2",
                         highlightthickness=0 if primary else 1, highlightbackground=BORDER)
        self.bind("<Enter>", lambda e: self.enabled and self.config(bg=self.hover_bg))
        self.bind("<Leave>", lambda e: self.config(bg=self.normal_bg))
        self.bind("<Button-1>", lambda e: self.enabled and self.cmd())

    def set_enabled(self, value):
        self.enabled = value
        self.config(fg=self.fg if value else "#b6bcc8", cursor="hand2" if value else "arrow")


class App:
    def __init__(self, root):
        self.root = root
        root.title("تشخیص اعداد فارسی و انگلیسی")
        root.geometry("980x760")
        root.minsize(860, 660)
        root.configure(bg=BG)

        self.last_path = None
        self.photo = None
        self.pil_img = None
        self.result_text = ""
        self.out_path = os.path.join(tempfile.gettempdir(), "digit_reader_out.png")
        self.preset = tk.StringVar(value="متوسط")
        self.mixed = tk.BooleanVar(value=False)

        self.build_header()
        self.build_toolbar()
        self.build_result()
        self.build_image_area()
        self.build_legend()
        self.draw_placeholder()

    # ---------------- ساخت صفحه ----------------
    def build_header(self):
        h = tk.Frame(self.root, bg=PRIMARY)
        h.pack(fill="x")
        tk.Label(h, text="تشخیص اعداد فارسی و انگلیسی", font=(FONT, 18, "bold"),
                 bg=PRIMARY, fg="white").pack(anchor="e", padx=24, pady=(14, 0))
        tk.Label(h, text="یک عکس انتخاب کنید؛ برنامه اعداد داخلش را می‌خواند (چاپی یا دست‌نویس)",
                 font=(FONT, 10), bg=PRIMARY, fg="#dbe6ff").pack(anchor="e", padx=24, pady=(2, 14))

    def build_toolbar(self):
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=20, pady=(16, 6))
        # راست‌چین: اولین دکمه سمت راست
        self.btn_open = FlatButton(bar, "انتخاب عکس", self.choose, primary=True)
        self.btn_open.pack(side="right", padx=(8, 0))
        self.btn_copy = FlatButton(bar, "کپی عدد", self.copy_result)
        self.btn_copy.pack(side="right", padx=(8, 0))
        self.btn_save = FlatButton(bar, "ذخیرهٔ عکس نتیجه", self.save_image)
        self.btn_save.pack(side="right", padx=(8, 0))
        self.btn_copy.set_enabled(False)
        self.btn_save.set_enabled(False)

        opt = tk.Frame(self.root, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        opt.pack(fill="x", padx=20, pady=(6, 6))
        tk.Label(opt, text="دقت تشخیص:", font=(FONT, 10, "bold"), bg=CARD, fg=TEXT
                 ).pack(side="right", padx=(12, 6), pady=10)
        for name in PRESETS:
            tk.Radiobutton(opt, text=name, value=name, variable=self.preset, font=(FONT, 10),
                           bg=CARD, fg=TEXT, activebackground=CARD, selectcolor=CARD,
                           command=self.rerun).pack(side="right", padx=4)
        tk.Checkbutton(opt, text="ارقام فارسی و انگلیسی در یک خط قاطی‌اند", variable=self.mixed,
                       font=(FONT, 10), bg=CARD, fg=TEXT, activebackground=CARD,
                       selectcolor=CARD, command=self.rerun).pack(side="left", padx=12)

    def build_result(self):
        card = tk.Frame(self.root, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill="x", padx=20, pady=6)
        tk.Label(card, text="عدد خوانده‌شده", font=(FONT, 10), bg=CARD, fg=MUTED
                 ).pack(anchor="e", padx=16, pady=(10, 0))
        self.result = tk.Label(card, text="—", font=(FONT, 34, "bold"), bg=CARD, fg=TEXT,
                               wraplength=900, justify="center")
        self.result.pack(pady=(0, 2))
        self.norm = tk.Label(card, text="", font=("Consolas", 13), bg=CARD, fg=MUTED,
                             wraplength=900, justify="center")
        self.norm.pack()
        self.status = tk.Label(card, text="هنوز عکسی انتخاب نشده", font=(FONT, 10), bg=CARD, fg=MUTED)
        self.status.pack(pady=(2, 10))

    def build_image_area(self):
        self.canvas = tk.Canvas(self.root, bg=CARD, highlightthickness=1,
                                highlightbackground=BORDER)
        self.canvas.pack(fill="both", expand=True, padx=20, pady=6)
        self.canvas.bind("<Configure>", lambda e: self.render_image())

    def build_legend(self):
        lg = tk.Frame(self.root, bg=BG)
        lg.pack(fill="x", padx=20, pady=(2, 12))
        items = [("#16a34a", "رقم مطمئن"), ("#dc2626", "مطمئن نیستم (؟)"),
                 ("#2563eb", "رقم نیست (مثلاً حرف)")]
        for color, label in items:
            tk.Label(lg, text=label, font=(FONT, 9), bg=BG, fg=MUTED).pack(side="right", padx=(0, 4))
            tk.Frame(lg, bg=color, width=14, height=14).pack(side="right", padx=(10, 2))
        tk.Label(lg, text="F = فارسی    E = انگلیسی", font=(FONT, 9), bg=BG, fg=MUTED
                 ).pack(side="left")

    # ---------------- منطق ----------------
    def draw_placeholder(self):
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.create_text(w // 2, h // 2, text="عکس نتیجه اینجا نمایش داده می‌شود",
                                font=(FONT, 13), fill="#9ca3af")

    def choose(self):
        path = filedialog.askopenfilename(
            title="انتخاب عکس",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                       ("All files", "*.*")])
        if path:
            self.run(path)

    def rerun(self):
        if self.last_path:
            self.run(self.last_path)

    def run(self, path):
        if not os.path.exists(Numberr.MODEL_PATH):
            messagebox.showerror("مدل پیدا نشد",
                                 "فایل digit_model.pt پیدا نشد.\nاول مدل را آموزش بدهید.")
            return
        self.last_path = path
        self.status.config(text="در حال خواندن...", fg=MUTED)
        self.root.update_idletasks()
        args = SimpleNamespace(ink="auto", min_conf=PRESETS[self.preset.get()],
                               space_ratio=0.8, mixed=self.mixed.get(), out=self.out_path)
        try:
            if os.path.exists(self.out_path):
                os.remove(self.out_path)
            results = Numberr.read_image(path, args)
        except SystemExit as e:
            messagebox.showerror("خطا", str(e))
            return
        except Exception as e:
            messagebox.showerror("خطا", repr(e))
            return

        name = os.path.basename(path)
        if not results:
            self.result_text = ""
            self.result.config(text="رقمی پیدا نشد")
            self.norm.config(text="")
            self.status.config(text=name, fg=MUTED)
            self.pil_img = None
            self.btn_copy.set_enabled(False)
            self.btn_save.set_enabled(False)
            self.draw_placeholder()
            return

        self.result_text = "\n".join(t for t, _ in results)
        self.result.config(text=self.result_text)
        extra = ["= " + n for t, n in results if t != n]
        self.norm.config(text="\n".join(extra))
        self.status.config(text=f"{name}   —   {sum(len(t.replace(' ', '')) for t, _ in results)} نویسه خوانده شد",
                           fg=MUTED)
        self.btn_copy.set_enabled(True)
        self.btn_save.set_enabled(True)
        try:
            self.pil_img = Image.open(self.out_path).convert("RGB")
        except Exception as e:
            self.pil_img = None
            messagebox.showerror("نمایش عکس ممکن نشد", repr(e))
        self.render_image()

    def render_image(self):
        if self.pil_img is None:
            self.draw_placeholder()
            return
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        img = self.pil_img.copy()
        img.thumbnail((cw - 16, ch - 16))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo)

    def copy_result(self):
        if not self.result_text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.result_text)
        self.status.config(text="عدد کپی شد", fg="#16a34a")

    def save_image(self):
        if not os.path.exists(self.out_path):
            return
        dest = filedialog.asksaveasfilename(title="ذخیرهٔ عکس نتیجه", defaultextension=".png",
                                            filetypes=[("PNG", "*.png")])
        if dest:
            shutil.copyfile(self.out_path, dest)
            self.status.config(text="عکس ذخیره شد", fg="#16a34a")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()