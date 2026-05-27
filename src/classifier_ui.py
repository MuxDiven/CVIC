import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk
import queue
import threading
import sys
import os
import subprocess
import random
import torch
import cv2
from pathlib import Path
from label import img_squash, label_dataset
from model import model_construction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent 
DATASET_PATH = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models"
FONT = "Helvetica"

def _label_event(dataset,callback):
        metadata = DATASET_PATH / f"{dataset.name}_metadata.npz"
        images = DATASET_PATH / f"{dataset.name}_images.npy"

        if images.exists() and metadata.exists():
            callback(f"vectorised forms already exist for {dataset.name}")
        else:
            label_dataset(dataset,log_callback=callback)

# App sustains the GUI execution context

class App(tk.Tk):
    def __init__(self,entry_point,mem_flag=False):
        super().__init__()
        self.title("Image Classifier")
        self.geometry("780x560")
        self.configure(bg="#f5f5f0")
        self.current_model = None
        self.entry_point = entry_point
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.mem_flag = mem_flag


        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",     background="#f5f5f0", borderwidth=0)
        style.configure("TNotebook.Tab", background="#e8e8e2", padding=[14, 6], font=(FONT, 11))
        style.map("TNotebook.Tab",       background=[("selected", "#f5f5f0")])
        style.configure("TFrame",        background="#f5f5f0")
        style.configure("TLabel",        background="#f5f5f0", font=(FONT, 11))
        style.configure("TButton",       font=(FONT, 11), padding=[10, 6])
        style.configure("TEntry",        font=(FONT, 11))

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=16, pady=16)
        self.train_tab = TrainTab(nb)
        self.pred_tab = PredictTab(nb)
        nb.add(self.train_tab,   text="  Build  ")
        nb.add(self.pred_tab, text="  Test  ")

    def _on_close(self):
        self.destroy()
        os._exit(0)

class TrainTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=16)
        self.POLL_TIME = 10000 
        self.log_q = queue.Queue()
        self.after(self.POLL_TIME, self._poll)
        self.root = self.master.master

        fields = ttk.Frame(self)
        fields.pack(fill="x", pady=(0, 12))

        # Row 0: Dataset name + Browse + Epochs
        ttk.Label(fields, text="Dataset path").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.dataset_var = tk.StringVar(value="No Dataset Selected")
        ttk.Entry(fields, textvariable=self.dataset_var, width=22).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Button(fields, text="Browse…", command=self._browse_dataset).grid(row=0, column=2, sticky="w", padx=(4, 16), pady=4)
        ttk.Label(fields, text="Epochs").grid(row=0, column=3, sticky="w", padx=(0, 8), pady=4)
        self.epochs_var = tk.IntVar(value=20)
        ttk.Spinbox(fields, from_=1, to=200, textvariable=self.epochs_var, width=6).grid(row=0, column=4, sticky="w", pady=4)

        # Row 1: Seed + Save
        ttk.Label(fields, text="Seed").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self.seed_var = tk.StringVar(value=42)
        ttk.Entry(fields, textvariable=self.seed_var, width=10).grid(row=1, column=1, sticky="w", pady=4)
        self.save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(fields, text="Save model after training", variable=self.save_var).grid(
            row=1, column=3, columnspan=2, sticky="w", pady=4)

        self.headless_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fields, text="Headless training (restarts app)", variable=self.headless_var).grid(
            row=2, column=3, columnspan=2, sticky="w", pady=4)

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", pady=(0, 10))
        self.train_btn = ttk.Button(btn_row, text="▶  Build Model", command=self._on_train)
        self.train_btn.pack(side="left")
        ttk.Button(btn_row, text="Clear log", command=self._clear_log).pack(side="left", padx=8)

        ttk.Label(self, text="Training log").pack(anchor="w")
        self.log = scrolledtext.ScrolledText(
            self, height=16, font=("Courier", 10),
            bg="#1e1e1e", fg="#d4d4d4", relief="flat", borderwidth=0)
        self.log.pack(fill="both", expand=True, pady=(4, 0))
        self.log.configure(state="disabled")

    def _browse_dataset(self):
        folder = filedialog.askdirectory(title="Select dataset folder")
        if folder:                          
            self.dataset_var.set(folder)

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_training(self, active):
        self.train_btn.configure(
            state="disabled" if active else "normal",
            text="Training…" if active else "▶  Start training")

    def _on_train(self):
        dataset = self.dataset_var.get().strip()
        epochs  = self.epochs_var.get() 
        seed_raw    = self.seed_var.get()
        save    = self.save_var.get()
        self.set_training(True)

        seed = int(seed_raw) if seed_raw else random.randint(0,2**32-1)
        dataset = Path(dataset)

        if self.headless_var.get():
            e_point = self.root.entry_point
            self.train_btn.configure(state="disabled")
            self.root.destroy()
            subprocess.Popen([sys.executable, e_point, "--headless", str(dataset), str(epochs), str(seed)])
            os._exit(0) #exit and clean up ui for more resources when training

        def run():
            _label_event(dataset,callback=self.log_q.put)

            self.root.current_model = model_construction(dataset.name, epochs, seed, save=save, log_callback=self.log_q.put)
            self.root.mem_flag = True
            self.root.pred_tab.model_status.configure(text="model in memory")
            self.after(0, lambda: self.set_training(False))

        threading.Thread(target=run, daemon=True).start()

    def _poll(self):
        while not self.log_q.empty():
            self.append_log(self.log_q.get_nowait())  
        self.after(self.POLL_TIME, self._poll)



class PredictTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=16)
        self.photo = None
        self.img_tensor = None
        self.model_name = ""
        self.root = self.master.master
        self.mem_flag = self.root.mem_flag

        # ── model picker ──────────────────────────────────────────────────────
        model_text = "no model loaded" if not self.mem_flag else "model in memory" 
        model_row = ttk.Frame(self)
        model_row.pack(fill="x", pady=(0, 12))
        ttk.Label(model_row, text="Model (.pt)").pack(side="left", padx=(0, 8))
        self.model_var = tk.StringVar()
        ttk.Entry(model_row, textvariable=self.model_var, width=36).pack(side="left")
        ttk.Button(model_row, text="Browse…", command=self._browse_model).pack(side="left", padx=6)
        ttk.Button(model_row, text="Load",    command=self._on_load).pack(side="left")
        self.model_status = ttk.Label(model_row, text=model_text, foreground="#999")
        self.model_status.pack(side="left", padx=10)

        # ── image preview + results ───────────────────────────────────────────
        content = ttk.Frame(self)
        content.pack(fill="both", expand=True)

        left = ttk.Frame(content)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self.canvas = tk.Canvas(left, width=280, height=280, bg="#e8e8e2",
                                highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack()
        self.canvas.create_text(140, 140, text="no image selected",
                                fill="#aaa", font=(FONT, 11))

        img_btn_row = ttk.Frame(left)
        img_btn_row.pack(fill="x", pady=8)
        ttk.Button(img_btn_row, text="Select image…", command=self._browse_image).pack(side="left")
        ttk.Button(img_btn_row, text="Predict",       command=self._on_predict).pack(side="left", padx=8)

        right = ttk.Frame(content)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(right, text="Result", font=(FONT, 12, "bold")).pack(anchor="w", pady=(0, 8))
        self.pred_label = ttk.Label(right, text="—", font=(FONT, 24))
        self.pred_label.pack(anchor="w")
        self.conf_label = ttk.Label(right, text="", foreground="#666")
        self.conf_label.pack(anchor="w", pady=(2, 16))

        ttk.Label(right, text="Class probabilities").pack(anchor="w", pady=(0, 6))
        self.bars_frame = ttk.Frame(right)
        self.bars_frame.pack(fill="x")

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select model",
            filetypes=[("TorchScript model", "*.pt"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")])
        if path:
            self.img_path = Path(path) 
            img = cv2.imread(self.img_path)
            img = img_squash(img)
            self.img_tensor = torch.from_numpy(img).to(device).unsqueeze(0)

            photo = Image.open(self.img_path)
            photo.thumbnail((280,280))
            self.photo = ImageTk.PhotoImage(photo) 
            self.canvas.delete("all") 
            self.canvas.create_image(140,140, anchor="center", image=self.photo)

    def _on_load(self):
        path = Path(self.model_var.get().strip()) 
        self.model_name = path.stem
        self.root.current_model = torch.jit.load(path).to(device)
        self.model_status.configure(text=f"{self.model_name} model active")

    def _on_predict(self):
        if self.root.current_model is not None and self.img_tensor is not None:
            logits = self.root.current_model(self.img_tensor).to(device)
            probs = torch.softmax(logits,dim=1).squeeze()
            self.show_result(self.root.current_model.classes,probs)

    def show_result(self, classes, probs,max_k=10):
        predicted_class = classes[probs.argmax()]
        self.pred_label.configure(text=predicted_class)
        top_conf = probs[probs.argmax()].item() 
        self.conf_label.configure(text=f"{top_conf * 100:.1f}% confidence")
        topk_p, topk_i = torch.topk(probs,min(max_k,len(probs)))

        for w in self.bars_frame.winfo_children():
            w.destroy()

        for tp, ti in zip(topk_p,topk_i):
            label = classes[ti]
            tp = tp.item()
            row = ttk.Frame(self.bars_frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=10, anchor="e").pack(side="left", padx=(0, 6))
            bar = tk.Canvas(row, height=14, bg="#e8e8e2", highlightthickness=0)
            bar.pack(side="left", fill="x", expand=True)
            bar.update_idletasks()
            fill = "#2a2a2a" if label == predicted_class else "#aaaaaa"
            bar.create_rectangle(0, 0, int(bar.winfo_width() * tp), 14, fill=fill, outline="")
            ttk.Label(row, text=f"{tp * 100:.1f}%", width=7).pack(side="left", padx=(6, 0))