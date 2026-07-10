from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf
import os
from glob import glob
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model_yolo = YOLO('Model/YOLO/best.pt')

MODEL_CNN_PATH = 'Model/CNN/cnn_best.keras'
if os.path.exists(MODEL_CNN_PATH):
    model_cnn = tf.keras.models.load_model(MODEL_CNN_PATH)
    print(f"Berhasil memuat model CNN: {MODEL_CNN_PATH}")
else:
    print(f"[ERROR] Model {MODEL_CNN_PATH} tidak ditemukan! Pastikan nama file sesuai.")
    exit()

IMAGE_X, IMAGE_Y = 128, 128
DATASET_DIR = 'Dataset/CNN_Images_Cleaned'
LABELS = sorted([os.path.basename(x) for x in glob(os.path.join(DATASET_DIR, '*'))])

COLOR_BG          = "#0f172a"   
COLOR_SIDEBAR     = "#111827"   
COLOR_CARD        = "#1e293b"   
COLOR_CARD_ALT    = "#172033"
COLOR_BORDER      = "#2d3b52"
COLOR_ACCENT      = "#6366f1"  
COLOR_ACCENT_HOV  = "#818cf8"  
COLOR_SUCCESS     = "#22c55e"  
COLOR_SUCCESS_HOV = "#4ade80"
COLOR_DANGER      = "#ef4444"  
COLOR_DANGER_HOV  = "#f87171"
COLOR_WARNING     = "#f59e0b"  
COLOR_TEXT        = "#f1f5f9"  
COLOR_TEXT_MUTED  = "#94a3b8"  
COLOR_DISABLED    = "#334155"  

FONT_FAMILY = "Segoe UI"

cap = None
is_playing = False
 
final_word = ""           
last_stable_letter = None    
CONFIDENCE_THRESHOLD = 75    
STABLE_FRAME_REQ = 3     
letter_streak_counter = 0   
last_seen_letter = None     
 
BLANK_LABEL = "_blank_" 

def decode_ctc_step(predicted_label, confidence):
    """Menerapkan prinsip CTC Decoding sejati menggunakan Class Pemisah (Blank)"""
    global final_word, last_stable_letter
    global letter_streak_counter, last_seen_letter
 
    if confidence < CONFIDENCE_THRESHOLD:
        return
 
    if predicted_label == last_seen_letter:
        letter_streak_counter += 1
    else:
        letter_streak_counter = 1
        last_seen_letter = predicted_label
 
    if letter_streak_counter >= STABLE_FRAME_REQ:
 
        if predicted_label == BLANK_LABEL:
            last_stable_letter = None 

        else:
            if predicted_label != last_stable_letter:
                final_word += predicted_label
                last_stable_letter = predicted_label
                lbl_word_output.config(text=final_word) 

def process_frame(img):
    """Memproses frame menggunakan YOLO dan CNN"""
    global last_stable_letter, last_seen_letter, letter_streak_counter
    
    results = model_yolo(img, stream=True, conf=0.5, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (99, 102, 241), 3)  # indigo box

            if y2 > y1 and x2 > x1:
                try:
                    hand_crop = img[y1:y2, x1:x2]
                    gray_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                    resized_crop = cv2.resize(gray_crop, (IMAGE_Y, IMAGE_X))
                    normalized_crop = resized_crop / 255.0
                    img_input = np.reshape(normalized_crop, (1, IMAGE_X, IMAGE_Y, 1))

                    predictions = model_cnn.predict(img_input, verbose=0)
                    class_idx = np.argmax(predictions)
                    predicted_label = LABELS[class_idx]
                    confidence = predictions[0][class_idx] * 100

                    display_label = "[ JEDA ]" if predicted_label == BLANK_LABEL else predicted_label
                    text = f"{display_label} ({confidence:.1f}%)"
                    
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(img, (x1, y1 - th - 18), (x1 + tw + 10, y1 - 4), (99, 102, 241), -1)
                    cv2.putText(img, text, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)

                    img_preview = cv2.resize(gray_crop, (140, 140))
                    img_preview = Image.fromarray(img_preview)
                    imgtk_preview = ImageTk.PhotoImage(image=img_preview)
                    lbl_hand_preview.imgtk = imgtk_preview
                    lbl_hand_preview.configure(image=imgtk_preview)

                    lbl_prediction_letter.config(text=display_label)
                    lbl_prediction_conf.config(text=f"Keyakinan: {confidence:.1f}%")
                    update_confidence_bar(confidence)

                    decode_ctc_step(predicted_label, confidence)

                except Exception:
                    pass
    return img


def update_confidence_bar(confidence):
    canvas_conf.delete("all")
    w = 200
    h = 14
    canvas_conf.create_rectangle(0, 0, w, h, fill=COLOR_DISABLED, outline="")
    fill_w = max(4, int(w * (confidence / 100)))
    bar_color = COLOR_SUCCESS if confidence >= 70 else (COLOR_WARNING if confidence >= 40 else COLOR_DANGER)
    canvas_conf.create_rectangle(0, 0, fill_w, h, fill=bar_color, outline="")


def update_video():
    global cap, is_playing
    if is_playing and cap is not None:
        success, img = cap.read()
        if success:
            img = process_frame(img)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resize = cv2.resize(img_rgb, (760, 480))

            img_pil = Image.fromarray(img_resize)
            imgtk = ImageTk.PhotoImage(image=img_pil)

            lbl_video.imgtk = imgtk
            lbl_video.configure(image=imgtk)

            lbl_video.after(20, update_video)
        else:
            stop_video()
            messagebox.showinfo("Info", "Video selesai diputar.")

def set_status(text, color=COLOR_TEXT_MUTED):
    status_dot.itemconfig(status_dot_circle, fill=color)
    lbl_status.config(text=text)

def select_video():
    global cap, is_playing
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
    if file_path:
        stop_video()
        cap = cv2.VideoCapture(file_path)
        lbl_source.config(text=f"📁 {os.path.basename(file_path)}")
        set_status("Memutar file video", COLOR_SUCCESS)
        start_video()

def start_camera():
    global cap, is_playing
    stop_video()
    cap = cv2.VideoCapture(0)
    lbl_source.config(text="📷 Live Webcam")
    set_status("Webcam aktif", COLOR_SUCCESS)
    start_video()

def start_video():
    global is_playing
    if cap is not None:
        is_playing = True
        btn_start.set_state(tk.DISABLED)
        btn_stop.set_state(tk.NORMAL)
        update_video()

def stop_video():
    global is_playing, cap
    is_playing = False
    if cap is not None:
        cap.release()
        cap = None
    btn_start.set_state(tk.NORMAL)
    btn_stop.set_state(tk.DISABLED)
    lbl_video.configure(image=img_placeholder)
    lbl_hand_preview.configure(image=img_hand_placeholder)
    lbl_prediction_letter.config(text="–")
    lbl_prediction_conf.config(text="Keyakinan: -")
    canvas_conf.delete("all")
    canvas_conf.create_rectangle(0, 0, 200, 14, fill=COLOR_DISABLED, outline="")
    lbl_source.config(text="Belum ada sumber")
    set_status("Berhenti", COLOR_TEXT_MUTED)

def clear_word():
    """Mengosongkan kata terjemahan yang terkumpul"""
    global final_word, last_stable_letter
    final_word = ""
    last_stable_letter = None
    lbl_word_output.config(text="...")

def add_space():
    """Menambahkan spasi pada kata"""
    global final_word
    final_word += " "
    lbl_word_output.config(text=final_word)

def on_closing():
    stop_video()
    root.destroy()

class ModernButton(tk.Canvas):
    def __init__(self, parent, text, command, base_color, hover_color,
                 icon="", width=210, height=46, disabled_color=COLOR_DISABLED, **kwargs):
        super().__init__(parent, width=width, height=height, bg=COLOR_SIDEBAR,
                         highlightthickness=0, **kwargs)
        self.command = command
        self.base_color = base_color
        self.hover_color = hover_color
        self.disabled_color = disabled_color
        self.width = width
        self.height = height
        self.text = f"{icon}  {text}" if icon else text
        self.state_ = tk.NORMAL

        self._draw(self.base_color)

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        points = [x1+r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y2-r, x2, y2,
                  x2-r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y1+r, x1, y1]
        return self.create_polygon(points, smooth=True, **kw)

    def _draw(self, color):
        self.delete("all")
        self._round_rect(2, 2, self.width - 2, self.height - 2, 14, fill=color, outline="")
        self.create_text(self.width / 2, self.height / 2, text=self.text,
                          fill="white", font=(FONT_FAMILY, 11, "bold"))

    def _on_enter(self, e):
        if self.state_ == tk.NORMAL:
            self._draw(self.hover_color)
            self.config(cursor="hand2")

    def _on_leave(self, e):
        if self.state_ == tk.NORMAL:
            self._draw(self.base_color)

    def _on_click(self, e):
        if self.state_ == tk.NORMAL and self.command:
            self.command()

    def set_state(self, state):
        self.state_ = state
        if state == tk.DISABLED:
            self._draw(self.disabled_color)
            self.config(cursor="arrow")
        else:
            self._draw(self.base_color)
            self.config(cursor="hand2")

root = tk.Tk()
root.title("Sign Language Detection Dashboard  •  YOLOv8 + CNN + True CTC Decoder")
root.geometry("1180x740")
root.minsize(1040, 680)
root.configure(bg=COLOR_BG)

header = tk.Frame(root, bg=COLOR_BG, height=64)
header.pack(side=tk.TOP, fill=tk.X, padx=24, pady=(18, 0))

tk.Label(header, text="🤟  Sign Language Detection", font=(FONT_FAMILY, 18, "bold"),
         fg=COLOR_TEXT, bg=COLOR_BG).pack(side=tk.LEFT)
tk.Label(header, text="YOLOv8 Detector + CNN Classifier + True CTC Logic", font=(FONT_FAMILY, 10),
         fg=COLOR_TEXT_MUTED, bg=COLOR_BG).pack(side=tk.LEFT, padx=(12, 0), pady=(6, 0))

body = tk.Frame(root, bg=COLOR_BG)
body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=24, pady=12)

frame_video_card = tk.Frame(body, bg=COLOR_CARD, bd=0, highlightbackground=COLOR_BORDER,
                             highlightthickness=1)
frame_video_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 18))

video_header = tk.Frame(frame_video_card, bg=COLOR_CARD)
video_header.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(14, 6))

lbl_source = tk.Label(video_header, text="Belum ada sumber", font=(FONT_FAMILY, 10, "bold"),
                       fg=COLOR_TEXT_MUTED, bg=COLOR_CARD)
lbl_source.pack(side=tk.LEFT)

status_dot = tk.Canvas(video_header, width=12, height=12, bg=COLOR_CARD, highlightthickness=0)
status_dot.pack(side=tk.RIGHT, pady=2)
status_dot_circle = status_dot.create_oval(2, 2, 12, 12, fill=COLOR_TEXT_MUTED, outline="")

lbl_status = tk.Label(video_header, text="Berhenti", font=(FONT_FAMILY, 10),
                       fg=COLOR_TEXT_MUTED, bg=COLOR_CARD)
lbl_status.pack(side=tk.RIGHT, padx=(0, 8))

video_wrap = tk.Frame(frame_video_card, bg="#000000")
video_wrap.pack(padx=16, pady=(0, 16), fill=tk.BOTH, expand=True)

img_placeholder = ImageTk.PhotoImage(Image.new("RGB", (760, 480), "#000000"))
lbl_video = tk.Label(video_wrap, image=img_placeholder, bg="#000000")
lbl_video.pack(fill=tk.BOTH, expand=True)

sidebar = tk.Frame(body, bg=COLOR_SIDEBAR, width=270, highlightbackground=COLOR_BORDER,
                    highlightthickness=1)
sidebar.pack(side=tk.RIGHT, fill=tk.Y)
sidebar.pack_propagate(False)

inner = tk.Frame(sidebar, bg=COLOR_SIDEBAR)
inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

tk.Label(inner, text="KONTROL", font=(FONT_FAMILY, 11, "bold"), fg=COLOR_TEXT_MUTED,
         bg=COLOR_SIDEBAR).pack(anchor=tk.W, pady=(0, 6))

btn_file = ModernButton(inner, "Pilih File Video", select_video, COLOR_ACCENT, COLOR_ACCENT_HOV, icon="📁")
btn_file.pack(pady=4)

btn_cam = ModernButton(inner, "Gunakan Webcam", start_camera, COLOR_ACCENT, COLOR_ACCENT_HOV, icon="📷")
btn_cam.pack(pady=4)

btn_start = ModernButton(inner, "Play", start_video, COLOR_SUCCESS, COLOR_SUCCESS_HOV, icon="▶")
btn_start.pack(pady=4)
btn_start.set_state(tk.DISABLED)

btn_stop = ModernButton(inner, "Stop", stop_video, COLOR_DANGER, COLOR_DANGER_HOV, icon="■")
btn_stop.pack(pady=4)
btn_stop.set_state(tk.DISABLED)

tk.Frame(inner, height=1, bg=COLOR_BORDER).pack(fill=tk.X, pady=12)

tk.Label(inner, text="HASIL DETEKSI HURUF", font=(FONT_FAMILY, 11, "bold"), fg=COLOR_TEXT_MUTED,
         bg=COLOR_SIDEBAR).pack(anchor=tk.W, pady=(0, 6))

result_card = tk.Frame(inner, bg=COLOR_CARD_ALT, highlightbackground=COLOR_BORDER, highlightthickness=1)
result_card.pack(fill=tk.X, pady=(0, 4))

result_inner = tk.Frame(result_card, bg=COLOR_CARD_ALT)
result_inner.pack(fill=tk.BOTH, padx=14, pady=10)

img_hand_placeholder = ImageTk.PhotoImage(Image.new("L", (140, 140), "#0b1220"))
hand_frame = tk.Frame(result_inner, bg="#0b1220", highlightbackground=COLOR_BORDER, highlightthickness=1)
hand_frame.pack()
lbl_hand_preview = tk.Label(hand_frame, image=img_hand_placeholder, bg="#0b1220")
lbl_hand_preview.pack(padx=2, pady=2)

lbl_prediction_letter = tk.Label(result_inner, text="–", font=(FONT_FAMILY, 26, "bold"),
                                  fg=COLOR_ACCENT_HOV, bg=COLOR_CARD_ALT)
lbl_prediction_letter.pack(pady=(6, 2))

lbl_prediction_conf = tk.Label(result_inner, text="Keyakinan: -", font=(FONT_FAMILY, 10),
                                fg=COLOR_TEXT_MUTED, bg=COLOR_CARD_ALT)
lbl_prediction_conf.pack(pady=(0, 6))

canvas_conf = tk.Canvas(result_inner, width=200, height=14, bg=COLOR_CARD_ALT, highlightthickness=0)
canvas_conf.pack()
canvas_conf.create_rectangle(0, 0, 200, 14, fill=COLOR_DISABLED, outline="")

word_panel = tk.Frame(root, bg=COLOR_CARD, highlightbackground=COLOR_BORDER, highlightthickness=1)
word_panel.pack(side=tk.TOP, fill=tk.X, padx=24, pady=(4, 12))

word_inner = tk.Frame(word_panel, bg=COLOR_CARD)
word_inner.pack(fill=tk.X, padx=16, pady=12)

tk.Label(word_inner, text="📝 HASIL TEKS (KATA):", font=(FONT_FAMILY, 11, "bold"), 
         fg=COLOR_TEXT_MUTED, bg=COLOR_CARD).pack(side=tk.LEFT, padx=(0, 12))

lbl_word_output = tk.Label(word_inner, text="...", font=(FONT_FAMILY, 16, "bold"), 
                           fg=COLOR_SUCCESS_HOV, bg=COLOR_CARD)
lbl_word_output.pack(side=tk.LEFT, fill=tk.X, expand=True, anchor=tk.W)

btn_space = tk.Button(word_inner, text="[ Spasi ]", font=(FONT_FAMILY, 9, "bold"),
                      command=add_space, bg=COLOR_DISABLED, fg=COLOR_TEXT, bd=0, padx=12, pady=4, cursor="hand2")
btn_space.pack(side=tk.RIGHT, padx=4)

btn_clear = tk.Button(word_inner, text="Hapus Kata", font=(FONT_FAMILY, 9, "bold"),
                      command=clear_word, bg=COLOR_DANGER, fg="white", bd=0, padx=12, pady=4, cursor="hand2")
btn_clear.pack(side=tk.RIGHT, padx=4)

footer = tk.Frame(root, bg=COLOR_BG)
footer.pack(side=tk.BOTTOM, fill=tk.X, padx=24, pady=(0, 16))
tk.Label(footer, text="Model: YOLOv8 (deteksi tangan)  •  CNN (klasifikasi huruf)  •  True CTC Decoder",
         font=(FONT_FAMILY, 9), fg=COLOR_TEXT_MUTED, bg=COLOR_BG).pack(side=tk.LEFT)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()