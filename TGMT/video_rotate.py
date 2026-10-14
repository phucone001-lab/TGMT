import argparse
import base64
import importlib
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import filedialog
from urllib.request import urlretrieve

import cv2
import numpy as np


class VideoApp:
    def __init__(self, root, video_path=None, initial_mode="rotate"):
        self.root = root
        self.root.title("PixelStudio")
        self.root.geometry("1100x760")
        self.root.minsize(760, 540)
        self.root.configure(bg="#141a17")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        self.capture = None
        self.video_path = None
        self.mode = "rotate"
        self.turns_clockwise = 0
        self.frame_interval_ms = 33
        self.duration_seconds = 0.0
        self.seeking = False
        self.face_detector = None
        self.face_cascade = None
        self.detector_loading = False
        self.balloon_session = None
        self.balloon_loading = False
        self.balloon_image = None
        self.balloon_image_path = None
        self.balloon_detections = []
        self.plate_session = None
        self.plate_loading = False
        self.plate_image = None
        self.plate_image_path = None
        self.current_frame = None
        self.current_faces = []
        self.photo = None
        self.image_item = None
        self.closing = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress>", self.on_key)
        self.set_mode(initial_mode)
        if video_path:
            if initial_mode == "balloons":
                self.open_balloon_image(video_path)
            elif initial_mode == "plate":
                self.open_plate_image(video_path)
            else:
                self.open_video(video_path)
        self.root.after(15, self.update_frame)

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#141a17", padx=24, pady=18)
        header.grid(row=0, column=0, sticky="ew")

        title = tk.Label(
            header,
            text="Pixel Studio",
            bg="#141a17",
            fg="#eef4ef",
            font=("Segoe UI", 18, "bold"),
        )
        title.pack(side="left")

        self.open_button = tk.Button(
            header,
            text="Mở video",
            command=self.choose_video,
            bg="#a6d96a",
            fg="#172014",
            activebackground="#b9e987",
            activeforeground="#172014",
            relief="flat",
            padx=18,
            pady=9,
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
        )
        self.open_button.pack(side="right")
        self.replay_button = self._make_button(
            header, "Phát lại", self.replay_video
        )
        self.replay_button.configure(state=tk.DISABLED)
        self.replay_button.pack(side="right", padx=(0, 8))

        video_area = tk.Frame(self.root, bg="#141a17", padx=24)
        video_area.grid(row=1, column=0, sticky="nsew")
        video_area.grid_rowconfigure(0, weight=1)
        video_area.grid_columnconfigure(0, weight=1)
        self.video_canvas = tk.Canvas(
            video_area,
            width=640,
            height=360,
            bg="#080c0a",
            highlightthickness=0,
        )
        self.video_canvas.grid(row=0, column=0, sticky="nsew")
        self.placeholder_item = self.video_canvas.create_text(
            320,
            180,
            text="Mở video để bắt đầu",
            fill="#94a49a",
            font=("Segoe UI", 16),
        )
        self.video_canvas.bind("<Configure>", self._resize_placeholder)

        seek_area = tk.Frame(self.root, bg="#141a17", padx=24, pady=6)
        self.seek_area = seek_area
        seek_area.grid(row=2, column=0, sticky="ew")
        seek_area.grid_columnconfigure(1, weight=1)
        self.position_var = tk.StringVar(value="00:00")
        self.duration_var = tk.StringVar(value="00:00")
        tk.Label(
            seek_area,
            textvariable=self.position_var,
            bg="#141a17",
            fg="#dce6de",
            width=6,
            anchor="w",
            font=("Segoe UI", 9),
        ).grid(row=0, column=0, padx=(0, 8))
        self.seek_var = tk.DoubleVar(value=0)
        self.seek_scale = tk.Scale(
            seek_area,
            from_=0,
            to=1,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            showvalue=False,
            variable=self.seek_var,
            command=self._seek_changed,
            bg="#141a17",
            fg="#eef4ef",
            troughcolor="#354139",
            activebackground="#a6d96a",
            highlightthickness=0,
            borderwidth=0,
            sliderlength=18,
            state=tk.DISABLED,
        )
        self.seek_scale.grid(row=0, column=1, sticky="ew")
        self.seek_scale.bind("<ButtonPress-1>", self._start_seek)
        self.seek_scale.bind("<ButtonRelease-1>", self._finish_seek)
        tk.Label(
            seek_area,
            textvariable=self.duration_var,
            bg="#141a17",
            fg="#dce6de",
            width=6,
            anchor="e",
            font=("Segoe UI", 9),
        ).grid(row=0, column=2, padx=(8, 0))

        controls = tk.Frame(self.root, bg="#202823", padx=20, pady=16)
        controls.grid(row=3, column=0, sticky="ew", padx=24, pady=(8, 0))

        mode_group = tk.Frame(controls, bg="#202823")
        mode_group.pack(side="left")
        tk.Label(
            mode_group,
            text="CHẾ ĐỘ",
            bg="#202823",
            fg="#aab7ae",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 7))

        mode_buttons = tk.Frame(mode_group, bg="#202823")
        mode_buttons.pack()
        mode_buttons.grid_columnconfigure((0, 1), weight=1)
        self.rotate_mode_button = self._make_button(
            mode_buttons, "Xoay video", lambda: self.set_mode("rotate")
        )
        self.rotate_mode_button.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.faces_mode_button = self._make_button(
            mode_buttons, "Nhận diện khuôn mặt", lambda: self.set_mode("faces")
        )
        self.faces_mode_button.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=2)
        self.balloons_mode_button = self._make_button(
            mode_buttons, "Đếm bóng bay", lambda: self.set_mode("balloons")
        )
        self.balloons_mode_button.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.plate_mode_button = self._make_button(
            mode_buttons, "Tách ký tự biển số", lambda: self.set_mode("plate")
        )
        self.plate_mode_button.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=2)

        self.action_area = tk.Frame(controls, bg="#202823")
        self.action_area.pack(side="right", anchor="s")

        self.rotate_actions = tk.Frame(self.action_area, bg="#202823")
        self._make_button(
            self.rotate_actions, "Xoay trái 90°", lambda: self.rotate(-1)
        ).pack(side="left", padx=(0, 7))
        self._make_button(
            self.rotate_actions, "Xoay phải 90°", lambda: self.rotate(1)
        ).pack(side="left")

        self.face_actions = tk.Frame(self.action_area, bg="#202823")
        self.save_button = self._make_button(
            self.face_actions,
            "Lưu khuôn mặt",
            self.save_faces,
            primary=True,
        )
        self.save_button.pack(side="left")

        self.balloon_actions = tk.Frame(self.action_area, bg="#202823")
        self.open_image_button = self._make_button(
            self.balloon_actions,
            "Mở ảnh và đếm",
            self.choose_balloon_image,
            primary=True,
        )
        self.open_image_button.pack(side="left")

        self.plate_actions = tk.Frame(self.action_area, bg="#202823")
        self.open_plate_button = self._make_button(
            self.plate_actions,
            "Mở ảnh và tách số",
            self.choose_plate_image,
            primary=True,
        )
        self.open_plate_button.pack(side="left")

        self.status_var = tk.StringVar(value="Chưa chọn video")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#141a17",
            fg="#aab7ae",
            anchor="w",
            padx=24,
            pady=12,
            font=("Segoe UI", 9),
        )
        self.status_label.grid(row=4, column=0, sticky="ew")

    def _resize_placeholder(self, event):
        if self.image_item is None:
            self.video_canvas.coords(
                self.placeholder_item, event.width // 2, event.height // 2
            )

    def _show_placeholder(self, text):
        if self.image_item is not None:
            self.video_canvas.delete(self.image_item)
            self.image_item = None
        self.video_canvas.delete(self.placeholder_item)
        self.placeholder_item = self.video_canvas.create_text(
            self.video_canvas.winfo_width() // 2,
            self.video_canvas.winfo_height() // 2,
            text=text,
            fill="#94a49a",
            font=("Segoe UI", 16),
        )

    @staticmethod
    def _make_button(parent, text, command, primary=False):
        background = "#a6d96a" if primary else "#354139"
        foreground = "#172014" if primary else "#eef4ef"
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=background,
            fg=foreground,
            activebackground="#b9e987" if primary else "#46564b",
            activeforeground=foreground,
            disabledforeground="#718076",
            relief="flat",
            padx=14,
            pady=10,
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
        )

    def set_mode(self, mode):
        self.mode = mode
        self.current_faces = []
        active = {"bg": "#a6d96a", "fg": "#172014"}
        inactive = {"bg": "#354139", "fg": "#eef4ef"}
        self.rotate_mode_button.configure(**(active if mode == "rotate" else inactive))
        self.faces_mode_button.configure(**(active if mode == "faces" else inactive))
        self.balloons_mode_button.configure(
            **(active if mode == "balloons" else inactive)
        )
        self.plate_mode_button.configure(**(active if mode == "plate" else inactive))

        self.rotate_actions.pack_forget()
        self.face_actions.pack_forget()
        self.balloon_actions.pack_forget()
        self.plate_actions.pack_forget()
        if mode == "rotate":
            self.rotate_actions.pack(side="right")
            self.open_button.configure(text="Mở video", command=self.choose_video)
            self.status_var.set(self._video_status("Chế độ xoay video"))
        elif mode == "faces":
            self.face_actions.pack(side="right")
            self.open_button.configure(text="Mở video", command=self.choose_video)
            self.status_var.set(self._video_status("Đang chuẩn bị nhận diện khuôn mặt"))
            self._ensure_face_detector()
        elif mode == "balloons":
            self.balloon_actions.pack(side="right")
            self.open_button.configure(
                text="Mở ảnh", command=self.choose_balloon_image
            )
            self.status_var.set("Chọn ảnh để đếm bóng bay")
            self._show_placeholder("Chọn ảnh để đếm bóng bay")
        else:
            self.plate_actions.pack(side="right")
            self.open_button.configure(
                text="Mở ảnh", command=self.choose_plate_image
            )
            self.status_var.set("Chọn ảnh biển số để tách ký tự")
            self._show_placeholder("Chọn ảnh biển số để bắt đầu")

        if mode in ("balloons", "plate"):
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            self.seek_scale.configure(state=tk.DISABLED)
            self.seek_area.grid_remove()
            self.replay_button.pack_forget()
        else:
            self.seek_area.grid()
            if self.video_path and self.replay_button.winfo_manager() == "":
                self.replay_button.pack(side="right", padx=(0, 8))
            self.seek_scale.configure(
                state=tk.NORMAL if self.duration_seconds > 0 else tk.DISABLED
            )

    def _video_status(self, message):
        if self.video_path:
            return f"{message}  |  {Path(self.video_path).name}"
        return message

    def choose_video(self):
        path = filedialog.askopenfilename(
            title="Chọn file video",
            filetypes=(
                ("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Tất cả file", "*.*"),
            ),
        )
        if path:
            self.open_video(path)

    def choose_balloon_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh để đếm bóng bay",
            filetypes=(
                ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Tất cả file", "*.*"),
            ),
        )
        if path:
            self.open_balloon_image(path)

    def open_balloon_image(self, path):
        image = cv2.imread(path)
        if image is None:
            self.status_var.set(f"Không mở được ảnh: {path}")
            return

        self.balloon_image = image
        self.balloon_image_path = path
        self.balloon_detections = []
        self._show_frame(image)
        self.status_var.set(f"Đang nhận diện bóng bay  |  {Path(path).name}")
        if self.balloon_session is None:
            self._ensure_balloon_detector()
        else:
            self._detect_and_show_balloons()

    def choose_plate_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh có biển số xe",
            filetypes=(
                ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Tất cả file", "*.*"),
            ),
        )
        if path:
            self.open_plate_image(path)

    def open_plate_image(self, path):
        image = cv2.imread(path)
        if image is None:
            self.status_var.set(f"Không mở được ảnh: {path}")
            return

        self.plate_image = image
        self.plate_image_path = path
        self._show_frame(image)
        self.status_var.set(f"Đang đọc biển số  |  {Path(path).name}")
        if self.plate_session is None:
            self._ensure_plate_detector()
        else:
            self._detect_and_crop_plate_characters()

    def _ensure_plate_detector(self):
        if self.plate_session is not None or self.plate_loading:
            return

        self.plate_loading = True
        self.status_var.set("Đang tải model biển số lần đầu")

        def load_detector():
            try:
                onnxruntime = importlib.import_module("onnxruntime")
                model_path = Path(__file__).with_name("license_plate_yolov8.onnx")
                if not model_path.exists():
                    model_url = (
                        "https://huggingface.co/ml-debi/"
                        "yolov8-license-plate-detection/resolve/main/best.onnx"
                    )
                    urlretrieve(model_url, model_path)
                session = onnxruntime.InferenceSession(
                    str(model_path), providers=["CPUExecutionProvider"]
                )
                self.root.after(0, self._plate_detector_ready, session, None)
            except Exception as error:
                try:
                    self.root.after(0, self._plate_detector_ready, None, str(error))
                except tk.TclError:
                    pass

        threading.Thread(target=load_detector, daemon=True).start()

    def _plate_detector_ready(self, session, error):
        self.plate_loading = False
        if error:
            self.status_var.set(f"Lỗi tải model biển số: {error}")
            return

        self.plate_session = session
        if self.mode == "plate" and self.plate_image is not None:
            self._detect_and_crop_plate_characters()

    def _detect_plates(self, image):
        image_height, image_width = image.shape[:2]
        scale = min(640 / image_width, 640 / image_height)
        resized_width = max(1, round(image_width * scale))
        resized_height = max(1, round(image_height * scale))
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        pad_left = (640 - resized_width) // 2
        pad_top = (640 - resized_height) // 2
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[
            pad_top:pad_top + resized_height,
            pad_left:pad_left + resized_width,
        ] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        input_name = self.plate_session.get_inputs()[0].name
        output = self.plate_session.run(None, {input_name: tensor})[0][0]
        predictions = output.T if output.shape[0] < output.shape[1] else output
        boxes = []
        scores = []
        for center_x, center_y, box_width, box_height, score in predictions:
            if score < 0.35:
                continue
            left = max(0, int((center_x - box_width / 2 - pad_left) / scale))
            top = max(0, int((center_y - box_height / 2 - pad_top) / scale))
            right = min(image_width, int((center_x + box_width / 2 - pad_left) / scale))
            bottom = min(image_height, int((center_y + box_height / 2 - pad_top) / scale))
            if right > left and bottom > top:
                boxes.append([left, top, right - left, bottom - top])
                scores.append(float(score))

        kept = cv2.dnn.NMSBoxes(boxes, scores, 0.35, 0.45)
        if len(kept) == 0:
            return []
        return [
            (
                boxes[index][0],
                boxes[index][1],
                boxes[index][0] + boxes[index][2],
                boxes[index][1] + boxes[index][3],
                scores[index],
            )
            for index in np.asarray(kept).reshape(-1)
        ]

    @staticmethod
    def _segment_plate_characters(plate):
        plate_height, plate_width = plate.shape[:2]
        if plate_height < 12 or plate_width < 24:
            return []

        scale = 160 / plate_height
        resized_width = max(1, round(plate_width * scale))
        resized = cv2.resize(plate, (resized_width, 160), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        masks = [
            binary,
            cv2.bitwise_not(binary),
            cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 7,
            ),
        ]

        candidates = []
        for mask in masks:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            boxes = []
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                aspect_ratio = width / height if height else 0
                if (
                    height >= 0.2 * 160
                    and height <= 0.94 * 160
                    and width >= 0.018 * resized_width
                    and width <= 0.22 * resized_width
                    and 0.08 <= aspect_ratio <= 1.15
                ):
                    boxes.append((x, y, x + width, y + height))
            if len(boxes) > len(candidates):
                candidates = boxes

        if not candidates:
            return []

        outer_candidates = []
        for box in sorted(
            candidates,
            key=lambda item: (item[2] - item[0]) * (item[3] - item[1]),
            reverse=True,
        ):
            left, top, right, bottom = box
            area = (right - left) * (bottom - top)
            is_inner_contour = any(
                left >= outer[0]
                and top >= outer[1]
                and right <= outer[2]
                and bottom <= outer[3]
                and area < (outer[2] - outer[0]) * (outer[3] - outer[1]) * 0.7
                for outer in outer_candidates
            )
            if not is_inner_contour:
                outer_candidates.append(box)
        candidates = outer_candidates

        candidates.sort(key=lambda box: (box[1] + box[3]) / 2)
        rows = []
        for box in candidates:
            center_y = (box[1] + box[3]) / 2
            row = next(
                (items for items in rows if abs(items[0] - center_y) < 0.22 * 160),
                None,
            )
            if row is None:
                rows.append([center_y, [box]])
            else:
                row[1].append(box)

        characters = []
        for _, row_boxes in sorted(rows, key=lambda row: row[0]):
            for left, top, right, bottom in sorted(row_boxes, key=lambda box: box[0]):
                x1 = max(0, int(left / scale) - 2)
                y1 = max(0, int(top / scale) - 2)
                x2 = min(plate_width, int(right / scale) + 2)
                y2 = min(plate_height, int(bottom / scale) + 2)
                if x2 > x1 and y2 > y1:
                    characters.append((x1, y1, x2, y2))
        return characters

    def _detect_and_crop_plate_characters(self):
        plates = self._detect_plates(self.plate_image)
        display = self.plate_image.copy()
        output_dir = Path(self.plate_image_path).with_name(
            f"{Path(self.plate_image_path).stem}_digits"
        )
        saved = 0
        character_boxes = []

        for plate_index, (left, top, right, bottom, score) in enumerate(plates, 1):
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"Bien so {score:.0%}",
                (left, max(20, top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            plate_crop = self.plate_image[top:bottom, left:right]
            for character_index, (x1, y1, x2, y2) in enumerate(
                self._segment_plate_characters(plate_crop), 1
            ):
                absolute_box = (left + x1, top + y1, left + x2, top + y2)
                character_boxes.append(absolute_box)
                output_dir.mkdir(parents=True, exist_ok=True)
                character_crop = plate_crop[y1:y2, x1:x2]
                output_path = output_dir / (
                    f"plate_{plate_index:02d}_char_{character_index:02d}.png"
                )
                if cv2.imwrite(str(output_path), character_crop):
                    saved += 1

        for left, top, right, bottom in character_boxes:
            cv2.rectangle(display, (left, top), (right, bottom), (255, 160, 0), 1)
        cv2.putText(
            display,
            f"Bien so: {len(plates)}  Ky tu da tach: {saved}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        self._show_frame(display)
        if saved:
            self.status_var.set(
                f"Đã tách {saved} ký tự, lưu trong {output_dir}"
            )
        elif plates:
            self.status_var.set(
                "Đã tìm thấy biển số nhưng chưa tách được ký tự rõ ràng"
            )
        else:
            self.status_var.set("Không tìm thấy biển số trong ảnh")

    def _ensure_balloon_detector(self):
        if self.balloon_session is not None or self.balloon_loading:
            return

        self.balloon_loading = True
        self.status_var.set("Đang tải model đếm bóng bay lần đầu")

        def load_detector():
            try:
                onnxruntime = importlib.import_module("onnxruntime")

                model_path = Path(__file__).with_name(
                    "balloon_detection_int8.onnx"
                )
                if not model_path.exists():
                    model_url = (
                        "https://huggingface.co/onnx-community/"
                        "BalloonDetectioDTR-ONNX/resolve/main/onnx/model_int8.onnx"
                    )
                    urlretrieve(model_url, model_path)
                session = onnxruntime.InferenceSession(
                    str(model_path), providers=["CPUExecutionProvider"]
                )
                self.root.after(0, self._balloon_detector_ready, session, None)
            except Exception as error:
                try:
                    self.root.after(
                        0, self._balloon_detector_ready, None, str(error)
                    )
                except tk.TclError:
                    pass

        threading.Thread(target=load_detector, daemon=True).start()

    def _balloon_detector_ready(self, session, error):
        self.balloon_loading = False
        if error:
            self.status_var.set(f"Lỗi tải model bóng bay: {error}")
            return

        self.balloon_session = session
        if self.mode == "balloons" and self.balloon_image is not None:
            self._detect_and_show_balloons()

    def _detect_and_show_balloons(self):
        image_height, image_width = self.balloon_image.shape[:2]
        scale = min(800 / min(image_height, image_width), 1333 / max(image_height, image_width))
        resized_width = max(1, round(image_width * scale))
        resized_height = max(1, round(image_height * scale))
        rgb = cv2.cvtColor(self.balloon_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC
        )
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = ((normalized - mean) / std).transpose(2, 0, 1)[None, ...]

        input_name = self.balloon_session.get_inputs()[0].name
        output_names = [output.name for output in self.balloon_session.get_outputs()]
        outputs = self.balloon_session.run(output_names, {input_name: tensor})
        predictions = dict(zip(output_names, outputs))
        logits = predictions["logits"][0]
        boxes = predictions["pred_boxes"][0]
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)

        detections = []
        for score, box in zip(probabilities[:, 0], boxes):
            if score < 0.6:
                continue
            center_x, center_y, box_width, box_height = box
            left = max(0, int((center_x - box_width / 2) * image_width))
            top = max(0, int((center_y - box_height / 2) * image_height))
            right = min(image_width, int((center_x + box_width / 2) * image_width))
            bottom = min(image_height, int((center_y + box_height / 2) * image_height))
            if right > left and bottom > top:
                detections.append((left, top, right, bottom, float(score)))

        self.balloon_detections = detections
        display = self.balloon_image.copy()
        for left, top, right, bottom, score in detections:
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"Bong bay {score:.0%}",
                (left, max(20, top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        cv2.putText(
            display,
            f"So bong bay: {len(detections)}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
        self._show_frame(display)
        self.status_var.set(
            f"Tìm thấy {len(detections)} bóng bay  |  "
            f"{Path(self.balloon_image_path).name}"
        )

    def open_video(self, path):
        new_capture = cv2.VideoCapture(path)
        if not new_capture.isOpened():
            new_capture.release()
            self.status_var.set(f"Không mở được video: {path}")
            return

        if self.capture is not None and self.mode != "balloons":
            self.capture.release()
        self.capture = new_capture
        self.video_path = path
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        frame_count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps > 0:
            self.frame_interval_ms = max(1, round(1000 / fps))
            self.duration_seconds = frame_count / fps
        else:
            self.duration_seconds = 0.0
        self.seek_scale.configure(
            to=max(self.duration_seconds, 0.1),
            state=tk.NORMAL if self.duration_seconds > 0 else tk.DISABLED,
        )
        self.seek_var.set(0)
        self.position_var.set("00:00")
        self.duration_var.set(self._format_time(self.duration_seconds))
        self.seeking = False
        self.replay_button.configure(state=tk.DISABLED)
        self.current_frame = None
        self.status_var.set(self._video_status("Đang phát"))

    def replay_video(self):
        if self.video_path:
            self.open_video(self.video_path)

    @staticmethod
    def _format_time(seconds):
        total_seconds = max(0, int(seconds))
        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _seek_changed(self, value):
        self.position_var.set(self._format_time(float(value)))

    def _start_seek(self, _event):
        self.seeking = True

    def _finish_seek(self, _event):
        target_seconds = self.seek_var.get()
        self.seeking = False
        if not self.video_path:
            return
        if self.capture is None:
            self.open_video(self.video_path)
        if self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_MSEC, target_seconds * 1000)
            self.current_frame = None
            self.current_faces = []
            self.replay_button.configure(state=tk.DISABLED)
            self.status_var.set(self._video_status("Đang phát"))

    def rotate(self, direction):
        self.turns_clockwise = (self.turns_clockwise + direction) % 4

    def on_key(self, event):
        key = event.keysym.lower()
        if key in ("q", "escape"):
            self.close()
            return "break"
        if self.mode == "rotate" and key == "a":
            self.rotate(-1)
            return "break"
        if self.mode == "rotate" and key == "d":
            self.rotate(1)
            return "break"
        if self.mode == "faces" and key == "s":
            self.save_faces()
            return "break"

    def _ensure_face_detector(self):
        if self.face_detector is not None or self.face_cascade is not None:
            return
        if self.detector_loading:
            return

        self.detector_loading = True
        self.status_var.set(self._video_status("Đang tải bộ nhận diện khuôn mặt"))

        def load_detector():
            try:
                if hasattr(cv2, "FaceDetectorYN_create"):
                    model_path = Path(__file__).with_name(
                        "face_detection_yunet_2023mar.onnx"
                    )
                    if not model_path.exists():
                        model_url = (
                            "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/"
                            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                        )
                        urlretrieve(model_url, model_path)
                    detector = cv2.FaceDetectorYN_create(
                        str(model_path), "", (320, 320), 0.8, 0.3, 5000
                    )
                    self.root.after(0, self._detector_ready, detector, None, None)
                else:
                    cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades
                        + "haarcascade_frontalface_default.xml"
                    )
                    if cascade.empty():
                        raise RuntimeError("Không tải được bộ nhận diện khuôn mặt.")
                    self.root.after(0, self._detector_ready, None, cascade, None)
            except Exception as error:
                try:
                    self.root.after(
                        0, self._detector_ready, None, None, str(error)
                    )
                except tk.TclError:
                    pass

        threading.Thread(target=load_detector, daemon=True).start()

    def _detector_ready(self, detector, cascade, error):
        self.detector_loading = False
        if error:
            self.status_var.set(f"Lỗi nhận diện: {error}")
            return

        self.face_detector = detector
        self.face_cascade = cascade
        if self.mode == "faces":
            self.status_var.set(self._video_status("Đang nhận diện khuôn mặt"))

    def _find_faces(self, frame):
        frame_height, frame_width = frame.shape[:2]
        if self.face_detector is not None:
            self.face_detector.setInputSize((frame_width, frame_height))
            _, detections = self.face_detector.detect(frame)
            detections = detections if detections is not None else []
            faces = [
                tuple(int(value) for value in detection[:4])
                for detection in detections
            ]
        elif self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
        else:
            return []

        bounded_faces = []
        for x, y, width, height in faces:
            left = max(0, x)
            top = max(0, y)
            right = min(frame_width, x + width)
            bottom = min(frame_height, y + height)
            if right > left and bottom > top:
                bounded_faces.append((left, top, right - left, bottom - top))
        return bounded_faces

    def _show_frame(self, frame):
        area_width = max(self.video_canvas.winfo_width(), 320)
        area_height = max(self.video_canvas.winfo_height(), 240)
        scale = min(
            area_width / frame.shape[1], area_height / frame.shape[0], 1.0
        )
        width = max(1, int(frame.shape[1] * scale))
        height = max(1, int(frame.shape[0] * scale))
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".png", resized)
        if not success:
            return

        image_data = base64.b64encode(encoded).decode("ascii")
        self.photo = tk.PhotoImage(data=image_data, format="png")
        position = (area_width // 2, area_height // 2)
        if self.image_item is None:
            self.video_canvas.delete(self.placeholder_item)
            self.image_item = self.video_canvas.create_image(
                *position, image=self.photo, anchor="center"
            )
        else:
            self.video_canvas.coords(self.image_item, *position)
            self.video_canvas.itemconfigure(self.image_item, image=self.photo)

    def update_frame(self):
        if self.closing:
            return

        frame_started = time.perf_counter()
        if self.capture is not None:
            success, frame = self.capture.read()
            if success:
                self.current_frame = frame
                self.current_faces = []
                if self.mode == "rotate":
                    display_frame = frame
                    for _ in range(self.turns_clockwise):
                        display_frame = cv2.rotate(
                            display_frame, cv2.ROTATE_90_CLOCKWISE
                        )
                else:
                    self.current_faces = self._find_faces(frame)
                    display_frame = frame.copy()
                    for x, y, width, height in self.current_faces:
                        cv2.rectangle(
                            display_frame,
                            (x, y),
                            (x + width - 1, y + height - 1),
                            (0, 255, 0),
                            2,
                        )
                    cv2.putText(
                        display_frame,
                        f"Faces: {len(self.current_faces)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                self._show_frame(display_frame)
                position_seconds = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                position_seconds = min(position_seconds, self.duration_seconds)
                if not self.seeking:
                    self.seek_var.set(position_seconds)
                    self.position_var.set(self._format_time(position_seconds))
            else:
                self.capture.release()
                self.capture = None
                self.seek_var.set(self.duration_seconds)
                self.position_var.set(self._format_time(self.duration_seconds))
                self.replay_button.configure(state=tk.NORMAL)
                self.status_var.set(self._video_status("Video đã phát hết"))

        if self.capture is None or self.mode == "balloons":
            delay_ms = 100
        else:
            elapsed_ms = (time.perf_counter() - frame_started) * 1000
            delay_ms = max(1, round(self.frame_interval_ms - elapsed_ms))
        self.root.after(delay_ms, self.update_frame)

    def save_faces(self):
        if self.current_frame is None or not self.current_faces:
            self.status_var.set(self._video_status("Không có khuôn mặt để lưu"))
            return

        saved = 0
        for index, (x, y, width, height) in enumerate(self.current_faces):
            face_image = self.current_frame[y:y + height, x:x + width]
            if cv2.imwrite(f"{index}.png", face_image):
                saved += 1
        self.status_var.set(
            self._video_status(f"Đã lưu {saved} ảnh khuôn mặt vào thư mục hiện tại")
        )

    def close(self):
        self.closing = True
        if self.capture is not None:
            self.capture.release()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Phát và xử lý video.")
    parser.add_argument("video", nargs="?", help="Đường dẫn đến file video")
    parser.add_argument(
        "--mode",
        choices=("rotate", "faces", "balloons", "plate"),
        default="rotate",
        help="Chế độ mở đầu: rotate, faces, balloons hoặc plate",
    )
    args = parser.parse_args()

    root = tk.Tk()
    VideoApp(root, args.video, args.mode)
    root.mainloop()


if __name__ == "__main__":
    main()