    import streamlit as st
import cv2
import time
import tempfile
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO
import matplotlib.pyplot as plt
import ffmpeg

st.set_page_config(page_title="YOLO11 vs YOLO12 Vehicle Counting", layout="wide")
st.title("YOLO11 vs YOLO12: Vehicle Detection & Counting")

# Video re-encoding for browser playback
def reencode_video(input_path, output_path):
    try:
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, output_path, vcodec='libx264', acodec='aac', format='mp4', pix_fmt='yuv420p')
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        return True
    except ffmpeg.Error:
        return False

# Process video using YOLO
def process_video(video_path, model_name, output_path, class_filter):
    try:
        model = YOLO(f"{model_name}.pt")
    except FileNotFoundError:
        st.error(f"Model {model_name}.pt not found.")
        return None, None, None

    class_indices = [i for i, name in model.names.items() if name in class_filter]
    class_list = model.names

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Could not open video for {model_name}.")
        return None, None, None

    w, h = int(cap.get(3)), int(cap.get(4))
    line_y = 430
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))

    count_class = defaultdict(int)
    crossed_ids = {}
    frame_num = 0
    fps_list = []
    inf_times = []
    prev_time = time.time()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    processed = 0

    results = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        if frame_num % 4 == 0:
            start = time.time()
            results = model.track(frame, persist=True, verbose=False, classes=class_indices)
            inf_times.append((time.time() - start) * 1000)

        cv2.line(frame, (550, line_y), (1100, line_y), (0, 255, 255), 3)

        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu()
            ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, class_idx in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                name = class_list[class_idx]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id} {name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                if track_id not in crossed_ids and cy > line_y:
                    crossed_ids[track_id] = True
                    count_class[name] += 1

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time
        fps_list.append(fps)
        cv2.putText(frame, f"FPS: {int(fps)}", (600, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        y_offset = 30
        for cls, count in count_class.items():
            cv2.putText(frame, f"{cls}: {count}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(frame, f"{cls}: {count}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_offset += 30

        writer.write(frame)
        processed += 1
        progress.progress(min(processed / total, 1.0))

    cap.release()
    writer.release()

    if os.path.exists(temp_out):
        if reencode_video(temp_out, output_path):
            os.unlink(temp_out)
        else:
            os.unlink(temp_out)
            return None, None, None

    return count_class, round(np.mean(fps_list), 2), round(np.mean(inf_times), 2)

# File upload
uploaded_file = st.file_uploader(" Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    output_y11 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    output_y12 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    classes = ["car", "motorcycle", "bus"]

    st.write("Processing with YOLO11...")
    counts11, fps11, inf11 = process_video(input_path, "yolo11l", output_y11, classes)

    st.write("Processing with YOLO12...")
    counts12, fps12, inf12 = process_video(input_path, "yolo12l", output_y12, classes)

    # Video results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("YOLO11 Output")
        if os.path.exists(output_y11):
            st.video(output_y11)
    with col2:
        st.subheader("YOLO12 Output")
        if os.path.exists(output_y12):
            st.video(output_y12)

    if counts11 and counts12:
        st.subheader("Performance Metrics")
        metrics = pd.DataFrame({
            "Model": ["YOLO11", "YOLO12"],
            "Average FPS": [fps11, fps12],
            "Inference Time (ms)": [inf11, inf12]
        })
        st.table(metrics.style.format({
            "Average FPS": "{:.2f}",
            "Inference Time (ms)": "{:.2f}"
        }).set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold')]}]))

        st.subheader("Vehicle Counts")
        count_df = pd.DataFrame({
            "Class": classes,
            "YOLO11": [counts11.get(cls, 0) for cls in classes],
            "YOLO12": [counts12.get(cls, 0) for cls in classes]
        })
        st.table(count_df.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold')]}]))

        # Bar chart
        fig, ax = plt.subplots()
        count_df.set_index("Class")[["YOLO11", "YOLO12"]].plot(kind="bar", ax=ax)
        ax.set_title("Vehicle Counts by Class")
        ax.set_ylabel("Count")
        st.pyplot(fig)

# Clean up
for f in ['input_path', 'output_y11', 'output_y12']:
    file_var = globals().get(f)
    if file_var and os.path.exists(file_var):
        try:
            os.unlink(file_var)
        except Exception as e:
            st.warning(f"Failed to delete {f}: {e}")
