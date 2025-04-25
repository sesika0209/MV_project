# streamlit_app.py

import streamlit as st
import cv2
import tempfile
import time
from collections import defaultdict
from ultralytics import YOLO
import psutil
import os
import plotly.graph_objects as go
import torch

st.title("YOLO11 vs YOLO12 Object Detection")


#  Model selection
model_choice = st.sidebar.selectbox("Choose YOLO Model", ["YOLO11", "YOLO12"])
model_path = "yolo11l.pt" if model_choice == "YOLO11" else "yolo12l.pt"

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

# Metrics for performance
def process_video(model_path, input_video_path):
    model = YOLO(model_path)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.to('device') 
    class_list = model.names
    cap = cv2.VideoCapture(input_video_path)
    
    count_class = defaultdict(int)
    crossed_ids = {}
    frame_number = 0
    total_fps = 0
    total_inference_time = 0
    line_yellow = 430

    fps_list = []
    memory_list = []

    process = psutil.Process(os.getpid())

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        if frame_number % 4 != 0:
            continue

        mem_before = process.memory_info().rss / (1024 ** 2)

        start_time = time.time()
        results = model.track(frame, persist=True, verbose=False, classes=[2, 3, 5])
        end_time = time.time()

        mem_after = process.memory_info().rss / (1024 ** 2)
        mem_used = mem_after - mem_before
        memory_list.append(mem_used)

        inference_time = end_time - start_time
        fps = 1 / inference_time if inference_time > 0 else 0
        fps_list.append(fps)

        total_inference_time += inference_time
        total_fps += fps

        if results and results[0].boxes and len(results[0].boxes.xyxy) > 0:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_indices = results[0].boxes.cls.int().cpu().tolist()

            for track_id, class_idx in zip(track_ids, class_indices):
                class_name = class_list[class_idx]
                if track_id not in crossed_ids:
                    crossed_ids[track_id] = True
                    count_class[class_name] += 1

    cap.release()

    total_frames = len(fps_list)
    avg_fps = total_fps / total_frames if total_frames > 0 else 0
    avg_inference_ms = (total_inference_time / total_frames * 1000) if total_frames > 0 else 0
    avg_memory_mb = sum(memory_list) / total_frames if total_frames > 0 else 0

    return dict(count_class), round(avg_fps, 2), round(avg_inference_ms, 2), round(avg_memory_mb, 2), fps_list, memory_list

# Run analysis
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(uploaded_file.read())
        tmp_path = tmpfile.name

    st.info(f"Processing video using {model_choice}...")
    with st.spinner("Running detection..."):
        class_counts, avg_fps, avg_inference_ms, avg_memory_mb, fps_list, memory_list = process_video(model_path, tmp_path)

    st.success("Detection complete.")
    st.subheader(" Performance Summary")
    st.markdown(f"**Model:** {model_choice}")
    st.markdown(f"**Average FPS:** {avg_fps}")
    st.markdown(f"**Average Inference Time:** {avg_inference_ms} ms/frame")
    st.markdown(f"**Average Memory Usage:** {avg_memory_mb} MB/frame")

    st.subheader(" Inference Time & Memory Usage Graphs")

    fig_fps = go.Figure()
    fig_fps.add_trace(go.Scatter(y=fps_list, mode='lines+markers', name='FPS/frame'))
    fig_fps.update_layout(title='Frame-wise FPS', xaxis_title='Frame', yaxis_title='FPS')

    fig_mem = go.Figure()
    fig_mem.add_trace(go.Scatter(y=memory_list, mode='lines+markers', name='Memory (MB)'))
    fig_mem.update_layout(title='Frame-wise Memory Usage', xaxis_title='Frame', yaxis_title='Memory (MB)')

    st.plotly_chart(fig_fps, use_container_width=True)
    st.plotly_chart(fig_mem, use_container_width=True)

    st.subheader(" Detected Object Counts")
    st.json(class_counts)
