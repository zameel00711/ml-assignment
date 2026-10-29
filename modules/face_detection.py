"""Application 1: Face Detection and Feature Extraction using OpenCV Haar Cascades."""

import io
import time
import zipfile
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from utils.sample_generator import generate_sample_portrait


@st.cache_resource
def get_haar_cascades() -> dict[str, cv2.CascadeClassifier]:
    """Loads OpenCV Haar Cascade classifiers."""
    base_dir = cv2.data.haarcascades
    return {
        "frontal_face": cv2.CascadeClassifier(base_dir + "haarcascade_frontalface_default.xml"),
        "frontal_face_alt": cv2.CascadeClassifier(base_dir + "haarcascade_frontalface_alt2.xml"),
        "eye": cv2.CascadeClassifier(base_dir + "haarcascade_eye.xml"),
        "smile": cv2.CascadeClassifier(base_dir + "haarcascade_smile.xml"),
    }


def hex_to_bgr(hex_str: str) -> tuple[int, int, int]:
    """Converts a hex color code to an OpenCV BGR tuple."""
    hex_clean = hex_str.lstrip("#")
    r = int(hex_clean[0:2], 16)
    g = int(hex_clean[2:4], 16)
    b = int(hex_clean[4:6], 16)
    return (b, g, r)


def run_vision_pipeline(
    rgb_image: np.ndarray,
    scale_factor: float = 1.12,
    min_neighbors: int = 4,
    min_size: int = 32,
    detect_eyes: bool = True,
    detect_smiles: bool = True,
    privacy_blur: bool = False,
    box_color_hex: str = "#2563EB",
    box_thickness: int = 2,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    """Runs face detection, eye and smile feature extraction, and optional privacy blur."""
    start_time = time.perf_counter()
    cascades = get_haar_cascades()

    annotated = rgb_image.copy()
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    faces = cascades["frontal_face"].detectMultiScale(
        gray_eq,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        faces = cascades["frontal_face_alt"].detectMultiScale(
            gray_eq,
            scaleFactor=scale_factor,
            minNeighbors=max(2, min_neighbors - 1),
            minSize=(min_size, min_size),
        )

    detections: list[dict[str, Any]] = []
    bgr_color = hex_to_bgr(box_color_hex)
    img_h, img_w = rgb_image.shape[:2]
    total_area = float(img_h * img_w)

    for idx, (x, y, w, h) in enumerate(faces, start=1):
        face_roi_gray = gray_eq[y : y + h, x : x + w]
        face_crop_rgb = rgb_image[y : y + h, x : x + w].copy()

        eyes_found: list[tuple[int, int, int, int]] = []
        smiles_found: list[tuple[int, int, int, int]] = []

        if detect_eyes and face_roi_gray.size > 0:
            upper_zone = face_roi_gray[0 : int(h * 0.65), :]
            raw_eyes = cascades["eye"].detectMultiScale(
                upper_zone,
                scaleFactor=1.1,
                minNeighbors=max(2, min_neighbors),
                minSize=(max(10, w // 10), max(10, h // 10)),
            )
            for ex, ey, ew, eh in raw_eyes:
                eyes_found.append((int(x + ex), int(y + ey), int(ew), int(eh)))

        if detect_smiles and face_roi_gray.size > 0:
            lower_zone = face_roi_gray[int(h * 0.50) : h, :]
            raw_smiles = cascades["smile"].detectMultiScale(
                lower_zone,
                scaleFactor=1.3,
                minNeighbors=max(7, min_neighbors * 2),
                minSize=(max(18, w // 4), max(8, h // 8)),
            )
            for sx, sy, sw, sh in raw_smiles:
                smiles_found.append((int(x + sx), int(y + int(h * 0.50) + sy), int(sw), int(sh)))

        if privacy_blur:
            ksize = max(15, (w // 6) * 2 + 1)
            blurred_roi = cv2.GaussianBlur(annotated[y : y + h, x : x + w], (ksize, ksize), 30)
            annotated[y : y + h, x : x + w] = blurred_roi

        cv2.rectangle(annotated, (x, y), (x + w, y + h), bgr_color, box_thickness)
        tag_text = f"Face {idx}"
        cv2.putText(
            annotated,
            tag_text,
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            bgr_color,
            2,
            cv2.LINE_AA,
        )

        if not privacy_blur:
            for ex, ey, ew, eh in eyes_found:
                cv2.circle(
                    annotated,
                    (ex + ew // 2, ey + eh // 2),
                    max(4, ew // 3),
                    (0, 220, 255),
                    2,
                )
            for sx, sy, sw, sh in smiles_found:
                cv2.rectangle(
                    annotated,
                    (sx, sy),
                    (sx + sw, sy + sh),
                    (200, 100, 255),
                    1,
                )

        area_pct = round(((w * h) / total_area) * 100, 2)
        detections.append(
            {
                "id": idx,
                "bbox": (int(x), int(y), int(w), int(h)),
                "crop": face_crop_rgb,
                "eye_count": len(eyes_found),
                "smiling": len(smiles_found) > 0,
                "area_pct": area_pct,
                "aspect_ratio": round(w / float(max(1, h)), 2),
            }
        )

    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    return annotated, detections, latency_ms


def return_to_home() -> None:
    """Callback to return to the home screen."""
    st.session_state["nav_selection"] = "Home"


def render_face_detection_app() -> None:
    """Renders the face detection interface."""
    col_back, _ = st.columns([1, 5])
    with col_back:
        st.button("<- Back to Home", on_click=return_to_home, key="btn_back_home_face")

    st.title("Face Detection and Feature Extraction")
    st.write(
        "Detect faces, eyes, and smiles in images using OpenCV Haar Cascade classifiers. "
        "Select parameters in the sidebar to adjust detection sensitivity."
    )

    with st.sidebar:
        st.subheader("Detector Parameters")
        scale_factor = st.slider(
            "Scale Factor",
            min_value=1.05,
            max_value=1.40,
            value=1.12,
            step=0.01,
            help="Image pyramid scaling rate per level.",
        )
        min_neighbors = st.slider(
            "Min Neighbors",
            min_value=2,
            max_value=10,
            value=4,
            help="Higher values reduce false positives.",
        )
        min_face_size = st.slider(
            "Min Face Size (pixels)",
            min_value=20,
            max_value=120,
            value=35,
            step=5,
        )

        st.divider()
        st.subheader("Options")
        box_color = st.color_picker("Bounding Box Color", value="#3B82F6")
        box_thickness = st.slider("Line Thickness", 1, 4, 2)
        detect_eyes = st.checkbox("Detect Eyes", value=True)
        detect_smiles = st.checkbox("Detect Smiles", value=True)
        privacy_blur = st.checkbox("Apply Privacy Blur (Mask Faces)", value=False)

    # Use radio selector so st.camera_input is only rendered when explicitly selected
    input_source = st.radio(
        "Choose Input Source:",
        ["Sample Images", "Upload Image", "Webcam Snapshot"],
        horizontal=True,
        key="face_input_mode",
    )

    image_to_process: np.ndarray | None = None

    if input_source == "Sample Images":
        st.write("Choose a built-in sample scene:")
        scene_choice = st.radio(
            "Sample scene:",
            ["Single Subject", "Two Subjects", "Group Scene"],
            horizontal=True,
            key="sample_scene_radio",
        )
        key_map = {
            "Single Subject": "single",
            "Two Subjects": "duo",
            "Group Scene": "crowd",
        }
        image_to_process = generate_sample_portrait(key_map[scene_choice])

    elif input_source == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="face_uploader",
        )
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            image_to_process = np.array(pil_img)

    elif input_source == "Webcam Snapshot":
        st.info("The browser will request camera permission when activating the camera below.")
        camera_snap = st.camera_input("Capture photo using webcam", key="cam_input_widget")
        if camera_snap is not None:
            pil_img = Image.open(camera_snap).convert("RGB")
            image_to_process = np.array(pil_img)

    if image_to_process is None:
        st.info("Select or upload an image to begin detection.")
        return

    annotated_img, detections, latency_ms = run_vision_pipeline(
        rgb_image=image_to_process,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size=min_face_size,
        detect_eyes=detect_eyes,
        detect_smiles=detect_smiles,
        privacy_blur=privacy_blur,
        box_color_hex=box_color,
        box_thickness=box_thickness,
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Faces Detected", len(detections))
    with kpi2:
        total_eyes = sum(d["eye_count"] for d in detections)
        st.metric("Eyes Tracked", total_eyes)
    with kpi3:
        smiling_count = sum(1 for d in detections if d["smiling"])
        st.metric("Smiles Detected", smiling_count)
    with kpi4:
        st.metric("Latency", f"{latency_ms} ms")

    st.divider()

    col_orig, col_annot = st.columns(2)
    with col_orig:
        st.subheader("Source Image")
        st.image(image_to_process, use_container_width=True)

    with col_annot:
        st.subheader("Processed Image")
        st.image(annotated_img, use_container_width=True)

        annotated_pil = Image.fromarray(annotated_img)
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        st.download_button(
            "Download Annotated Image",
            data=buf.getvalue(),
            file_name="detected_faces.png",
            mime="image/png",
            use_container_width=True,
        )

    if detections:
        st.subheader("Extracted Face Crops")
        crop_cols = st.columns(min(len(detections), 6))
        for col, det in zip(crop_cols, detections):
            with col:
                st.caption(f"Face #{det['id']}")
                st.image(det["crop"], use_container_width=True)
                st.caption(f"Eyes: {det['eye_count']} | Smile: {'Yes' if det['smiling'] else 'No'}")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for det in detections:
                crop_pil = Image.fromarray(det["crop"])
                c_buf = io.BytesIO()
                crop_pil.save(c_buf, format="PNG")
                zf.writestr(f"face_{det['id']}.png", c_buf.getvalue())

        st.download_button(
            "Download All Face Crops (ZIP)",
            data=zip_buf.getvalue(),
            file_name="face_crops.zip",
            mime="application/zip",
        )

        st.subheader("Detection Telemetry")
        table_rows = []
        for d in detections:
            x, y, w, h = d["bbox"]
            table_rows.append(
                {
                    "Face ID": f"#{d['id']}",
                    "X": x,
                    "Y": y,
                    "Width": w,
                    "Height": h,
                    "Aspect Ratio": d["aspect_ratio"],
                    "Frame Area (%)": f"{d['area_pct']}%",
                    "Eyes Detected": d["eye_count"],
                    "Smile Detected": "Yes" if d["smiling"] else "No",
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    else:
        st.warning("No faces detected with current thresholds. Adjust Scale Factor or Min Neighbors.")
