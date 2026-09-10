import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Virtual Fundus Camera",
    page_icon="👁️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("👁️ Virtual Fundus Camera")

st.caption(
    "Phase 1 – Fundus Image Capture & Processing"
)

st.warning(
    "Educational / research prototype only. "
    "This application is not intended for medical diagnosis."
)


# ============================================================
# SESSION STATE
# ============================================================

if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

if "capture_count" not in st.session_state:
    st.session_state.capture_count = 0


# ============================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================

def decode_image(image_bytes):

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Unable to read the captured image."
        )

    return image


def enhance_image(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge(
        (l, a, b)
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


def circular_crop(image):

    h, w = image.shape[:2]

    center_x = w // 2
    center_y = h // 2

    radius = int(
        min(h, w) * 0.45
    )

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    cv2.circle(
        mask,
        (center_x, center_y),
        radius,
        255,
        -1
    )

    result = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    return result


def calculate_quality(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    brightness = float(
        np.mean(gray)
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    if brightness < 40:

        brightness_status = "Too Dark"

    elif brightness > 220:

        brightness_status = "Too Bright"

    else:

        brightness_status = "Good"

    if sharpness < 30:

        focus_status = "Low Focus"

    else:

        focus_status = "Good"

    if (
        brightness_status == "Good"
        and focus_status == "Good"
    ):

        overall = "GOOD"

    else:

        overall = "RETAKE"

    return (
        brightness,
        sharpness,
        brightness_status,
        focus_status,
        overall
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Image Processing")

enable_enhancement = st.sidebar.checkbox(
    "Enable Enhancement",
    value=True
)

enable_circular_crop = st.sidebar.checkbox(
    "Circular Fundus Crop",
    value=False
)


# ============================================================
# CAPTURE SECTION
# ============================================================

st.header("📷 Camera")

st.write(
    "Click **Capture New Image** to open your "
    "mobile or computer camera."
)


# ------------------------------------------------------------
# Capture New Image
# ------------------------------------------------------------

capture = st.camera_input(
    "Capture New Image",
    key=f"camera_{st.session_state.capture_count}"
)


# ============================================================
# WHEN NEW IMAGE IS CAPTURED
# ============================================================

if capture is not None:

    try:

        # Convert captured image
        image = decode_image(
            capture.getvalue()
        )

        # Store image
        st.session_state.captured_image = image

        st.session_state.capture_count += 1

    except Exception as e:

        st.error(
            f"Camera error: {e}"
        )


# ============================================================
# DISPLAY CAPTURED IMAGE
# ============================================================

if st.session_state.captured_image is not None:

    image = st.session_state.captured_image

    st.divider()

    st.header("📸 Captured Fundus Image")

    col1, col2 = st.columns(2)


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    with col1:

        st.subheader(
            "Original Capture"
        )

        original_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            original_rgb,
            use_container_width=True
        )


    # ========================================================
    # PROCESS IMAGE
    # ========================================================

    processed = image.copy()


    if enable_circular_crop:

        processed = circular_crop(
            processed
        )


    if enable_enhancement:

        processed = enhance_image(
            processed
        )


    # ========================================================
    # PROCESSED IMAGE
    # ========================================================

    with col2:

        st.subheader(
            "Processed Image"
        )

        processed_rgb = cv2.cvtColor(
            processed,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            processed_rgb,
            use_container_width=True
        )


    # ========================================================
    # IMAGE QUALITY
    # ========================================================

    st.divider()

    st.header(
        "🔍 Image Quality"
    )

    (
        brightness,
        sharpness,
        brightness_status,
        focus_status,
        overall
    ) = calculate_quality(
        processed
    )


    q1, q2, q3, q4 = st.columns(4)


    with q1:

        st.metric(
            "Brightness",
            f"{brightness:.1f}"
        )


    with q2:

        st.metric(
            "Sharpness",
            f"{sharpness:.1f}"
        )


    with q3:

        st.metric(
            "Brightness Status",
            brightness_status
        )


    with q4:

        st.metric(
            "Overall",
            overall
        )


    if overall == "GOOD":

        st.success(
            "✓ Image quality is acceptable."
        )

    else:

        st.warning(
            "⚠️ Image should be captured again."
        )


    # ========================================================
    # RETAKE BUTTON
    # ========================================================

    st.divider()

    st.header(
        "🔄 Capture Again"
    )

    if st.button(
        "🔄 Retake Image",
        use_container_width=True
    ):

        st.session_state.captured_image = None

        st.session_state.capture_count += 1

        st.rerun()


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    st.divider()

    st.header(
        "💾 Save Image"
    )


    success, encoded = cv2.imencode(
        ".jpg",
        processed
    )


    if success:

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            f"fundus_{timestamp}.jpg"
        )

        st.download_button(
            label="⬇️ Download Fundus Image",
            data=encoded.tobytes(),
            file_name=filename,
            mime="image/jpeg",
            use_container_width=True
        )


# ============================================================
# NO IMAGE YET
# ============================================================

else:

    st.info(
        "📷 No image captured yet. "
        "Click **Capture New Image** above to start."
    )


# ============================================================
# FUTURE AI SECTION
# ============================================================

st.divider()

st.header(
    "🤖 AI Analysis – Coming in Phase 2"
)

st.write(
    """
    After the camera capture and image-quality pipeline
    is working correctly, we can connect a deep-learning
    model for retinal image analysis.
    """
)


st.markdown(
    """
    **Planned pipeline**

    Camera
    → Fundus Image
    → Image Quality Check
    → OpenCV Processing
    → Deep Learning Model
    → Research Result
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Virtual Fundus Camera V1 | "
    "Educational / Research Prototype"
)
