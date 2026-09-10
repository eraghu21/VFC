import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Virtual Fundus Camera",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Virtual Fundus Camera")
st.caption("Phase 1 — Virtual retinal imaging simulator")

st.warning(
    "Educational/research simulation only. "
    "This is not a medical diagnostic system."
)


# =========================================================
# CREATE SIMULATED RETINA
# =========================================================

def create_retina(
    size=700,
    illumination=1.0,
    blur=0.0,
    exposure=1.0,
    vessel_strength=1.0
):

    img = np.zeros(
        (size, size, 3),
        dtype=np.float32
    )

    y, x = np.mgrid[0:size, 0:size]

    cx = size // 2
    cy = size // 2

    # -----------------------------------------------------
    # Circular fundus field
    # -----------------------------------------------------

    distance = np.sqrt(
        (x - cx) ** 2 +
        (y - cy) ** 2
    )

    radius = size * 0.43

    mask = distance < radius

    # -----------------------------------------------------
    # Basic retinal background
    # -----------------------------------------------------

    radial = 1 - distance / radius
    radial = np.clip(radial, 0, 1)

    base = (
        75
        + 80 * radial
    )

    # Slight natural variation
    noise = np.random.normal(
        0,
        5,
        (size, size)
    )

    red = base + noise
    green = base * 0.45 + noise
    blue = base * 0.25 + noise

    img[:, :, 0] = red
    img[:, :, 1] = green
    img[:, :, 2] = blue

    # -----------------------------------------------------
    # Optic disc
    # -----------------------------------------------------

    optic_x = int(cx + size * 0.20)
    optic_y = int(cy - size * 0.02)

    cv2.ellipse(
        img,
        (optic_x, optic_y),
        (
            int(size * 0.07),
            int(size * 0.10)
        ),
        -15,
        0,
        360,
        (210, 145, 120),
        -1
    )

    # -----------------------------------------------------
    # Macula
    # -----------------------------------------------------

    macula_x = int(cx - size * 0.13)
    macula_y = int(cy)

    cv2.circle(
        img,
        (macula_x, macula_y),
        int(size * 0.035),
        (45, 25, 20),
        -1
    )

    # -----------------------------------------------------
    # Blood vessels
    # -----------------------------------------------------

    vessel_img = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    vessel_color = int(
        120 * vessel_strength
    )

    # Main vessels
    directions = [
        (-1, -0.35),
        (-1, 0.35),
        (-0.75, -0.65),
        (-0.75, 0.65),
        (-0.45, -0.85),
        (-0.45, 0.85),
    ]

    for dx, dy in directions:

        points = []

        start_x = optic_x
        start_y = optic_y

        for i in range(15):

            t = i / 14

            px = int(
                start_x +
                dx * size * 0.42 * t
            )

            py = int(
                start_y +
                dy * size * 0.42 * t
            )

            # curved vessel
            py += int(
                np.sin(t * 8) *
                size * 0.025
            )

            points.append(
                (px, py)
            )

        pts = np.array(
            points,
            dtype=np.int32
        )

        cv2.polylines(
            vessel_img,
            [pts],
            False,
            vessel_color,
            max(2, int(size * 0.006))
        )

        # Smaller branches
        for j in range(2, 5):

            idx = min(
                j * 3,
                len(points) - 1
            )

            sx, sy = points[idx]

            ex = sx + int(
                dx * size * 0.18
            )

            ey = sy + int(
                dy * size * 0.18
            )

            cv2.line(
                vessel_img,
                (sx, sy),
                (ex, ey),
                vessel_color,
                max(1, int(size * 0.003))
            )

    vessel_img = cv2.GaussianBlur(
        vessel_img,
        (5, 5),
        0
    )

    vessel_effect = (
        vessel_img.astype(np.float32)
        * vessel_strength
    )

    img[:, :, 0] -= vessel_effect * 0.40
    img[:, :, 1] -= vessel_effect * 0.25
    img[:, :, 2] -= vessel_effect * 0.20

    # -----------------------------------------------------
    # Camera illumination falloff
    # -----------------------------------------------------

    illumination_map = (
        0.65 +
        0.35 * radial
    )

    img *= illumination_map[:, :, None]

    # -----------------------------------------------------
    # Exposure
    # -----------------------------------------------------

    img *= exposure
    img *= illumination

    # -----------------------------------------------------
    # Circular mask
    # -----------------------------------------------------

    img[~mask] = 0

    img = np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

    # -----------------------------------------------------
    # Focus simulation
    # -----------------------------------------------------

    if blur > 0:

        kernel = int(
            blur * 2 + 1
        )

        if kernel % 2 == 0:
            kernel += 1

        img = cv2.GaussianBlur(
            img,
            (kernel, kernel),
            0
        )

    return img


# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.header("Virtual Camera Controls")

illumination = st.sidebar.slider(
    "Illumination",
    0.3,
    2.0,
    1.0,
    0.05
)

exposure = st.sidebar.slider(
    "Exposure",
    0.5,
    2.0,
    1.0,
    0.05
)

focus = st.sidebar.slider(
    "Focus / Blur",
    0.0,
    8.0,
    0.0,
    0.5
)

vessel_strength = st.sidebar.slider(
    "Vessel Visibility",
    0.3,
    2.0,
    1.0,
    0.1
)

image_size = st.sidebar.selectbox(
    "Resolution",
    [512, 700, 900],
    index=1
)


# =========================================================
# GENERATE IMAGE
# =========================================================

if "fundus" not in st.session_state:

    st.session_state.fundus = create_retina(
        size=image_size
    )


if st.sidebar.button(
    "🔄 Generate New Retina"
):

    st.session_state.fundus = create_retina(
        size=image_size,
        illumination=illumination,
        blur=focus,
        exposure=exposure,
        vessel_strength=vessel_strength
    )


# =========================================================
# DISPLAY
# =========================================================

image = create_retina(
    size=image_size,
    illumination=illumination,
    blur=focus,
    exposure=exposure,
    vessel_strength=vessel_strength
)


col1, col2 = st.columns(2)


with col1:

    st.subheader("Virtual Fundus View")

    st.image(
        image,
        use_container_width=True
    )


with col2:

    st.subheader("Camera Information")

    mean_brightness = np.mean(image)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    st.metric(
        "Brightness",
        f"{mean_brightness:.1f}"
    )

    st.metric(
        "Sharpness",
        f"{sharpness:.1f}"
    )

    if mean_brightness < 40:

        st.error(
            "Image too dark"
        )

    elif mean_brightness > 220:

        st.warning(
            "Image too bright"
        )

    elif sharpness < 20:

        st.warning(
            "Image appears out of focus"
        )

    else:

        st.success(
            "Image quality acceptable"
        )


# =========================================================
# CAPTURE
# =========================================================

st.divider()

st.subheader("📸 Capture Image")

if st.button(
    "Capture Fundus Image"
):

    st.session_state.captured = image

    st.success(
        "Fundus image captured"
    )


if "captured" in st.session_state:

    captured = st.session_state.captured

    st.image(
        captured,
        caption="Captured Fundus Image",
        use_container_width=True
    )

    pil_image = Image.fromarray(
        captured
    )

    buffer = BytesIO()

    pil_image.save(
        buffer,
        format="JPEG",
        quality=95
    )

    st.download_button(
        label="⬇️ Save Fundus Image",
        data=buffer.getvalue(),
        file_name="virtual_fundus.jpg",
        mime="image/jpeg"
    )


# =========================================================
# NEXT STAGE
# =========================================================

st.divider()

st.subheader(
    "🚀 Phase 2 — AI Integration"
)

st.write(
    """
    The next stage can connect this virtual fundus camera
    to a deep-learning model for retinal image quality
    assessment and research classification.
    """
)

st.info(
    "Normal → Image Quality → AI Model → Research Result"
)
