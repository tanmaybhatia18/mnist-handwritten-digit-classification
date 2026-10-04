from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from scipy import ndimage
from streamlit_drawable_canvas import st_canvas

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mnist_model.pkl"


@st.cache_resource
def load_model():
    """Load the model once and reuse it across reruns."""
    return joblib.load(MODEL_PATH)


def preprocess(img: Image.Image):
    """Convert a white-digit-on-black PIL image into MNIST format.

    Returns a (28, 28) float array in [0, 1], or None if the image is blank.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)       # 1. grayscale

    ys, xs = np.where(gray > 20)                              # 2. find the ink
    if len(ys) == 0:
        return None
    digit = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = digit.shape                                        # 3. fit into 20x20,
    scale = 20.0 / max(h, w)                                  #    keep aspect ratio
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    digit = Image.fromarray(digit.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)

    canvas = np.zeros((28, 28), dtype=np.float32)             # 4. paste into 28x28
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = np.array(digit)

    cy, cx = ndimage.center_of_mass(canvas)                   # 5. center by mass
    canvas = ndimage.shift(canvas, (14 - cy, 14 - cx), order=1, mode="constant")

    return np.clip(canvas, 0, 255) / 255.0                    # 6. normalize to 0-1


def predict(model, img28):
    """Flatten to (1, 784) and predict. Returns (digit, probabilities or None)."""
    x = img28.reshape(1, -1).astype("float32")
    digit = int(model.predict(x)[0])
    probs = model.predict_proba(x)[0] if hasattr(model, "predict_proba") else None
    return digit, probs


def show_result(model, img28):
    if img28 is None:
        st.info("Nothing detected yet. Draw or upload a digit.")
        return
    digit, probs = predict(model, img28)
    col1, col2 = st.columns(2)
    with col1:
        st.write("What the model sees (28×28):")
        st.image(img28, width=140, clamp=True)
    with col2:
        st.metric("Predicted digit", digit)
        if probs is not None:
            st.write(f"Confidence: {probs.max():.1%}")
    if probs is not None:
        st.bar_chart(probs)


st.title("Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) or upload an image, and the model will predict it.")

model = load_model()
tab_draw, tab_upload = st.tabs(["Draw", "Upload"])

with tab_draw:
    canvas = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        return_image_data=True,
    )
    if canvas.image_data is not None:
        pil_img = Image.fromarray(canvas.image_data.astype("uint8"))
        show_result(model, preprocess(pil_img))

with tab_upload:
    invert = st.checkbox("My image has a dark digit on a light background (invert it)", value=True)
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("L")
        if invert:
            pil_img = ImageOps.invert(pil_img)
        st.image(uploaded, caption="Your upload", width=140)
        show_result(model, preprocess(pil_img))