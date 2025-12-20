"""Streamlit web app for MNIST digit recognition."""

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from dpl import as_variable, no_grad
import dpl.functions as F

from model import load_model

st.set_page_config(page_title="MNIST Digit Recognition", layout="wide")

st.title("MNIST Digit Recognition")
st.write("Draw a digit (0-9) on the canvas and see the model's prediction!")


@st.cache_resource
def get_model():
    """Load the model (cached)."""
    try:
        model = load_model("mnist_cnn_weights.npz")
        return model, None
    except FileNotFoundError:
        return None, "Model weights not found. Please run train.py first."


model, error = get_model()

if error:
    st.error(error)
    st.info("Run `uv run python train.py` to train the model first.")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw here")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Clear", type="primary"):
        st.rerun()

with col2:
    st.subheader("Prediction")

    if canvas_result.image_data is not None:
        # Get the drawn image
        img_data = canvas_result.image_data

        # Convert to grayscale
        img = Image.fromarray(img_data.astype("uint8"), "RGBA")
        img = img.convert("L")

        # Resize to 28x28 (MNIST size)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Check if canvas has any drawing
        if img_array.max() > 0.1:
            # Reshape for CNN: (1, 1, 28, 28)
            input_data = img_array.reshape(1, 1, 28, 28)

            # Run inference
            with no_grad():
                x = as_variable(input_data)
                logits = model(x)
                probs = F.softmax(logits)
                probs_data = probs.data_required[0]

            # Get prediction
            pred = int(np.argmax(probs_data))
            confidence = float(probs_data[pred])

            # Display result
            st.metric("Predicted Digit", pred, f"{confidence:.1%} confidence")

            # Show probability distribution
            st.write("**Probability distribution:**")

            # Create a horizontal bar chart
            for digit in range(10):
                prob = float(probs_data[digit])
                bar_color = "🟩" if digit == pred else "⬜"
                bar_length = int(prob * 20)
                bar = bar_color * bar_length
                st.write(f"`{digit}`: {bar} {prob:.1%}")

            # Show preprocessed image
            st.write("**Preprocessed input (28x28):**")
            st.image(img_array, width=140, clamp=True)
        else:
            st.info("Draw a digit on the canvas to see predictions.")
    else:
        st.info("Draw a digit on the canvas to see predictions.")
