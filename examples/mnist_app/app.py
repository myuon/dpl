"""Streamlit web app for MNIST digit recognition."""

import glob

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from dpl import as_variable, no_grad
import dpl.functions as F

from model import load_model

st.set_page_config(page_title="MNIST Digit Recognition", layout="wide")

st.title("MNIST Digit Recognition")


def find_model_files():
    """Find all npz files in the current directory (excluding hard_set.npz)."""
    npz_files = glob.glob("*.npz")
    # Exclude hard_set.npz
    model_files = [f for f in npz_files if f != "hard_set.npz"]
    return sorted(model_files)


@st.cache_resource
def get_model(weights_path: str):
    """Load the model (cached)."""
    try:
        model = load_model(weights_path)
        return model, None
    except FileNotFoundError:
        return None, f"Model weights not found: {weights_path}"


@st.cache_data
def load_hard_set():
    """Load the hard set (cached)."""
    try:
        data = np.load("hard_set.npz", allow_pickle=True)
        return data["X"], data["y"], None
    except FileNotFoundError:
        return None, None, "Hard set not found. Run collect_hard_set.py first."


# Model selector
model_files = find_model_files()

if not model_files:
    st.error("No model weights found in the current directory.")
    st.info("Run `uv run python train.py` to train a model first.")
    st.stop()

# Default to mnist_cnn_weights.npz if available, otherwise use the first file
default_index = 0
if "mnist_cnn_weights.npz" in model_files:
    default_index = model_files.index("mnist_cnn_weights.npz")

selected_model = st.selectbox(
    "Select model weights",
    model_files,
    index=default_index,
    key="model_selector",
)

model, error = get_model(selected_model)

if error:
    st.error(error)
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["Draw", "Hard Set"])

with tab1:
    st.write("Draw a digit (0-9) on the canvas and see the model's prediction!")

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

with tab2:
    st.write("View hard cases - samples the model struggles with.")

    hard_X, hard_y, hard_error = load_hard_set()

    if hard_error:
        st.warning(hard_error)
        st.info("Run `uv run python collect_hard_set.py` to create the hard set.")
    else:
        # Evaluate on hard set
        @st.cache_data
        def evaluate_hard_set(_model, X, y, model_name: str):
            """Evaluate model on hard set."""
            all_preds = []
            all_confs = []
            batch_size = 100

            with no_grad():
                for i in range(0, len(X), batch_size):
                    X_batch = X[i:i + batch_size]
                    x = as_variable(X_batch)
                    logits = _model(x)
                    probs = F.softmax(logits).data_required

                    all_preds.append(probs.argmax(axis=1))
                    all_confs.append(probs.max(axis=1))

            pred = np.concatenate(all_preds)
            conf = np.concatenate(all_confs)
            wrong_mask = pred != y

            return {
                "acc": float((pred == y).mean()),
                "mean_conf": float(conf.mean()),
                "wrong_conf": float(conf[wrong_mask].mean()) if wrong_mask.any() else 0.0,
                "num_correct": int((pred == y).sum()),
                "num_wrong": int(wrong_mask.sum()),
            }

        results = evaluate_hard_set(model, hard_X, hard_y, selected_model)

        # Display evaluation metrics
        st.write(f"**Total hard cases:** {len(hard_X)}")

        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Accuracy", f"{results['acc']:.1%}")
        with metric_cols[1]:
            st.metric("Mean Confidence", f"{results['mean_conf']:.1%}")
        with metric_cols[2]:
            st.metric("Wrong Pred Conf", f"{results['wrong_conf']:.1%}")
        with metric_cols[3]:
            st.metric("Correct / Wrong", f"{results['num_correct']} / {results['num_wrong']}")

        # Initialize session state for random indices
        if "hard_set_indices" not in st.session_state:
            st.session_state.hard_set_indices = np.random.choice(
                len(hard_X), size=min(10, len(hard_X)), replace=False
            )

        # Shuffle button
        if st.button("Shuffle", type="primary", key="shuffle_hard"):
            st.session_state.hard_set_indices = np.random.choice(
                len(hard_X), size=min(10, len(hard_X)), replace=False
            )
            st.rerun()

        # Display samples in a grid
        indices = st.session_state.hard_set_indices
        cols = st.columns(5)

        for i, idx in enumerate(indices):
            col = cols[i % 5]

            with col:
                # Get image and label
                img = hard_X[idx]
                if img.ndim == 3:  # (1, 28, 28)
                    img = img[0]
                true_label = int(hard_y[idx])

                # Run inference
                input_data = img.reshape(1, 1, 28, 28)
                with no_grad():
                    x = as_variable(input_data)
                    logits = model(x)
                    probs = F.softmax(logits)
                    probs_data = probs.data_required[0]

                pred = int(np.argmax(probs_data))
                confidence = float(probs_data[pred])

                # Display
                st.image(img, width=100, clamp=True)
                if pred == true_label:
                    st.success(f"True: {true_label}, Pred: {pred} ({confidence:.0%})")
                else:
                    st.error(f"True: {true_label}, Pred: {pred} ({confidence:.0%})")
