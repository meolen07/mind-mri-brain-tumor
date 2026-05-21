"""MIND — MRI-based Intelligent Neural Detection (Streamlit demo)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.inference import CLASS_NAMES, load_model, predict

st.set_page_config(
    page_title="MIND",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best_model.pth"
DEVICE = "cpu"

AUTHOR_NAME = "Huynh Mai Linh Nguyen"
YEAR = "2026"

WEIGHTS_MISSING_MSG = (
    "Model weights not found. Please place best_model.pth inside the weights/ folder."
)
MEDICAL_DISCLAIMER = (
    "For educational and research use only. "
    "Not a substitute for professional medical diagnosis or clinical judgment."
)


@st.cache_resource(show_spinner="Loading model...")
def get_model(weights_path: str):
    return load_model(weights_path=weights_path, device=DEVICE)


def probabilities_dataframe(prob_dict: dict) -> pd.DataFrame:
    rows = [{"Class": name, "Probability": prob_dict[name]} for name in CLASS_NAMES]
    return pd.DataFrame(rows).sort_values("Probability", ascending=False).reset_index(drop=True)


def render_author_footer() -> None:
    st.divider()
    st.caption(f"Author: {AUTHOR_NAME}")
    st.caption(f"Copyright {YEAR}")


def main() -> None:
    st.title("MIND — MRI-based Intelligent Neural Detection")
    st.write(
        "Educational brain MRI tumor classification "
        "(Glioma, Meningioma, No Tumor, Pituitary Tumor)."
    )

    weights_available = WEIGHTS_PATH.is_file()
    if not weights_available:
        st.error(WEIGHTS_MISSING_MSG)

    uploaded = st.file_uploader(
        "Upload brain MRI image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        st.info("Upload an image to run classification.")
        st.caption(MEDICAL_DISCLAIMER)
        render_author_footer()
        return

    image = Image.open(uploaded).convert("RGB")
    col_img, col_result = st.columns(2)

    with col_img:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)
        st.caption(uploaded.name)

    with col_result:
        st.subheader("Prediction")
        if not weights_available:
            st.warning("Classification disabled until model weights are available.")
        else:
            try:
                class_name, confidence, prob_dict = predict(
                    get_model(str(WEIGHTS_PATH)), image, device=DEVICE
                )
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                st.caption(MEDICAL_DISCLAIMER)
                render_author_footer()
                return

            st.metric("Predicted class", class_name)
            st.metric("Confidence", f"{confidence:.1%}")
            st.progress(float(confidence))

            st.subheader("Class probabilities")
            prob_df = probabilities_dataframe(prob_dict)
            chart_df = prob_df.set_index("Class")[["Probability"]]
            st.bar_chart(chart_df)
            prob_df["Percentage"] = prob_df["Probability"].map(lambda p: f"{p:.1%}")
            st.dataframe(
                prob_df[["Class", "Probability", "Percentage"]],
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Interpretation")
    st.markdown(
        "- The predicted class has the highest probability.\n"
        "- Low confidence may indicate an unclear image or model uncertainty."
    )
    st.caption("Grad-CAM: coming soon")
    st.caption(MEDICAL_DISCLAIMER)
    render_author_footer()


if __name__ == "__main__":
    main()
