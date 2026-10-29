"""Main entrypoint for the Machine Learning and Computer Vision application."""

import streamlit as st

from modules.bike_sharing_ml import render_bike_sharing_app
from modules.face_detection import render_face_detection_app

st.set_page_config(
    page_title="Machine Learning Applications",
    layout="wide",
    initial_sidebar_state="expanded",
)


def navigate_to(page_name: str) -> None:
    """Updates active page in session state matching the sidebar radio key."""
    st.session_state["nav_selection"] = page_name


def render_home() -> None:
    """Renders a simple, clean overview of both assignment applications."""
    st.title("Machine Learning and Computer Vision Applications")
    st.caption("Select an application from the options below or using the sidebar.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("1. Face Detection Application")
            st.write(
                "Detect faces, eyes, and smiles in images using OpenCV Haar Cascades. "
                "Supports device camera capture, file upload, privacy blur masking, "
                "and bounding box telemetry."
            )
            st.write("**Key Capabilities:**")
            st.write("- Multi-scale face detection with fallback cascade")
            st.write("- Eye and smile feature extraction")
            st.write("- Optional face blur for privacy protection")
            st.write("- Downloadable annotated images and cropped face archive")
            st.button(
                "Open Face Detection Application",
                on_click=navigate_to,
                args=("1. Face Detection Application",),
                type="primary",
                use_container_width=True,
                key="btn_home_face",
            )

    with col2:
        with st.container(border=True):
            st.subheader("2. Bike Sharing Demand Prediction")
            st.write(
                "An end-to-end regression machine learning workflow using the UCI Bike Sharing "
                "Dataset to forecast hourly rental demand based on environmental and calendar factors."
            )
            st.write("**Key Capabilities:**")
            st.write("- Multi-model comparison across 5 regression algorithms")
            st.write("- Stratified 5-fold cross-validation with automatic model selection")
            st.write("- Exportable evaluation plots (actual vs predicted, residuals, importance)")
            st.write("- Formal A4 PDF evaluation report generation")
            st.write("- Real-time demand simulation and operational recommendations")
            st.button(
                "Open Bike Sharing Prediction",
                on_click=navigate_to,
                args=("2. Bike Sharing Demand Prediction",),
                type="primary",
                use_container_width=True,
                key="btn_home_bike",
            )

    st.divider()
    st.caption("Author: Zameel  |  Environment: Python, Streamlit, Scikit-Learn, OpenCV Headless")


def main() -> None:
    """Top-level navigation router."""
    options = [
        "Home",
        "1. Face Detection Application",
        "2. Bike Sharing Demand Prediction",
    ]

    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = "Home"

    with st.sidebar:
        st.subheader("Navigation")
        st.radio(
            "Go to page:",
            options=options,
            key="nav_selection",
        )
        st.divider()
        st.caption("Project Details:")
        st.caption("Author: Zameel")
        st.caption("Framework: Streamlit")

    active_page = st.session_state.get("nav_selection", "Home")

    if active_page == "1. Face Detection Application":
        render_face_detection_app()
    elif active_page == "2. Bike Sharing Demand Prediction":
        render_bike_sharing_app()
    else:
        render_home()


if __name__ == "__main__":
    main()
