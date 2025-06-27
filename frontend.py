import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="ML / DL Prediction App", layout="wide")

# ---------- Title ----------
st.title("🤖 ML / DL Prediction Interface")

# ---------- Two Side-by-Side Containers ----------
col1, col2 = st.columns(2)

# ---------- Iris Prediction Form ----------
with col1:
    st.subheader("🌸 Iris Species Prediction (ML MODEL)")
    with st.form("iris_form"):
        sl = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
        sw = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
        pl = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
        pw = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")
        iris_submit = st.form_submit_button("Predict Iris")

    if iris_submit:
        iris_payload = {
            "sepal_length": sl,
            "sepal_width": sw,
            "petal_length": pl,
            "petal_width": pw
        }

        try:
            res = requests.post("http://127.0.0.1:8000/predict_iris", json=iris_payload)
            if res.status_code == 200:
                prediction = res.json()["predicted_category"]
                st.success(f"🌿 Predicted Iris Species: `{prediction}`")
            else:
                st.error("❌ Error in prediction")
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Could not connect to FastAPI server.")

# ---------- Digit Prediction Form ----------
with col2:
    st.subheader("🔢 Digit Recognition (DL MODEL)")
    uploaded_file = st.file_uploader("Upload a 28x28 grayscale digit image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, width=150, caption="Uploaded Image")
        if st.button("Predict Digit"):
            try:
                files = {"file": uploaded_file.getvalue()}
                res = requests.post("http://127.0.0.1:8000/predict_digit", files={"file": uploaded_file})
                if res.status_code == 200:
                    prediction = res.json()["predicted_category"]
                    st.success(f"🔢 Predicted Digit: `{prediction}`")
                else:
                    st.error("❌ Error in digit prediction")
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Could not connect to FastAPI server.")

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    Created with ❤️ by <strong>Muhammad Hamza</strong> | © 2025<br>
    Backend: FastAPI | Frontend: Streamlit | Models: Scikit-learn + TensorFlow
</div>
""", unsafe_allow_html=True)
