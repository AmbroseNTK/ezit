import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
from workflow_processor import process_image_with_ai_workflow

st.set_page_config(page_title="AI Image Workflow Processor", layout="centered")
st.title("🖼️ AI Image Workflow Processor")

st.markdown("""
Upload an image, describe what you want to do in natural language, and let AI build and execute an OpenCV workflow for you!
""")

# API Key input (hidden or via env)
def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter your OpenAI API Key", type="password")
    return api_key

api_key = get_api_key()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
prompt = st.text_area("Describe your image processing workflow (e.g. 'Convert to grayscale and blur the image'):")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    if image_np.ndim == 2:
        st.image(image, caption="Uploaded Image (Grayscale)", use_column_width=True)
    else:
        st.image(image, caption="Uploaded Image", use_column_width=True)
else:
    image_np = None

if st.button("Generate & Run Workflow", disabled=(uploaded_file is None or not prompt or not api_key)):
    if image_np is not None and prompt and api_key:
        with st.spinner("Generating workflow and processing image..."):
            try:
                # Convert RGBA to RGB if needed
                if image_np.shape[-1] == 4:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
                elif image_np.shape[-1] == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                result_image, workflow_json = process_image_with_ai_workflow(image_np, prompt, api_key)
                # Convert result to displayable format
                if result_image.ndim == 2:
                    st.image(result_image, caption="Result Image (Grayscale)", use_column_width=True)
                else:
                    st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption="Result Image", use_column_width=True)
                st.subheader("Generated Workflow JSON")
                st.json(workflow_json)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please upload an image, enter a prompt, and provide your OpenAI API key.")
