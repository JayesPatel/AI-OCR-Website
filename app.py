import streamlit as st
import numpy as np
from PIL import Image
import os

# Disable oneDNN
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"

from paddleocr import PaddleOCR

st.set_page_config(
    page_title="AI OCR Website",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI OCR - Text Extractor")
st.caption("Extract text from images using PaddleOCR")

@st.cache_resource
def load_ocr():
    return PaddleOCR(
        lang="en",
        enable_mkldnn=False
    )

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Extract Text", type="primary"):
        with st.spinner("Extracting text..."):
            try:
                ocr = load_ocr()
                image_array = np.array(image)

                result = ocr.predict(image_array)

                extracted_text = []

                for page in result:
                    page_data = page.json
                    if callable(page_data):
                        page_data = page_data()

                    if isinstance(page_data, dict):
                        page_data = page_data.get("res", page_data)

                    texts = page_data.get("rec_texts", [])
                    extracted_text.extend(texts)

                if extracted_text:
                    st.success("Text extracted successfully!")

                    final_text = "\n".join(extracted_text)

                    st.text_area(
                        "Extracted Text",
                        value=final_text,
                        height=250
                    )

                    st.download_button(
                        "Download Text",
                        data=final_text,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("No text detected in this image.")

            except Exception as e:
                st.error(f"OCR Error: {e}")
