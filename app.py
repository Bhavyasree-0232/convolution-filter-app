import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from datetime import datetime

st.set_page_config(page_title="🧠 Custom Convolution Filter App", layout="wide")

st.title("🧠 Custom Convolution Filter App")
st.markdown("Upload an image and apply custom filters like Edge Detection, Blur, Emboss, and more.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Filter options
filter_options = [
    "Sobel Edge Detection", 
    "Sharpen", 
    "Blur",
    "Emboss",
    "Outline",
    "Custom 3x3 Kernel"
]

# Filter dropdown
selected_filter = st.selectbox("Choose a filter", filter_options)

# Optional custom kernel input
custom_kernel = None
if selected_filter == "Custom 3x3 Kernel":
    st.markdown("### Enter Custom 3x3 Kernel")
    custom_kernel = []
    cols = st.columns(3)
    for i in range(3):
        row = []
        for j in range(3):
            val = cols[j].number_input(f"({i+1},{j+1})", value=0.0, key=f"k{i}{j}")
            row.append(val)
        custom_kernel.append(row)
    custom_kernel = np.array(custom_kernel, dtype=np.float32)

# Process image
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define preset kernels
    kernels = {
        "Sobel Edge Detection": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "Blur": np.ones((3, 3), np.float32) / 9.0,
        "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
        "Outline": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    }

    # Choose kernel
    kernel = custom_kernel if selected_filter == "Custom 3x3 Kernel" else kernels[selected_filter]

    # Apply filter
    filtered = cv2.filter2D(src=image_rgb, ddepth=-1, kernel=kernel)

    # Show side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Original Image", use_container_width=True)
    with col2:
        st.image(filtered, caption=f"Filtered Image - {selected_filter}", use_container_width=True)

    # Download Button
    st.markdown("### 📥 Download Filtered Image")
    pil_img = Image.fromarray(filtered)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    st.download_button("Download as PNG", data=byte_im, file_name=f"filtered_{timestamp}.png", mime="image/png")
