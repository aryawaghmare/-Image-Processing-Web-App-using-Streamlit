import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Set Streamlit app title and description
st.set_page_config(
    page_title="Advanced Image Processing Web App",
    page_icon="📷",
    layout="wide"
)

st.title("Advanced Image Processing Web App")
st.markdown(
    "Upload an image and apply advanced image processing techniques. "
    "Enhance, transform, segment, and visualize your images."
)

# File upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Sidebar options
st.sidebar.title("Image Processing Options")

# Add more image processing techniques and parameters here
selected_option = st.sidebar.selectbox(
    "Choose an image processing technique",
    ["None", "Grayscale", "Blur", "Custom Filter", "Histogram Equalization", "Image Enhancement", "Edge Detection", "Image Segmentation"]
)

# Function to process the image based on the selected option
def process_image(input_image, option, params=None):
    img = np.array(input_image)
    
    if option == "Grayscale":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif option == "Blur":
        blur_level = params["blur_level"]
        img = cv2.GaussianBlur(img, (blur_level, blur_level), 0)
    elif option == "Custom Filter":
        kernel_size = params["kernel_size"]
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        img = cv2.filter2D(img, -1, kernel)
    elif option == "Histogram Equalization":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.equalizeHist(img)
    elif option == "Image Enhancement":
        alpha = params["alpha"]
        beta = params["beta"]
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    elif option == "Edge Detection":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.Canny(img, 100, 200)
    elif option == "Image Segmentation":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    
    return img

# Process and display the image based on the selected option
if selected_option != "None" and uploaded_image:
    original_image = np.array(image)
    
    if selected_option in ["Edge Detection", "Image Segmentation"]:
        params = None  # No additional parameters needed for these techniques
    else:
        params = {}
        if selected_option == "Blur":
            params["blur_level"] = st.sidebar.slider("Select blur level", 1, 11, 3)
        elif selected_option == "Custom Filter":
            params["kernel_size"] = st.sidebar.slider("Select kernel size", 3, 11, 3)
        elif selected_option == "Image Enhancement":
            params["alpha"] = st.sidebar.slider("Select alpha (contrast)", 0.1, 3.0, 1.0)
            params["beta"] = st.sidebar.slider("Select beta (brightness)", 0, 100, 0)
    
    processed_image = process_image(original_image, selected_option, params)

    # Display side-by-side comparison of original and processed images
    col1, col2 = st.columns(2)
    col1.header("Original Image")
    col1.image(original_image, use_column_width=True, channels="RGB")
    col2.header("Processed Image")
    col2.image(processed_image, use_column_width=True, channels="RGB")

# Download the processed image
if selected_option != "None" and uploaded_image:
    st.sidebar.markdown("### Download Processed Image")
    download_button = st.sidebar.button("Download")
    if download_button:
        im = Image.fromarray(processed_image)
        im.save("processed_image.png")
        st.sidebar.success("Image Downloaded Successfully")

# Footer and credits
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Arya")
