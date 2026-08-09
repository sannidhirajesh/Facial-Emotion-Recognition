import streamlit as st
import cv2
import numpy as np
import torch

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊"
)

st.title("Facial Emotion Recognition")
st.write("Upload a clear image and the AI will detect the facial emotion.")


MODEL_NAME = "dima806/facial_emotions_image_detection"


@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


processor, model = load_model()


face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png","jfif"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:

        st.error(
            "No face detected. Please upload a clear, "
            "front-facing image."
        )

    else:

        # Select the largest face
        x, y, w, h = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        # Crop face
        face = image_array[y:y+h, x:x+w]
        face_image = Image.fromarray(face)

        # Prepare image
        inputs = processor(
            images=face_image,
            return_tensors="pt"
        )

        # Predict emotion
        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

        prediction = torch.argmax(probabilities).item()

        emotion = model.config.id2label[prediction]

        confidence = (
            probabilities[prediction].item() * 100
        )

        # Draw result
        result_image = image_array.copy()

        cv2.rectangle(
            result_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            result_image,
            f"{emotion} ({confidence:.1f}%)",
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        st.subheader("Prediction")

        st.image(
            result_image,
            caption="Emotion Detection Result",
            use_container_width=True
        )

        st.success(
            f"Detected Emotion: {emotion}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )