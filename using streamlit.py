import streamlit as st
import cv2
import numpy as np
import dlib
from imutils import face_utils
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

class DriverDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def compute(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            landmarks = self.predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = self.blinked(landmarks[36], landmarks[37],
                                      landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = self.blinked(landmarks[42], landmarks[43],
                                       landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if left_blink == 0 or right_blink == 0:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
            elif left_blink == 1 or right_blink == 1:
                status = "Drowsy !"
                color = (0, 0, 255)
            else:
                status = "Active :)"
                color = (0, 255, 0)

            cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            for n in range(0, 68):
                (x, y) = landmarks[n]
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        return frame

    def blinked(self, a, b, c, d, e, f):
        up = self.compute_distance(b, d) + self.compute_distance(c, e)
        down = self.compute_distance(a, f)
        ratio = up / (2.0 * down)

        if ratio > 0.25:
            return 2
        elif 0.21 <= ratio <= 0.25:
            return 1
        else:
            return 0

    @staticmethod
    def compute_distance(ptA, ptB):
        return np.linalg.norm(ptA - ptB)

def main():
    st.title("Driver Drowsiness Detection")

    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=DriverDetectionTransformer,
        async_transform=True,
    )

    if not webrtc_ctx.video_transformer:
        st.warning("Webcam not available")

if __name__ == "__main__":
    main()
