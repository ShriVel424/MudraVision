import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "hand_landmarker.task"


def main():
    # Create the MediaPipe hand landmarker.
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Open the Mac's webcam.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    with vision.HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                print("Failed to read webcam frame.")
                break

            # Mirror the webcam.
            frame = cv2.flip(frame, 1)

            # Convert BGR → RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to a MediaPipe image.
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # Detect the hand.
            result = landmarker.detect(mp_image)

            # Draw landmarks manually.
            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    for landmark in hand:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])

                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            cv2.imshow("MudraVision", frame)

            # Press Q to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()