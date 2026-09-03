import csv
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "hand_landmarker.task"
OUTPUT_FILE = "mudra_data.csv"

# The mudras we'll collect.
MUDRAS = [
    "Pataka",
    "Tripataka",
    "Ardhapataka",
    "Kartarimukha",
    "Alapadma",
    "Musti",
]


def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Create the CSV file.
    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)

        # 21 landmarks × 3 coordinates = 63 features.
        header = ["label"]

        for i in range(21):
            header.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}",
            ])

        writer.writerow(header)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    with vision.HandLandmarker.create_from_options(options) as landmarker:

        print("\nMudraVision Data Collector")
        print("---------------------------")
        print("Press 1-6 to select a mudra.")
        print("Press Q to quit.\n")

        current_mudra = None

        while True:
            success, frame = cap.read()

            if not success:
                print("Failed to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            result = landmarker.detect(mp_image)

            # Display the currently selected mudra.
            if current_mudra:
                cv2.putText(
                    frame,
                    f"Recording: {current_mudra}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Press 1-6 to select a mudra",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            # Draw hand landmarks.
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                for landmark in hand:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1,
                    )

            cv2.imshow("MudraVision Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            # Select mudra.
            if key in [ord(str(i)) for i in range(1, 7)]:
                index = int(chr(key)) - 1
                current_mudra = MUDRAS[index]
                print(f"Now recording: {current_mudra}")

            # Save a sample.
            elif key == ord("s"):
                if current_mudra and result.hand_landmarks:
                    hand = result.hand_landmarks[0]

                    row = [current_mudra]

                    for landmark in hand:
                        row.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z,
                        ])

                    with open(OUTPUT_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(row)

                    print(f"Saved sample: {current_mudra}")

            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nData saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()