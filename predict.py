import cv2
import mediapipe as mp
import numpy as np
import joblib

from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "hand_landmarker.task"
CLASSIFIER_PATH = "mudra_model.pkl"


# ─────────────────────────────────────────────
# MUDRA INFORMATION
# ─────────────────────────────────────────────

MUDRA_INFO = {
    "Pataka": {
        "meaning": "Flag",
        "uses": "Clouds  •  Wind  •  Forests  •  Blessings",
    },
    "Tripataka": {
        "meaning": "Three-part flag",
        "uses": "Crowns  •  Trees  •  Arrows  •  Thunder",
    },
    "Ardhapataka": {
        "meaning": "Half flag",
        "uses": "Leaves  •  Writing Boards  •  Riverbanks",
    },
    "Kartarimukha": {
        "meaning": "Scissor",
        "uses": "Separation  •  Opposition  •  Lightning",
    },
    "Alapadma": {
        "meaning": "Fully blossomed lotus",
        "uses": "Beauty  •  Full Moon  •  Blossoming",
    },
    "Musti": {
        "meaning": "Fist",
        "uses": "Strength  •  Determination  •  Holding objects",
    },
}


# ─────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────

MAROON = (48, 12, 25)
DARK_MAROON = (27, 7, 15)
CARD = (63, 18, 31)
GOLD = (218, 170, 75)
LIGHT_GOLD = (245, 213, 137)
CREAM = (247, 239, 218)
MUTED = (185, 165, 145)
GREEN = (92, 190, 140)
WHITE = (255, 255, 255)


# ─────────────────────────────────────────────
# FONTS
# ─────────────────────────────────────────────

def get_font(size, bold=False):
    paths = [
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ]

    if bold:
        paths = [
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        ] + paths

    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue

    return ImageFont.load_default()


# ─────────────────────────────────────────────
# LANDMARK NORMALIZATION
# ─────────────────────────────────────────────

def normalize_landmarks(landmarks):
    points = np.array([
        [landmark.x, landmark.y, landmark.z]
        for landmark in landmarks
    ])

    # Wrist becomes origin
    wrist = points[0]
    points = points - wrist

    # Normalize hand size
    distances = np.linalg.norm(points, axis=1)
    hand_size = np.max(distances)

    if hand_size > 0:
        points = points / hand_size

    return points.flatten().reshape(1, -1)


# ─────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────

def rounded_rectangle(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )


def center_text(draw, text, y, font, fill, width):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]

    x = (width - text_width) // 2

    draw.text(
        (x, y),
        text,
        font=font,
        fill=fill,
    )


def draw_lotus(draw, center_x, center_y, scale=1):
    """
    Minimal lotus-inspired decorative motif.
    """

    petals = [
        (-32, -48, -15, 0),
        (-18, -58, 0, 0),
        (0, -58, 18, 0),
        (15, -48, 32, 0),
    ]

    for x1, y1, x2, y2 in petals:
        draw.ellipse(
            (
                center_x + x1 * scale,
                center_y + y1 * scale,
                center_x + x2 * scale,
                center_y + y2 * scale,
            ),
            outline=GOLD,
            width=max(1, int(2 * scale)),
        )

    draw.arc(
        (
            center_x - 42 * scale,
            center_y - 20 * scale,
            center_x + 42 * scale,
            center_y + 30 * scale,
        ),
        180,
        360,
        fill=GOLD,
        width=max(1, int(2 * scale)),
    )


def draw_corner_motif(draw, x, y, flip_x=1, flip_y=1):
    """
    Small geometric motif inspired by traditional rangoli/kolam forms.
    """

    size = 22

    for i in range(3):
        offset = i * 9

        x1 = x + flip_x * offset
        y1 = y + flip_y * offset

        draw.ellipse(
            (
                x1 - 3,
                y1 - 3,
                x1 + 3,
                y1 + 3,
            ),
            fill=GOLD,
        )

    draw.line(
        [
            (x, y + flip_y * size),
            (x + flip_x * size, y),
        ],
        fill=GOLD,
        width=1,
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    model = joblib.load(CLASSIFIER_PATH)

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Window dimensions
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 760

    with vision.HandLandmarker.create_from_options(options) as landmarker:

        while True:

            success, camera_frame = cap.read()

            if not success:
                print("Failed to read webcam frame.")
                break

            camera_frame = cv2.flip(camera_frame, 1)

            # ─────────────────────────────────
            # CAMERA / MEDIAPIPE
            # ─────────────────────────────────

            rgb_frame = cv2.cvtColor(
                camera_frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect(mp_image)

            prediction = None
            confidence = 0
            meaning = ""
            uses = ""

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                # Draw landmarks directly onto camera
                h, w = camera_frame.shape[:2]

                for landmark in hand:

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(
                        camera_frame,
                        (x, y),
                        5,
                        (218, 170, 75),
                        -1
                    )

                features = normalize_landmarks(hand)

                probabilities = model.predict_proba(features)[0]

                best_index = np.argmax(probabilities)

                prediction = model.classes_[best_index]

                confidence = probabilities[best_index] * 100

                info = MUDRA_INFO.get(prediction, {})

                meaning = info.get("meaning", "")
                uses = info.get("uses", "")

            # ─────────────────────────────────
            # CREATE UI
            # ─────────────────────────────────

            ui = Image.new(
                "RGB",
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                DARK_MAROON
            )

            draw = ImageDraw.Draw(ui)

            # ─────────────────────────────────
            # HEADER
            # ─────────────────────────────────

            draw.rectangle(
                (0, 0, WINDOW_WIDTH, 105),
                fill=MAROON
            )

            # Gold divider
            draw.rectangle(
                (0, 103, WINDOW_WIDTH, 105),
                fill=GOLD
            )

            # Decorative corners
            draw_corner_motif(draw, 30, 28)
            draw_corner_motif(
                draw,
                WINDOW_WIDTH - 30,
                28,
                flip_x=-1
            )

            title_font = get_font(38, bold=True)
            subtitle_font = get_font(16)
            small_font = get_font(14)

            center_text(
                draw,
                "MUDRAVISION",
                17,
                title_font,
                LIGHT_GOLD,
                WINDOW_WIDTH
            )

            center_text(
                draw,
                "BHARATANATYAM  •  HAND GESTURE RECOGNITION",
                63,
                subtitle_font,
                CREAM,
                WINDOW_WIDTH
            )

            # ─────────────────────────────────
            # CAMERA PANEL
            # ─────────────────────────────────

            camera_x = 45
            camera_y = 140
            camera_w = 750
            camera_h = 560

            rounded_rectangle(
                draw,
                (
                    camera_x,
                    camera_y,
                    camera_x + camera_w,
                    camera_y + camera_h
                ),
                18,
                fill=CARD,
                outline=GOLD,
                width=2
            )

            # Resize camera feed
            camera_rgb = cv2.cvtColor(
                camera_frame,
                cv2.COLOR_BGR2RGB
            )

            camera_image = Image.fromarray(camera_rgb)

            camera_image.thumbnail(
                (camera_w - 20, camera_h - 20)
            )

            paste_x = camera_x + (camera_w - camera_image.width) // 2
            paste_y = camera_y + (camera_h - camera_image.height) // 2

            ui.paste(
                camera_image,
                (paste_x, paste_y)
            )

            # ─────────────────────────────────
            # RIGHT INFO PANEL
            # ─────────────────────────────────

            panel_x = 830
            panel_y = 140
            panel_w = 405
            panel_h = 560

            rounded_rectangle(
                draw,
                (
                    panel_x,
                    panel_y,
                    panel_x + panel_w,
                    panel_y + panel_h
                ),
                18,
                fill=CARD,
                outline=GOLD,
                width=2
            )

            label_font = get_font(14)
            mudra_font = get_font(38, bold=True)
            meaning_font = get_font(23, bold=True)
            body_font = get_font(15)

            draw.text(
                (panel_x + 30, panel_y + 35),
                "RECOGNIZED MUDRA",
                font=label_font,
                fill=MUTED
            )

            if prediction:

                draw.text(
                    (panel_x + 30, panel_y + 75),
                    prediction.upper(),
                    font=mudra_font,
                    fill=LIGHT_GOLD
                )

                # Confidence
                draw.text(
                    (panel_x + 30, panel_y + 140),
                    f"{confidence:.1f}% confidence",
                    font=body_font,
                    fill=GREEN
                )

                # Confidence bar
                bar_x = panel_x + 30
                bar_y = panel_y + 170
                bar_w = panel_w - 60
                bar_h = 7

                rounded_rectangle(
                    draw,
                    (
                        bar_x,
                        bar_y,
                        bar_x + bar_w,
                        bar_y + bar_h
                    ),
                    4,
                    fill=(90, 55, 65)
                )

                filled = int(
                    bar_w * min(confidence, 100) / 100
                )

                rounded_rectangle(
                    draw,
                    (
                        bar_x,
                        bar_y,
                        bar_x + filled,
                        bar_y + bar_h
                    ),
                    4,
                    fill=GOLD
                )

                # Meaning divider
                draw.line(
                    (
                        panel_x + 30,
                        panel_y + 215,
                        panel_x + panel_w - 30,
                        panel_y + 215
                    ),
                    fill=(110, 55, 70),
                    width=1
                )

                draw.text(
                    (panel_x + 30, panel_y + 245),
                    "MEANING",
                    font=label_font,
                    fill=MUTED
                )

                draw.text(
                    (panel_x + 30, panel_y + 280),
                    meaning,
                    font=meaning_font,
                    fill=CREAM
                )

                draw.text(
                    (panel_x + 30, panel_y + 335),
                    "TRADITIONAL USES",
                    font=label_font,
                    fill=MUTED
                )

                # Wrap uses if necessary
                words = uses.split("•")
                y = panel_y + 370

                for word in words:
                    word = word.strip()

                    draw.text(
                        (panel_x + 30, y),
                        f"•  {word}",
                        font=body_font,
                        fill=CREAM
                    )

                    y += 30

            else:

                center_text(
                    draw,
                    "Place your hand",
                    panel_y + 220,
                    meaning_font,
                    CREAM,
                    WINDOW_WIDTH
                )

                center_text(
                    draw,
                    "inside the camera frame",
                    panel_y + 260,
                    body_font,
                    MUTED,
                    WINDOW_WIDTH
                )

                draw_lotus(
                    draw,
                    panel_x + panel_w // 2,
                    panel_y + 380,
                    1
                )

            # ─────────────────────────────────
            # FOOTER
            # ─────────────────────────────────

            footer_y = 720

            draw.line(
                (
                    45,
                    footer_y,
                    WINDOW_WIDTH - 45,
                    footer_y
                ),
                fill=(90, 45, 60),
                width=1
            )

            draw.text(
                (WINDOW_WIDTH - 150, footer_y + 15),
                "Q  •  EXIT",
                font=small_font,
                fill=LIGHT_GOLD
            )

            # ─────────────────────────────────
            # DISPLAY
            # ─────────────────────────────────

            final_frame = cv2.cvtColor(
                np.array(ui),
                cv2.COLOR_RGB2BGR
            )

            cv2.imshow(
                "MudraVision",
                final_frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()