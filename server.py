
import streamlit as st
import sys
st.write(sys.version)
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import easyocr
from ultralytics import YOLO

st.set_page_config(layout='wide')


# ---------------------------
# MAIN
# ---------------------------
def main():
    car_m, lp_m, reader = load_model()

    st.title("자동차 번호판 인식")
    file = st.file_uploader('이미지를 올려주세요')

    if file:
        im, text = detect(car_m, lp_m, reader, file)
        st.write(text)
        st.image(im)


# ---------------------------
# LOAD MODELS
# ---------------------------
@st.cache_resource
def load_model():
    car_m = YOLO("yolov5s.pt")        # vehicle detector
    lp_m = YOLO("lp_det.pt")          # license plate detector

    reader = easyocr.Reader(
        ['en'],
        detect_network='craft',
        recog_network='best_acc',
        user_network_directory='lp_models/user_network',
        model_storage_directory='lp_models/models'
    )

    return car_m, lp_m, reader


# ---------------------------
# DETECTION PIPELINE
# ---------------------------
def detect(car_m, lp_m, reader, file):
    fontpath = "SpoqaHanSansNeo-Light.ttf"
    font = ImageFont.truetype(fontpath, 40)

    # Load image correctly
    im = Image.open(file).convert("RGB")
    to_draw = np.array(im)

    # Detect cars (filter classes)
    car_results = car_m(im, classes=[2, 3, 5, 7])[0]
    car_boxes = car_results.boxes.xyxy if car_results.boxes is not None else []

    result_text = []

    # ---------------------------
    # CASE 1: No car detected
    # ---------------------------
    if len(car_boxes) == 0:
        lp_results = lp_m(im)[0]
        lp_boxes = lp_results.boxes.xyxy if lp_results.boxes is not None else []

        if len(lp_boxes) == 0:
            result_text.append('검출된 차 없음')
        else:
            for box in lp_boxes:
                x2, y2, x3, y3 = map(int, box.tolist())

                try:
                    crop = to_draw[y2:y3, x2:x3]
                    gray = cv2.cvtColor(
                        cv2.resize(crop, (224, 128)),
                        cv2.COLOR_BGR2GRAY
                    )
                    text = reader.readtext(gray)[0][1]
                    result_text.append(text)
                except:
                    continue

                # Draw
                img_pil = Image.fromarray(to_draw)
                draw = ImageDraw.Draw(img_pil)
                draw.text((x2, y2 - 40), text, font=font, fill=(255, 0, 255))
                to_draw = np.array(img_pil)

                cv2.rectangle(to_draw, (x2, y2), (x3, y3), (255, 0, 255), 3)

        return cv2.resize(to_draw, (1280, 1280)), result_text

    # ---------------------------
    # CASE 2: Cars detected
    # ---------------------------
    for car_box in car_boxes:
        x, y, x1, y1 = map(int, car_box.tolist())
        car_crop = to_draw[y:y1, x:x1]

        lp_results = lp_m(Image.fromarray(car_crop))[0]
        lp_boxes = lp_results.boxes.xyxy if lp_results.boxes is not None else []

        if len(lp_boxes) == 0:
            result_text.append("차는 검출됐으나 번호판이 검출되지 않음")

        for lp_box in lp_boxes:
            x2, y2, x3, y3 = map(int, lp_box.tolist())

            try:
                crop = to_draw[y + y2:y + y3, x + x2:x + x3]
                gray = cv2.cvtColor(
                    cv2.resize(crop, (224, 128)),
                    cv2.COLOR_BGR2GRAY
                )
                text = reader.readtext(gray)[0][1]
                result_text.append(text)
            except:
                continue

            # Draw text
            img_pil = Image.fromarray(to_draw)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x + x2, y + y2 - 40), text, font=font, fill=(255, 0, 255))
            to_draw = np.array(img_pil)

            # Draw box
            cv2.rectangle(
                to_draw,
                (x + x2, y + y2),
                (x + x3, y + y3),
                (255, 0, 255),
                3
            )

    return cv2.resize(to_draw, (1280, 1280)), result_text


# ---------------------------
# ENTRY POINT
# ---------------------------
if __name__ == '__main__':
    main()
