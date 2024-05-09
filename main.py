import cv2
import numpy as np
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import tensorflow_hub as hub

app = FastAPI()
templates = Jinja2Templates(directory="templates")


def get_label(predict):
    labels = {
        0: "llorando",
        1: "estresado"
    }
    predict = np.argmax(predict, axis=-1)
    if predict < 0.6:
        return 'sin expresion'
    return labels[np.argmax(predict, axis=-1)]


@app.get("/video")
async def video_feed():
    cap = cv2.VideoCapture(0)
    with custom_object_scope({"KerasLayer": hub.KerasLayer}):
        modelo = load_model('modelo_entrenado.h5')
    cont = 0

    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                img = cv2.resize(
                    frame, (224, 224)
                )
                prediccion = modelo.predict(
                    img.reshape(-1, 224, 224, 3)
                )

                frame = cv2.putText(
                    frame, f" Prediccion: {prediccion[0]}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2,
                    cv2.LINE_AA)

                frame = cv2.putText(
                    frame, f" Label: {get_label(prediccion[0])}",
                    (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2,
                    cv2.LINE_AA)

                ret, buffer = cv2.imencode('.jpg', frame)

                if not ret:
                    continue
                texto_para_html = str(cont + 1)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                                                                                b'Content-Type: text/plain\r\n\r\n' +
                       texto_para_html.encode() + b'\r\n')

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace;boundary=frame")


@app.get("/")
def get_html(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})




