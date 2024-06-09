import cv2
import numpy as np
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import tensorflow_hub as hub
from PIL import Image
from io import BytesIO

app = FastAPI()
templates = Jinja2Templates(directory="templates")

modelo = None

@app.on_event("startup")
async def startup_event():
    global modelo
    with custom_object_scope({"KerasLayer": hub.KerasLayer}):
        modelo = load_model('modelo_entrenado.h5')

def get_label(predict):
    labels = {0: "billete de 100 mil", 1: "billete de 10 mil", 2: "billete de 1 mil", 3: "billete de 10 mil", 4: "billete de 20 mil", 5: "billete de 2 mil", 6: "billete de 50 mil", 7: "billete de 5 mil"}
    return labels[predict]

def categorizar(img_bytes):
    img = Image.open(BytesIO(img_bytes))
    img = np.array(img).astype(float) / 255
    img = cv2.resize(img, (224, 224))
    prediccion = modelo.predict(img.reshape(-1, 224, 224, 3))
    return np.argmax(prediccion[0], axis=-1), prediccion

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .png, .jpg, and .jpeg are allowed.")
    img_bytes = await file.read()
    label_index, prediction = categorizar(img_bytes)
    label = get_label(label_index)
    return {"label": label, "prediction": prediction.tolist()}

@app.get("/video")
async def video_feed():
    cap = cv2.VideoCapture(0)
    cont = 0

    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                if modelo is not None:
                    img = cv2.resize(frame, (224, 224))
                    prediccion = modelo.predict(img.reshape(-1, 224, 224, 3))

                    frame = cv2.putText(
                        frame, f"Prediccion: {prediccion[0]}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2,
                        cv2.LINE_AA)

                    frame = cv2.putText(
                        frame, f"Label: {get_label(np.argmax(prediccion[0]))}",
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
                       b'Content-Type: text/plain\r\n\r\n' + texto_para_html.encode() + b'\r\n')

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )

@app.get("/")
def get_html(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
