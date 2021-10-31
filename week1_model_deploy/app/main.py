import io

import numpy as np
import cvlib as cv
import cv2
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse


class Model(str, Enum):
    yolov4tiny = "yolov4-tiny"
    yolov4 = "yolov4"

app = FastAPI(title='Simple Object detection model with fastapi')


@app.get("/")
async def root():
    return {"message": "You've reached the model home. Please pass in arguments"}


@app.post("/predict")
async def prediction(model: Model, confidence_level=0.5, file: UploadFile = File(...)):
    """

    :param model: one of the predefined moddels from Model enum
    https://fastapi.tiangolo.com/tutorial/request-files/#file-parameters-with-uploadfile
    :param file: UploadFile type from fastapi -- works well for larger files since it only stores up to a max file size
    before saving to memory
    :return:
    """
    filename = file.filename
    content_type = file.content_type
    mimetype = content_type.split('/')[0]
    media_type = content_type.split('/')[1]
    if mimetype != 'image' or media_type not in ('jpg', 'jpeg', 'png'):
        raise HTTPException(status_code=415, detail="unsupported file format")
    try:
        image_data = io.BytesIO(await file.read())
        image_data.seek(0)

        # Write the stream of bytes into a numpy array
        file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # detect_common_objects from cvlib uses yolov3 by default-- you can also use yolov4 variants
        # to use custom weights https://github.com/arunponnusamy/cvlib#custom-trained-yolo-weights
        bbox, label, conf = cv.detect_common_objects(image, model=model, confidence_level=confidence_level)

        # Create image that includes bounding boxes and labels
        output_image = cv.object_detection.draw_bbox(image, bbox, label, conf)

        # Save it in a folder within the server
        cv2.imwrite(f'{filename}', output_image)

        # Open the saved image for reading in binary mode
        file_image = open(f'{filename}', mode="rb")

        # Return the image as a stream specifying media type
        return StreamingResponse(file_image, media_type="image/jpeg")
    except Exception as e:
        print('error!!', e)
        raise HTTPException(status_code=500, detail="prediction error")