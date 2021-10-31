This is a quick sample template of deploying a ML model using FastApi and Docker. 
[FastApi](https://fastapi.tiangolo.com/) has a lot of nice stuff in it, including async support, 
a prebuilt swagger spec ui and default type declarations. 

Here, we are using a simple setup:
- docker: using the official fastapi docker image from `tiangolo/uvicorn-gunicorn-fastapi:python3.7`
- ML model: we are using the pretrained yolov4 from cvlib

Running instructions:
1. in this directory, (ie, same level as `Dockerfile`), run `docker build -t yolo_sample .`
2. to run the container `docker run  --name yolo -p 80:80 yolo_sample`
3. go to http://127.0.0.1/docs#/default/prediction_predict_post where you can find the fastapi docs ui, where you 
   can test the api out