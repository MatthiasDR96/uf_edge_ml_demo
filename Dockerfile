# Using the base image with LT4-Pytorch for Jetpack 5.1.1
#FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
FROM ultralytics/ultralytics:latest-jetson-jetpack5

# Otherwise there is an error importing OpenCV ('ImportError: /lib/aarch64-linux-gnu/libgstreamer-1.0.so.0: file too short')
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y 

# Install editor
RUN apt-get install nano

# Set our working directory as app
WORKDIR /uf_edge_ml_demo

# Copy the necessary files and directories into the container
COPY data/ /uf_edge_ml_demo/data/
COPY models/ /uf_edge_ml_demo/models/
COPY src/ /uf_edge_ml_demo/src/
COPY static/ /uf_edge_ml_demo/static/
COPY templates/ /uf_edge_ml_demo/templates/
COPY requirements.txt yolov8n-cls.pt yolov8n.pt /uf_edge_ml_demo/

# Set file permissions
RUN chmod +x /uf_edge_ml_demo/src/app.py

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Exposing port 8000 from the container
EXPOSE 8000
EXPOSE 5000
EXPOSE 6006

# To run the application directly
#ENTRYPOINT [ "python" ]
#CMD ["app.py"]