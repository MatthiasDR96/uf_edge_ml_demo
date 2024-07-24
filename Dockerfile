# Using the base image with LT4-Pytorch for Jetpack 5.1.1 and Ultralytics
FROM ultralytics/ultralytics:latest-jetson-jetpack5

# Install nano editor
RUN apt-get install nano

# Set the working directory 
WORKDIR /uf_edge_ml_demo

# Copy the necessary files and directories into the container
COPY data/ /uf_edge_ml_demo/data/
COPY models/ /uf_edge_ml_demo/models/
COPY src/ /uf_edge_ml_demo/src/
COPY static/ /uf_edge_ml_demo/static/
COPY templates/ /uf_edge_ml_demo/templates/ 
COPY start.sh label_studio.sqlite3 requirements.txt yolov8n-seg.pt yolov8n-cls.pt yolov8n.pt /uf_edge_ml_demo/

# Label studio local file sync variables makes automatic syncing of data possible
ENV LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
ENV LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="/uf_edge_ml_demo/data"

# Set file permissions
RUN chmod +x /uf_edge_ml_demo/src/app.py
RUN chmod +x /uf_edge_ml_demo/start.sh

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Exposing port 8000 from the container
EXPOSE 8000
EXPOSE 8080
EXPOSE 5000
EXPOSE 6006

# To run the application directly
CMD ["./start.sh"]



