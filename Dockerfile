# Using the base image with LT4-Pytorch for Jetpack 5.1.1
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Otherwise there is an error importing OpenCV ('ImportError: /lib/aarch64-linux-gnu/libgstreamer-1.0.so.0: file too short')
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y 

# Install editor
RUN apt-get install nano

# Set our working directory as app
WORKDIR /app

# Copy the necessary files and directories into the container
COPY data/ /app/data/
COPY models/ /app/models/
COPY static/ /app/static/
COPY templates/ /app/templates/
COPY app.py model.py utils.py requirements.txt /app/

# Set file permissions
RUN chmod +x app.py

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Exposing port 8000 from the container
EXPOSE 8000

# To run the application directly
#ENTRYPOINT [ "python" ]
#CMD ["app.py"]