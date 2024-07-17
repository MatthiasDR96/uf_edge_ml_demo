#!/bin/bash

# Build docker image
docker buildx build --platform linux/arm64 -t my-app:latest .  

# Register docker image
docker login
docker tag my-app:latest username/my-app:latest
docker push username/my-app:latest

# Pull on remote machine
ssh user@remote-server
docker login
docker pull username/my-app:latest

# Run on remote machine
docker run --rm -it -p 5000:5000 username/my-app:latest
docker run --rm -it -v "$(pwd)/app.py,target=/app/scripts" username/my-app:latest

sudo docker run --runtime nvidia -it --rm -p 8000:8000 --device /dev/video0 matthiasdr96/app:nano