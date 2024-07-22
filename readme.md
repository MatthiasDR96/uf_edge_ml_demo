# Ultimate Factory Edge ML Demo

## Installation

Clone this repository to your local working directory.

```bash
git clone https://github.com/MatthiasDR96/uf_edge_ml_demo.git
```

Download and install Docker on your local machine. Build a Docker image and push it to the Docker registry.

```bash
docker login
docker build -t my-app:latest .  
docker tag my-app:latest username/my-app:latest 
docker push username/my-app:latest     
```

On the remote host, pull the image

```bash
ssh user@remote-server
docker login
docker pull username/my-app:latest
docker run --gpus all --runtime nvidia -it --rm -p 8000:8000 --ipc=host --device /dev/video0 username/my-app:latest 
```

## Usage

Once the Docker container runs on the remote host, the webserver can be accessed via: <host-ip>:8000. It can take a while before the stream starts.