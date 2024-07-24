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

Once the Docker container runs on the remote host, the webserver can be accessed via: hostname:8000. It can take a while before the stream starts.

## Debugging

For debugging, you can use the Visual Studio Code’s Remote - SSH extension to connect your local machine to your remote machine and use it as a development environment. This allows you to write code on your local machine, but run it on the remote machine. Copy the files in e.g. the src folder into your remote machine in a folder called e.g. uf_edge_ml_model via ssh:

```bash
scp -r ..\uf_edge_ml_demo\ nano@10.43.11.11:uf_edge_ml_demo  
```

When starting the docker container, mount the working volume consisting of the files in / uf_edge_ml_model on the remote machine to the folder in the docker container where these files are located, e.g. /uf_edge_ml_demo/src. 

```bash
sudo docker run --gpus all --rm -it --runtime nvidia -p 8000:8000 -p 8080:8080 -p 6006:6006 -v $(pwd):/uf_edge_ml_demo/src --device="/dev/video0:/dev/video0" matthiasdr96/app:nano
```

When running the container, the files will now be synced with the ones on the remote machine which can be modified from the local machine. 