# Ultimate Factory Edge ML Demo

## Installation

Clone this repository to your working directory

```bash
git clone https://github.com/MatthiasDR96/uf_edge_ml_demo.git
```

Build a Docker image and push it to the Docker registry

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
docker run --runtime nvidia -it --rm -p 8000:8000 --device /dev/video0 username/my-app:latest 
```

## Usage

Once the Docker container runs on the remote host, the webserver can be accessed via <host-ip>:8000. To change the classes of the model, the classes.txt file in './data/' can be modified. 