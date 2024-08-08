# Ultimate Factory Edge ML Demo

![screenshot](screenshot_interface.png)

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
docker run --gpus all --runtime nvidia -it --rm -p 8000:8000 -p 8080:8080 -p 6006:6006 --device="/dev/video0:/dev/video0" username/my-app:latest 
```
## Usage

Once the Docker container runs on the remote host, the webserver can be accessed via: hostname:8000. It can take a while before the stream starts. Label studio can be accessed via: hostname:8080.

* Inference: When the webserver is started, images get captured and the inference results are directly shown on the captured frames. By default, there are three pretrained systems that can be selected for inference from the drop-down menu next to the training button. One can choose from classification, detection, and segmentation. 

* Data-collection: When one wants to train custom models, data needs to be captured. On the GUI, in the left plane, classes can be entered in the text field. When clicking the submit button, image folders for each class will be created and are accessible via the drop-down menu. When clicking the capture button, images get stored in the image folder corresponding to the selected category. All image folders can also be deleted using the delete button. 

* Training: After the data collection, training can be started using the training button. The type of model to train can be selected from the drop-down menu. For classification, no extra labelling is required. All captured data is stored in folders with the corresponding class names. When clicking the training button, the classifier starts training. Once done, the new classifier inferences on the new images. For detection and segmentation, first some additional labelling is required. 

* Labelling: Once all images are stored in the correct folders, one can start labelling. Pressing on the 'Start labels' button creates a new project with the correct classes in Label Studio and imports all images. When one wants to import some additional images, some new images can be captured and the 'updata labels' button will add them to Label Studio. In Label Studio, all images can be labelled using the userfriendly labelling tools. When done labelling, clicking the 'Download labels' button will download the labels locally. Once done, training can be started by pressing the train button. 

## Debugging

For debugging, you can use the Visual Studio Code’s Remote - SSH extension to connect your local machine to your remote machine and use it as a development environment. This allows you to write code on your local machine, but run it on the remote machine. Copy the files in e.g. the src folder into your remote machine in a folder called e.g. uf_edge_ml_model via ssh:

```bash
scp -r ..\uf_edge_ml_demo\ nano@hostname:uf_edge_ml_demo  
```

When starting the docker container, mount the working volume consisting of the files in / uf_edge_ml_model on the remote machine to the folder in the docker container where these files are located, e.g. /uf_edge_ml_demo/src. 

```bash
sudo docker run --gpus all --rm -it --runtime nvidia -p 8000:8000 -p 8080:8080 -p 6006:6006 -v $(pwd):/uf_edge_ml_demo --device="/dev/video0:/dev/video0" matthiasdr96/app:nano
```

When running the container, the files will now be synced with the ones on the remote machine which can be modified from the local machine. 