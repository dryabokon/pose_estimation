docker_imge="conda-image-pose"
sudo docker run -v ${PWD}:/home --workdir /home -it $docker_imge python3 main.py -i KZ.mp4
