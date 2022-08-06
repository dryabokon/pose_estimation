docker_imge="conda-image-pose-p36"

#sudo docker run -v ${PWD}:/home --workdir /home -it $docker_imge python3 main.py -i KZ.mp4 --cam_fov_x_deg 14.0 
#sudo docker run -v ${PWD}:/home --workdir /home -it $docker_imge python3 main.py -i TNO-7180R_20220418134537.avi 
sudo docker run -v ${PWD}:/home --workdir /home -it $docker_imge python3 main.py -i TNO-7180R_20220418134537.avi --cam_fov_x_deg 7 
#sudo docker run -v ${PWD}:/home --workdir /home -it $docker_imge python3 main.py -i TNO-7180R_20220418134537.avi --cam_height_m 7.5

