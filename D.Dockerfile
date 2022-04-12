FROM continuumio/miniconda3

#WORKDIR /app

COPY environment.yml .
RUN conda env update --file environment.yml  --prune
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN conda install -c menpo opencv

