FROM tensorflow/tensorflow:latest-gpu-py3

ENV NVIDIA_VISIBLE_DEVICES=0

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . /usr/src/app
WORKDIR /usr/src/app