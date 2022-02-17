FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN apt-get update
RUN apt-get install wget curl -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN mkdir app
WORKDIR /app
RUN pip3 install pandas
RUN pip3 install sklearn
RUN pip3 install pyswarms
RUN pip3 install pyyaml==6.0
RUN pip3 install networkx
COPY . .
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser