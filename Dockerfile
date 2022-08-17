FROM tensorflow/tensorflow:latest-gpu
RUN /usr/local/bin/python -m pip install --upgrade pip
COPY requirements.txt /mia/requirements.txt
WORKDIR /mia
RUN ["pip","install","-r","requirements.txt","--prefer-binary","--no-cache"]
