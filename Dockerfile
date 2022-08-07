FROM tensorflow/tensorflow:latest-gpu
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN ["pip","install","tensorflow"]
RUN ["pip","install","numpy"]
COPY requirements.txt /mia/requirements.txt
WORKDIR /mia
RUN ["pip","install","-r","requirements.txt","--prefer-binary","--no-cache"]
COPY . /mia
