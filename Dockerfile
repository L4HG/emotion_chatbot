FROM python:3

RUN apt-get -y update && apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    libavcodec-dev \
    libavformat-dev \
    libjpeg-dev \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

WORKDIR /opt/emotion_chatbot

COPY requirements.txt ./

RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python3", "./main.py" ]
