FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04

# Install Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Install Python packages
WORKDIR /app
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 6873
COPY app.py models/ /app/

CMD python3.10 app.py
