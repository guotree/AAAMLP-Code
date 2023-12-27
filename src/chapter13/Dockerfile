FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-pip
RUN useradd -m ubuntu
RUN chown -R ubuntu:ubuntu /home/ubuntu/
COPY --chown=ubuntu *.* /home/ubuntu/app/
USER ubuntu
RUN mkdir /home/ubuntu/data/
RUN cd /home/ubuntu/app/ && pip3 install -r requirements.txt
RUN pip3 install mkl
WORKDIR /home/ubuntu/app