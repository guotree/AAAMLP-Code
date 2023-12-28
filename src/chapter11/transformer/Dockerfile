FROM ubuntu:18.04
RUN apt-get update && apt-get install -y \
git \
curl \
ca-certificates \
python3 \
python3-pip \
sudo \
&& rm -rf /var/lib/apt/lists/*
RUN useradd -m abhishek
RUN chown -R abhishek:abhishek /home/abhishek/
COPY --chown=abhishek *.* /home/abhishek/app/
USER abhishek
RUN mkdir /home/abhishek/data/
RUN cd /home/abhishek/app/ && pip3 install -r requirements.txt
RUN pip3 install mkl
WORKDIR /home/abhishek/app