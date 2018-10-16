FROM python:3.6.6-slim-jessie
# installing packages like pandas is non-trivial in alphine or stretch, therefore jessie is used
# use docker-compose build again after changing this file

RUN apt-get update
RUN apt-get install -y git-core
RUN pip install deeppavlov==0.0.8

COPY configs/intents_snips.json .

# TESTING WHETHER PRE-INSTALL IN DOCKER WORKS
RUN echo "deb http://ftp.us.debian.org/debian unstable main contrib non-free" > /etc/apt/sources.list.d/unstable.list
RUN apt-get install -y gcc  # 4.9.2
RUN gcc --version

RUN /usr/local/bin/python -m pip install git+https://github.com/deepmipt/fastText.git#egg=fastText==0.8.22

RUN python -m deeppavlov install ./intents_snips.json
# END OF TEST
