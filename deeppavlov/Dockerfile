# FROM debian:buster-20181011-slim
FROM python:3.7.0-stretch
# installing packages like pandas is non-trivial in alphine or stretch, therefore debian is used
# only debian jessie available for python on hub, so using buster for gcc 5
# use docker-compose build again after changing this file

RUN python3 --version
RUN python --version
# RUN apt-get install -y git-core
RUN pip3 install scipy
RUN pip3 install pandas==0.23.1  # THIS IS THE PROBLEM THIS PANDAS IS NOT AVAILABLE IN REPO OF STRETCH
RUN pip3 install deeppavlov==0.0.8

# ln -s python3 python

# COPY configs/intents_snips.json .

# TESTING WHETHER PRE-INSTALL IN DOCKER WORKS
#RUN echo "deb http://ftp.us.debian.org/debian unstable main contrib non-free" > /etc/apt/sources.list.d/unstable.list
#RUN apt-get install -y gcc  # 4.9.2
#RUN gcc --version

RUN /usr/local/bin/python -m pip install git+https://github.com/deepmipt/fastText.git#egg=fastText==0.8.22

#RUN python -m deeppavlov install ./intents_snips.json
# END OF TEST
