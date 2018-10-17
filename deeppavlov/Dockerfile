FROM debian:buster-20181011-slim
# FROM python:3.7.0-stretch
# installing packages like pandas is non-trivial in alphine or stretch, therefore debian is used
# only debian stretch does not include the specific pandas required by deeppavlov
# use docker-compose build again after changing this file

RUN apt-get update
RUN apt-get install -y python3-pip
RUN apt-get install -y git-core

RUN pip3 install deeppavlov==0.0.8

COPY configs/intents_snips.json .
RUN python3 -m deeppavlov install ./intents_snips.json
# RUN python3 -m deeppavlov download ./intents_snips.json
# RUN ln -s python3 python
# RUN ln -s python3 /usr/local/bin/python  # these do not seem to work

# RUN /usr/local/bin/python -m pip install git+https://github.com/deepmipt/fastText.git#egg=fastText==0.8.22

# END OF TEST
