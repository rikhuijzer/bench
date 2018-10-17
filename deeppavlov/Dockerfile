FROM debian:buster-20181011-slim
# FROM python:3.7.0-stretch
# installing packages like pandas is non-trivial in alphine or stretch, therefore debian is used
# debian stretch does not include the specific pandas required by deeppavlov
# debian jessie does not include gcc 5 which is required by deeppavlov

RUN apt-get update
RUN apt-get install -y python3-pip
RUN apt-get install -y git-core

RUN pip3 install deeppavlov==0.0.8

COPY configs/intents_snips.json .
RUN python3 -m deeppavlov install ./intents_snips.json
RUN python3 -m deeppavlov download ./intents_snips.json
