FROM debian:buster-20181011-slim
# installing packages like pandas is non-trivial in alphine, therefore debian is used
# debian stretch does not include a prebuild pandas package for the pandas version required by deeppavlov
# debian jessie does not include gcc 5 which is required by deeppavlov

RUN apt-get update
RUN apt-get install -y python3-pip
RUN apt-get install -y git-core

RUN pip3 install deeppavlov==0.0.8

# pre-installing and downloading many files and packages
# these lines make the Dockerfile unsuitable for Docker Hub
COPY configs/intents_snips.json .
RUN python3 -m deeppavlov install ./intents_snips.json
RUN python3 -m deeppavlov download ./intents_snips.json