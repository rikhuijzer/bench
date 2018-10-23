FROM python:3.7.0-alpine3.8

RUN apk add git make build-base
RUN pip install git+https://github.com/mit-nlp/MITIE.git
RUN apk add freetype-dev libpng-dev
RUN pip install rasa_nlu[mitie]

# RUN pip install git+git://github.com/mit-nlp/MITIE.git@v0.2
# RUN pip install rasa_nlu[mitie]==0.5
