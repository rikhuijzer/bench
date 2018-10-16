# bench

when terminal is inside same folder as Dockerfile the docker image deeppavlov-jessie can be build via 
```
docker build -t deeppavlov0.0.8-jessie . 
docker run -it deeppavlov0.0.8-jessie
```

To run Rasa and DeepPavlov run `docker-compose up` in the folder where `docker-compose.yml` is located. 

Packages used in the benchmarks are listed in `requirements.txt` and can be installed by using `pip install -r requirements.txt`.

