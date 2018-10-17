# bench

When terminal is inside same folder as `docker-compose.yml` the docker image can be build via 
```
docker build -t deeppavlov0.0.8-buster deeppavlov 
```
To test the docker file use `docker run -it deeppavlov0.0.8-buster`.

To run Rasa and DeepPavlov at the same time use
```
docker-compose up
``` 

Packages used in the benchmarks are listed in `requirements.txt` and can be installed by using `pip install -r requirements.txt`.

