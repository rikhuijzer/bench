# bench

Set terminal current directory to the project root (where `bench.py` is). The docker 
image can then be build via 
```
docker build -t deeppavlov0.0.8-buster . 
```
To test the docker file use `docker run -it deeppavlov0.0.8-buster`.

To run Rasa and DeepPavlov at the same time use
```
docker-compose up
``` 

Packages used in the benchmarks are listed in `requirements.txt` and can be installed 
by using `pip install -r requirements.txt`.

Docker-compose is used to avoid starting Docker containers from Python. One big issue 
of starting Docker containers  from Python is that Docker requires root privileges for 
that.