# bench
Benchmarking tool for various intent and entity classification systems.

## Installation
### Open-source systems (Rasa, DeepPavlov)
Set terminal current directory to the project root (where `bench.py` is). The docker images can then be build for each `Dockerfile` having location `systems/<system>/Dockerfile` where `<system>` is the folder name of some system using:
```
docker build -t <system_tag> systems/<system> 
```
For example `docker build -t rasa0.5-mitie0.2 systems/rasa-mitie`.

To test the docker file use `docker run -it <system_tag>`.

To run all the build and tagged Dockers at the same time use
```
docker-compose up
``` 

Packages used in the benchmarks are listed in `requirements.txt` and can be installed by using `pip install -r requirements.txt`.

Docker-compose is used to avoid starting various Docker containers from Python. Multiple containers are needed to benchmark systems with different configurations (for example, Rasa MITIE and Rasa spaCy + sklearn). One big issue 
of starting Docker containers from Python is that Docker requires root privileges.

### Cloud services 
#### Watson
Specify Watson API key via environment variable `WATSON_USERNAME` and `WATSON_PASSWORD`. For Ubuntu this can 
be done via changing `nano /etc/environment`. Validation via `printenv 
<var name (optional)>`

#### DialogFlow
See the DialogFlow v2 API documentation.

TODO: Set `project id` via environment variable.