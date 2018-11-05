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

### Cloud services (Watson)
Specify Watson API key via environment variable `WATSON_USERNAME` and `WATSON_PASSWORD`. For Ubuntu this can 
be done via changing `nano /etc/environment`. Validation via `printenv 
<var name (optional)>`

### Code remarks
The code is written in Python, since it is the default choice for machine learning. Both open-
source systems (Rasa and DeepPavlov) are built using Python. This is especially useful in the 
case of Rasa. The code uses some methods and data representations defined by the Rasa team. 

Python is mainly object oriented. Over time it has included more and more functional programming
ideas. The code in this project will aim to be adhering to functional programming. 
Reasons are pedagogic value, improved modularity, expressiveness, ease of testing, and brevity. 

The functional programming (FP) constraints for the project is that we do not defining any new 
classes. Specifically, not using the class keyword. Only enums are allowed. This result in some
changes to the code which are explained next.

#### NamedTuple
Pure functions by definition cannot rely on information stored somewhere in the system. We 
provide one example from the code where this created a problem and how this can be solved using
NamedTuples. 

The benchmarking tools communicates with a system called Rasa. Rasa starts in 
a default, untrained, state. To measure its performance we train Rasa and then send many 
sentences to the system. Since you want your functions to be as generic as possible
it makes sense to have one function which takes some sentence, sends it to Rasa to be 
classified and returns all information from the response we want. To avoid re-training Rasa
for each system we have to remember whether Rasa is already trained. Passing just a flag
'retrain' to the system is insufficient, since the function does not know where Rasa should 
train on. To make it all work we need the following parameters:
- `sentence`. The sentence text.
- `sentence_corpus`. The corpus the sentence is taken from.
- `system_name`. Used to call the function which can train the specific system we are interested in.
- `system_knowledge`. Used in combination with sentence_corpus to determine whether we need to re-train.
- `system_data`. Sometimes we need even more information. For example, when we want to enforce 
re-training the system to check whether its outputs differ.

When this function has decided to train the system the system_knowledge changes. So as output
we need to return ``

Since Python 3.5 a NamedTuple with type hints is available. 

To allow for better type checking and reduce the number of function parameters 
use is made of `typing.NamedTuples`. 

TODO: Explain how NamedTuples help type checking on Factory Design pattern replacement.

#### Function caching
Functions can be cached using `functools.lru_cache`. This is mainly used for reducing 
the number of filesystem operations. A typical example is as follows. Suppose we write some
text to a file iteratively by calling `write()` multiple times. Since we try to avoid 
storing a state `write()` does not know whether the file already exists. To solve we can do
two things. The first option is passing parameters telling the function whether the file 
already exists. This is cumbersome, since this state needs to be passed through all the 
functions to the function which is calling the loop over `write()` (directly by calling 
`write()` or indirectly by calling some other function). The second option is defining
a function to create a file if it does not yet exists `create_file()`. We call this function
every time `write()` is called. This does mean that the filesystem is accessed to check the
folder each time `write()` is called. To avoid all those filesystem operations `create_file()`
can be decorated using `functools.lru_cache`. Now on all but the first calls to `create_file()`
just query memory.

##### Caveats
Make sure when using `functools.lru_cache` to not try to mimic state. In other words the
program should not change behaviour if the cache is removed. Reason for this is that any 
state introduced via the cache is similar to doing object oriented programming but without
all the constructs from OOP.

#### Map, filter and reduce
short explanation of these and how they compare to lists

#### Iterators
By default Python is not interested in performance and advises to use a list for every 
collection. However, lists are mutable and therefore not suitable for hashing. Since hashing 
is not possible any function taking lists as input is not suitable for function caching. 

Also, in many cases the list might not be the final structure we need. Consider the following
use cases where the output of type list is used:
- Only unique values are required, so the list is casted to a set.
- The x first elements are required. 
- Only the values satisfying x are required.
- Only an output which is transformed is required.

Considering all these use cases it makes more sense to return an iterator by default instead 
of a collection. One practical example for the bench project which supports this notion is
using an iterator on classification requests. 

Suppose we want to measure the performance of some cloud service. Suppose we wrote some code 
which takes a sentence from some corpus and performs the following operations on this 
sentence:
1. Send the sentence to some cloud service.
2. Transforms the response to the pieces of information we need.
3. Store this information.
Suppose one of the last two operations contains a mistake causing the program to crash. 
When not using an iterator all sentences will have been sent to the cloud service after 
the first operation. Since the post-processing did not succeed we did not obtain results and
need to redo this operation. In effect the programming error caused us to waste about as
many API calls as there are sentences in the corpus we are testing. This is a problem since
the API calls cost money and take time to execute.