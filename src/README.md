# Code remarks

The code is written in Python, since it is the default choice for machine learning. Both open-
source systems (Rasa and DeepPavlov) are built using Python. This is especially useful in the 
case of Rasa. The code uses some methods and data representations defined by the Rasa team. 

Python is mainly object oriented. Over time it has included more and more functional programming
ideas. The code in this project will aim to be adhering to functional programming. 
Reasons are pedagogic value, improved modularity, expressiveness, ease of testing, and brevity. 

The functional programming constraints for the project are that we do not defining any new 
classes. Specifically, not using the class keyword. Notable code design considerations 
are explained next.

#### Imports
When importing an attempt is made to explicitly import using `from <module> import <class>`. 
When more implicit imports are used `import <module>` this is either caused by the appearance
of circular imports, by the fact that some names are too common or to avoid reader confusion.
An example for the latter are the types defined in `src.typ`. The names are quite generic
and could cause name clashing or confusion when imported explicitly.

#### Mapping to functions
In code we often have a function which calls other functions depending on some conditionals.
For example in `system.py` the factory design pattern is replaced by a more functional design.
In this design `system.py` behaves like a super and delegates the work based on what system
we currently interested in. We give an example for two systems. The delegation 
could be done via conditional statements.
```
if 'mock' in system.name:
    response = src.systems.mock.get_response(tp.Query(system, message.text))
elif 'rasa' in system.name:
    response = src.systems.rasa.get_response(tp.Query(system, message.text))
elif ...
```
This introduces a lot of code duplication. Therefore a dict is created. 
```
get_intent_systems = {
    'mock': src.systems.mock.get_response,
    'rasa': src.systems.rasa.get_response,
    ...
}
```
Now we can just get the correct function from the dict and call it.
```
func: Callable[[tp.Query], tp.Response] = get_substring_match(get_intent_systems, system.name)
query = tp.Query(system, message.text)
response = func(query)
```
Note that `get_substring_match()` implements the substring matching used in the conditional
code (`if 'mock' in system.name:`). Since the code can return any of the functions contained
in the mapping they should all have the same signature and output. The currently used IDE
(PyCharm 2018.2.4)is not able to check this. Therefore, functions from the mapping 
`func` get a type hint. This allows the IDE to check types again and it allows developers 
to see what signature should be used for all the functions in the mapping. 

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
Short explanation of these and how they compare to collections.

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