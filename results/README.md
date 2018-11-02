# Results 

Results are stored here. The goal of these files is to be easy to process for a computer and human. 
Therefore the results are in YAML and CSV files. For each system and corpus the following files are created.

This files can be used by the system to report system performance. Performance statistics include:
- F1 score (micro, macro scores for intent and entity)
- Average response time
- Training time

## General
For the general information YAML is used. 
```yaml
system name: mock-server

corpus: Mock

training time [seconds]: 10

f1 scores:
  micro: 0.99
  macro: 0.99

response_time [ms]:
  average: 15
  standard deviation: 2
```

## Intents
Below is an example of a system which classified two sentences. 

The first sentence incorrectly classified in half a second. The system was not so confident 
in the classification.

| id | sentence | intent | classification | confidence [%] | time |
| --- | --- | --- | --- | --- | --- | 
| 0 | When is tomorrows first flight to London? | departure time | greet | 20.3 | 0.5 |
| 1 | Will it rain on monday? | get weather | get weather | 99.1 | 0.4

## Entities
The system also classified entities. For convenience the correct entities have been added as well.

| id | sentence id | source | entity | value | start | stop | confidence [%]
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | gold standard | date | tomorrows | 8 | 17 | 
| 1 | 0 | gold standard | location | London | 36 | 43 | 
| 2 | 0 | classification | location | London | 36 | 43 | 83.3
| 3 | 1 | gold standard | weather | rain | 8 | 13 | 
| 4 | 1 | gold standard | date | monday | 18 | 25 |
| 5 | 1 | classification | weather | rain on | 8 | 16 | 62.8
| 6 | 1 | classification | date | monday | 18 | 25 |
