# Results 

TODO: Check what data formats others have used for this problem. 

TODO: Consider storing it all in one JSON to improve post-processing ease.

Results are stored here. The goal of these files is to be easy to process for a computer and human. 
Therefore the results are in YAML and CSV files. For each system and corpus the following files are created.

This files can be used by the system to report system performance. Performance statistics include:
- F1 score (micro, macro scores for intent and entity)
- Average response time
- Training time

## `general.yml`
For the general information YAML is used. 
```yaml
system name: mock-server

corpus: Mock

training time [seconds]: float

f1 scores:
  micro: float
  macro: float

response_time [ms]:
  average: int
  standard deviation: int
```

## `intents.csv`
Below is an example from a system which classified two sentences. 

| id | sentence | intent | classification | confidence [%] | time [ms] |
| --- | --- | --- | --- | --- | --- | 
| 0 | When is tomorrows first flight to London? | departure time | greet | 20.3 | 50 |
| 1 | Will it rain on monday? | get weather | get weather | 99.1 | 40

## `entities.csv`
The system also classified entities. For reader convenience the correct entities have been added as well.

| id | sentence id | source | entity | value | start | stop | confidence [%]
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | gold standard | date | tomorrows | 8 | 17 | 
| 1 | 0 | gold standard | location | London | 36 | 43 | 
| 2 | 0 | classification | location | London | 36 | 43 | 83.3
| 3 | 1 | gold standard | weather | rain | 8 | 13 | 
| 4 | 1 | gold standard | date | monday | 18 | 25 |
| 5 | 1 | classification | weather | rain on | 8 | 16 | 62.8
| 6 | 1 | classification | date | monday | 18 | 25 |

## `summary.csv`
The micro F1 score calculation is omitted since it provides little interesting information and since
it could be confusing in combination with the macro F1 score.

entity type / intent | type | true + | false - | false + | precision | recall
| --- | --- | --- | --- | --- | --- | --- |
| departure time | intent | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 0 
| weather | entity | 1 | 1 | 2 | 0 | 0