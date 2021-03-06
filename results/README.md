# Results 

Results are stored here. The goal of these files is to be easy to process for a computer 
and human. Therefore the results are in YAML and CSV files. For each system and corpus a folder
is created containing the files listed below.

## `summary.yml`
The benchmark results are summarised in `summary.yml`. For example the file could contain:
```yaml
Corpus.CHATBOT:
  f1_intent_scores:
    macro:
      rasa-spacy: '0.939'
      other_system: '0.843'
```

##  Subfolders
This files can be used by the system to report system performance. Performance statistics include:
- F1 score (micro, macro scores for intent and entity)
- Average response time
- Training time

## `intents.csv`
Below is an example from a system which classified two sentences. 

| id | timestamp | sentence | gold standard | classification | confidence [%] | time [ms] |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2018-01-01 00:00:00 | When is tomorrows first flight to London? | departure time | greet | 20.3 | 50 | 0 |
| 1 | 2018-01-01 00:00:00 | Will it rain on monday? | get weather | get weather | 99.1 | 40 | 0 |

## `entities.csv`
The system also classified entities. For reader convenience the correct entities have been added as well.

| id | intent id | timestamp | source | entity | value | start | end | confidence [%]
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 2018-01-01 00:00:00 | gold standard | date | tomorrows | 8 | 17 | 
| 1 | 0 | 2018-01-01 00:00:00 | gold standard | location | London | 36 | 43 | 
| 2 | 0 | 2018-01-01 00:00:00 | classification | location | London | 36 | 43 | 83.3
| 3 | 1 | 2018-01-01 00:00:00 | gold standard | weather | rain | 8 | 13 | 
| 4 | 1 | 2018-01-01 00:00:00 | gold standard | date | monday | 18 | 25 |
| 5 | 1 | 2018-01-01 00:00:00 | classification | weather | rain on | 8 | 16 | 62.8
| 6 | 1 | 2018-01-01 00:00:00 | classification | date | monday | 18 | 25 |

## `statistics.yml`
For the general statistics YAML is used. These statistics are calculated for the most
recent run in `intents.csv` and `entities.csv`.

TODO: Consider reporting memory usage.

```yaml
system name: mock
corpus: Corpus.MOCK
run: '2018-01-01 00:00:00'
training time [seconds]: '0.2'
f1 scores:
  micro: '0.823'
  macro: '0.930'
  weighted: '0.670'
response_time [ms]:
  average: '20'
  standard deviation: '4'
```

## `classification report`
Generated by `sklearn.metrics.classification_report`