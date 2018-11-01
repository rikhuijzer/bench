# Results 

Results are stored here. The goal of these files is to be easy to process for a computer and human. 
Therefore the results are in CSV files. For each system and corpus the following CSV files are created.

## Intents
Below is an example of a system which classified one sentence and in half a second 
generated a completely incorrect response. The system was not so confident in its response.

| id | sentence | intent | classification | confidence [%] | time |
| --- | --- | --- | --- | --- | --- | 
| 0 | When is the tomorrows first flight to London? | departure time | greet | 20.3 | 0.5 |

## Entities
The system also classified some entities, these classifications are stored in the entities CSV. 
Note that the correct answer is not added in this file. 

### NO I DON'T LIKE  THIS. THE GOLD STANDARD IS NOT INCLUDED AND NO LINK IS MADE TO THE EARLIER TABLE
| id | sentence | entity | value | start | stop |
| --- | --- | --- | --- | --- | --- |
| 0 
