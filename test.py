print(33)

export PYTHONPATH="$(pwd)/lmms-eval:$PYTHONPATH"


decode eviction 0
|  Tasks  |Version|Filter|n-shot|      Metric       |   |Value|   |Stderr|
|---------|------:|------|-----:|-------------------|---|----:|---|------|
|convbench|    0.1|none  |     0|convbench_PPL      |↓  |4.419|±  |   N/A|
|convbench|    0.1|none  |     0|convbench_PPL_turn1|↓  |5.338|±  |   N/A|
|convbench|    0.1|none  |     0|convbench_PPL_turn2|↓  |4.257|±  |   N/A|
|convbench|    0.1|none  |     0|convbench_PPL_turn3|↓  |4.037|±  |   N/A|


decode eviction 1
|  Tasks  |Version|Filter|n-shot|      Metric       |   |Value |   |Stderr|
|---------|------:|------|-----:|-------------------|---|-----:|---|------|
|convbench|    0.1|none  |     0|convbench_PPL      |↓  | 6.526|±  |   N/A|
|convbench|    0.1|none  |     0|convbench_PPL_turn1|↓  | 5.481|±  |   N/A|
|convbench|    0.1|none  |     0|convbench_PPL_turn2|↓  | 4.541|±  |   N/A|
|convbench|    0.1|none  |     0|convbench_PPL_turn3|↓  |10.208|±  |   N/A|
