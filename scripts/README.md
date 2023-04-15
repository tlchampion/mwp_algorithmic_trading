# Contents

This folder contains: 

* a maintenance script that should be run on a periodic basis to ensure the optimal functioning of the application 
* a script to regenerate the machine learing models for making buy/sell predictions


## Usage Instructions

### Data Refresh

You may refresh the data at anytime by running the script:

```
python create_data_files.py
```

Do the the volume of data and images being prepared, this process can take some time so please be patient when choosing to refresh.


### Model Retrainging

To retrain and save the machine learning models please run the save_best_models.py script:

```
python save_best_models.py
```

Alternatively, you may run the save_best_models.ipynb notebook in jupyter lab