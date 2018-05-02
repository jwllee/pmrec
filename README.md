# pmrec
PMRec - Next event prediction model adapted from factorization machine

### How to run code?
Each experimental configuration is set up as a python script with a suffix of runX.py. For example, to run the experiments for the step size 2, simply execute:
```
python ./PMRec/run.py
```

One thing to note is that you would need to download the data and put them as a csv. Please download the data from https://data.4tu.nl/repository/collection:event_logs_synthetic.

To do your own experiments, feel free to use the converter to transform event data from a csv format to a process step format. Have a look at the ```dataRep``` directory for the different event data representation.
