# BanditsFramework


The framework implement general purpose functionalities to optimise a specific KPI using bandits alghoritms based on customers data.
Standard algos are implemented and can be tested within the simulation tool. 
A new tree-based learner alghoritm is proposed (see Documentation); capable to auto generate optimised 
segmentations using customers data.


## DEVELOPMENT

* The application will automatically run in dev mode if you haven't set the environment variable ENVIRONMENT. It will basically run the same code, expect that it writes to local file storage instead of S3.

``` shell
$ python main.py AT
```

## Simulation Tool

Run it using the following: `python simulation_tool.py <HORIZON> <NUM_SEED_WEEKS>`  
The parameters for the Segmented Epsilon-greedy algorithm is set in the `parameter_grid` variable.  
The parameters for the standard Epsilon-greedy algorithm can be set in the `eps_greedy_params` variable.  

