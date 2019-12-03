# BanditsFramework
===================================

This project aims to optimize the discount on the second box, for customers obtained by a gir channel.

It outlines the latest models, algorithms, data sources, and configuration required to run the project.

## DEVELOPMENT

* The application will automatically run in dev mode if you haven't set the environment variable ENVIRONMENT. It will basically run the same code, expect that it writes to local file storage instead of S3.

``` shell
$ python main.py AT
```

## Simulation Tool

Run it using the following: `python simulation_tool.py <HORIZON> <NUM_SEED_WEEKS>`  
The parameters for the Segmented Epsilon-greedy algorithm is set in the `parameter_grid` variable.  
The parameters for the standard Epsilon-greedy algorithm can be set in the `eps_greedy_params` variable.  

