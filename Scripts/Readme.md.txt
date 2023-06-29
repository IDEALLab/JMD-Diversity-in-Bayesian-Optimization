# Scripts folder to generate data and/or replicate results produced for "How Diverse Initial Samples Help and Hurt Bayesian Optimizers."

> Plots in the main paper and supplement can be generated using the scripts in this directory. Use --help argument to look at the needed arguments for each script. Below is the table that can be used to see which script is reponsible for which Figures. 

| Sciript          | Figures                              |
|------------------|:------------------------------------:|
|dppcomparison     |SupFig1a, SupFig1b                    |
|Experiment2       |Fig4                                  |
|Experiment3-1     |SupFig3                               |
|Experiment3-2-1   |SupFig4                               |
|Experiment3-2-2   |SupFig5, SupFig6, Fig5                |
|Experiment3-2-3   |Fig6                                  |
|Experiment4	   |Fig1                                  |
|Experiment5-1	   |Fig2                                  |
|ExperimentA1      |SupFig8, SupFig11                     |
|ExperimentA3      |SupFig9, SupFig12                     |
|ExperimentA4      |SupFig10, SupFig13                    |
|ExperimentA5      |SupFig7, SupFig14, SupFig15, SupFig16 |
|wildcatwells-grid |Fig2                                  |

Each script needs certain arguments, most arguments default to the needed argument for the figures in the paper. You can find all the arguments needed by using --help argument for each script. Below you can see a table describing the needed argument to plot data using each script, this is after the data for that plot has been generated.

Common arg for all scripts:
>intent: plot-data or data-gen

Further specifications for each figure can be found below:

| Figures                              |needed args												   |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------|
|SupFig1a	                       |N/A		    										           |
|SupFig1b	                       |N/A		      											   |
|Fig4                                  | if intent: plot-data, grid_type: difference, plot_type: grid      					   |
|SupFig3                               | if intent: plot-data, grid_type: difference								   |		   
|SupFig4	                       |N/A													   |
|SupFig5                     	       |smoothness_range: (0.6,0.7,0.2), ruggedness_range: (0.4,0.5,0.2), intent: plot-data, plot_type: individual |
|Fig5                     	       |N/A			|
|Fig6                                  | if intent: plot-data : grid_type: difference, plot_type: grid       |
|Fig1                                  |N/A			|
|Fig2                                  |training_size: 5	|
|SupFig8	                       |fun_name: wildcatwells, intent: plot-data, plot_type: grid, grid_type: difference, dimension_mat: (2,4,1) |
|SupFig11	                       |fun_name: Rastrigin, intent: plot-data, plot_type: grid, grid_type: difference |
|SupFig9		               |fun_name: wildcatwells, intent: plot-data, dimension_mat: (2,4,1)|
|SupFig12		               |fun_name: Rastrigin, intent: plot-data|
|SupFig10	                       |fun_name: wildcatwells, intent: plot-data, plot_type: grid, grid_type: difference, dimension_mat: (2,4,1) |
|SupFig13	                       |fun_name: Rastrigin, intent: plot-data, plot_type: grid, grid_type: difference |
|SupFig7 			       |fun_name: wildcatwells, intent: plot-data, dimension_mat: (2,4,1)|
|SupFig14			       |fun_name: Rastrigin, intent: plot-data, dimension_mat: (2,6,1)|
|SupFig15			       |fun_name: Rosenbrock, intent: plot-data, dimension_mat: (2,6,1)|
|SupFig16			       |fun_name: Sphere, intent: plot-data, dimension_mat: (2,6,1)|
|Fig2                                  |N/A			|

For any questions, please create an issue and the authors will do their best to answer these questions as quickly as possible.