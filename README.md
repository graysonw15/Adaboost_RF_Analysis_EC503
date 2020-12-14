# EC503Proj - Instructions to Reproduce results

* To run decision trees demo:
    
    * run the main under the decision trees folder with "python main.py" and this will input some questions to the user on how to proceed. To make the dataset (once you get to that part) simply click on the purple grid.

* To run random forests demo:
    * run the "python gaussian_classifier.py" and "linear_data_prediction.py" to get the evaluation of random forests on the datasets shown in the report that contain a gaussian distribution or a easily separable dataset. Note for the "python linear_data_prediction.py" you have to alter the name of the csv file to experiment with all the linearly separable datasets. The linearly separable datasets are "linearly_separable_in_one_d.csv" and ""linearly_separable_in_two_d.csv".
    * run the "python spiral_classifier.py" and "real_world_classifier.py" to get the evaluation of random forests on the datasets shown in the report that contain a spiral distribution or real-world cardiovascular dataset. The "python spiral_classifier.py" finds the ideal Hyperparameters as well as tests the number of trees v. CCR (to do so, you will have to uncomment the block at the end). The "real_world_classifier.py" uses SkLearn to identify ideal Hyperparameters for the CVD dataset.

* To run adaboost demo:
    * run "python gaussian_classifier.py" and "linear_data_prediction.py" to get the evaluation of adaboost on the datasets shown in the report that contain a gaussian distribution or a easily separable dataset. Note for the "python linear_data_prediction.py" you have to alter the name of the csv file to experiment with all the linearly separable datasets. The linearly separable datasets are "linearly_separable_in_one_d.csv" and ""linearly_separable_in_two_d.csv".
    * run ada_spirals.py to run the adaboost algorithm on the spiral dataset
    * run ada_rw.py to run the adaboost algorithm on the real world dataset
