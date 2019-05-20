# SemEval-2017 Task 7

The link to the task page is given here: http://alt.qcri.org/semeval2017/task7/

The task contains 3 subtasks, of which the first subtask is a binary classification task to detect whether or not a given context contains a pun. You are required to perform only the first subtask. You may try the second and third subtasks for bonus marks.

The link to the codalab platform for the first subtask is given here:

https://competitions.codalab.org/competitions/15705

You may use the trial data provided in the codalab page (also attached here for convenience).

As usual you are required to submit codes. Please also include a readme file with instructions on how to run your code.

http://alt.qcri.org/semeval2017/task7/data/uploads/semeval2017_task7.tar.xz


# RUN


- To run the code : Open the terminal in this directory and copy paste this command 'python3 subtask_1.py'

- Keep all the files in the same directory

- The code uses homographic and heterographic test data and homographic and heterographic gold data to train 2 types of classifiers and predict

- The output of the code is a table of accuracy and f1 score of the two classifiers: SVM and RandomForest
