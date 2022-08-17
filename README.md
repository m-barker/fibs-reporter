# ML FIBS Reporter
**FIBS** (**F**eature **I**mportance, **B**aseline-models and **S**purious correlation) Reporter is an open-source toolkit aimed at assisting users in understanding their machine learning dataset , by extracting the most important features, training and evaluating baseline model performance, and highlighting potential feature spurious correlation and outputting the toolkit's findings as a digestible pdf report. The toolkit is run through one-command line command; **all of the calculations, model training, and report generation happen automatically**.

All that is required as input on the command line is the path to the CSV file containing the ML data, and the name of the output (dependent) variable within the dataset. The toolkit will automatiicaly detrmine whether the task is regression or classification. Optionally, the toolkit can process and extract audio data, provided the name of the variable wihtin the CSV that contains the audio file for each observation is specified,

## Key features that are generated automatically:
- A traffic light score overview of potential bias within the dataset
- The most important features within a given dataset
- Baseline model performance on the given dataset 
- Recall plots by class or model residual plots
- Visuals of the most important features
- TSNE feature plot
- PCA baseline model to estimate complexity of ML task
- (Optionally) extract audio data features and run the above on these features

## Showcase of report features
![Feature Importance Example](images/feature_importance_example.png)

## Installation
The easiest way to install FIBS is by running the pip command:
```
pip install fibs-reporter
```

Note that FIBS requires Python >= V.3.8

## Usage
It is assumed that the ML data to be analysed is contained within a CSV file, where each column contains a variable, and each row contains an obervation. 

If the CSV file path is _path_to_my_csv_file_ and the output variable is called _target_ then the following command line command will output a report called _data_report.pdf_ in the current working directory:

```
fibs path_to_my_csv_file target --report_name data_report
```

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
