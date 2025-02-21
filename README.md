# Neurocast - Workload Forecasting
Neurocast - Workload Forecasting is a system designed to predict workload patterns in Amazon Redshift clusters. It analyzes query spikes, execution time, and resource usage using multiple baseline models and deep learning models. The goal is to enable autonomous resource allocation for Redshift clusters, improving cost-effectiveness and system efficiency.

This project is built using the dataset - [Redset information](https://github.com/amazon-science/redset)

## Features
- User-friendly interface to load, train and predict the work load pattern built on gradIO
- User can select the particular instance on which one wants to predict the workload
- Complete visualization implemented for pre-trained analysis and also for the prediction data.
- There available totally 4 models namely - DeepAR, Deepseq, Naiveseasonal, PatchTST.
- Compares the results using different error metrics for all the models.
- Mainly focuses on instances which have high workload.

## How to use
- Clone this repository to your local machine
- Use any code editor to open the folder in a specific path and run the forecasting_app.py to launch the application and work around in that or also to see the machine learning model result directly one can run the workload_forecasting.ipynb. 
- At first select the model using which user want to predict the load.
- Once model has been selected, load the data and select visualize data option to see the data distribution of time series data for all three metrics - query count, bytes scanned and execution time.
- User can also select the instance for which one wants to forecast the result. 
- Then train the model using the option given for the user to train and choose the predict button to see the final predicted result for next one week.
- Visualize the prediction error metrics and values on the application.



## Installation
To install the required dependencies and setup App Engine environment, run the following command in your terminal:

```bash
git clone https://gitos.rrze.fau.de/utn-machine-intelligence/teaching/ml-ws2425-final-projects/g8.git
```
- For python pip environment
```bash
python -m venv ./env
source env/bin/activate
pip install -r piprequirements.txt
```

- For conda environments
```bash
conda create --name myenv python=3.10 -y
conda activate myenv
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt
```

## Running the application
Follow these steps to run the Workload Forecasting Gradio App:
```bash
venv\Scripts\activate
cd path/to/your/project
python -m w_forecasting_app.py
```
Running on local URL:  http://127.0.0.1:7860


## Miscelleanous
### Swagger UI

## License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.