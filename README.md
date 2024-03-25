# Code to Accompany the Journal Article: Estimation of Ground Reaction Forces Using Commercial Wearable Sensors and Deep Learning During Varied Treadmill Running

Setup Anaconda Environment using the accompanying environment.yml file (Python Version: 3.8.12)

When moving this code and accompanying dataset onto your local or remote machine ensure you have the following file structure: 

project_root (LSTM_GRF_Estimation)/  
│   
├── Dataset/   
│   ├── Info_Files/   
│   ├── P001/   
│   ├── P002/   
│   └── ...   
│   
├── **Run_LOSO.py**    
├── **Hyperparameter_Optimisation.py**    
├── ML_Models.py    
├── Dataset_Generator.py    
└── Training_Utility_Functions.py    

The functionality provided: 
* Run model training and evaluation using Leave One Subject Out validation **(Run_LOSO.py)** with the provided model architecture. There is the ability to change the variables being used as input into the model and the target variable being estimated by the model at the top of this script. 

* Run hyperparameter optimisation for a given set of input variables and target output **(Hyperparameter_Optimisation.py)**. The search space for each hyperparameter can be adjusted in the objective function at the top of the script. 

Before running the functional scripts the only variable you should need to change for it to work on your machine is to adjust the 'base_path' variable defined at the top of the current script to match your project_root/LSTM_GRF_Estimation folder path. 
*Model training is currently supported on CPU and CUDA-powered GPUs and will automatically use the CUDA-powered GPU if it is available*
