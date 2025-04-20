# 🧪 Electrical Conductivity Predictor

This project compares two machine learning models — a **Hybrid Quantum-Classical Neural Network (HQCNN)** and a **Classical Neural Network (MLP)** — to predict the **electrical conductivity** of materials.

## 📁 Project Structure
  📁 electrical-conductivity-predictor/
  ├── classical.py                              # Classical MLP model using PyTorch
  ├── hqcnn.py                                  # Hybrid Quantum-Classical model using PennyLane
  ├── comparemode.py                            # Script to compare outputs from both models
  ├── materials.csv                             # Input dataset with material properties
  ├── predicted_electrical_conductivity.csv     # Output from HQCNN model
  ├── classical_predicted_electrical_conductivity.csv  # Output from classical model
  ├── .gitignore                                # Git ignore rules
  ├── README.md                                 # Project overview and instructions
  └── venv/
  Before running the models, make sure you have the following installed:

# Steps to Run the Python Code
  1)Copy paste the python codes to the editor along with the csv file
  2)Change the directory to the location of the file
  3)Make sure the following is installed 
        1)python >= 3.8 
        2)pip install torch pandas matplotlib pennylane
  4)Start a virtual environment
    -python -m venv venv
    -source venv/bin/activate 
  5)Run the Hybrid model first then the Classical then the Comparision model
  6)All the required csv files will be updated on the files.
