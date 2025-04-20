# 🧪 Electrical Conductivity Predictor

This project compares two machine learning models — a **Hybrid Quantum-Classical Neural Network (HQCNN)** and a **Classical Neural Network (MLP)** — to predict the **electrical conductivity** of materials.

---

# 📁 Project Structure

├── classical.py                               # Classical MLP model using PyTorch  
├── hqcnn.py                                   # Hybrid Quantum-Classical model using PennyLane  
├── comparemode.py                             # Script to compare outputs from both models  
├── materials.csv                              # Input dataset with material properties  
├── predicted_electrical_conductivity.csv      # Output from HQCNN model  
├── classical_predicted_electrical_conductivity.csv  # Output from classical model  
├── .gitignore                                 # Git ignore rules  
├── README.md                                  # Project overview and instructions  
└── venv/                                      # Virtual environment (ignored in Git)  

## ⚙️ Requirements

- Python ≥ 3.8
- Install dependencies:

```bash
pip install torch pandas matplotlib pennylane
🚀 How to Run the Project
Clone the repo or copy the files into your project folder.

Change to the project directory:
cd electrical-conductivity-predictor
(Optional but recommended) Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate

Install the required packages:
pip install torch pandas matplotlib pennylane

Run the scripts in the following order:
python hqcnn.py        # Runs the Hybrid Quantum-Classical model
python classical.py    # Runs the Classical MLP model
python comparemode.py  # Compares predictions from both models
All output .csv files will be generated in the same directory for review and analysis.
