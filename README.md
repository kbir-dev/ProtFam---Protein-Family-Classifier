# ProtFam - Protein Family Classifier

A full-stack web application for classifying proteins into their respective families based on physicochemical properties. The application uses machine learning to predict protein families and provides both single-protein analysis and bulk classification capabilities.

## Features

- **Single Protein Analysis**: Interactive interface for analyzing individual proteins
- **Bulk Analysis**: Process multiple proteins through CSV upload
- **Visualization**: Interactive charts showing prediction probabilities and feature importance
- **Downloadable Results**: Export classification results as CSV files

## Installation

### 1. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Backend Server

```bash
# Start the FastAPI backend server
uvicorn backend:app --reload
```

The backend API will be available at `http://localhost:8000`.

### Starting the Frontend

```bash
# Start the Streamlit frontend
streamlit run frontend.py
```

The frontend will be available at `http://localhost:8501`.

## Input Parameters

### Basic Properties
- Nneg: Number of Negative Charges
- Npos: Number of Positive Charges
- Exc1: Excitation Coefficient 1
- Exc2: Excitation Coefficient 2
- I.Index: Instability Index
- A.Index: Aliphatic Index

### Advanced Properties
- GRAVY: GRAVY Score
- Ser: Serine Count
- Thr: Threonine Count
- Tyr: Tyrosine Count
- ExpAA: Expected AA
- PredHel: Predicted Helices

### Tools used to get above params
- Nneg, Npos, Exc1, Exc2, I_Index, A_Index, GRAVY, Ser, Thr, Tyr - Expasy ProtParam
- ExpAA, PredHel - THMMM by DTU

## Bulk Analysis

For bulk analysis, prepare a CSV file with the following columns:
- Nneg, Npos, Exc1, Exc2, I_Index, A_Index, GRAVY, Ser, Thr, Tyr, ExpAA, PredHel

The application will process the file and provide a downloadable CSV with predictions.

## Model Information

The application uses a Random Forest Classifier trained on protein physicochemical properties to predict protein families. The model can classify proteins into the following families:
- DNA Repair Protein
- Decarboxylase
- Defensin
- Heat Shock Protein
- RNA Binding Protein
- Voltage Gated Channel

## Technologies Used

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Machine Learning**: scikit-learn
- **Data Processing**: Pandas
- **Visualization**: Plotly
 