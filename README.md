# Machine-Learning-in-AlloraNetwork
This Python script is a Machine Learning preprocessing and evaluation pipeline designed for use with data collected in the Allora Network. It prepares real-time or historical telemetry data (e.g., latency, performance metrics, agent evaluations) for model training using pandas and scikit-learn.
The script handles:

### Data Preprocessing

Feature engineering

Encoding of categorical variables

Normalization of numerical values

Train/test splitting

Ready-to-train output for ML models

---

### 1. Create a file named:
```
nano allora_ml_pipeline.py
```
### 2. Paste the full Python code:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load dataset
data = pd.read_csv('path_to_your_data.csv')

# Step 1: Cleaning
data.drop_duplicates(inplace=True)
data.fillna(data.mean(), inplace=True)  # Handle missing values

# Step 2: Feature Engineering (e.g., extract hour from timestamp)
data['hour'] = pd.to_datetime(data['timestamp']).dt.hour

# Step 3: Define features and target
X = data.drop('target_column', axis=1)
y = data['target_column']

# Step 4: Identify column types
categorical_cols = ['categorical_feature1', 'categorical_feature2']
numerical_cols = ['numerical_feature1', 'numerical_feature2', 'hour']

# Step 5: Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Step 6: Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Fit-transform training data and transform test data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```
### 3. Save and Close the File
#### If you're using nano:

Press CTRL + O to save

Press Enter to confirm

Press CTRL + X to exit

### 3.Prepare the Environment
```
pip install pandas scikit-learn
```
### 4.Run the Script
```
python allora_ml_pipeline.py
```
---
Output
<img width="966" height="95" alt="image" src="https://github.com/user-attachments/assets/429282d3-223d-47ee-b1a2-177adf87d7e4" />

