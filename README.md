# Fire Alarm Testing Price Prediction System

A machine learning system for predicting prices for fire alarm testing services. This is implemented as a regression problem using various ML algorithms.

## Project Structure

```
Avarn/
├── data/                    # Data directory
│   └── training_data.csv    # Training dataset
├── models/                  # Saved models and preprocessors
│   ├── best_model.pkl       # Best performing model
│   ├── preprocessor.pkl     # Data preprocessor
│   └── training_results.json # Training metrics
├── scripts/                 # Python scripts
│   ├── preprocess.py        # Data preprocessing module
│   ├── train_model.py       # Model training script
│   └── predict.py           # Prediction script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

Your training data CSV file should contain:
- **Features**: Various attributes that affect pricing (e.g., building size, number of alarms, location, service type, etc.)
- **Target**: A `price` column with the actual testing prices

Example features might include:
- `building_size` (numeric): Size of the building in square meters
- `num_alarms` (numeric): Number of fire alarms to test
- `location` (categorical): Location/city
- `service_type` (categorical): Type of testing service
- `building_age` (numeric): Age of the building
- `price` (numeric): Target variable - the testing price

## Usage

### 1. Prepare Your Data

Create a CSV file with your training data and place it in the `data/` directory. Ensure it has a `price` column as the target variable.

### 2. Train Models

Train all available regression models:

```bash
cd scripts
python train_model.py --data ../data/training_data.csv --output ../models
```

Options:
- `--data`: Path to training data CSV (default: `../data/training_data.csv`)
- `--output`: Directory to save models (default: `../models`)
- `--test-size`: Proportion of data for testing (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)

The script will:
- Train multiple regression models (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, SVR)
- Evaluate each model's performance
- Save the best model automatically
- Save training results and metrics

### 3. Make Predictions

Use the trained model to make predictions on new data:

```bash
cd scripts
python predict.py --input ../data/new_data.csv --output ../data/predictions.csv
```

Options:
- `--model`: Path to model file (default: `../models/best_model.pkl`)
- `--preprocessor`: Path to preprocessor file (default: `../models/preprocessor.pkl`)
- `--input`: Path to input CSV file with features (required)
- `--output`: Path to save predictions (optional)
- `--format`: Output format - `simple` (just predictions) or `detailed` (with input features) (default: `detailed`)

## Available Models

The system trains and compares the following regression algorithms:

1. **Linear Regression**: Basic linear model
2. **Ridge Regression**: L2 regularization
3. **Lasso Regression**: L1 regularization
4. **Random Forest**: Ensemble of decision trees
5. **Gradient Boosting**: Sequential ensemble method
6. **Support Vector Regression (SVR)**: Kernel-based regression

The best model (lowest test RMSE) is automatically selected and saved.

## Model Evaluation Metrics

The training script evaluates models using:
- **RMSE** (Root Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better
- **R² Score** (Coefficient of Determination): Higher is better (closer to 1.0)

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your training data in data/training_data.csv

# 3. Train models
cd scripts
python train_model.py --data ../data/training_data.csv

# 4. Make predictions on new data
python predict.py --input ../data/new_data.csv --output ../data/predictions.csv
```

## Customization

### Adding New Features

Simply add new columns to your training CSV. The preprocessor will automatically handle:
- Numeric features: Scaled using StandardScaler
- Categorical features: Encoded using LabelEncoder

### Modifying Models

Edit `scripts/train_model.py` to:
- Add new models to the `models` dictionary
- Adjust hyperparameters
- Change model selection criteria

### Data Preprocessing

Modify `scripts/preprocess.py` to:
- Add custom feature engineering
- Change scaling methods
- Implement different encoding strategies

## Notes

- The preprocessor automatically handles missing values and categorical encoding
- All models are saved for comparison, but the best one is used by default
- The system ensures feature consistency between training and prediction
- Unknown categories in prediction data are handled gracefully

## Troubleshooting

**Error: "Feature columns not defined"**
- Make sure you've trained a model first before making predictions

**Error: "Missing features"**
- Ensure your prediction data has all the same feature columns as training data

**Poor model performance**
- Check data quality and feature engineering
- Try collecting more training data
- Consider feature selection or dimensionality reduction
