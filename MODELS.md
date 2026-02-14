# Model Artifacts

This directory contains trained model artifacts used for production inference.

## Files

- `best_model.pkl`: The best performing model (currently Gradient Boosting) with conformal prediction statistics
- `preprocessor.pkl`: Feature pipeline (OneHotEncoder + ColumnTransformer) for preprocessing

## Why These Files Are in Git

These model files are included in the repository for:
1. **Reproducibility**: Anyone cloning the repo can immediately run the API without training
2. **Demo/Portfolio**: Quick demonstration of the complete system
3. **Size**: Model files are ~500KB, which is acceptable for Git

## For Production

In a production environment, consider:
- **Git LFS**: For larger models (>100MB)
- **Model Registry**: Use MLflow model registry or cloud storage (S3, GCS)
- **CI/CD**: Automatically download models during deployment
- **Versioning**: Use semantic versioning for model artifacts

## Regenerating Models

To retrain and update models:

```bash
python app/models/train_with_mlflow.py
```

This will:
1. Train multiple models and compare performance
2. Save the best model to `models/best_model.pkl`
3. Save the feature pipeline to `models/preprocessor.pkl`
4. Log experiments to MLflow

## MLflow Tracking

MLflow experiment tracking is stored in `models/mlruns/` (gitignored). To view:

```bash
python start_mlflow_ui.py
```

Then open http://localhost:5000
