# Heart Disease Prediction

A machine learning project for heart disease prediction with a custom Flask web dashboard.

## Structure

- `notebook/Heart.ipynb` - project notebook
- `data/heart.csv` - dataset
- `models/` - trained models, scaler, feature columns and comparison results
- `templates/index.html` - dashboard pages
- `static/style.css` - dashboard styling
- `app.py` - Flask application

## Run

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

The dashboard contains Home, EDA, Model Comparison, Confusion Matrix, ROC Curve and Explainability pages.
