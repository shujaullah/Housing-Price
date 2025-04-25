# Housing Price Prediction Project

This project implements a machine learning solution for predicting house prices using various features. The project includes data preprocessing, model training, and a web application for predictions.

## Project Structure

```
├── pre-process-data/
│   └── team_pre_process.py    # Data preprocessing code
├── model-training/
│   └── best_model.pkl         # Trained model file
├── Model_Prediction_House_price.ipynb  # Model training and evaluation
├── app.py                     # Streamlit web application
└── README.md                  # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/housing-price-prediction.git
cd housing-price-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

1. Navigate to the preprocessing directory:
```bash
cd pre-process-data
```

2. Run the preprocessing script:
```bash
python team_pre_process.py
```

This will:
- Clean and preprocess the raw data
- Handle missing values
- Engineer new features
- Save processed datasets

### Model Training and Prediction

1. Open and run the Jupyter notebook:
```bash
jupyter notebook Model_Prediction_House_price.ipynb
```

The notebook contains:
- Model training code
- Evaluation metrics
- Prediction implementation

### Web Application

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

## Features

- Comprehensive data preprocessing pipeline
- Multiple model implementations
- Interactive web interface
- Real-time predictions

## Data Preprocessing Steps

1. Missing Value Treatment
   - Numerical features: Median imputation
   - Categorical features: Mode/None imputation

2. Feature Engineering
   - Total Square Footage
   - Total Bathrooms
   - House Age
   - Quality Score

3. Data Transformation
   - Categorical encoding
   - Feature scaling
   - Target variable transformation

## Model Training

The model training process includes:
- Data splitting
- Feature selection
- Hyperparameter tuning
- Model evaluation
- Model saving

## Web Application

The Streamlit app provides:
- User-friendly interface
- Input form for house features
- Real-time price predictions
- Visualization of results

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Contact

Your Name - ahsan.s@northeastern.edu, sudharsan.s@northeastern.edu, ingale.p@northeastern.edu

