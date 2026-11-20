# Fake News Detection Using NLP and Machine Learning

A machine learning-based web application that analyzes news text and predicts whether it is likely to be **Real or Fake** using Natural Language Processing (NLP).

## Project Overview

The Fake News Detection System uses text preprocessing and machine learning techniques to classify news articles based on patterns learned from a labeled dataset.

The application provides an interactive interface built using Streamlit.

## Technologies Used

- Python
- Natural Language Processing (NLP)
- Machine Learning
- Streamlit
- Pandas
- Scikit-learn

## Project Files

| File | Description |
|---|---|
| `app.py` | Streamlit user interface and model training code |
| `dataset.csv` | Illustrative dataset for demonstration |
| `requirements.txt` | Required Python packages |
| `README.txt` | Additional setup instructions |

## Features

- Interactive Streamlit interface
- News headline/article input
- Text preprocessing
- Machine learning-based classification
- Real or Fake prediction

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/saniya-d568/fake-news-detection.git
```

### 2. Open the project folder

```bash
cd fake-news-detection
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## Dataset Information

The included dataset is a small illustrative dataset intended for demonstration purposes.

For meaningful evaluation, use a larger, documented and appropriately licensed dataset with a separate test set.

## Limitations

This application is a text classifier, not a professional fact-checking service. Its predictions are based on patterns learned from training data and may be incorrect.

A high confidence score does not guarantee that a news article is true.

## Future Improvements

- Train using a larger real-world dataset.
- Improve model accuracy through proper evaluation.
- Explore advanced NLP models such as BERT.
- Add multilingual news classification.

## Author

**Saniya**

GitHub: [@saniya-d568](https://github.com/saniya-d568)
