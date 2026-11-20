# Fake News Detection Using NLP and Machine Learning

## Included files
- `app.py` — Streamlit user interface and model training code.
- `dataset.csv` — small illustrative dataset to make the demo run immediately.
- `requirements.txt` — Python packages needed.
- `README.txt` — setup instructions.

## Run on Windows (VS Code)

1. Install Python 3.10 or 3.11 from https://www.python.org/downloads/ .
   During installation, select **Add Python to PATH**.
2. Extract this ZIP file.
3. Open the extracted folder in VS Code.
4. Open Terminal > New Terminal.
5. Create a virtual environment (recommended):

   `python -m venv .venv`

6. Activate it:

   PowerShell:
   `.\.venv\Scripts\Activate.ps1`

   If PowerShell blocks activation, open Command Prompt and use:
   `.venv\Scripts\activate.bat`

7. Install packages:

   `python -m pip install -r requirements.txt`

8. Start the app:

   `python -m streamlit run app.py`

9. A browser tab should open. Paste a headline/article and click **Analyze News**.
   If it does not open automatically, copy the local URL shown in the terminal (usually
   `http://localhost:8501`) into your browser.

## Dataset format
`dataset.csv` must have exactly these required column names:
- `text`: headline/article text
- `label`: `real` or `fake`

The included dataset is only a small educational example. For a serious project evaluation,
replace it with a larger, documented, appropriately licensed dataset. Keep labels consistent.
Do not claim high accuracy based on the illustrative sample. Use a held-out test set and discuss
dataset limitations in your report.

## Important limitation
This is a text classifier, not a fact-checking service. It learns statistical patterns from its
training examples. It may label true stories as fake or fake stories as real. A high confidence
score is not proof of truth.
