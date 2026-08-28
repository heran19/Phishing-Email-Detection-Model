# Phishing-Email-Detection-Model
A machine learning model built with Scikit-learn that classifies emails as **Phishing** or **Safe** based on textual content and URL-related features. This project was completed as part of the **Thiranex Cybersecurity Internship** (Task-03).
 
## About
 
Phishing emails remain one of the most common attack vectors used to steal credentials and deliver malware. This project trains a text classification model on a labeled dataset of phishing and legitimate emails, using TF-IDF vectorization and Logistic Regression, and evaluates its performance using accuracy and a confusion matrix. It also extracts simple URL-based signals (URL count, suspicious top-level domains) to support the analysis.
 
## Features
 
- Trains on a labeled dataset of phishing and legitimate emails (auto-generates a sample dataset if none is provided)
- Extracts and analyzes email text using TF-IDF vectorization
- Detects URL-based signals such as suspicious top-level domains (`.xyz`, `.tk`, `.click`, etc.)
- Classifies emails as "Phishing" or "Safe" using a Logistic Regression classifier
- Displays accuracy, a confusion matrix, and a full classification report
- Allows interactive testing of custom email text after training
## How It Works
 
1. The dataset (email text + label) is loaded, or a sample dataset is generated if none exists.
2. Text is split into training and test sets (70/30 split).
3. A Scikit-learn `Pipeline` combines a `TfidfVectorizer` with a `LogisticRegression` classifier.
4. The model is trained and evaluated on the held-out test set.
5. Accuracy, a confusion matrix, and a classification report are printed.
6. The user can then enter custom email text to see a live prediction with confidence score and URL signal breakdown.
## Requirements
 
- Python 3.x
- `scikit-learn`
- `pandas`
Install dependencies:
 
```bash
pip install scikit-learn pandas
```
 
## Usage
 
1. Clone this repository or download the script.
```bash
git clone https://github.com/herusprady/phishing-email-detection.git
cd phishing-email-detection
```
 
2. Run the script:
```bash
python phishing_email_detection.py
```
 
3. Example output:
```
Phishing Email Detection Model
Dataset loaded: 20 emails (10 phishing, 10 safe)
 
Accuracy: 100.00%
 
Confusion Matrix (rows = actual, columns = predicted):
                Phishing   Safe
Actual Phishing       3         0
Actual Safe           0         3
 
Try classifying your own email text (leave blank to exit):
Enter email text: Urgent! Verify your account now at http://secure-verify-portal.xyz
Prediction: Phishing (confidence: 61.92%)
URL signals: {'url_count': 1, 'has_suspicious_tld': 1}
```
 
## Example
 
| Email Snippet | Prediction |
|---------------|-----------|
| "Your account has been suspended, click here to verify..." | Phishing |
| "Please find attached the quarterly report for review." | Safe |
 
## File Structure
 
```
phishing-email-detection/
│
├── phishing_email_detection.py     # Main program
├── phishing_email_dataset.csv      # Auto-generated sample dataset
└── README.md                       # Project documentation
```
 
## Learning Outcomes
 
- Applying supervised machine learning to a text classification problem
- Feature engineering with TF-IDF vectorization
- Model evaluation using accuracy, confusion matrices, and classification reports
- Identifying URL-based heuristics commonly used in phishing detection
## Acknowledgements
 
This project was completed as part of the Cybersecurity Internship offered by **Thiranex**.
 
## License
 
This project is open source and available for educational purposes.
 
