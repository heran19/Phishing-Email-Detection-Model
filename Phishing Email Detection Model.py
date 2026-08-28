"""
Phishing Email Detection Model

Builds a machine learning model using Scikit-learn to classify emails
as "Phishing" or "Safe" based on their text content and URL-related
features. Trains on a labeled dataset, evaluates performance using
accuracy and a confusion matrix, and allows classification of new
sample emails.
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

DATASET_FILE = "phishing_email_dataset.csv"


def build_sample_dataset(path=DATASET_FILE):
    """
    Creates a small labeled sample dataset of phishing and legitimate
    emails for demonstration purposes. In a real-world scenario, this
    would be replaced with a larger dataset (e.g. from Kaggle).
    """
    data = {
        "text": [
            "Urgent! Your account has been suspended. Click here to verify: http://secure-bank-login.tk/verify",
            "Congratulations! You have won a $1000 gift card. Claim now at http://free-prize-claim.xyz",
            "Dear user, your password will expire today. Update immediately at http://account-update-portal.info",
            "verify your identity now or your account will be locked http://paypa1-secure.com/login",
            "You have a pending refund of $500. Confirm your bank details at http://refund-claim-now.net",
            "Your Apple ID has been locked for security reasons. Unlock it now: http://apple-id-verify.support",
            "Action required: unusual sign-in activity detected. Confirm your identity: http://login-alert-secure.ru",
            "Get free Netflix premium for a year, click to activate: http://netflix-free-offer.click",
            "Your package could not be delivered. Reschedule here: http://delivery-reschedule-now.top",
            "Final notice: your invoice is overdue. Pay immediately to avoid penalty: http://invoice-payment-portal.biz",
            "Hi team, please find attached the quarterly report for review before Friday's meeting.",
            "Reminder: our project sync call is scheduled for 3 PM tomorrow in the main conference room.",
            "Thanks for your email. I'll review the document and get back to you by end of day.",
            "Here are the notes from today's stand-up meeting. Let me know if I missed anything.",
            "Please find attached the invoice for last month's consulting services. Let us know if you have questions.",
            "Happy to catch up next week, does Tuesday afternoon work for a quick call?",
            "The design mockups have been updated based on your feedback, please take a look when you can.",
            "Your subscription renewal receipt is attached. No action is needed at this time.",
            "Great meeting you at the conference! Looking forward to staying in touch.",
            "The server maintenance window is scheduled for Sunday 2 AM to 4 AM as planned.",
        ],
        "label": [
            "Phishing", "Phishing", "Phishing", "Phishing", "Phishing",
            "Phishing", "Phishing", "Phishing", "Phishing", "Phishing",
            "Safe", "Safe", "Safe", "Safe", "Safe",
            "Safe", "Safe", "Safe", "Safe", "Safe",
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df


def extract_url_features(text):
    """Extract simple URL-based signals that often correlate with phishing."""
    urls = re.findall(r"https?://[^\s]+", text)
    suspicious_tlds = (".tk", ".xyz", ".info", ".click", ".top", ".biz", ".ru", ".support")
    has_suspicious_tld = any(url.lower().endswith(suspicious_tlds) or
                              any(tld in url.lower() for tld in suspicious_tlds) for url in urls)
    return {
        "url_count": len(urls),
        "has_suspicious_tld": int(has_suspicious_tld),
    }


def load_dataset(path=DATASET_FILE):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print("Dataset not found. Generating a sample dataset for demonstration...")
        return build_sample_dataset(path)


def train_model(df):
    """Train a Logistic Regression classifier on TF-IDF text features."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.3, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=500)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=["Phishing", "Safe"])
    report = classification_report(y_test, predictions, labels=["Phishing", "Safe"])

    return pipeline, accuracy, matrix, report


def classify_email(pipeline, email_text):
    prediction = pipeline.predict([email_text])[0]
    probability = pipeline.predict_proba([email_text]).max()
    url_features = extract_url_features(email_text)
    return prediction, probability, url_features


def main():
    print("=" * 55)
    print("Phishing Email Detection Model")
    print("=" * 55)

    df = load_dataset()
    print(f"\nDataset loaded: {len(df)} emails "
          f"({(df['label'] == 'Phishing').sum()} phishing, {(df['label'] == 'Safe').sum()} safe)")

    print("\nTraining model...")
    pipeline, accuracy, matrix, report = train_model(df)

    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print("                Phishing   Safe")
    print(f"Actual Phishing    {matrix[0][0]:>4}      {matrix[0][1]:>4}")
    print(f"Actual Safe        {matrix[1][0]:>4}      {matrix[1][1]:>4}")
    print("\nClassification Report:")
    print(report)

    print("-" * 55)
    print("Try classifying your own email text (leave blank to exit):")
    while True:
        sample = input("\nEnter email text: ").strip()
        if not sample:
            break
        prediction, probability, url_features = classify_email(pipeline, sample)
        print(f"Prediction: {prediction} (confidence: {probability:.2%})")
        print(f"URL signals: {url_features}")


if __name__ == "__main__":
    main()
