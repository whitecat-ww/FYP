# src/train.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from feature_extractor import extract_features, features_to_vector, FEATURE_ORDER
from tqdm import tqdm
import os
import time

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/phishing_dataset.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "../models/phishing_rf.pkl")

def load_data(path):
    df = pd.read_csv(path)
    # expect columns: url,label
    if 'label' not in df.columns:
        raise ValueError("CSV must contain 'label' column with 1=phishing,0=legit")
    df = df[['url', 'label']].dropna()
    return df

def build_feature_df(df):
    rows = []
    times = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        url = r['url']
        t0 = time.time()
        feats = extract_features(url)
        t1 = time.time()
        times.append(t1 - t0)
        vec = features_to_vector(feats)
        rows.append(vec)
    feat_df = pd.DataFrame(rows, columns=FEATURE_ORDER)
    feat_df['label'] = df['label'].values
    print("Average feature extraction time per URL: {:.2f}s".format(sum(times)/len(times)))
    return feat_df

def impute_missing(df):
    # simple imputation: replace -1 and -999 with median
    df = df.copy()
    for col in df.columns:
        if col == 'label':
            continue
        med = df.loc[~df[col].isin([-1, -999]), col].median()
        df[col] = df[col].replace([-1, -999], med)
        df[col] = df[col].fillna(med)
    return df

def main():
    df = load_data(DATA_PATH)
    feat_df = build_feature_df(df)
    feat_df = impute_missing(feat_df)
    X = feat_df.drop('label', axis=1)
    y = feat_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training RandomForest...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # save model and feature order
    os.makedirs(os.path.join(os.path.dirname(__file__), "../models"), exist_ok=True)
    joblib.dump({'model': clf, 'feature_order': FEATURE_ORDER}, MODEL_OUT)
    print("Saved model to", MODEL_OUT)

if __name__ == "__main__":
    main()
