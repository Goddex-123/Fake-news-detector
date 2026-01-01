"""
Text Classifier for Fake News Detection

Implements both TF-IDF + Logistic Regression and DistilBERT classifiers.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import NLP_CONFIG, PATHS, RANDOM_SEED
from nlp.preprocessing import TextPreprocessor, FeatureEngineer, TfidfFeatureExtractor


class NewsDataset(Dataset):
    """PyTorch Dataset for BERT model."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = 1 if self.labels[idx] == 'fake' else 0
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TfidfClassifier:
    """TF-IDF + Logistic Regression baseline classifier."""
    
    def __init__(self):
        self.feature_extractor = TfidfFeatureExtractor()
        self.feature_engineer = FeatureEngineer()
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight='balanced'
        )
        self.preprocessor = TextPreprocessor()
    
    def train(self, X_text, y):
        """Train the TF-IDF classifier."""
        print("\n" + "="*50)
        print("Training TF-IDF Classifier")
        print("="*50)
        
        # Extract TF-IDF features
        X_tfidf = self.feature_extractor.fit_transform(X_text)
        
        # Extract linguistic features
        print("Extracting linguistic features...")
        df_temp = pd.DataFrame({'text': X_text, 'label': y})
        df_features = self.feature_engineer.add_features_to_dataframe(df_temp, 'text')
        
        # Get linguistic feature columns
        feature_cols = [col for col in df_features.columns if col not in ['text', 'label']]
        X_linguistic = df_features[feature_cols].values
        
        # Combine TF-IDF and linguistic features
        from scipy.sparse import hstack, csr_matrix
        X_combined = hstack([X_tfidf, csr_matrix(X_linguistic)])
        
        print(f"Combined feature shape: {X_combined.shape}")
        
        # Train classifier
        print("Training logistic regression...")
        self.classifier.fit(X_combined, y)
        
        # Get feature importance
        self._analyze_feature_importance(X_text, y)
        
        print("✅ TF-IDF classifier training complete!")
        
        return self
    
    def predict(self, X_text):
        """Predict labels for new texts."""
        # Extract features
        X_tfidf = self.feature_extractor.transform(X_text)
        
        df_temp = pd.DataFrame({'text': X_text})
        df_features = self.feature_engineer.add_features_to_dataframe(df_temp, 'text')
        feature_cols = [col for col in df_features.columns if col != 'text']
        X_linguistic = df_features[feature_cols].values
        
        from scipy.sparse import hstack, csr_matrix
        X_combined = hstack([X_tfidf, csr_matrix(X_linguistic)])
        
        return self.classifier.predict(X_combined)
    
    def predict_proba(self, X_text):
        """Predict probabilities for new texts."""
        X_tfidf = self.feature_extractor.transform(X_text)
        
        df_temp = pd.DataFrame({'text': X_text})
        df_features = self.feature_engineer.add_features_to_dataframe(df_temp, 'text')
        feature_cols = [col for col in df_features.columns if col != 'text']
        X_linguistic = df_features[feature_cols].values
        
        from scipy.sparse import hstack, csr_matrix
        X_combined = hstack([X_tfidf, csr_matrix(X_linguistic)])
        
        return self.classifier.predict_proba(X_combined)
    
    def _analyze_feature_importance(self, X_text, y):
        """Analyze top features for each class."""
        print("\n🔍 Top features by class:")
        
        X_tfidf = self.feature_extractor.vectorizer.transform(
            X_text.apply(self.preprocessor.preprocess)
        )
        top_features = self.feature_extractor.get_top_features_by_class(X_tfidf, y, top_n=10)
        
        print("\nTop fake news indicators:")
        for feature, score in top_features['fake'][:10]:
            print(f"  {feature}: {score:.4f}")
        
        print("\nTop real news indicators:")
        for feature, score in top_features['real'][:10]:
            print(f"  {feature}: {score:.4f}")
    
    def save(self, path):
        """Save the model."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Model saved to {path}")
    
    @staticmethod
    def load(path):
        """Load a saved model."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class BertClassifier:
    """DistilBERT-based classifier for fake news detection."""
    
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            NLP_CONFIG['bert_model_name']
        )
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def train(self, X_text, y, validation_split=0.1):
        """Train the BERT classifier."""
        print("\n" + "="*50)
        print("Training DistilBERT Classifier")
        print("="*50)
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_text.values, y.values,
            test_size=validation_split,
            random_state=RANDOM_SEED,
            stratify=y
        )
        
        # Create datasets
        train_dataset = NewsDataset(
            X_train, y_train, self.tokenizer, NLP_CONFIG['bert_max_length']
        )
        val_dataset = NewsDataset(
            X_val, y_val, self.tokenizer, NLP_CONFIG['bert_max_length']
        )
        
        # Initialize model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            NLP_CONFIG['bert_model_name'],
            num_labels=2
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(PATHS['bert_model']),
            num_train_epochs=NLP_CONFIG['bert_epochs'],
            per_device_train_batch_size=NLP_CONFIG['bert_batch_size'],
            per_device_eval_batch_size=NLP_CONFIG['bert_batch_size'],
            learning_rate=NLP_CONFIG['bert_learning_rate'],
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            seed=RANDOM_SEED,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train
        print(f"Training on {len(train_dataset)} samples...")
        print(f"Validating on {len(val_dataset)} samples...")
        trainer.train()
        
        print("✅ BERT classifier training complete!")
        
        return self
    
    def predict(self, X_text):
        """Predict labels for new texts."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in X_text:
                encoding = self.tokenizer(
                    str(text),
                    add_special_tokens=True,
                    max_length=NLP_CONFIG['bert_max_length'],
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()
                predictions.append('fake' if pred == 1 else 'real')
        
        return np.array(predictions)
    
    def predict_proba(self, X_text):
        """Predict probabilities for new texts."""
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for text in X_text:
                encoding = self.tokenizer(
                    str(text),
                    add_special_tokens=True,
                    max_length=NLP_CONFIG['bert_max_length'],
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                probabilities.append(probs)
        
        return np.array(probabilities)
    
    def save(self, path):
        """Save the model."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"✅ Model saved to {path}")
    
    @staticmethod
    def load(path):
        """Load a saved model."""
        classifier = BertClassifier()
        classifier.model = DistilBertForSequenceClassification.from_pretrained(path)
        classifier.model.to(classifier.device)
        return classifier


def evaluate_classifier(y_true, y_pred, y_pred_proba, model_name="Model"):
    """Evaluate classifier performance."""
    print("\n" + "="*50)
    print(f"{model_name} Evaluation Results")
    print("="*50)
    
    # Accuracy and F1
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label='fake')
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['real', 'fake'])
    print("\nConfusion Matrix:")
    print(cm)
    
    # ROC AUC if probabilities available
    if y_pred_proba is not None:
        y_true_binary = (y_true == 'fake').astype(int)
        fake_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba
        auc = roc_auc_score(y_true_binary, fake_proba)
        print(f"\nROC AUC: {auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'confusion_matrix': cm
    }


def train_and_evaluate_models(data_path=None):
    """Train both TF-IDF and BERT classifiers and compare results."""
    
    # Load data
    if data_path is None:
        data_path = PATHS['news_articles']
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Combine headline and body for classification
    df['text'] = df['headline'] + ' ' + df['body']
    
    print(f"Loaded {len(df)} articles")
    print(f"  Real: {len(df[df['label']=='real'])}")
    print(f"  Fake: {len(df[df['label']=='fake'])}")
    
    # Split data
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=NLP_CONFIG['test_size'],
        random_state=RANDOM_SEED,
        stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train TF-IDF classifier
    print("\n" + "🤖 Training TF-IDF Classifier")
    tfidf_clf = TfidfClassifier()
    tfidf_clf.train(X_train, y_train)
    
    # Evaluate TF-IDF
    y_pred_tfidf = tfidf_clf.predict(X_test)
    y_pred_proba_tfidf = tfidf_clf.predict_proba(X_test)
    tfidf_metrics = evaluate_classifier(
        y_test, y_pred_tfidf, y_pred_proba_tfidf, "TF-IDF Classifier"
    )
    
    # Save TF-IDF model
    tfidf_clf.save(PATHS['tfidf_model'])
    
    # Train BERT classifier (optional, can be slow)
    print("\n" + "🤖 Training BERT Classifier")
    print("Note: This may take 10-30 minutes depending on hardware...")
    
    bert_clf = BertClassifier()
    bert_clf.train(X_train, y_train)
    
    # Evaluate BERT
    y_pred_bert = bert_clf.predict(X_test.values)
    y_pred_proba_bert = bert_clf.predict_proba(X_test.values)
    bert_metrics = evaluate_classifier(
        y_test, y_pred_bert, y_pred_proba_bert, "BERT Classifier"
    )
    
    # Save BERT model
    bert_clf.save(PATHS['bert_model'])
    
    # Compare models
    print("\n" + "="*50)
    print("📊 Model Comparison")
    print("="*50)
    print(f"{'Metric':<20} {'TF-IDF':<15} {'BERT':<15}")
    print("-" * 50)
    print(f"{'Accuracy':<20} {tfidf_metrics['accuracy']:<15.4f} {bert_metrics['accuracy']:<15.4f}")
    print(f"{'F1 Score':<20} {tfidf_metrics['f1']:<15.4f} {bert_metrics['f1']:<15.4f}")
    
    return tfidf_clf, bert_clf, tfidf_metrics, bert_metrics


if __name__ == "__main__":
    train_and_evaluate_models()
