import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import logging
from datetime import datetime
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class SentimentModelTrainer:
    def __init__(self, input_path="data/komentar_dengan_sentimen.csv", model_dir="model"):
        """
        Initialize model trainer
        
        Args:
            input_path (str): Path ke file CSV hasil sentiment analysis
            model_dir (str): Directory untuk menyimpan model
        """
        self.input_path = input_path
        self.model_dir = model_dir
        self.vectorizer = None
        self.best_model = None
        self.model_name = None
        
        # Buat direktori model jika belum ada
        os.makedirs(self.model_dir, exist_ok=True)
        
    def load_and_validate_data(self):
        """Load dan validasi data"""
        logger.info(f"Loading data dari: {self.input_path}")
        
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"File tidak ditemukan: {self.input_path}")
        
        # Load data dengan error handling
        try:
            df = pd.read_csv(self.input_path, encoding='utf-8')
            logger.info(f"Berhasil load {len(df)} baris data")
        except Exception as e:
            logger.error(f"Gagal membaca file: {e}")
            raise
        
        # Validasi kolom yang dibutuhkan
        required_cols = ['komentar_bersih', 'label_sentimen']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            # Coba kolom alternatif untuk backward compatibility  
            if 'label' in df.columns and 'label_sentimen' not in df.columns:
                df['label_sentimen'] = df['label']
                logger.info("Menggunakan kolom 'label' sebagai 'label_sentimen'")
            else:
                raise ValueError(f"Kolom yang dibutuhkan tidak ditemukan: {missing_cols}")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocessing dan balancing data"""
        logger.info("Memulai preprocessing data...")
        
        # Hapus data kosong atau null
        initial_count = len(df)
        df = df.dropna(subset=['komentar_bersih', 'label_sentimen'])
        df = df[df['komentar_bersih'].str.strip() != '']
        df = df[df['label_sentimen'].str.strip() != '']
        
        logger.info(f"Data setelah cleaning: {len(df)} baris (dihapus: {initial_count - len(df)} baris)")
        
        # Filter hanya 3 kelas yang diinginkan
        valid_labels = ['positif', 'negatif', 'netral']
        df = df[df['label_sentimen'].isin(valid_labels)]
        
        logger.info(f"Data setelah filtering label: {len(df)} baris")
        
        # Cek minimal 2 kelas dengan data yang cukup
        label_counts = df['label_sentimen'].value_counts()
        valid_labels_with_data = label_counts[label_counts >= 10].index.tolist()
        
        if len(valid_labels_with_data) < 2:
            raise ValueError(f"Dataset harus memiliki minimal 2 kelas dengan minimal 10 sampel. "
                           f"Label yang valid: {valid_labels_with_data}")
        
        # Filter hanya label yang memiliki data cukup
        df = df[df['label_sentimen'].isin(valid_labels_with_data)]
        
        return df
    
    def balance_data(self, df, strategy='undersample'):
        """
        Balance data dengan berbagai strategi
        
        Args:
            df: DataFrame
            strategy: 'undersample', 'oversample', atau 'none'
        """
        logger.info(f"Distribusi label sebelum balancing:")
        label_counts = df['label_sentimen'].value_counts()
        print(tabulate(
            [(label, count, f"{count/len(df)*100:.2f}%") for label, count in label_counts.items()],
            headers=['Label', 'Count', 'Percentage'],
            tablefmt='grid'
        ))
        
        if strategy == 'undersample':
            # Undersample ke jumlah kelas minoritas
            min_count = label_counts.min()
            df_balanced = (
                df.groupby('label_sentimen', group_keys=False)
                  .apply(lambda x: x.sample(min(len(x), min_count), random_state=42))
                  .reset_index(drop=True)
            )
            
        elif strategy == 'oversample':
            # Simple oversample dengan replacement
            max_count = label_counts.max()
            df_balanced = (
                df.groupby('label_sentimen', group_keys=False)
                  .apply(lambda x: x.sample(max_count, replace=True, random_state=42))
                  .reset_index(drop=True)
            )
            
        else:  # strategy == 'none'
            df_balanced = df.copy()
        
        if strategy != 'none':
            logger.info(f"Distribusi label setelah {strategy}:")
            balanced_counts = df_balanced['label_sentimen'].value_counts()
            print(tabulate(
                [(label, count, f"{count/len(df_balanced)*100:.2f}%") for label, count in balanced_counts.items()],
                headers=['Label', 'Count', 'Percentage'],
                tablefmt='grid'
            ))
        
        return df_balanced
    
    def create_features(self, X_text):
        """Create TF-IDF features"""
        logger.info("Membuat TF-IDF features...")
        
        # TF-IDF dengan parameter yang dioptimalkan
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Unigram dan bigram
            max_df=0.9,          # Hapus kata yang terlalu umum
            min_df=2,            # Hapus kata yang terlalu jarang
            max_features=10000,  # Batasi jumlah feature
            sublinear_tf=True,   # Logarithmic scaling
            stop_words=None      # Tidak menggunakan stop words (sudah dibersihkan di preprocessing)
        )
        
        X = self.vectorizer.fit_transform(X_text)
        logger.info(f"TF-IDF features shape: {X.shape}")
        
        return X
    
    def train_multiple_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models dan pilih yang terbaik"""
        logger.info("Training multiple models...")
        
        # Calculate class weights untuk handling imbalanced data
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        # Define models
        models = {
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight='balanced',
                max_depth=10
            ),
            'SVM': SVC(
                kernel='linear', 
                random_state=42, 
                class_weight='balanced',
                probability=True
            )
        }
        
        best_score = 0
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Train dan test
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'predictions': y_pred
            }
            
            logger.info(f"{name} - CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f}), Test: {accuracy:.4f}")
            
            # Update best model
            if accuracy > best_score:
                best_score = accuracy
                self.best_model = model
                self.model_name = name
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Hyperparameter tuning untuk model terbaik"""
        logger.info(f"Hyperparameter tuning untuk {self.model_name}...")
        
        if self.model_name == 'Naive Bayes':
            param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]}
            base_model = MultinomialNB()
            
        elif self.model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
            
        elif self.model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
            base_model = SVC(random_state=42, class_weight='balanced', probability=True)
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def evaluate_model(self, X_test, y_test):
        """Evaluasi model yang sudah di-train"""
        logger.info("Evaluating final model...")
        
        y_pred = self.best_model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return accuracy, report, cm, y_pred
    
    def save_model(self):
        """Save model dan vectorizer"""
        model_path = os.path.join(self.model_dir, "sentiment_model.pkl")
        vectorizer_path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'training_date': datetime.now().isoformat(),
            'feature_count': len(self.vectorizer.get_feature_names_out()) if hasattr(self.vectorizer, 'get_feature_names_out') else 'N/A'
        }
        
        import json
        metadata_path = os.path.join(self.model_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model disimpan di: {model_path}")
        logger.info(f"Vectorizer disimpan di: {vectorizer_path}")
        logger.info(f"Metadata disimpan di: {metadata_path}")
    
    def plot_confusion_matrix(self, cm, labels):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save plot
        plot_path = os.path.join(self.model_dir, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Confusion matrix plot disimpan di: {plot_path}")

def main():
    """Main function untuk training model"""
    # Path yang sesuai dengan output dari sentiment analysis sebelumnya
    input_path = "data/komentar_dengan_sentimen.csv"
    model_dir = "model"
    
    try:
        # Initialize trainer
        trainer = SentimentModelTrainer(input_path, model_dir)
        
        # Load dan validasi data
        df = trainer.load_and_validate_data()
        
        # Preprocessing
        df_clean = trainer.preprocess_data(df)
        
        # Balance data (pilih strategi: 'undersample', 'oversample', atau 'none')
        df_balanced = trainer.balance_data(df_clean, strategy='undersample')
        
        # Prepare features
        X_text = df_balanced["komentar_bersih"].astype(str)
        y = df_balanced["label_sentimen"].astype(str)
        
        # Create TF-IDF features
        X = trainer.create_features(X_text)
        
        # Train-test split dengan stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Train multiple models
        results = trainer.train_multiple_models(X_train, X_test, y_train, y_test)
        
        # Hyperparameter tuning untuk model terbaik
        best_params = trainer.hyperparameter_tuning(X_train, y_train)
        
        # Final evaluation
        accuracy, report, cm, y_pred = trainer.evaluate_model(X_test, y_test)
        
        # Display results
        print("\n" + "="*80)
        print("🎯 HASIL TRAINING MODEL SENTIMENT ANALYSIS")
        print("="*80)
        
        print(f"\n📊 Model Terbaik: {trainer.model_name}")
        print(f"🎯 Akurasi: {accuracy:.4f}")
        print(f"⚙️  Best Parameters: {best_params}")
        
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\n📋 Confusion Matrix:")
        labels = sorted(y.unique())
        print(tabulate(
            cm,
            headers=labels,
            showindex=labels,
            tablefmt='grid'
        ))
        
        # Model comparison
        print("\n🔍 Perbandingan Model:")
        comparison_data = []
        for name, result in results.items():
            comparison_data.append([
                name,
                f"{result['cv_mean']:.4f} ± {result['cv_std']:.4f}",
                f"{result['test_accuracy']:.4f}"
            ])
        
        print(tabulate(
            comparison_data,
            headers=['Model', 'CV Score', 'Test Accuracy'],
            tablefmt='grid'
        ))
        
        # Save model
        trainer.save_model()
        
        # Plot confusion matrix
        trainer.plot_confusion_matrix(cm, labels)
        
        print(f"\n✅ Training selesai! Model terbaik ({trainer.model_name}) disimpan di folder '{model_dir}'")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()