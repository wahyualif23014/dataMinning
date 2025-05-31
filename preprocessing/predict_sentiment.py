import pandas as pd
import os
from preprocessing.text_cleaner import preprocess_text
from transformers import pipeline
import torch
from tabulate import tabulate
import logging
from tqdm import tqdm
import warnings
import re

# Setup sek
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class ImprovedSentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analyzer dengan multiple methods"""
        self.device = 0 if torch.cuda.is_available() else -1
        self.model_name = "indobenchmark/indobert-base-p1"
        self.sentiment_pipeline = None
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()
        self._load_model()
    
    def _load_positive_words(self):
        """Load positive words for Indonesian"""
        return {
            'bagus', 'baik', 'enak', 'seru', 'mantap', 'keren', 'oke', 'ok', 'top',
            'hebat', 'luar biasa', 'sempurna', 'indah', 'cantik', 'menarik', 
            'recommended', 'rekomendasi', 'suka', 'senang', 'puas', 'memuaskan',
            'worth it', 'worthit', 'amazing', 'awesome', 'good', 'great', 'best',
            'terbaik', 'favorit', 'love', 'like', 'asik', 'asyik', 'menyenangkan',
            'berkualitas', 'berkelas', 'juara', 'mantul', 'kece', 'gokil', 'wow'
        }
    
    def _load_negative_words(self):
        """Load negative words for Indonesian"""
        return {
            'jelek', 'buruk', 'tidak enak', 'sepi', 'kotor', 'jorok', 'rusak',
            'mahal', 'mengecewakan', 'kecewa', 'zonk', 'bosan', 'membosankan',
            'tidak suka', 'benci', 'hate', 'bad', 'worst', 'terrible', 'awful',
            'gak bagus', 'ga bagus', 'nggak bagus', 'tidak bagus', 'jeblok',
            'payah', 'ancur', 'parah', 'amit-amit', 'najis', 'ngeri', 'seram',
            'tidak worth it', 'rugi', 'sia-sia', 'percuma', 'kacau', 'berantakan'
        }
    
    def _load_model(self):
        """Load IndoBERT sentiment analysis model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            logger.info(f"Using device: {'GPU' if self.device >= 0 else 'CPU'}")
            
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
                return_all_scores=True
            )
            logger.info("Model berhasil dimuat")
        except Exception as e:
            logger.warning(f"Gagal memuat IndoBERT model: {e}")
            logger.info("Akan menggunakan rule-based sentiment analysis")
            self.sentiment_pipeline = None
    
    def rule_based_sentiment(self, text):
        """Rule-based sentiment analysis untuk fallback"""
        if not text or text.strip() == "":
            return 'netral', 0.5
        
        text_lower = text.lower()
        
        # elek ro apik e
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # score 
        if positive_count > negative_count:
            confidence = min(0.9, 0.6 + (positive_count - negative_count) * 0.1)
            return 'positif', confidence
        elif negative_count > positive_count:
            confidence = min(0.9, 0.6 + (negative_count - positive_count) * 0.1)
            return 'negatif', confidence
        else:
            return 'netral', 0.5
    
    def detect_sentiment_model(self, text):
        """Deteksi sentimen menggunakan IndoBERT model"""
        if not text or text.strip() == "":
            return 'netral', 0.0
        
        try:
            text_input = str(text)[:512]
            results = self.sentiment_pipeline(text_input)[0]
            
            best_result = max(results, key=lambda x: x['score'])
            label_raw = best_result['label'].lower()
            confidence = best_result['score']
            
            # label mapping
            if any(keyword in label_raw for keyword in ['positive', 'positif', 'pos', 'label_1']):
                sentiment_label = 'positif'
            elif any(keyword in label_raw for keyword in ['negative', 'negatif', 'neg', 'label_0']):
                sentiment_label = 'negatif'
            else:
                sentiment_label = 'netral'
            
            return sentiment_label, round(confidence, 4)
            
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")
            return None, 0.0
    
    def detect_sentiment(self, text):
        """
        Hybrid sentiment detection: Model + Rule-based
        """
        if not text or text.strip() == "":
            return 'netral', 0.0
        
        # model 1
        if self.sentiment_pipeline:
            model_result, model_confidence = self.detect_sentiment_model(text)
            
            # model jika
            if model_result and model_confidence > 0.7:
                return model_result, model_confidence
        
        # Fallback 
        rule_result, rule_confidence = self.rule_based_sentiment(text)
        
        if self.sentiment_pipeline and model_result:
            if model_result == rule_result:
                combined_confidence = min(0.95, (model_confidence + rule_confidence) / 2 + 0.1)
                return model_result, combined_confidence
            else:
                # If they disagree, use rule-based for Indonesian text
                return rule_result, rule_confidence
        
        return rule_result, rule_confidence
    
    def process_comments(self, df):
        """Process all comments in DataFrame"""
        logger.info("Memulai preprocessing dan analisis sentimen...")
        
        # loading
        df['komentar'] = df['komentar'].astype(str)
        
        # Text processing text
        logger.info("Melakukan text preprocessing...")
        tqdm.pandas(desc="Preprocessing")
        df['komentar_bersih'] = df['komentar'].progress_apply(preprocess_text)
        
        # bar scentimen
        logger.info("Melakukan analisis sentimen...")
        tqdm.pandas(desc="Sentiment Analysis")
        sentiment_results = df['komentar_bersih'].progress_apply(self.detect_sentiment)
        
        # Split result sentimen
        df['label_sentimen'] = sentiment_results.apply(lambda x: x[0])
        df['confidence_score'] = sentiment_results.apply(lambda x: x[1])
        
        return df
    
    def generate_summary(self, df):
        """Generate sentiment summary statistics"""
        sentiment_counts = df['label_sentimen'].value_counts()
        total_comments = len(df)
        
        summary = {
            'Total Komentar': total_comments,
            'Positif': sentiment_counts.get('positif', 0),
            'Negatif': sentiment_counts.get('negatif', 0),
            'Netral': sentiment_counts.get('netral', 0),
            'Persentase Positif': f"{(sentiment_counts.get('positif', 0) / total_comments * 100):.2f}%",
            'Persentase Negatif': f"{(sentiment_counts.get('negatif', 0) / total_comments * 100):.2f}%",
            'Persentase Netral': f"{(sentiment_counts.get('netral', 0) / total_comments * 100):.2f}%"
        }
        
        return summary

def main():
    """Main function untuk menjalankan analisis sentimen"""
    input_path = "data/komentar_instagram.csv"
    output_path = "data/komentar_dengan_sentimen.csv"
    
    # Validasi file
    if not os.path.exists(input_path):
        logger.error(f"File tidak ditemukan: {input_path}")
        return
    
    # Read CSV 
    df = None
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            logger.info(f"Mencoba membaca file dengan encoding: {encoding}")
            df = pd.read_csv(input_path, on_bad_lines='skip', encoding=encoding)
            logger.info(f"Berhasil membaca {len(df)} baris data dengan encoding {encoding}")
            break
        except Exception as e:
            logger.warning(f"Gagal dengan encoding {encoding}: {e}")
            continue
    
    if df is None:
        logger.error("Gagal membaca file dengan semua encoding yang dicoba")
        return
    
    # Validasi kolom
    if "komentar" not in df.columns:
        logger.error("Kolom 'komentar' tidak ditemukan dalam dataset.")
        logger.info(f"Kolom yang tersedia: {list(df.columns)}")
        return
    
    # Clean comments
    initial_count = len(df)
    df = df.dropna(subset=['komentar'])
    df = df[df['komentar'].str.strip() != '']
    logger.info(f"Data setelah cleaning: {len(df)} baris (dihapus: {initial_count - len(df)} baris)")
    
    if len(df) == 0:
        logger.error("Tidak ada data valid untuk diproses")
        return
    
    # Initialize analyzer
    try:
        analyzer = ImprovedSentimentAnalyzer()
    except Exception as e:
        logger.error(f"Gagal menginisialisasi analyzer: {e}")
        return
    
    # Process analysis
    df_processed = analyzer.process_comments(df)
    
    # Save 
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_processed.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Hasil disimpan di: {output_path}")
    except Exception as e:
        logger.error(f"Gagal menyimpan file: {e}")
        return
    
    # Generate 
    summary = analyzer.generate_summary(df_processed)
    
    print("\n" + "="*60)
    print("📊 RINGKASAN ANALISIS SENTIMEN INSTAGRAM COMMENTS")
    print("="*60)
    
    for key, value in summary.items():
        print(f"{key:<20}: {value}")
    
    print("\n" + "="*60)
    print("📋 SAMPLE HASIL ANALISIS")
    print("="*60)
    
    # Show sample result
    sample_data = df_processed[['komentar', 'komentar_bersih', 'label_sentimen', 'confidence_score']].head(10)
    
    sample_data_display = sample_data.copy()
    sample_data_display['komentar'] = sample_data_display['komentar'].apply(
        lambda x: x[:60] + "..." if len(str(x)) > 60 else x
    )
    sample_data_display['komentar_bersih'] = sample_data_display['komentar_bersih'].apply(
        lambda x: x[:40] + "..." if len(str(x)) > 40 else x
    )
    
    print(tabulate(
        sample_data_display,
        headers=['Komentar Asli', 'Komentar Bersih', 'Sentimen', 'Confidence'],
        tablefmt='grid',
        showindex=False
    ))
    
    # cheklis all in one
    label_dist = df_processed['label_sentimen'].value_counts()
    print(f"\n🔍 DISTRIBUSI LABEL:")
    for label, count in label_dist.items():
        print(f"  {label.capitalize()}: {count} ({count/len(df_processed)*100:.1f}%)")
    
    print(f"\n✅ Analisis selesai! Total {len(df_processed)} komentar berhasil diproses.")
    print(f"📁 File hasil: {output_path}")

if __name__ == "__main__":
    main()