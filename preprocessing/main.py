import pandas as pd
import os
from preprocessing.text_cleaner import preprocess_text
from transformers import pipeline
import torch
from tabulate import tabulate
import logging
from tqdm import tqdm
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

class InstagramSentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analyzer dengan IndoBERT model"""
        self.device = 0 if torch.cuda.is_available() else -1
        self.model_name = "indobenchmark/indobert-base-p1"
        self.sentiment_pipeline = None
        self._load_model()
    
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
            logger.error(f"Gagal memuat model: {e}")
            raise
    
    def detect_sentiment(self, text):
        """
        Deteksi sentimen dari teks dengan confidence score
        
        Args:
            text (str): Teks yang akan dianalisis
            
        Returns:
            tuple: (label_sentimen, confidence_score)
        """
        if not text or text.strip() == "":
            return 'netral', 0.0
        
        try:
            # Truncate text untuk menghindari error token limit
            text_input = str(text)[:512]
            
            # Prediksi sentimen
            results = self.sentiment_pipeline(text_input)[0]
            
            best_result = max(results, key=lambda x: x['score'])
            label_raw = best_result['label'].lower()
            confidence = best_result['score']
            
            if any(keyword in label_raw for keyword in ['positive', 'positif', 'pos']):
                sentiment_label = 'positif'
            elif any(keyword in label_raw for keyword in ['negative', 'negatif', 'neg']):
                sentiment_label = 'negatif'
            else:
                sentiment_label = 'netral'
            
            return sentiment_label, round(confidence, 4)
            
        except Exception as e:
            logger.warning(f"Gagal mendeteksi sentimen untuk teks: '{text[:50]}...'. Error: {e}")
            return 'netral', 0.0
    
    def process_comments(self, df):
        """
        Proses semua komentar dalam DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame dengan kolom 'komentar'
            
        Returns:
            pandas.DataFrame: DataFrame dengan kolom tambahan sentiment
        """
        logger.info("Memulai preprocessing dan analisis sentimen...")
        
        df['komentar'] = df['komentar'].astype(str)
        
        # Preprocessing teks
        logger.info("Melakukan text preprocessing...")
        tqdm.pandas(desc="Preprocessing")
        df['komentar_bersih'] = df['komentar'].progress_apply(preprocess_text)
        
        # Analisis sentimen dengan progress bar
        logger.info("Melakukan analisis sentimen...")
        tqdm.pandas(desc="Sentiment Analysis")
        sentiment_results = df['komentar_bersih'].progress_apply(self.detect_sentiment)
        
        # Split hasil sentimen dan confidence
        df['label_sentimen'] = sentiment_results.apply(lambda x: x[0])
        df['confidence_score'] = sentiment_results.apply(lambda x: x[1])
        
        return df
    
    def generate_summary(self, df):
        """Generate summary statistik sentimen"""
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
    
    # Validasi file input
    if not os.path.exists(input_path):
        logger.error(f"File tidak ditemukan: {input_path}")
        return
    
    # Baca data CSV
    try:
        logger.info(f"Membaca file: {input_path}")
        df = pd.read_csv(input_path, on_bad_lines='skip', encoding='utf-8')
        logger.info(f"Berhasil membaca {len(df)} baris data")
    except Exception as e:
        logger.error(f"Gagal membaca file CSV: {e}")
        return
    
    # Validasi kolom
    if "komentar" not in df.columns:
        logger.error("Kolom 'komentar' tidak ditemukan dalam dataset.")
        logger.info(f"Kolom yang tersedia: {list(df.columns)}")
        return
    
    # Hapus baris dengan komentar kosong atau null
    initial_count = len(df)
    df = df.dropna(subset=['komentar'])
    df = df[df['komentar'].str.strip() != '']
    logger.info(f"Data setelah membersihkan komentar kosong: {len(df)} baris (dihapus: {initial_count - len(df)} baris)")
    
    if len(df) == 0:
        logger.error("Tidak ada data valid untuk diproses")
        return
    
    # Inisialisasi analyzer
    try:
        analyzer = InstagramSentimentAnalyzer()
    except Exception as e:
        logger.error(f"Gagal menginisialisasi analyzer: {e}")
        return
    
    # Proses analisis sentimen
    df_processed = analyzer.process_comments(df)
    
    # Simpan hasil
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_processed.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Hasil disimpan di: {output_path}")
    except Exception as e:
        logger.error(f"Gagal menyimpan file: {e}")
        return
    
    # Generate dan tampilkan summary
    summary = analyzer.generate_summary(df_processed)
    
    print("\n" + "="*60)
    print("📊 RINGKASAN ANALISIS SENTIMEN INSTAGRAM COMMENTS")
    print("="*60)
    
    for key, value in summary.items():
        print(f"{key:<20}: {value}")
    
    print("\n" + "="*60)
    print("📋 SAMPLE HASIL ANALISIS (10 KOMENTAR PERTAMA)")
    print("="*60)
    
    # Tampilkan sample hasil dengan format yang rapi
    sample_data = df_processed[['komentar', 'komentar_bersih', 'label_sentimen', 'confidence_score']].head(10)
    
    # Potong komentar yang terlalu panjang untuk display
    sample_data_display = sample_data.copy()
    sample_data_display['komentar'] = sample_data_display['komentar'].apply(
        lambda x: x[:50] + "..." if len(str(x)) > 50 else x
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
    
    print(f"\n✅ Analisis selesai! Total {len(df_processed)} komentar berhasil diproses.")
    print(f"📁 File hasil: {output_path}")

if __name__ == "__main__":
    main()