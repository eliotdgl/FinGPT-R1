**:brain: Enhancing Large Language Models for Financial Sentiment Analysis**  
A framework for fine-tuning and evaluating transformer-based models on domain-specific financial sentiment tasks, with improved handling of numerical.

---

**:file_folder: Repository Structure**
```
FinGPT‑R1/
├── BERT/
│   └── bert_training.py         # BERT-specific training
├── data/
│   └── FinancialPhraseBank-v1.0/     
│   └── financial_news/          
│   └── import_save_HFdata.py    # Import (and save) Hugging Face datasets
│   └── local_data_loader.py     # Custom dataset loader
├── FinGPTR1_pipeline/
│   └── training/                # LoRA and other training processes
│   └── custom_embeddings.py     # Embedding logic (for new special tokens)
│   └── delt_training.py         # Delimiters (DelT) method training
│   └── FGPTR1_tokenizer.py      # Tokenizer setup and customization
├── sentiment_analysis/
│   └── controlled_environment.py  # Generated data
│   └── sentiment_model_class.py   # Wrapper class
├── tokenization/
│   └── financial_vocab/
│   └── numericals/
│   └── stock_tickers/
│   └── vocabulary/
│   └── preprocess_text.py      # Token preprocessing and cleaning
├── cache_models.py             # Pre-load HF models for offline use
├── train.py                    # CLI entry for training
├── test.py                     # CLI entry for testing
├── models/                     # Trained models (not saved in the Github repository)
├── results/                   
└── requirements.txt        
```

---

**:rocket: Getting Started**
```
git clone https://github.com/eliotdgl/FinGPT-R1.git
cd FinGPT-R1/
pip install -r requirements.txt
```
If you are using a cluster–where internet access may be limited or unavailable–ensure you have preloaded the `bert-base-uncased`–and `yiyanghkust/finbert-tone` for baseline results–models. You can do this by running `python cache_models.py` locally on a machine with internet access.  

---

**:heavy_check_mark: Available models**
```
- Bert: Standard BERT with LoRA fine-tuning
- BertEC: Standard BERT with LoRA, plus fine-tuning of the embedding layer and classification head
- DelT: BERT augmented with the three delimiter tokens and LoRA fine-tuning
- DelTEC: DelT with additional fine-tuning of the embedding layer and classification head
- HashT: BERT augmented with Hashtags tokens and LoRA fine-tuning
- HashTEC: HashT with additional fine-tuning of the embedding layer and classification head
```

---

**:arrows_counterclockwise: Import Data**  
```
python data/local_data_loader.py
```

---

**:bulb: Train**  
Run the full training for all or specific models via CLI:
```
python train.py --model all    # train all models
python train.py --model Bert BertEC ...
```
Trained models are saved under `models/`.

---

**:bar_chart: Evaluate**
```
python test.py --model all    # evaluate all trained models
python test.py --model Bert BertEC ...
python test.py --model ... --baseline    # include FinBERT baseline scores
```
Results are saved under `results/`; including accuracy, calibration and reliability plots under `results/plots/`.
