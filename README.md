**:brain:** Enhancing LLMs for Financial Sentiment Analysis

–––––––––––––––––––––––––

**:file_folder: Repository Structure**
```
FinGPT‑R1/
├── BERT/
│   └── bert_training.py 
├── data/
│   └── FinancialPhraseBank-v1.0/     
│   └── financial_news/          #
│   └── import_save_HFdata.py    #
│   └── local_data_loader.py     #
├── FinGPTR1_pipeline/
│   └── training/                #
│   └── custom_embeddings.py
│   └── delt_training.py         #
│   └── FGPTR1_tokenizer.py
├── sentiment_analysis/
│   └── controlled_environment.py  #
│   └── sentiment_model_class.py   #
├── tokenization/
│   └── financial_vocab/
│   └── numericals/
│   └── stock_tickers/
│   └── vocabulary/
│   └── preprocess_text.py      #
├── cache_models.py             
├── train.py                    #
├── test.py                     #
├── models/                     # Trained models (not saved in the Github repository
├── results/                   
└── requirements.txt/            
```

**:rocket: Getting Started**
```
git clone -b Tokenization-Branch https://github.com/eliotdgl/FinGPT‑R1.git
cd FinGPT‑R1
pip install -r requirements.txt
```
If you are using a cluster–where internet access may be limited or unavailable–ensure you have preloaded the `bert-base-uncased`–and `yiyanghkust/finbert-tone` for baseline results–models. You can do this by running `cache_models.py` locally on a machine with internet access.

**:heavy_check_mark: Available models**
```
- Bert: Standard BERT with LoRA fine-tuning
- BertEC: Standard BERT with LoRA, plus fine-tuning of the embedding layer and classification head
- DelT: BERT augmented with the three delimiter tokens and LoRA fine-tuning
- DelTEC: DelT with additional fine-tuning of the embedding layer and classification head
- HashT: BERT augmented with Hashtags tokens and LoRA fine-tuning
- HashTEC: HashT with additional fine-tuning of the embedding layer and classification head
```

**:bulb: Train**  
Run the full training for all or specific models via CLI:
```
python train.py --model all    # train all models
python train.py --model Bert BertEC ...
```
Trained models are saved under `models/`.

**:bar_chart: Evaluate**
```
python test.py --model all    # evaluate all trained models
python test.py --model Bert BertEC ...
python test.py --model ... --baseline    # include FinBERT baseline scores
```
Results are saved under `results/`; including accuracy, calibration and reliability plots under `results/plots/`.
