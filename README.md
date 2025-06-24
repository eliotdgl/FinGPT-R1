**:file_folder: Repository Structure**
```
FinGPT‑R1/
├── data/                   # CSVs and pickled datasets
│   └── local_data/
├── BERT/
│   └── bert_train.py       # BERT + LoRA training script
├── sentiment_analysis/
│   ├── sentiment_model_class.py  # wrapper for PEFT model
│   └── controlled_environment.py
├── tokenization/
│   └── preprocess_text.py
├── train.py             # train script for all models
├── test.py             # test & evaluation pipeline
├── models/                 # Saved models
└── results/                # Evaluation results and plots
```

**:rocket: Getting Started**
```
git clone -b Tokenization-Branch https://github.com/eliotdgl/FinGPT‑R1.git
cd FinGPT‑R1
pip install -r requirements.txt
```
If you are using a cluster–where internet access may be limited or unavailable–make sure to preload the `bert-base-uncased`–and `yiyanghkust/finbert-tone` for baseline results–models. It can be simply done by running `cache_models.py` locally on a machine with internet access.

**:bulb: Training**  
Run the full training for all or specific models via CLI:
```
python train.py --model all    # train all models
python train.py --model Bert BertEC ...
```
The available models are: Bert, BertEC, HashT, HashTEC, DelT, DelTEC

