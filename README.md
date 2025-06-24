📂 Repository Structure

FinGPT‑R1/ 
├── BERT/ 
│   └── bert_train.py       # BERT + LoRA training script 
├── sentiment_analysis/ 
│   ├── sentiment_model_class.py  # wrapper for PEFT model 
│   └── controlled_environment.py   
├── tokenization/ 
│   └── preprocess_text.py 
├── pipeline.py             # train script for all models 
├── evaluate.py             # test & evaluation pipeline 
├── data/local_data/        # CSVs and pickled datasets 
├── models/                 # Saved models 
└── results/                # Evaluation results and plots 
