import nltk
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

# # Load the tokenizer, model, and data collator
# MODEL_NAME = "google/flan-t5-base"
MODEL_PATH = "../autodl-tmp/models/flan-t5-base" # 从本地加载模型

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Acquire the training data from Hugging Face
# DATA_NAME = "yahoo_answers_qa"
# DATA_PATH = "../datasets/yahoo_answers_qa/nfL6.json"
DATA_PATH = "./nfL6.json"
# yahoo_answers_qa = load_dataset('json', DATA_PATH)
yahoo_answers_qa = load_dataset('json', data_files={'train': DATA_PATH})
print(yahoo_answers_qa)

yahoo_answers_qa = yahoo_answers_qa["train"].train_test_split(test_size=0.3)
# FOR TEST
# yahoo_answers_qa = yahoo_answers_qa["train"].train_test_split(train_size=0.01, test_size=0.01)
# Check the length of the data and its structure
# yahoo_answers_qa

# We prefix our tasks with "answer the question"
prefix = "Please answer this question: "

# Define the preprocessing function
def preprocess_function(examples):
   """Add prefix to the sentences, tokenize the text, and set the labels"""
   # The "inputs" are the tokenized answer:
   inputs = [prefix + doc for doc in examples["question"]]
   model_inputs = tokenizer(inputs, max_length=128, truncation=True)
  
   # The "labels" are the tokenized outputs:
   labels = tokenizer(text_target=examples["answer"], 
                      max_length=512,         
                      truncation=True)

   model_inputs["labels"] = labels["input_ids"]
   return model_inputs

# Map the preprocessing function across our dataset
tokenized_dataset = yahoo_answers_qa.map(preprocess_function, batched=True)

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def nltk_based_rouge(predictions, references, use_stemmer=True):
    """
    基于 NLTK 的 ROUGE 实现
    """
    try:
        # 确保 nltk 资源可用
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    stemmer = PorterStemmer() if use_stemmer else None
    
    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        if stemmer:
            tokens = [stemmer.stem(token) for token in tokens]
        return tokens

    def longest_common_subsequence(a, b):
        """
        计算两个序列的最长公共子序列长度
        """
        m, n = len(a), len(b)
        # 创建DP表
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
        
    results = {
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0
    }
    
    for pred, ref in zip(predictions, references):
        pred_tokens = preprocess_text(pred)
        ref_tokens = preprocess_text(ref)
        
        # ROUGE-1
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)
        overlap_1 = len(pred_set & ref_set)
        rouge1 = overlap_1 / len(ref_set) if len(ref_set) > 0 else 0.0
        
        # ROUGE-2
        pred_bigrams = set(zip(pred_tokens[:-1], pred_tokens[1:]))
        ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
        overlap_2 = len(pred_bigrams & ref_bigrams)
        rouge2 = overlap_2 / len(ref_bigrams) if len(ref_bigrams) > 0 else 0.0
        
        # ROUGE-L
        lcs_len = longest_common_subsequence(pred_tokens, ref_tokens)
        rougeL = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
        results['rouge1'] += rouge1
        results['rouge2'] += rouge2
        results['rougeL'] += rougeL
    
    n = len(predictions)
    results = {k: round(v / n, 4) for k, v in results.items()}
    results['rougeLsum'] = results['rougeL']
    
    return results

def compute_metrics(eval_preds):
   preds, labels = eval_preds

   # decode preds and labels
   preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
   labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
   vocab_size = tokenizer.vocab_size
   preds = np.clip(preds, 0, vocab_size - 1)
   decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

   # rougeLSum expects newline after each sentence
   decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
   decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

   # result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
   result = nltk_based_rouge(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
  
   return result

# Global Parameters
L_RATE = 3e-4
BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 3

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
   output_dir="../autodl-tmp/results",
#    evaluation_strategy="epoch",
   eval_strategy="epoch",
   learning_rate=L_RATE,
   per_device_train_batch_size=BATCH_SIZE,
   per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
   weight_decay=WEIGHT_DECAY,
   save_total_limit=SAVE_TOTAL_LIM,
   num_train_epochs=NUM_EPOCHS,
   predict_with_generate=True,
   push_to_hub=False,
)

trainer = Seq2SeqTrainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_dataset["train"],
   eval_dataset=tokenized_dataset["test"],
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics
)

trainer.train()
