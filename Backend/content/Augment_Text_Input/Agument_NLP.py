import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from googletrans import Translator
import random
import nltk
from nltk.corpus import wordnet
from transformers import BertTokenizer, BertForMaskedLM
import re

nltk.download('wordnet')
nltk.download('omw-1.4')

# Load models
t5_model = T5ForConditionalGeneration.from_pretrained("ramsrigouthamg/t5_paraphraser")
t5_tokenizer = T5Tokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
translator = Translator()

# Từ khóa NSFW - tránh thay thế
NSFW_KEYWORDS = ['nude', 'topless', 'naked', 'lingerie', 'erotic', 'sensual', 'cleavage']

def has_nsfw_keyword(text):
    return any(word in text.lower() for word in NSFW_KEYWORDS)

### 1. T5 paraphrasing
def t5_paraphrase(text, num_return_sequences=3):
    input_text = f"paraphrase: {text} </s>"
    encoding = t5_tokenizer.encode_plus(input_text, padding='max_length', return_tensors="pt", max_length=128, truncation=True)
    outputs = t5_model.generate(
        input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask'],
        max_length=128,
        num_return_sequences=num_return_sequences,
        num_beams=5,
        early_stopping=True
    )
    return [t5_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

### 2. Back Translation (EN → DE → EN)
def back_translate(text):
    try:
        de = translator.translate(text, src='en', dest='de').text
        en = translator.translate(de, src='de', dest='en').text
        return en
    except Exception as e:
        print("Translation error:", e)
        return text

### 3. BERT Word Substitution (chỉ từ không nhạy cảm)
def bert_word_substitute(text, top_k=3):
    words = text.split()
    new_words = words[:]
    for i, word in enumerate(words):
        if word.lower() in NSFW_KEYWORDS:
            continue
        tokenized_text = words[:]
        tokenized_text[i] = '[MASK]'
        input_ids = bert_tokenizer.encode(" ".join(tokenized_text), return_tensors='pt')
        mask_token_index = torch.where(input_ids == bert_tokenizer.mask_token_id)[1]
        with torch.no_grad():
            token_logits = bert_model(input_ids).logits
        mask_token_logits = token_logits[0, mask_token_index, :]
        top_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()
        replacement = bert_tokenizer.decode([top_tokens[0]]).strip()
        if replacement.lower() not in NSFW_KEYWORDS:
            new_words[i] = replacement
            break  # chỉ thay một từ cho mỗi caption
    return " ".join(new_words)

### 🚀 Test caption
def augment_nsfw_caption(caption):
    print(f"\n📌 Original: {caption}")

    if not has_nsfw_keyword(caption):
        print("⚠️  Warning: No NSFW keyword found. Skipping augmentation.")
        return

    print("\n🔁 T5 Paraphrase:")
    for p in t5_paraphrase(caption):
        print("→", p)

    print("\n🔁 Back Translation:")
    print("→", back_translate(caption))

    print("\n🔁 BERT Word Substitution:")
    print("→", bert_word_substitute(caption))


### 🧪 Test với một caption nhạy cảm
caption = "A nude woman is lying on a red velvet sofa."
augment_nsfw_caption(caption)
