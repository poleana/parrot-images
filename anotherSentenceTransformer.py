#pip install -U sentence-transformers


from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('{MODEL_NAME}')
embeddings = model.encode(sentences)
print(embeddings)

###################################

from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('{MODEL_NAME}')
model = AutoModel.from_pretrained('{MODEL_NAME}')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)

## Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="kietnt0603/xlmr-semantic-textual-relatedness")
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("kietnt0603/xlmr-semantic-textual-relatedness")
model = AutoModelForSequenceClassification.from_pretrained("kietnt0603/xlmr-semantic-textual-relatedness")

###############################################################
import spacy

nlp = spacy.load("en_core_web_sm")

def remove_stop_words(text: str) -> str:
  doc = nlp(text)
  text_parts = [token.text for token in doc if not token.is_stop]
  return "".join(text_parts)

def split_sentences(text: str) -> List[str]:
  doc = nlp(text)
  sentences = [sent.text for sent in doc.sents]
  return sentences

def group_sentences_semantically(sentences: List[str], threshold: int) -> List[str]:
  docs = [nlp(sentence) for sentence in sentences]
  segments = []

  start_idx = 0
  end_idx = 1
  segment = [sentences[start_idx]]
  while end_idx < len(docs):
    if docs[start_idx].similarity(docs[end_idx]) >= threshold:
      segment.append(docs[end_idx])
    else:
      segments.append(" ".join(segment))
      start_idx = end_idx
      segment = [sentences[start_idx]]
    end_idx += 1

  if segment:
    segments.append(" ".join(segment))

  return segments

def split_text(text: str) -> List[str]:
  text_no_stop_words = remove_stop_words(text)
  sentences = split_sentences(text_no_stop_words)
  return group_sentences_semantically(sentences, 0.8)