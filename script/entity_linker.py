#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Much of the world's healthcare data is stored in free-text documents, usually clinical notes taken by doctors. This unstructured data can be challenging to analyze and extract meaningful insights from. However, by applying a standardized terminology like SNOMED CT, healthcare organizations can convert this free-text data into a structured format that can be readily analyzed by computers, in turn stimulating the development of new medicines, treatment pathways, and better patient outcomes.
# 
# One way to analyze clinical notes is to identify and label the portions of each note that correspond to specific medical concepts. This process is called entity linking because it involves identifying candidate spans in the unstructured text (the entities) and linking them to a particular concept in a knowledge base of medical terminology.
# 
# However, clinical entity linking is hard!  Medical notes are often rife with abbreviations (some of them context-dependent) and assumed knowledge. Furthermore, the target knowledge bases can easily include hundreds of thousands of concepts, many of which occur infrequently leading to a “long tail” effect in the distribution of concepts.
# 
# The objective of the competition is to link spans of text in clinical notes with specific topics in the SNOMED CT clinical terminology. Participants will train models based on real-world doctors' notes which have been de-identified and annotated with SNOMED CT concepts by medically trained professionals.
# 
# In this post, we build a straightforward entity linking model and prepare it for submission.  
# 
# Typically, an entity linker contains two components:
# 
# - The Clinical Entity Recognizer (CER) model is responsible for detecting candidate clinical entities from within the text.
# - The Linker is responsible for connecting the entities to the knowledge base.  Often (as here) the liner's tasks are split into two steps:
#     - In the Candidate Generation step, the Linker retrieves a handful of candidate concepts that it thinks may match to the entity.
#     - In the Candidate Selection step, the linker selects the best candidate.

from snomed_graph import *
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from more_itertools import chunked
from gensim.models.keyedvectors import KeyedVectors
from tqdm.notebook import tqdm
from itertools import combinations
from sentence_transformers import (
    SentenceTransformer, models, InputExample, losses
)
from ipymarkup import show_span_line_markup
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
from datasets import Dataset
import evaluate
from collections import Counter
import scipy.sparse as sp
import numpy as np
import dill as pickle
from transformers import (
    AutoTokenizer, 
    pipeline,
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification,
    DebertaV2ForTokenClassification 
)


random_seed = 42                                                    # For reproducibility
max_seq_len = 512                                                   # Maximum sequence length for (BERT-based) encoders 
cer_model_id = "microsoft/deberta-v3-large"                         # Base model for Clinical Entity Recogniser
kb_embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"    # base model for concept encoder
use_LoRA = False                                                    # Whether to use a LoRA to fine-tune the CER model


torch.manual_seed(random_seed)
assert torch.cuda.is_available()


# # 1. Load the data

notes_df = (
    pd.read_csv("data/training_notes.csv")
    .set_index("note_id")
)
print(f"{notes_df.shape[0]} notes loaded.")


annotations_df = (
    pd.read_csv("data/training_annotations.csv")
    .set_index("note_id")
)
print(f"{annotations_df.shape[0]} annotations loaded.")
print(f"{annotations_df.concept_id.nunique()} unique concepts seen.")
print(f"{annotations_df.index.nunique()} unique notes seen.")


# ## 1.1 Split the data into training and test sets

training_notes_df, test_notes_df = train_test_split(notes_df, test_size=32, random_state=random_seed)
training_annotations_df = annotations_df.loc[training_notes_df.index]
test_annotations_df = annotations_df.loc[test_notes_df.index]

print(f"There are {training_annotations_df.shape[0]} total annotations in the training set.")
print(f"There are {test_annotations_df.shape[0]} total annotations in the test set.")
print(f"There are {training_annotations_df.concept_id.nunique()} distinct concepts in the training set.")
print(f"There are {test_annotations_df.concept_id.nunique()} distinct concepts in the test set.")
print(f"There are {training_notes_df.shape[0]} notes in the training set.")
print(f"There are {test_notes_df.shape[0]} notes in the test set.")


# # 2. Train the CER model
# 
# This will be a token classifier, based on the widely-used BERT architecture.

# ## 2.1 Define the token types
# 
# A token classifier is typically looking to tag tokens according to the part of speech or entity type.  We have quite a simple task here: locate tokens that are part of clinical entities.  We define the following token labels:
# 
# - *O*.  This token is not part of an entity.
# - *B-clinical_entity*. This token is the beginning (first part of the first word) of a clinical entity.
# - *I-clinical_entity*. This token is inside a clinical entity - i.e. not the first word but a subsequent word.

label2id = {
    'O': 0, 
    'B-clinical_entity': 1, 
    'I-clinical_entity': 2
}

id2label = {v: k for k,v in label2id.items()}


# ## 2.2 Load a tokenizer
# 
# We'll use the tokenizer for our chosen NER model.

cer_tokenizer = AutoTokenizer.from_pretrained(
    cer_model_id, 
    model_max_length=max_seq_len
)


# ## 2.3 Construct training and test datasets for the CER model
# 
# The annotation dataset contains tuples of the form `(note_id, concept_id, start, end)`.
# 
# To create a dataset for the token classifier we need to make two transformations to the data:
# 
# 1. We have to split the discharge notes into chunks of 512 characters (the input dimension for BERT-based models).
# 2. We have to tokenize the discharge notes and determine which of the resulting tokens fall within the span of an annotation according to the `label2id` map defined above.
# 
# We will create a dataset consisting of 512-token chunks, along with a length-512 vector flagging the tokens which appear within an annotation.
# 
# One further consideration is that the tokenizer will tokenize to a sub-word level.  For example, this tokenizer will split the word `tokenization` into two sub-words: `__token` and `ization`.  We will always flag the first token of each word with the appropriate entity type ("B", "I" or "O") but need to decide how to flag subsequent sub-words.  One way is to flag these with a `-100` value, which is interpreted used by `pytorch` loss functions as "ignore this value".  This involves complicating the alignment logic, however.  Instead, the approach taken below is to flag all subwords with the appropriate "I" or "B" label.  (The tokenizer offers a handy `word_ids()` function which we can use to determine whether a particular token represents the start of a new word or the continuation of the previous word.)
# 
# The logic for the CER tokenizer is therefore as follows:
# 
# - First token of the first word within an annotation: `B-clinical_entity`
# - First token a subsequent word within an annotation: `I-clinical_entity`
# - First token of a word not within an annotation: `O`
# - Special token ([CLS], [SEP]): `-100`
# 
# The first token of an input to a BERT-based model must be the classificiation (`[CLS]`) token and the last must be the separator (`[SEP]`).  We add these manually.

# Step through the annotation spans for a given note.  When they're exhausted,
# return (1000000, 1000000).  This will avoid a StopIteration exception.

def get_annotation_boundaries(note_id, annotations_df):
    for row in annotations_df.loc[note_id].itertuples():
        yield row.start, row.end, row.concept_id
    yield 1000000, 1000000, None


def generate_ner_dataset(notes_df, annotations_df):

    for row in notes_df.itertuples():
        
        tokenized = cer_tokenizer(
            row.text, 
            return_offsets_mapping=False,   # Avoid misalignments due to destructive tokenization
            return_token_type_ids=False,    # We're going to construct these below
            return_attention_mask=False,    # We'll construct this by hand
            add_special_tokens=False,       # We'll add these by hand
            truncation=False,               # We'll chunk the notes ourselves
        )

        # Prime the annotation generator and fetch the token <-> word_id map
        annotation_boundaries = get_annotation_boundaries(row.Index, annotations_df)
        ann_start, ann_end, concept_id = next(annotation_boundaries)
        word_ids = tokenized.word_ids()

        # The offsets_mapping returned by the tokenizer will be misaligned vs the original text.
        # This is due to the fact that the tokenization scheme is destructive, for example it 
        # drops spaces which cannot be recovered when decoding the inputs.
        # In the following code snippet we create an offset mapping which is aligned with the 
        # original text; hence we can accurately locate the annotations and match them to the
        # tokens.
        global_offset = 0
        global_offset_mapping = []
        
        for input_id in tokenized["input_ids"]:
            token = cer_tokenizer.decode(input_id)
            pos = row.text[global_offset:].find(token)
            start = global_offset + pos
            end = global_offset + pos + len(token)
            global_offset = end
            global_offset_mapping.append((start, end))        

        # Note the max_seq_len - 2.
        # This is because we will have to add [CLS] and [SEP] tokens once we're done.
        it = zip(
            chunked(tokenized["input_ids"], max_seq_len-2),
            chunked(global_offset_mapping, max_seq_len-2),
            chunked(word_ids, max_seq_len-2)
        )

        # Since we are chunking the discharge notes, we need to maintain the start and
        # end character index for each chunk so that we can align the annotations for
        # chunks > 1
        chunk_start_idx = 0
        chunk_end_idx = 0
        
        for chunk_id, chunk in enumerate(it):
            input_id_chunk, offset_mapping_chunk, word_id_chunk = chunk
            token_type_chunk = list()
            concept_id_chunk = list()
            prev_word_id = -1
            concept_word_number = 0
            chunk_start_idx = chunk_end_idx
            chunk_end_idx = offset_mapping_chunk[-1][1]
            
            for offsets, word_id in zip(offset_mapping_chunk, word_id_chunk):
                token_start, token_end = offsets
                
                # Check whether we need to fetch the next annotation
                if token_start >= ann_end:
                    ann_start, ann_end, concept_id = next(annotation_boundaries)  
                    concept_word_number = 0
            
                # Check whether the token's position overlaps with the next annotation
                if token_start < ann_end and token_end > ann_start:

                    if prev_word_id != word_id:
                        concept_word_number += 1
                    
                    # If so, annotate based on the word number in the concept
                    if concept_word_number == 1:
                        token_type_chunk.append(label2id["B-clinical_entity"])
                    else:
                        token_type_chunk.append(label2id["I-clinical_entity"])

                    # Add the SCTID (we'll use this later to train the Linker)
                    concept_id_chunk.append(concept_id)
        
                # Not part of an annotation
                else:
                    token_type_chunk.append(label2id["O"])
                    concept_id_chunk.append(None)
            
                prev_word_id = word_id

            # Manually adding the [CLS] and [SEP] tokens.
            token_type_chunk = [-100] + token_type_chunk + [-100]
            input_id_chunk = [cer_tokenizer.cls_token_id] + input_id_chunk + [cer_tokenizer.sep_token_id]
            attention_mask_chunk = [1] * len(input_id_chunk)
            offset_mapping_chunk = [(None, None)] + offset_mapping_chunk + [(None, None)]
            concept_id_chunk = [None] + concept_id_chunk + [None]
            
            yield {
                # These are the fields we need
                "note_id": row.Index,
                "input_ids": input_id_chunk,
                "attention_mask": attention_mask_chunk,
                "labels": token_type_chunk,
                # These fields are helpful for debugging
                "chunk_id": chunk_id,
                "chunk_span": (chunk_start_idx, chunk_end_idx),
                "offset_mapping": offset_mapping_chunk,
                "text": row.text[chunk_start_idx : chunk_end_idx],                
                "concept_ids": concept_id_chunk,
            }


# We can ignore the "Token indices sequence length is longer than the specified maximum sequence length"
# warning because we are chunking by hand.
train = pd.DataFrame(list(generate_ner_dataset(training_notes_df, training_annotations_df)))
train = Dataset.from_pandas(train)
train


test = pd.DataFrame(list(generate_ner_dataset(test_notes_df, test_annotations_df)))
test = Dataset.from_pandas(test)
test


# The data collator handles batching for us.
data_collator = DataCollatorForTokenClassification(tokenizer=cer_tokenizer)


# ## 2.4 Define some training metrics for the fine-tuning run
# 
# It's always easier to be able to track some meaningful performance metrics during a training run, rather than simple watching a cross-entropy loss function change.  This is a standard, boilerplate function taken directly from a HuggingFace tutorial that is useful for any classifier fine-tuning.

seqeval = evaluate.load("seqeval")

def compute_metrics(p):

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# ## 2.5 Define and train the model
# 
# The `deberta-v3-large` model (model card: https://huggingface.co/microsoft/deberta-v3-large) has 304M parameters.  To speed up the fine-tuning can use a LoRA, which will greatly reduce the number of trainable parameters.

cer_model = DebertaV2ForTokenClassification.from_pretrained(
    cer_model_id, 
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)   

if use_LoRA:
    lora_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="TOKEN_CLS",
    )
    
    cer_model = get_peft_model(cer_model, lora_config)
    
    cer_model.print_trainable_parameters()


training_args = TrainingArguments(
    output_dir="~/temp/cer_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    seed=random_seed
)

trainer = Trainer(
    model=cer_model,
    args=training_args,
    train_dataset=train,
    eval_dataset=test,
    tokenizer=cer_tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


trainer.save_model("cer_model")
cer_tokenizer.save_pretrained("cer_model")


# ## 2.6 CER Inference

# We can ignore the warning message.  This is simply due to the fact that
# DebertaV2ForTokenClassification loads the DebertaV2 model first, then 
# initializes a random header model before restoring the states of the 
# TokenClassifer.  So we *do* have our fine-tuned model available. 

if use_LoRA:
    config = PeftConfig.from_pretrained("cer_model")

    cer_model = DebertaV2ForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=config.base_model_name_or_path,
        num_labels=3, 
        id2label=id2label, 
        label2id=label2id
    )  
    cer_model = PeftModel.from_pretrained(cer_model, "cer_model")
else:
    cer_model = DebertaV2ForTokenClassification.from_pretrained(
        pretrained_model_name_or_path="cer_model",
        num_labels=3, 
        id2label=id2label, 
        label2id=label2id
    )  


# If using the adaptor, ignore the warning: 
# "The model 'PeftModelForTokenClassification' is not supported for token-classification."
# The PEFT model is wrapped just fine and will work within the pipeline.
# N.B. moving model to CPU makes inference slower, but enables us to feed the pipeline 
# directly with strings.
cer_pipeline = pipeline(
    task="token-classification", 
    model=cer_model, 
    tokenizer=cer_tokenizer, 
    aggregation_strategy="first",
    device="cpu"
)


# Visualise the predicted clinical entities against the actual annotated entities.
# N.B. only the first 512 tokens of the note will contain predicted spans.
# Not run due to sensitivity of MIMIC-IV notes

note_id = "10807423-DS-19"
text = test_notes_df.loc[note_id].text

# +1 to offset the [CLS] token which will have been added by the tokenizer
predicted_annotations = [
    (span["start"]+1, span["end"], "PRED") for span in cer_pipeline(text)
]

gt_annotations = [
    (row.start, row.end, "GT")
    for row in test_annotations_df.loc[note_id].itertuples()
]

show_span_line_markup(text, predicted_annotations + gt_annotations)


# # 3. Linking Model
# 
# The second part of the Entity Linker is the Linking model.  This component is charged with selecting the concepts from the knowledge base that best match the detected entity.
# 
# We will build a simple, multi-level indexer for the task, drawing upon an encoder-only transformer that has been fine-tuned across the SNOMED CT concepts.
# 
# The first index will find the most similar entity seen during training.  The second will use the context surrounding the entity to find the most likely concept matching said entity.

# ## 3.1 Load the knowledge base

# To load from a SNOMED RF2 folder, use:
# 
# ```SG = SnomedGraph.from_rf2("SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z")```
# 
# Here, we will load a previously constructed concept graph and filter to the concepts that were in scope of the annotation exercise.

SG = SnomedGraph.from_serialized("../snomed_graph/full_concept_graph.gml")


# If we want to load all of the concepts that were in scope of the annotation exercise, it's this:
concepts_in_scope = SG.get_descendants(71388002) | SG.get_descendants(123037004) | SG.get_descendants(404684003)
print(f"{len(concepts_in_scope)} concepts have been selected.")


# If we want to simply use concepts for which we have a training example, it's this:
concepts_in_scope = [
    SG.get_concept_details(a)
    for a in annotations_df.concept_id.unique()
]

print(f"{len(concepts_in_scope)} concepts have been selected.")


# ## 3.2 Fine-tune the Linker's Encoder
# 
# To fine-tune the encoder, we'll collect each in-scope concept from SNOMED CT and generate a training example from each pairwise combination of synonyms.  We train with a multiple negative-rankings loss.  This calculates the distance between each example pair and also the distance between the first example in the pair and _all other_ first examples in the batch.  The loss is constructed from the ranking of these distances.  We want the distance between an example and itself to be the minimum of all distances in the batch.  This should result in an embedding in which synonyms for the SNOMED concepts are encoded into close proximity.
# 
# Note that this is a relatively trivial exploitation of the SNOMED CT graph.  We could experiment with other ways to generate pairs too, for example: by generating pairs that consist of parent and child concepts.

kb_model = SentenceTransformer(kb_embedding_model_id)

kb_sft_examples = [
    InputExample(texts=[syn1, syn2], label=1)
    for concept in tqdm(concepts_in_scope)
    for syn1, syn2 in combinations(concept.synonyms, 2)
]

kb_sft_dataloader = DataLoader(kb_sft_examples, shuffle=True, batch_size=32)

kb_sft_loss = losses.ContrastiveLoss(kb_model)

kb_model.fit(
    train_objectives=[(kb_sft_dataloader, kb_sft_loss)], 
    epochs=2, 
    warmup_steps=100,
    checkpoint_path="~/temp/ke_encoder",
)

kb_model.save("kb_model")


# ## 3.3 Construct the Linker
# 
# The simplest linker would simply map an entity (as extracted by the CER model) to the associated concept in the training dataset.  Two problems with this approach present themselves:
# 
# 1. We might encounter entities that have not been seen during training.
# 2. Some entities might be mapped to >1 concept.  Why would this happen?  Consider the entity "ABD".  This is an abbreviation for "Acute behavioural disorder".  However, it is also shorthand for "Abdomen".
# 
# To resolve the first problem our linker keeps an index of entities seen during training.  At inference time, it selects the known entity that is closest to the entity it is presented with.  (This is the "candidate generation" step.)
# 
# To resolve the second problem, the linker builds a "second level" index for each entity.  This second level index maps each occurance of an entity + its surrounding context to the SNOMED concept it was annotated with.  At inference time, we encode the \[entity + context\] and find the most similar result in the second level index.  We return the associated SCTID.  (This is the "candidate selection" step.)
# 
# We perform a simple grid search over context window sizes.
# 
# As a further enhancement, we not only train the linker using entities seen in the training dataset but also with all of the synonyms for the in-scope SNOMED concepts (here there is no "context" for each of the entities, so we simply use the entity as it's own context.)  You can run an ablation experiment by not passing the Linker any SNOMED concepts.  The performance will drop!

class Linker():
    
    def __init__(self, encoder, context_window_width=0):
        self.encoder = encoder
        self.entity_index = KeyedVectors(self.encoder[1].word_embedding_dimension)
        self.context_index = dict()
        self.history = dict()
        self.context_window_width = context_window_width

    def add_context(self, row):
        window_start = max(0, row.start-self.context_window_width)
        window_end = min(row.end+self.context_window_width, len(row.text))
        return row.text[window_start : window_end]

    def add_entity(self, row):
        return row.text[row.start : row.end]       

    def fit(self, df=None, snomed_concepts=None):
        # Create a map from the entities to the concepts and contexts in which they appear
        if df is not None:
            for row in df.itertuples():
                entity = self.add_entity(row)
                context = self.add_context(row)
                map_ = self.history.get(entity, dict())
                contexts = map_.get(row.concept_id, list())
                contexts.append(context)
                map_[row.concept_id] = contexts
                self.history[entity] = map_

        # Add SNOMED CT codes for lookup
        if snomed_concepts is not None:
            for c in snomed_concepts:
                for syn in c.synonyms:
                    map_ = self.history.get(syn, dict())
                    contexts = map_.get(c.sctid, list())
                    contexts.append(syn)
                    map_[c.sctid] = contexts
                    self.history[syn] = map_            
            
        # Create indexes to help disambiguate entities by their contexts
        for entity, map_ in tqdm(self.history.items()):
            keys = [
                (concept_id, occurance)
                for concept_id, contexts in map_.items()
                for occurance, context in enumerate(contexts)
            ]
            contexts = [
                context 
                for contexts in map_.values() 
                for context in contexts
            ]
            vectors = self.encoder.encode(contexts)
            index = KeyedVectors(self.encoder[1].word_embedding_dimension)
            index.add_vectors(keys, vectors)
            self.context_index[entity] = index

        # Now create the top-level entity index
        keys = list(self.history.keys())
        vectors = self.encoder.encode(keys)
        self.entity_index.add_vectors(keys, vectors)

    def link(self, row):
        entity = self.add_entity(row)
        context = self.add_context(row)        
        vec = self.encoder.encode(entity)
        nearest_entity = self.entity_index.most_similar(vec, topn=1)[0][0]     
        index = self.context_index.get(nearest_entity, None)
        
        if index:
            vec = self.encoder.encode(context)
            key, score = index.most_similar(vec, topn=1)[0]
            sctid, _ = key
            return sctid
        else:
            return None 


linker_training_df = training_notes_df.join(training_annotations_df)
linker_test_df = test_notes_df.join(test_annotations_df)


def evaluate_linker(linker, df):
    n_correct = 0
    n_examples = df.shape[0]

    for _, row in tqdm(df.iterrows(), total=n_examples):
        sctid = linker.link(row)
        if row["concept_id"] == sctid:
            n_correct += 1
    
    return n_correct / n_examples


for context_window_width in tqdm([5, 8, 10, 12]):
    linker = Linker(kb_model, context_window_width)
    linker.fit(linker_training_df, concepts_in_scope)
    acc = evaluate_linker(linker, linker_test_df)
    print(f"Context Window Width: {context_window_width}\tAccuracy: {acc}")


linker = Linker(kb_model, 12)
linker.fit(linker_training_df, concepts_in_scope)

with open("linker.pickle", "wb") as f:
    pickle.dump(linker, f)


# We can then re-load the linker with:
with open("linker.pickle", "rb") as f:
    linker = pickle.load(f)


# # 4. Evaluation
# 
# Here we glue the Clinical Entity Recogniser model to the Linker model and show how to generate and evaluate predictions over our test set.

# ## 4.1 Prediction pipeline

def predict(df):

    # One note at a time...
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        
        # Tokenize the entire discharge note
        tokenized = cer_tokenizer(
            row.text, 
            return_offsets_mapping=False,    
            add_special_tokens=False,   
            truncation=False,       
        )

        global_offset = 0
        global_offset_mapping = []

        # Adjust the token offsets so that they match the original document 
        for input_id in tokenized["input_ids"]:
            token = cer_tokenizer.decode(input_id)
            pos = row.text[global_offset:].find(token)
            start = global_offset + pos
            end = global_offset + pos + len(token)
            global_offset = end
            global_offset_mapping.append((start, end))     

        chunk_start_idx = 0
        chunk_end_idx = 0            
            
        # Process the document in chunks of 512 tokens chunk at a time
        for offset_chunk in chunked(global_offset_mapping, max_seq_len-2):
            chunk_start_idx = chunk_end_idx
            chunk_end_idx = offset_chunk[-1][1]
            chunk_text = row.text[chunk_start_idx:chunk_end_idx]

            # Iterate through the detected entities and link them
            for entity in cer_pipeline(chunk_text):
                example = pd.Series({
                    # +1 to account for the [CLS] token
                    "start": entity["start"] + chunk_start_idx + 1, 
                    "end": entity["end"] + chunk_start_idx,            
                    "text": row.text
                })
                sctid = linker.link(example)

                # Only yield matches where the Linker returned something
                if sctid:
                    yield {
                        'note_id': row.Index,
                        'start': example["start"],  
                        'end': example["end"],
                        'concept_id': sctid,
                        # The following are useful for debugging and analysis
                        'FSN': SG.get_concept_details(sctid).fsn,
                        'entity': row.text[example["start"]:example["end"]],
                        'tokenizer_word': entity["word"]
                    }

preds_df = pd.DataFrame(list(predict(test_notes_df)))


# ## 4.3 Visualisation
# 
# The following code will compare the ground truth ("GT_") annotations to the predicted ("P_") annotations.  Since we cannot share the text of these notes, the outputs of this code have been hidden.

note_id = "10807423-DS-19"
text = test_notes_df.loc[note_id].text

predicted_annotations = [
    (row.start, row.end, f'P_{row.concept_id}')
    for row in preds_df.set_index("note_id").loc[note_id].itertuples()
]

gt_annotations = [
    (row.start, row.end, f'GT_{row.concept_id}')
    for row in test_annotations_df.loc[note_id].itertuples()
]

show_span_line_markup(text, predicted_annotations + gt_annotations)


# ## 4.3 Scoring
# 
# We apply a token-level scorer function, which is what the competition will use to evaluate solutions.  We run this over our reserved test set to get a sense for out-of-sample performance.

def iou_per_class(user_annotations: pd.DataFrame, target_annotations: pd.DataFrame) -> List[float]:
    """
    Calculate the IoU metric for each class in a set of annotations.
    """
    # Get mapping from note_id to index in array
    docs = np.unique(np.concatenate([user_annotations.note_id, target_annotations.note_id]))
    doc_index_mapping = dict(zip(docs, range(len(docs))))

    # Identify union of categories in GT and PRED
    cats = np.unique(np.concatenate([user_annotations.concept_id, target_annotations.concept_id]))

    # Find max character index in GT or PRED
    max_end = np.max(np.concatenate([user_annotations.end, target_annotations.end]))

    # Populate matrices for keeping track of character class categorization
    def populate_char_mtx(n_rows, n_cols, annot_df):
        mtx = sp.lil_array((n_rows, n_cols), dtype=np.uint64)
        for row in annot_df.itertuples():
            doc_index = doc_index_mapping[row.note_id]
            mtx[doc_index, row.start : row.end] = row.concept_id  # noqa: E203
        return mtx.tocsr()

    gt_mtx = populate_char_mtx(docs.shape[0], max_end, target_annotations)
    pred_mtx = populate_char_mtx(docs.shape[0], max_end, user_annotations)

    # Calculate IoU per category
    ious = []
    for cat in cats:
        gt_cat = gt_mtx == cat
        pred_cat = pred_mtx == cat
        # sparse matrices don't support bitwise operators, but the _cat matrices
        # have bool dtypes so when we multiply/add them we end up with only T/F values
        intersection = gt_cat * pred_cat
        union = gt_cat + pred_cat
        iou = intersection.sum() / union.sum()
        ious.append(iou)

    return ious


ious = iou_per_class(preds_df, test_annotations_df.reset_index())
print(f"macro-averaged character IoU metric: {np.mean(ious):0.4f}")


# # 5. Preparing for Submission
# 
# Here we wrap the model up into a compliant submission format. (Note that, before submitting, we'd want to re-fit both the CER model (using the optimal number of training epochs) and the Linker on _all_ of the data.)
# 
# Before we do so, it's a good idea to re-train the entity linker on all of the available notes, just to squeeze out every last drop of performance.
# 
# The contents of `solution.py` are as follows:

# ## 5.1 Finalise the CER model
# 
# We'll give a final epoch of supervised fine-tuning over the held-out notes.

training_args.num_train_epochs = 1

trainer = Trainer(
    model=cer_model,
    args=training_args,
    train_dataset=test,
    eval_dataset=test,
    tokenizer=cer_tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("cer_model")


# ## 5.2 Finalise the Linker

kb_model = SentenceTransformer("kb_model")
linker = Linker(kb_model, 12)
linker.fit(notes_df.join(annotations_df), concepts_in_scope)

with open("linker.pickle", "wb") as f:
    pickle.dump(linker, f)


"""Benchmark submission for Entity Linking Challenge."""
from pathlib import Path
from loguru import logger
import pandas as pd
from more_itertools import chunked
from peft import PeftConfig, PeftModel
from transformers import (
    DebertaV2ForTokenClassification, AutoTokenizer, pipeline
)
import dill as pickle

NOTES_PATH = Path("data/test_notes.csv")
SUBMISSION_PATH = Path("submission.csv")
LINKER_PATH = Path("linker.pickle")
CER_MODEL_PATH = Path("cer_model")

CONTEXT_WINDOW_WIDTH = 20
MAX_SEQ_LEN = 512
USE_LORA = False

def load_cer_pipeline():

    label2id = {
        'O': 0, 
        'B-clinical_entity': 1, 
        'I-clinical_entity': 2
    }    

    id2label = {v: k for k,v in label2id.items()}

    cer_tokenizer = AutoTokenizer.from_pretrained(
        CER_MODEL_PATH, model_max_length=MAX_SEQ_LEN
    )

    if USE_LORA:
        config = PeftConfig.from_pretrained(CER_MODEL_PATH)

        cer_model = DebertaV2ForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=config.base_model_name_or_path,
            num_labels=3, 
            id2label=id2label, 
            label2id=label2id
        )  
        cer_model = PeftModel.from_pretrained(cer_model, CER_MODEL_PATH)
    else:
        cer_model = DebertaV2ForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=CER_MODEL_PATH,
            num_labels=3, 
            id2label=id2label, 
            label2id=label2id
        )  

    cer_pipeline = pipeline(
        task="token-classification", 
        model=cer_model, 
        tokenizer=cer_tokenizer, 
        aggregation_strategy="first",
        device="cpu"
    )  
    return cer_pipeline


def main():    
    # columns are note_id, text
    logger.info("Reading in notes data.")
    notes = pd.read_csv(NOTES_PATH)
    logger.info(f"Found {notes.shape[0]} notes.")
    spans = []

    # Load model components
    logger.info("Loading CER pipeline.")
    cer_pipeline = load_cer_pipeline()
    cer_tokenizer = cer_pipeline.tokenizer
    
    logger.info("Loading Linker")
    with open(LINKER_PATH, "rb") as f:
        linker = pickle.load(f)
    
    # Process one note at a time...
    logger.info("Processing notes.")
    for row in notes.itertuples():

        # Tokenize the entire discharge note
        tokenized = cer_tokenizer(
            row.text, 
            return_offsets_mapping=False,    
            add_special_tokens=False,       
            truncation=False,       
        )

        global_offset = 0
        global_offset_mapping = []

        # Adjust the token offsets so that they match the original document 
        for input_id in tokenized["input_ids"]:
            token = cer_tokenizer.decode(input_id)
            pos = row.text[global_offset:].find(token)
            start = global_offset + pos
            end = global_offset + pos + len(token)
            global_offset = end
            global_offset_mapping.append((start, end))     

        chunk_start_idx = 0
        chunk_end_idx = 0            
            
        # Process the document in chunks of 512 tokens chunk at a time
        for offset_chunk in chunked(global_offset_mapping, MAX_SEQ_LEN-2):
            chunk_start_idx = chunk_end_idx
            chunk_end_idx = offset_chunk[-1][1]
            chunk_text = row.text[chunk_start_idx:chunk_end_idx]

            # ...one matched clinical entity at a time
            # Iterate through the detected entities and link them
            for entity in cer_pipeline(chunk_text):
                example = pd.Series({
                    # +1 to account for the [CLS] token
                    "start": entity["start"] + chunk_start_idx + 1, 
                    "end": entity["end"] + chunk_start_idx,            
                    "text": row.text
                })
                sctid = linker.link(example)
                if sctid:
                    spans.append({
                        'note_id': row.Index,
                        'start': example["start"],
                        'end': example["end"],
                        'concept_id': sctid
                    })
    
    logger.info(f"Generated {len(spans)} annotated spans.")
    spans_df = pd.DataFrame(spans)
    spans_df.to_csv(SUBMISSION_PATH, index=False)
    logger.info("Finished.")

if __name__ == "__main__":
    main()


# # Parting Words
# 
# There's a fair amount that goes into an entity linker.  The approach we took here - using transformer encoders - has the virtue of being quick to fine-tune and easy to experiment with; on the flip-side, it's difficult to get good performance from a 300M parameter encoder for the CER step using "out of the box" fine-tuning.  Furthermore, the requirement to chunk the documents and align the annotations with the tokenization scheme adds unwelcome complexity to the code. Entity linkers that use modern, decoder-based transformers - having the virtue of longer context windows and a deeper "understanding" of natural language - should be able to beat this benchmark.
# 
# Furthermore, the model constructed in notebook does not take full advantage of the knowledge encoded within the SNOMED Clinical Terminology.  We used synonyms to fine-tune the Knowledge Base Encoder but made no use of either the hierarchy or the defining relationships in constructing fine-tuning datasets. For example, in a decoder-based model, we can imagine developing _retrieval augmented generation_ techniques for candidate selection.
# 
# The full power of SNOMED CT is an underexplored area for the development of Clinical Entity Linking models.  We wish you all the best in your experiments!



