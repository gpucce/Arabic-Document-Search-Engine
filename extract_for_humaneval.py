'''
    Example how to get similar vectors from index giving a phrase
'''


from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
from copy import deepcopy
import json
from utils.index_utils import extract_tlg_texts, extract_sentences_from_texts, encode_sentences
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from transformers import pipeline
import faiss
import pandas as pd
import sqlite3
import numpy as np
from tqdm import tqdm


# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

def tokenize_and_encode(text, model, tokenizer, H, device):
    tokenized = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=H.model.model_max_length).to(device)
    with torch.inference_mode():
        out = model(**tokenized, output_hidden_states=True)
    return out.hidden_states[-1][:,0,:].cpu().numpy().astype(np.float32)

def compute_embeddings(model, tokenizer, H, texts, device):
    # out = model(**tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=H.model.model_max_length).to(device), output_hidden_states=True)
    with torch.inference_mode():
        # out = [j.reshape(1, -1) for j in out['hidden_states'][-1][:,0,:].cpu().numpy().astype(np.float32)]
        out = [tokenize_and_encode(text, model, tokenizer, H, device) for text in tqdm(texts)]
        for _idx in range(len(out)):
            target_encoding = np.zeros((1, H.data.len_embedding)).astype(np.float32)
            target_encoding[0] = out[_idx][0]
            out[_idx] = target_encoding
            faiss.normalize_L2(out[_idx])
    out = np.concatenate(out, axis=0)
    return out

def main(argv):

    H = FLAGS.config

    device = H.run.device if torch.cuda.is_available() else -1

    ref_data = pd.read_csv(H.reference.path, sep="\t")
    ref_queries = ref_data.loc[:, "Query"].to_list()
    n_queries = len(ref_queries)
    ref_targets = pd.DataFrame()
    if "Target #1" in ref_data.columns:
        ref_targets = ref_data.loc[:, [f"Target #{i}" for i in range(1, 6)]].to_dict(orient="list")

    #load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(H.model.tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(H.model.model).to(device)

    #load index
    index = faiss.read_index(os.path.join(H.index.index_path,f"{H.index.index_name}.index"))

    #create or open db
    path_to_load_db = os.path.join(H.db.db_path,f"{H.db.db_name}.db")
    connection = sqlite3.connect(path_to_load_db)

    m = connection.total_changes

    assert m == 0, "ERROR: cannot open database."

    cursor = connection.cursor()

    encodings = [np.zeros((1, H.data.len_embedding)).astype(np.float32) for _ in range(n_queries)]

    # Tokenize sentence
    inputs = tokenizer(ref_queries, return_tensors="pt", truncation=True, padding=True, max_length=H.model.model_max_length).to(device)

    # Get embeddings
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)


    target_outputs = {}
    for i in range(1, 6):
        if f"Target #{i}" not in ref_targets:
            continue
        target_outputs[i] = compute_embeddings(model, tokenizer, H, ref_targets[f"Target #{i}"], device=device)

    # Use [CLS] token embedding as sentence encoding
    sentence_embedding = outputs['hidden_states'][-1][:,0,:].cpu().numpy().astype(np.float32)

    # encoding = sentence_embedding
    # Normalize
    for idx in range(len(encodings)):
        encodings[idx][0] = sentence_embedding[idx]
        faiss.normalize_L2(encodings[idx])

    output = {"queries": [], "corpus_results": [], "targets_results": []}
    for query, encoding in zip(ref_queries, encodings):

        query_target_results = []
        for i in range(1,6):
            if f"Target #{i}" not in ref_targets:
                continue
            # target_results = encoding @ target_outputs[i].T
            target_results = 1 - np.linalg.norm(encoding - target_outputs[i], axis=1) / 2
            assert target_results.shape == (n_queries,)
            target_i_results = []
            for idx, target in enumerate(ref_targets[f"Target #{i}"]):
                target_i_results.append({"target": target, "similarity": target_results[idx].item()})
            query_target_results.append(target_i_results)

        output["targets_results"].append(query_target_results)

        #search best matches in index
        distances, ann = index.search(encoding, k=H.retrieval.num_matches)
        query_corpus_results = []
        #retrieve best results from db
        output["queries"].append(query)
        for d, idx in zip(distances[0], ann[0]):
            s = 1 - d / 2
            cursor.execute(f"SELECT * FROM {H.db.db_name} WHERE row_id = {idx+1}")
            rows = cursor.fetchall()
            assert len(rows) == 1, "mmmh"
            query_corpus_results.append({"similarity": s.item(), "row": rows[0]})
            for row in rows:
                print(s, row)
                print("")

        output["corpus_results"].append(query_corpus_results)
        print("")

    with open(H.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    corpus_results_texts = [[i["row"][4] for i in j] for j in output["corpus_results"]]
    pd.DataFrame(
        np.concatenate([np.array(output["queries"]).reshape(-1, 1), np.array(corpus_results_texts)], axis=1),
        columns = ["Query"] + [f"Result #{i}" for i in range(1, len(corpus_results_texts[0]) + 1)]
    ).to_csv(H.output_path.replace(".json", "_corpus_comparison.tsv"), sep="\t", index=False)

    models = [
        "bowphs/GreBerta",
        "cabrooks/LOGION-50k_wordpiece",
        "pranaydeeps/Ancient-Greek-BERT",
        "bowphs/SPhilBerta",
        "kevinkrahn/shlm-grc-en", # vediamo se funziona
    ]

    model_ablation_dict = {"queries": output["queries"], "corpus_results": output["corpus_results"]}
    for model_name in models:
        model = AutoModelForMaskedLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        new_target_results = []
        for query, query_results in zip(model_ablation_dict["queries"], model_ablation_dict["corpus_results"]):
            with torch.inference_mode():
                _encoding = model(**tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=H.model.model_max_length).to(device), output_hidden_states=True)['hidden_states'][-1][:,0,:].cpu().numpy()
                encoding = np.zeros((1, H.data.len_embedding)).astype(np.float32)
                encoding[0] = _encoding
                faiss.normalize_L2(encoding)
                del _encoding

            assert encoding.shape == (1, H.data.len_embedding)
            texts = [result["row"][4] for result in query_results]
            new_texts_embeddings = compute_embeddings(model, tokenizer, H, texts, device)
            new_target_similarity = 1 - (np.linalg.norm(encoding - new_texts_embeddings, axis=1) ** 2) / 2
            new_query_target_results = []
            for idx, result in enumerate(query_results):
                _result = deepcopy(result)
                _result["similarity"] = new_target_similarity[idx].item()
                new_query_target_results.append(_result)
            new_target_results.append(new_query_target_results)
        model_ablation_dict[model_name] = new_target_results
                # model_ablation_dict[model].append(result["similarity"])

        with open(H.output_path.replace(".json", "_model_ablation.json"), "w") as jf:
            json.dump(model_ablation_dict, jf, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    app.run(main)