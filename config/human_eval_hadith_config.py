# pylint: disable=pointless-string-statement
from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()

    config.output_path = "./human_eval_hadith.json"
    config.run = run = ConfigDict()

    #gpu device
    run.device = 3

    config.data = data = ConfigDict()
    data.json_dataset_path = "/home/gpucce/Repos/arabo_panzeca/all_data/fonti_arabo_wp8"
    data.min_words_in_phrase = 5

    #lenght of the embedding of each sentence. Will be used to inizialize the index
    data.len_embedding = 768

    #dove si trova la lista contenente le opere TLG (create prima con lo script save_name_of_the_works)
    data.name_of_the_works_path = "/home/gpucce/Greek-Document-Search-Engine/name_of_the_tlg_works.npy"

    config.model = model = ConfigDict()

    model.tokenizer = "aubmindlab/bert-base-arabertv2"
    model.model = "aubmindlab/bert-base-arabertv2"
    #max number of tokens the model can handle
    model.model_max_length = 512
    #when decide to join two words w1-w2, check if w2 is in the top_k next words after w1
    model.top_k = 10


    config.index = index = ConfigDict()
    index.index_path = "/home/gpucce/Greek-Document-Search-Engine"
    index.index_name ="Faiss_Hadith"

    config.db = db = ConfigDict()
    db.db_path = "/home/gpucce/Greek-Document-Search-Engine"
    db.db_name ="DB_Hadith"

    config.reference = reference = ConfigDict()
    reference.path = "/home/gpucce/Greek-Document-Search-Engine/frasi_baseline_hadith.tsv"

    #not important. Just if you want to execute test_index.py
    config.retrieval = retrieval = ConfigDict()
    retrieval.num_matches = 100

    return config
