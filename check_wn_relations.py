from typing import Dict, List
import wn
import urllib3
import random

excluded_ids = ["cili"]
ids = sorted(set([f"{p['id']}:{p['version']}" for p in wn.projects() if p["id"] not in excluded_ids]))
fcts = {
    "hypernyms": (lambda x: x.hypernyms()),
    "hyponyms": (lambda x: x.hyponyms()),
    "meronyms": (lambda x: x.meronyms()),
    "holonyms": (lambda x: x.holonyms()),
}
lang_to_relations: Dict[str, List[str]] = {id: {k: [] for k in fcts.keys()} for id in ids}
lang_to_relation_count: Dict[str, List[str]] = {id: {k: 0 for k in fcts.keys()} for id in ids}
for wn_id in ids:
    try:
        wn.download(wn_id)
        wnet = wn.Wordnet(lexicon=wn_id)
        for f_name, f in fcts.items():
            synsets = wnet.synsets()
            random.shuffle(synsets)
            synsets = synsets[:10000]
            for synset in synsets:
                if f(synset):
                    lang_to_relations[wn_id][f_name].append(synset)
                    lang_to_relation_count[wn_id][f_name] += 1
                    # print(synset)
    except (ConnectionError, urllib3.exceptions.ReadTimeoutError, TimeoutError, wn.Error):
        print(f"Unable to download wordnet {wn_id}.")
        lang_to_relations[wn_id] = "UNABLE TO DOWNLOAD"

# print(lang_to_relations)
print(lang_to_relation_count)
