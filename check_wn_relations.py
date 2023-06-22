from typing import Dict, List
import wn
import urllib3

ids = sorted(set([f"{p['id']}:{p['version']}" for p in wn.projects()]))
lang_to_relations: Dict[str, List[str]] = {id: [] for id in ids}
for wn_id in ids:
    try:
        wn.download(wn_id)
        wnet = wn.Wordnet(lexicon=wn_id)
        fcts = {
            "hypernyms": (lambda x: x.hypernyms()),
            "hyponyms": (lambda x: x.hyponyms()),
            "meronyms": (lambda x: x.meronyms()),
            "holonyms": (lambda x: x.holonyms()),
        }
        for f_name, f in fcts.items():
            synsets = wnet.synsets()[:10000]
            for synset in wnet.synsets():
                if f(synset):
                    lang_to_relations[wn_id].append(f_name)
                    break
    except (ConnectionError, urllib3.exceptions.ReadTimeoutError, TimeoutError) as e:
        print(f"Unable to download wordnet {wn_id}.")
        lang_to_relations[wn_id].append(f"UNABLE TO DOWNLOAD because of error {e}.")

print(lang_to_relations)
