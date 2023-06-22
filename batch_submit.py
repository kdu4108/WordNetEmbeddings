import subprocess

to_keep = ["1000", "10000", "20000", "30000", "60000"]
langs = ["Arabic", "Hebrew", "German", "Polish", "Spanish"]
vec_dims = [100, 200]
for lang in langs:
    for k in to_keep:
        for v in vec_dims:
            subprocess.check_call(["sbatch", "submit.cluster", f"{lang}", f"{k}", f"{v}"])
