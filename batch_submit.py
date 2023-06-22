import subprocess

to_keep = ["1000", "10000", "20000", "30000", "60000"]
langs = ["Arabic", "Hebrew", "German", "Polish", "Spanish"]
for lang in langs:
    for k in to_keep:
        subprocess.check_call(['sbatch', 'submit.cluster', f'{lang}', f'{k}'])

