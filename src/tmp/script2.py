from methode1 import methode as methode1
from methode2 import methode as methode2
import os

files_path = "/Users/grzegorzstermedia/Downloads/Rzuty_dachow"

methods = {1: methode1, 2: methode2}
methode = 2
chosen_methode = methods[methode]
dump_path = f"./outputs_fb_{methode}"
os.makedirs(dump_path, exist_ok=True)

for path in [p for p in os.listdir(".") if p.endswith("krok_3b.png")]:
    print(path)
    chosen_methode(
        path, f"{dump_path}/out_{path}"
    )