from methode1 import methode as methode1
from methode2 import methode as methode2
import os

files_path = "/Users/grzegorzstermedia/Downloads/Rzuty_dachow"

start = 46
stop = 69
methods = {1: methode1, 2: methode2}
methode = 2
chosen_methode = methods[methode]
dump_path = f"./outputs_{methode}"
os.makedirs(dump_path, exist_ok=True)

for path_id in range(start, stop):
    chosen_methode(
        f"{files_path}/{path_id}.pdf", f"{dump_path}/out_{path_id}.jpg"
    )