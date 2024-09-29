import sys
import json
import zlib
import numpy as np
from alive_progress import alive_bar
try:
    fname = sys.argv[1]
    print(fname)
except IndexError:
    fname = input("Enter filepath: ")
fname = fname.strip()
assert fname.endswith(".cgolcache")
fname = fname[:-10]
initposfile = fname+".txt"
with open(initposfile) as f:
    position = f.read()
position = position.replace("O","1").replace(".","0").split("\n")
maxsize = max([len(row) for row in position])
position = [[int(char) for char in row.ljust(maxsize,"0")] for row in position]
with open(fname+".cgolcache", "rb") as cachefile:
    cache = json.loads(zlib.decompress(cachefile.read()))
class Saver:
    def __init__(self,celldata):
        self.changes = celldata.flatten().tolist()
        self.frames = []
        self.repeatframe = -1

    pass
saveobject = Saver(np.array(position))
with alive_bar(len(cache.keys())) as bar:
    while True:
        original = np.array(position)
        position = cache["".join(["".join([str(cell) for cell in row]) for row in position])]
        shortened = ("".join(["".join([str(cell) for cell in row]) for row in position]))
        if shortened in saveobject.frames:
            saveobject.repeatframe = saveobject.frames.index(shortened)
            saveobject.repeatframe += 1
            x = np.array(saveobject.changes)
            where = np.flatnonzero
            n = len(x)
            starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
            lengths = np.diff(np.r_[starts, n])
            values = x[starts]
            starts = starts[values==1]
            lengths = lengths[values==1]
            ends = starts + lengths
            fname = fname.split("\\") if "\\" in fname else fname.split("/")
            fname = fname[-1]
            with open(f"histories/{fname}.cgolhist","x"):
                pass
            with open(f"histories/{fname}.cgolhist","a") as f:
                celldata = np.array(position)
                f.write(f"{celldata.shape[0]},{celldata.shape[1]},{saveobject.repeatframe},{n}\n")
                for start,end in zip(starts,ends):
                    f.write(f"{start},{end}\n")
            print("done")
            quit()
        else:
            change = 1 * (original != np.array(position))
            saveobject.changes.extend(change.flatten().tolist())
            saveobject.frames.append(shortened)
        bar()