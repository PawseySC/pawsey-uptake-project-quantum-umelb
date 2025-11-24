import glob
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

fname = sys.argv[1]
prefix = "outdir/"

dirs = glob.glob("q*", root_dir=prefix)

# regex matching
num = r"([0-9]+)"
pattern = re.compile("q" + num + "l" + num + "r" + num)

plt.figure(layout="constrained")
for dir in dirs:
    match = pattern.match(dir)
    if match is None:
        print(f"pattern did not match, directory {dir}")
    else:
        qubit = match.group(1)
        layer = match.group(2)
        rots = match.group(3)
        cost = np.load(prefix + dir + "/cost.npy")
        match int(qubit):
            case 6:
                color = "r"
            case 8:
                color = "b"
            case 10:
                color = "g"
            case 14:
                color = "c"
            case 18:
                color = "m"
            case _:
                color = ""

        match int(layer):
            case 2:
                marker = "d"
            case 3:
                marker = "D"
            case 4:
                marker = "x"
            case 5:
                marker = "p"
            case 6:
                marker = "^"
            case 8:
                marker = "v"
            case 10:
                marker = "o"
            case 12:
                marker = "."
            case 14:
                marker = "s"
            case _:
                marker = ""

        match int(rots):
            case 3:
                linestyle = "--"
            case _:
                linestyle = "-"

        plt.plot(
            cost,
            color=color,
            marker=marker,
            label=dir,
            linestyle=linestyle,
            markevery=10,
        )
plt.legend()
plt.title(fname)
print(f"saving to {fname}.svg")
plt.savefig(f"tmp/{fname}.svg")
