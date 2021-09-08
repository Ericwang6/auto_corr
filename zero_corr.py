import numpy as np


class xvg(object):
    def __init__(self, fname):
        self.data = []
        self.comments = []
        self.parse(fname)

    def parse(self, fname):
        with open(fname, 'r') as f:
            for line in f.readlines():
                if line.startswith("#") or line.startswith("@"):
                    self.comments.append(line)
                else:
                    self.data.append([float(x) for x in line.split()[:2]])
        self.data = np.array(self.data)
    
    def write(self, fname):
        with open(fname, 'w') as f:
            for cmt in self.comments:
                f.write(cmt)
            for cc in self.data:
                time, ener = f"{cc[0]:.6f}", f"{cc[1]:.6f}"
                f.write(f"{time:>12}  {ener}\n")


md = xvg("hrex.0.0.xvg")
deepmd = xvg("hrex.1.1.xvg")
md_rerun = xvg("hrex.0.1.xvg")
deepmd_rerun = xvg("hrex.1.0.xvg")

zero = np.mean(deepmd_rerun.data[:, 1] - md.data[:, 1])
deepmd.data[:, 1] -= zero
deepmd_rerun.data[:, 1] -= zero

deepmd.write("hrex.1.1.xvg")
deepmd_rerun.write("hrex.1.0.xvg")

