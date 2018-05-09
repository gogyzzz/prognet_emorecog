import numpy as np
from fileutils import htk
import sys
import pickle

lld = htk.readHtk(sys.argv[1])

pickle.dump(lld,open(sys.argv[2],'wb'))
