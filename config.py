"""Constants for use in various modules"""

# DO NOT change at runtime
# dtype for np.ndarray
FIELD = complex

# Which variant of the numpy, scipy stack
STACK = 0

# ===============
# Can be changed at runtime
# Tolerance for treating degeneracies in SVD
degen_tolerance = 1e-6
# Maximum memory
memcap = None
# Ratio of memory to maximum number of elements in tensor to be produced
memratio = 10
# Whether output is to be flushed
flush = True
# Number of times to try linalg.svd before resorting to eigenvalue methods
svd_retry = 1
# 'Load-only mode' (don't change/fix anything found incomplete during unpickling
loadonly = False

def restrict_memory(newmemcap):
  import resource
  assert isinstance(newmemcap,int)
  soft,hard = resource.getrlimit(resource.RLIMIT_AS)
  resource.setrlimit(resource.RLIMIT_AS,(newmemcap,hard))
  global memcap
  memcap = newmemcap

# Verbosity levels
linalg_verbose = 1 # Verbosity for linear-algebraic functions
lin_iter_verbose = 0 # Verbosity for iterative linear-algebraic functions
VDEBUG = 5 # Debugging verbosity level
verbose = 1 # General verbosity
opt_verbose = 0 # Verbosity for optimization routines
import logging,sys
# TODO integrate existing messaging into logging
logger = logging.getLogger('tensorlog')
# Separate log for messages intended to go directly to stdout
streamlog = logger.getChild('stdout')
streamhandler = logging.StreamHandler(sys.stdout)
streamhandler.setLevel(logging.INFO)
streamhandler.setFormatter(logging.Formatter('%(message)s'))
streamlog.addHandler(streamhandler)
linalg_log = logger.getChild('linalg')
opt_log = logger.getChild('optimization')
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stderr)
stdout_handler.setLevel(logging.ERROR)
stdout_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
logger.addHandler(stdout_handler)
def setlogging(logfile, fmtstring, level=logging.DEBUG, datefmt=None,mode='a'):
  handler = logging.FileHandler(logfile,mode=mode)
  formatter = logging.Formatter(fmt=fmtstring, datefmt=datefmt)
  handler.setLevel(level)
  handler.setFormatter(formatter)
  logger.addHandler(handler)

# Signalling
haltsig = False
