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
