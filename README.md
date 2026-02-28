My current tensor-network library

## Design principles
1. ### Data encapsulation and basis independence:
   Many tensor-network operations create a new basis (for a "virtual" or "bond"
   space) that doesn't have a lot of immediate physical meaning and often even
   has a certain amount of arbitrariness. This means that relatively few of the
   many operations one might normally perform on multidimensional arrays are valid,
   which is part of why I find it preferable to keep the internal array
   representation of a tensor largely hidden from the user.
2. ### Diagrammatics:
   When Einstein notation becomes unwieldy on paper, we turn to tensor-network
   diagrams; and if it's gotten to that point then keeping track of
   axes by their order alone will be even
   harder. So I label axes (usually based on either their function or placement
   on a diagram) and manipulate those tensors through those labels, keeping
   the internal axis order largely hidden as well. This also allows for a sort
   of mini-language defined for building and contracting networks.
3. ### Let someone else do the heavy lifting:
   The vast majority of time in a given
   tensor-network algorithm is usually spent in a small number of routines
   (and sometimewhere the processes of saving
   progress s just one, namely tensor contraction). Fast tensor-network
   code is usually about figuring out how to call those routines efficiently (e.g.
   by optimizing contraction order) and effectively (e.g. anything that reduces
   the number of iterations) and then getting a LAPACK interface, like what `numpy`
   offers, to handle them for you. A lot of people have put a lot of work into
   making fast, stable numerical linear algebra routines, and that frees up
   someone like me to focus on the higher-level stuff.
4. ### Modularity:
   A major goal of mine is to make it easier to tweak algorithms
   (partly to find the most efficient version, but maybe more than that out of the
   recognition that what's most efficient may depend on context). In recent
   versions of this codebase, that has meant adding a
   ["management" layer](#high-level-management-and-algorithm-control), largely
   decoupled from lower-level tensor operations, to separate saving progress
   from the actual operation of the algorithm; systematize setting (potentially
   many) parameters; and even allow the user to override subroutines with
   custom code.

## Components
### Tensors and vector spaces
* `tensors.py` implements a basic tensor (i.e. no additional structure)
* `links.py` defines the vector spaces corresponding to the indices of those
tensors. In the basic case, all that matters is the dimension; it is also
possible (using `setindextracking.py`) to switch to a much stricter paradigm,
useful for debugging, where tensor operations are only possible when
indices correspond to vector spaces that are explicitly identified with
each other (or, in more cases, each other's dual space).

### Networks
* `networks.py` implements a network class, methods for constructing networks,
and a dynamic-programming algorithm for optimizing contraction order (unlike
[the older version(https://github.com/nikkopomata/np-old-tensors/blob/main/network_optimization_parallel.py)],
currently lacks parallelization and does not save contraction orders to disk)
* `operators.py` turns those networks into
`scipy.sparse.linalg.LinearOperator`s in order to use `ARPACK` iterative
linear algebra methods

### Symmetries
* `groups.py` implements a `Group` base class that encapsulates the basic
information (primarily about the group's representations) necessary to
enforce invariance under group actions on a tensor's indices. It also
defines some of the most commonly-invoked groups, U(1), O(2), and SU(2),
and (direct) product groups.
* `invariant.py` implements tensors invariant under a group acting on its
indices in a well-defined way, as well as tensors which transform as one
of the group's irreps under its actions.

In the `moregroups` submodule:
* `finitegroups.py` implements a `FiniteGroup` class that
accepts the representations of a finite group, and implements this for
cyclic, dihedral, symmetric, and Dirac groups
  * `finitefromrandom.py` obtains the representations of a finite
  group from its composition table (though, because this uses random numbers,
  it is not guaranteed that the same basis will be obtained every time) and
  uses this to instantiate the classical groups SL(n,p) (for prime p) 
* `liegroups.py` implements simple Lie algebras
  * `rationallie.py` computes matrix elements and decompositions
  of highest-weight representations in non-orthonormal bases where said
  matrix elements are rational, using a rational-element extension of
  `numpy` arrays defined in `rationallinalg.py`
  * `symboliclie.py` instead performs symbolic computation using `sympy`
  (a prior approach that turned out to be too inefficient)
  * `racah.py` defines the "Racah algebra" of generalized 3j symbols
  (also in a non-normalized basis to ensure rational coefficients),
  including the recoupling "F" symbol

### High-level management and algorithm control
The classes defined in `management.py` and `settingsman.py` are meant to
address the problems of, first, how to easily save and resume frequently
enough in a relatively complicated algorithm, and second, how to tweak the
parameters of those algorithms in a somewhat systematic fashion.

### MPS implementation and algorithms
* `mpsabstract.py`, `mpsbasic.py`, and `impsbasic.py` implement finite and
infinite Matrix Product States (MPS and iMPS) ansatzë
* `DMRGalgo.py` implements finite Density-Matrix Renormalization Group (DMRG)
ground-state optimization and orthogonal (low-energy) DMRG within
"managers" described above
* `iDMRGalgo.py` similarly implements infinite DMRG
* `TEBDalgo.py` similarly implements Time-Evolving Block Decimation (TEBD)
ground-state optimization and time evolution, as well as Minimally-Entangled
Typical Thermal States (METTS) (incomplete)
* `dmrgdefaults.toml` and `tebddefaults.toml` provide default settings for
the above algorithms (as read in `settingsman.py`)
* `metts.py` implements the METTS algorithm outside of "managers"

### PEPS
* `squarepeps.py` implements a finite, square-lattice Projected Entangled Pair
States (PEPS) ansatz, and ground-state optimization and time-evolution
algorithms. Uses a somewhat unconventional form of corner transfer matrix
for computation of expectation values (may not capture entanglement as well as
standard version in typical cases)

