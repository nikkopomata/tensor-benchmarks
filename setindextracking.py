"""Import first in order to employ strict verification that vector spaces
match"""

from . import links
if links._tensorimported:
  raise ValueError('Must set index tracking before initializing Tensor class')

links.VSpace = links.VectorSpaceTracked
