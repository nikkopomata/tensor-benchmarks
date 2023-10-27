"""Import first in order to employ strict verification that vector spaces
match"""

from . import links
if links._tensorimported:
  raise ValueError('Must set index tracking before initializing Tensor class')

links.VSpace = links.VectorSpaceTracked
links.spacetype = 'tracked'

def weak():
  links.VSpace = links.VectorSpaceTrackedWeak
  links.spacetype = 'weak'

def ided(): # TODO make default?
  links.VSpace = links.VectorSpaceIDed
  links.spacetype = 'uuid'
