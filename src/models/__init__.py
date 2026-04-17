from .base_model import BaseModel

__all__ = [
	'BaseModel',
	'GASelector',
	'PSOSelector',
	'ABCSelector',
	'HybridSelector',
	'JointSelector',
]


def __getattr__(name):
	if name == 'GASelector':
		from .ga_selector import GASelector
		return GASelector
	if name == 'PSOSelector':
		from .pso_selector import PSOSelector
		return PSOSelector
	if name == 'ABCSelector':
		from .abc_selector import ABCSelector
		return ABCSelector
	if name == 'HybridSelector':
		from .hybrid_selector import HybridSelector
		return HybridSelector
	if name == 'JointSelector':
		from .joint_selector import JointSelector
		return JointSelector

	raise AttributeError(f"module 'src.models' has no attribute '{name}'")
