# File: models/__init__.py
from .framework_baseline import Network as BaselineNetwork
from .framework_gnn import Network as GNNNetwork
# framework_gnn_message is likely redundant now if framework_gnn handles msg_type