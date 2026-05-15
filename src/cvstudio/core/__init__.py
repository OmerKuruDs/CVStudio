"""Domain primitives: operations, parameters, pipelines, registry."""

from cvstudio.core.operation import OperationSpec, Parameter, ParamKind
from cvstudio.core.pipeline import Pipeline, PipelineNode, Roi
from cvstudio.core.registry import (
    all_operations,
    get_operation,
    register_operation,
)

__all__ = [
    "OperationSpec",
    "ParamKind",
    "Parameter",
    "Pipeline",
    "PipelineNode",
    "Roi",
    "all_operations",
    "get_operation",
    "register_operation",
]
