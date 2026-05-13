"""Domain primitives: operations, parameters, pipelines, registry."""

from cvsandbox.core.operation import OperationSpec, Parameter, ParamKind
from cvsandbox.core.pipeline import Pipeline, PipelineNode
from cvsandbox.core.registry import (
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
    "all_operations",
    "get_operation",
    "register_operation",
]
