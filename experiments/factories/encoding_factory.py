"""State encoding factory definition"""
from typing import Any, Type

from pennylane.operation import Operation


class EncodingFactory(object):
    """State encoding factory"""

    def __init__(self, class_obj: Type[Operation], **kwargs: Any):
        self.class_obj = class_obj
        self.kwargs = kwargs

    def create(self, state: Any, wires: Any) -> Operation:
        """Creates a state encoding instance"""
        return self.class_obj(state, wires=wires, **self.kwargs)
