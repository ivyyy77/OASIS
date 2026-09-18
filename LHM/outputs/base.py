"""Common mapping behavior for renderer output dataclasses."""

from collections import OrderedDict
from dataclasses import fields, is_dataclass
from typing import Any, Tuple


class BaseOutput(OrderedDict):
    """Dataclass output that supports both attributes and mapping access."""

    def __post_init__(self) -> None:
        class_fields = fields(self)
        if not class_fields:
            raise ValueError(f"{self.__class__.__name__} has no fields")

        first = getattr(self, class_fields[0].name)
        if all(getattr(self, field.name) is None for field in class_fields[1:]) and isinstance(first, dict):
            for key, value in first.items():
                self[key] = value
        else:
            for field in class_fields:
                value = getattr(self, field.name)
                if value is not None:
                    self[field.name] = value

    def __getitem__(self, key: Any) -> Any:
        return dict(self.items())[key] if isinstance(key, str) else self.to_tuple()[key]

    def __setattr__(self, name: Any, value: Any) -> None:
        if name in self and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __delitem__(self, *args, **kwargs):
        raise TypeError(f"cannot delete fields from {self.__class__.__name__}")

    def setdefault(self, *args, **kwargs):
        raise TypeError(f"cannot add defaults to {self.__class__.__name__}")

    def pop(self, *args, **kwargs):
        raise TypeError(f"cannot remove fields from {self.__class__.__name__}")

    def update(self, *args, **kwargs):
        raise TypeError(f"cannot replace fields through update on {self.__class__.__name__}")

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable_, _args, *remaining = super().__reduce__()
        return callable_, tuple(getattr(self, field.name) for field in fields(self)), *remaining

    def to_tuple(self) -> Tuple[Any, ...]:
        return tuple(self[key] for key in self.keys())
