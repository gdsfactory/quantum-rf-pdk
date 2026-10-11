"""Singleton metaclass for classes intended to have a single instance."""

import threading
from typing import Any, ClassVar

__all__ = ["SingletonMeta"]


class SingletonMeta(type):
    """Metaclass that hands out the same instance on every instantiation.

    A class can define ``_check_singleton_args(self, *args, **kwargs)`` to vet the
    arguments of a later call against the cached instance; it should raise if they
    ask for a different configuration than the instance has.
    """

    _instances: ClassVar[dict[type, Any]] = {}
    # Lock because the check-then-create pair is racy across threads
    _lock = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Construct cls on first call, then return the cached instance.

        Returns:
            The single instance of ``cls``.
        """
        with SingletonMeta._lock:
            if cls not in SingletonMeta._instances:
                SingletonMeta._instances[cls] = super().__call__(*args, **kwargs)
                return SingletonMeta._instances[cls]
            instance = SingletonMeta._instances[cls]
        check = getattr(instance, "_check_singleton_args", None)
        if check is not None:
            check(*args, **kwargs)
        return instance
