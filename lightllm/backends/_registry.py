"""Process-local backend selection, completed before inference modules bind ops."""

from threading import RLock


class BackendRegistry:
    def __init__(self, kind, default_factory):
        self.kind = kind
        self.factories = {"cuda": default_factory}
        self.selected = "cuda"
        self.instance = None
        self.lock = RLock()

    def register(self, name, factory):
        with self.lock:
            if name in self.factories:
                raise ValueError(f"{self.kind} backend {name!r} is already registered")
            self.factories[name] = factory

    def select(self, name):
        with self.lock:
            if name not in self.factories:
                raise ValueError(f"Unknown {self.kind} backend: {name!r}")
            if self.instance is not None and name != self.selected:
                raise RuntimeError(f"Select the {self.kind} backend before importing inference modules")
            self.selected = name

    def get(self):
        with self.lock:
            if self.instance is None:
                self.instance = self.factories[self.selected]()
            return self.instance
