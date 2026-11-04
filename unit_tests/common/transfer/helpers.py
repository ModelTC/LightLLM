from contextlib import nullcontext


class Event:
    def __init__(self, ready=False):
        self.ready = ready
        self.waits = 0

    def query(self):
        return self.ready

    def synchronize(self):
        self.waits += 1
        self.ready = True


class Runtime:
    def __init__(self):
        self.events = []

    def use_stream(self, stream=None):
        return nullcontext()

    def record_event(self):
        event = Event()
        self.events.append(event)
        return event
