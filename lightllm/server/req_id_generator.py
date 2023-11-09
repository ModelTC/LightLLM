import threading

class ReqIDGenerator:
    def __init__(self, start_id=0):
        self.current_id = start_id
        self.lock = threading.Lock()

    def generate_id(self):
        with self.lock:
            id = self.current_id
            self.current_id += 1
        return id