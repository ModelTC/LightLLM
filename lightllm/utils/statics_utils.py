class MovingAverage:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def add(self, value):
        self.total += value
        self.count += 1

    def average(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count
