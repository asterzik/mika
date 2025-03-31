import cProfile


class ProfileContext:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.disable()
        self.profiler.dump_stats(self.filename)
