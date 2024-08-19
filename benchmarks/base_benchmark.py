from abc import ABC, abstractmethod
import os
import tempfile
import shutil

class BaseBenchmark(ABC):
    def __init__(self):
        self.id = None
        self.temp_dir = None

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def run(self, model: str, client) -> float:
        pass

    @abstractmethod
    def cleanup(self):
        pass

    def create_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp()

    def remove_temp_dir(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)