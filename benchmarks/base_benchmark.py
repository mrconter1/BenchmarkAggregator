from abc import ABC, abstractmethod
import os
import tempfile
import shutil
import aiofiles
import aiohttp

class BaseBenchmark(ABC):
    def __init__(self):
        self.id = None
        self.temp_dir = None

    @abstractmethod
    async def setup(self):
        pass

    @abstractmethod
    async def run(self, model: str, client) -> float:
        pass

    @abstractmethod
    async def cleanup(self):
        pass

    def create_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp()

    def remove_temp_dir(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @staticmethod
    async def download_file(url: str, local_path: str):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    async with aiofiles.open(local_path, 'wb') as file:
                        while True:
                            chunk = await response.content.read(8192)
                            if not chunk:
                                break
                            await file.write(chunk)
            print(f"File downloaded successfully and saved to {local_path}")
        except Exception as e:
            print(f"An error occurred while downloading the file: {e}")
            raise