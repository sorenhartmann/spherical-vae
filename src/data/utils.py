from pygtrie import StringTrie
import inspect
import zipfile
from pathlib import Path
import io

class Singleton(type):
    _instances = {}
    _init = {}

    def __init__(cls, name, bases, dct):
        cls._init[cls] = dct.get("__init__", None)

    def __call__(cls, *args, **kwargs):
        init = cls._init[cls]
        if init is not None:
            key = (
                cls,
                frozenset(inspect.getcallargs(init, None, *args, **kwargs).items()),
            )
        else:
            key = cls

        if key not in cls._instances:
            cls._instances[key] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[key]

class ZippedData(metaclass=Singleton):

    def __init__(self, file_path):

        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(self.file_path)

        with self.open_zip() as archive:
            file_names = [
                file_.filename for file_ in archive.filelist if not file_.is_dir()
            ]

        self.file_trie = StringTrie.fromkeys(file_names)

    def iter_files(self, directory:str, ext:str=None):

        assert self.file_trie.has_subtrie(directory), f"Directory {directory} not found"

        file_iter = self.file_trie.iterkeys(directory)
        files = [file_ for file_ in file_iter if ext is None or file_.endswith(ext)]
        with self.open_zip() as archive:
            for file_ in files:
                with archive.open(file_) as f:
                    bytes_ = f.read()
                    buffer = io.BytesIO(bytes_)
                    
                yield file_, io.TextIOWrapper(buffer)

    def open_zip(self):
        return zipfile.ZipFile(self.file_path)


if __name__ == "__main__":

    zipped_data = ZippedData("data/raw/allasfamc.zip")
    for buffer in zipped_data.iter_files("all_asfamc/subjects/10", ".amc"):
        print(buffer)
