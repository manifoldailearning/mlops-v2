from src.config.core import package_root


with open(package_root / "src" / "VERSION") as file:
    __version__ = file.read().strip()
    
print(__version__)