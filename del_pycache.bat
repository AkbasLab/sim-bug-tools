python -Bc "for p in __import__('pathlib').Path('.').rglob('*.py[co]'): p.unlink()"
python -Bc "for p in __import__('pathlib').Path('.').rglob('__pycache__'): p.rmdir()"