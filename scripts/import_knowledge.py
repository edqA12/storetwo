"""Run from the project: python scripts/import_knowledge.py path/to/file.pdf"""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.knowledge import store
from services.knowledge_import import import_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='本地导入知识资料，不发送文档至云端')
    parser.add_argument('file', nargs='?', type=Path)
    args = parser.parse_args()
    if args.file:
        import_file(store(), args.file, print)
    store().embed_missing(print)
