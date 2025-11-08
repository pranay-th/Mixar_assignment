import os, json, logging
from datetime import datetime

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dump_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    

def init_logger(out_dir):
    ensure_dir(out_dir)
    logger = logging.getLogger('mixar')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(out_dir, 'run.log'))
        fm = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        fh.setFormatter(fm)
        logger.addHandler(fh)
        # console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fm)
        logger.addHandler(ch)
    return logger
