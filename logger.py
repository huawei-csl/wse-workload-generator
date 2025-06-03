
import logging 
import sys

def init_logger(level):
    logging.basicConfig(
        filename='out.log',
        level=level,
        filemode="w",
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
    )
    stdout_handler = logging.StreamHandler(
        stream=sys.stdout,
    )
    logging.getLogger().addHandler(stdout_handler)
