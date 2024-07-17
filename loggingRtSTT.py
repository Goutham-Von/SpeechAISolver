import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('speectRtSTT.log')
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

def log(message) :
    logger.log(msg=message, stack_info=True, level=logging.DEBUG)