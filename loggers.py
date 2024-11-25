import logging
import time
import os

# Ensure logs are written to the script's directory
log_file_path = os.path.join(os.getcwd(), "all_logs.log")

# Function to set up the logger
def setup_logger():
    logger = logging.getLogger("custom_logger")
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # Set the logging level
        
        # Create a file handler
        file_handler = logging.FileHandler(log_file_path, mode='a')  # Append mode
        
        # Custom formatter to include UTC offset
        class UTCFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                local_time = time.localtime(record.created)
                utc_offset = time.strftime('%z', local_time)
                formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
                return f"{formatted_time} {utc_offset}"
        
        # Create and set the custom formatter
        formatter = UTCFormatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(file_handler)
    
    return logger

# Instantiate the logger
logger = setup_logger()

# Function to log messages
def logger_func(message, log_type):
    log_type = log_type.lower()  # Normalize the log type input
    if log_type in ['i', 'info']:
        logger.info(message)
    elif log_type in ['d', 'debug']:
        logger.debug(message)
    elif log_type in ['w', 'warning']:
        logger.warning(message)
    elif log_type in ['e', 'error']:
        logger.error(message)
    elif log_type in ['c', 'critical']:
        logger.critical(message)
    else:
        logger.error(f"Unknown log type '{log_type}' for message: {message}")

