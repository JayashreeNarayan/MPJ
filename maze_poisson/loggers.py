import logging
import os
import time

# Ensure logs are written to the script's directory
log_file_path = os.path.join(os.getcwd(), "all_logs.log")

# Function to set up the logger
def setup_logger():
    logger = logging.getLogger("custom_logger")
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # Set the logging level
        
        # Create a file handler
        
        # Custom formatter to include UTC offset
        class UTCFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                local_time = time.localtime(record.created)
                utc_offset = time.strftime('%z', local_time)
                formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
                return f"{formatted_time} {utc_offset}"
        
        # Create and set the custom formatter
        file_handler = logging.FileHandler(log_file_path, mode='a')  # Append mode
        file_formatter = UTCFormatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        # Add a stream handler to print to console
        stream_formatter = logging.Formatter('%(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(stream_formatter)
        stream_handler.setLevel(logging.INFO)  # Ensure debug messages are not printed to the console
        logger.addHandler(stream_handler)
    
    return logger

# Instantiate the logger
logger = setup_logger()
