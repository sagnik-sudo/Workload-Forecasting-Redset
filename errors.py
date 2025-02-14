from datetime import datetime

class DataLoadError(Exception):
    """Custom error raised when there is a problem loading the data."""
    
    def __init__(self, message, code=None):
        """
        Args:
            message (str): A descriptive error message.
            code (int, optional): An optional error code for more granular error handling.
        """
        # Capture the current timestamp
        self.timestamp = datetime.now()
        self.message = message
        self.code = code
        
        # Format the error message with timestamp and error code (if provided)
        timestamp_str = self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        if code is not None:
            full_message = f"[{timestamp_str}] [Error {code}] {message}"
        else:
            full_message = f"[{timestamp_str}] {message}"
            
        super().__init__(full_message)
    
    def log_error(self):
        """Log the error details along with the timestamp."""
        print(f"Error Logged at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {self}")

class DataSplitError(Exception):
    """Custom error raised when there is an issue splitting the dataset."""
    
    def __init__(self, message, code=None):
        """
        Args:
            message (str): A descriptive error message.
            code (int, optional): An optional error code for more granular error handling.
        """
        self.timestamp = datetime.now()
        self.message = message
        self.code = code
        
        timestamp_str = self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        if code is not None:
            full_message = f"[{timestamp_str}] [Error {code}] {message}"
        else:
            full_message = f"[{timestamp_str}] {message}"
            
        super().__init__(full_message)
    
    def log_error(self):
        """Log the error details along with the timestamp."""
        print(f"Error Logged at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {self}")