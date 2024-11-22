
import logging
from datetime import datetime

class ActivityLogger:
    """Logs assistant activities for monitoring and debugging."""

    def __init__(self, log_file="activity.log"):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger()

    def log_info(self, message):
        """Log an informational message."""
        self.logger.info(message)

    def log_error(self, message):
        """Log an error message."""
        self.logger.error(message)

    def log_activity(self, activity_type, details):
        """Log a specific activity with details."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"{timestamp} - {activity_type}: {details}")

if __name__ == "__main__":
    # Example usage
    logger = ActivityLogger()
    logger.log_info("This is an info log.")
    logger.log_error("This is an error log.")
    logger.log_activity("TEST_ACTIVITY", "Testing the activity logger.")
