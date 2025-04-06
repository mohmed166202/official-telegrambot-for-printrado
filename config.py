import os
from typing import Any, Dict
from dotenv import load_dotenv

class ConfigurationError(Exception):
    """Raised when there's an error in configuration."""
    pass

class Config:
    """Configuration management for the Book Recommendation Bot."""
    
    REQUIRED_ENV_VARS = {
        'API_TOKEN': 'Telegram Bot API Token',
        'DB_PASS': 'Database Password',
        'DB_USER': 'Database Username',
        'DB_HOST': 'Database Host'
    }

    def __init__(self):
        """Initialize configuration with environment variables and validation."""
        # Check if .env file exists
        if not os.path.exists('.env'):
            print("Warning: .env file not found in current directory")
            print(f"Current working directory: {os.getcwd()}")
        else:
            print(f".env file found in current directory: {os.getcwd()}")
        
        load_dotenv()
        self._validate_environment()
        self._load_configuration()

    def _validate_environment(self) -> None:
        """Validate that all required environment variables are present."""
        missing_vars = [
            var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)
        ]
        if missing_vars:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
        print("Environment variables validation successful.")

    def _load_configuration(self) -> None:
        """Load and validate all configuration values."""
        try:
            # Bot Configuration
            self.API_TOKEN = os.getenv("API_TOKEN")
            self.BOT_NAME = os.getenv("BOT_NAME", "BookRecommendationBot")
            self.WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
            
            # Database Configuration
            self.DB_HOST = os.getenv("DB_HOST")
            self.DB_PORT = self._validate_port(os.getenv("DB_PORT", "3306"))
            self.DB_NAME = os.getenv("DB_NAME", "Printrado_books")
            self.DB_USER = os.getenv("DB_USER")
            self.DB_PASSWORD = os.getenv("DB_PASS")
            
            # Database Connection Settings
            self.DB_CONNECTION_TIMEOUT = int(os.getenv("DB_CONNECTION_TIMEOUT", "30"))
            self.DB_MAX_RETRIES = int(os.getenv("DB_MAX_RETRIES", "3"))
            self.DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
            self.DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
            
            # SSL/TLS Configuration for AWS RDS
            self.DB_SSL = {
                'ssl': {
                    'ssl_verify_identity': True,
                    'ssl_verify_cert': True
                }
            }
            
            # Model Configuration
            self.MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
            self.EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "book_embeddings.pkl")
            self.DEVICE = os.getenv("DEVICE", "cpu")
            self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
            
            # Rate Limiting
            self.RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
            self.RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

            # Log configuration for debugging
            self._log_database_configuration()
            
            print("\nLoaded Environment Variables:")
            print(f"API_TOKEN: {self.API_TOKEN}")
            print(f"DB_HOST: {self.DB_HOST}")
            print(f"DB_PORT: {self.DB_PORT}")
            print(f"DB_NAME: {self.DB_NAME}")
            print(f"DB_USER: {self.DB_USER}")
            print(f"DB_PASSWORD: {self.DB_PASSWORD}")
            
        except ValueError as e:
            raise ConfigurationError(f"Invalid configuration value: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {str(e)}")

    def _log_database_configuration(self) -> None:
        """Log database configuration for debugging purposes."""
        print("\nDatabase Configuration:")
        print("-" * 20)
        print(f"Host: {self.DB_HOST}")
        print(f"Port: {self.DB_PORT}")
        print(f"Database: {self.DB_NAME}")
        print(f"User: {self.DB_USER}")
        print(f"SSL Enabled: {bool(self.DB_SSL)}")
        print("-" * 20 + "\n")

    @staticmethod
    def _validate_port(port_str: str) -> int:
        """Validate and convert port number."""
        try:
            port = int(port_str)
            if not 1 <= port <= 65535:
                raise ValueError(f"Port number must be between 1 and 65535, got {port}")
            return port
        except ValueError as e:
            raise ConfigurationError(f"Invalid port number: {port_str}. {str(e)}")