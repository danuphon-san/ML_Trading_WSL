"""
Configuration loader with environment variable support

Supports variable interpolation using ${VAR_NAME} syntax in YAML files.
"""
import os
import re
import yaml
from typing import Any, Dict
from loguru import logger
from pathlib import Path


def load_env_file(env_file: str = ".env"):
    """
    Load environment variables from .env file if it exists

    Args:
        env_file: Path to .env file (default: .env in project root)
    """
    env_path = Path(env_file)

    if not env_path.exists():
        logger.debug(f"No .env file found at {env_path}")
        return

    logger.info(f"Loading environment variables from {env_path}")

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Set environment variable if not already set
                if key not in os.environ:
                    os.environ[key] = value
                    logger.debug(f"Set {key} from .env file")


def interpolate_env_vars(value: Any) -> Any:
    """
    Recursively interpolate environment variables in config values

    Supports ${VAR_NAME} or $VAR_NAME syntax
    Falls back to variable name if not found in environment

    Args:
        value: Configuration value (can be str, dict, list, etc.)

    Returns:
        Value with environment variables interpolated
    """
    if isinstance(value, str):
        # Pattern: ${VAR_NAME} or $VAR_NAME
        pattern = r'\$\{([^}]+)\}|\$([A-Z_][A-Z0-9_]*)'

        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            env_value = os.getenv(var_name)

            if env_value is None:
                logger.warning(
                    f"Environment variable '{var_name}' not found, "
                    f"keeping placeholder"
                )
                return match.group(0)  # Keep original ${VAR}

            return env_value

        return re.sub(pattern, replace_var, value)

    elif isinstance(value, dict):
        return {k: interpolate_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [interpolate_env_vars(item) for item in value]

    else:
        return value


def load_config(config_path: str = "config/config.yaml", load_env: bool = True) -> Dict:
    """
    Load configuration from YAML file with environment variable support

    Args:
        config_path: Path to config YAML file
        load_env: Whether to load .env file first (default: True)

    Returns:
        Configuration dictionary with env vars interpolated

    Example:
        # In config.yaml:
        fundamentals:
          simfin:
            api_key: ${SIMFIN_API_KEY}

        # In .env:
        SIMFIN_API_KEY=your-actual-key-here

        # Usage:
        config = load_config()
        # config['fundamentals']['simfin']['api_key'] == 'your-actual-key-here'
    """
    # Load .env file if requested
    if load_env:
        load_env_file()

    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Interpolate environment variables
    config = interpolate_env_vars(config)

    return config


def validate_required_keys(config: Dict, required_keys: Dict[str, str]) -> bool:
    """
    Validate that required configuration keys are set and not placeholders

    Args:
        config: Configuration dictionary
        required_keys: Dict of {config_path: description} to validate
                      Example: {'fundamentals.simfin.api_key': 'SimFin API key'}

    Returns:
        True if all required keys are valid

    Raises:
        ValueError if any required key is missing or still a placeholder
    """
    errors = []

    for key_path, description in required_keys.items():
        # Navigate nested dict using dot notation
        keys = key_path.split('.')
        value = config

        try:
            for key in keys:
                value = value[key]
        except (KeyError, TypeError):
            errors.append(f"Missing required config: {key_path} ({description})")
            continue

        # Check if value is still a placeholder
        if isinstance(value, str) and ('${' in value or value.startswith('$')):
            errors.append(
                f"Unresolved environment variable in {key_path}: {value}\n"
                f"  Please set the environment variable or update .env file"
            )

        # Check for placeholder text
        if isinstance(value, str) and (
            'your-api-key' in value.lower() or
            'replace-with' in value.lower() or
            'change-me' in value.lower()
        ):
            errors.append(
                f"Placeholder value in {key_path}: {value}\n"
                f"  Please set a real {description}"
            )

    if errors:
        error_msg = "\n".join(errors)
        raise ValueError(f"Configuration validation failed:\n{error_msg}")

    return True


# Convenience function for common use case
def load_config_with_validation(
    config_path: str = "config/config.yaml",
    provider: str = None
) -> Dict:
    """
    Load config and validate based on provider

    Args:
        config_path: Path to config file
        provider: Provider to validate for (simfin, alpha_vantage, etc.)
                 If None, detects from config

    Returns:
        Validated configuration dictionary
    """
    config = load_config(config_path)

    # Detect provider if not specified
    if provider is None:
        provider = config.get('fundamentals', {}).get('provider', 'yfinance')

    # Validate based on provider
    if provider == 'simfin':
        validate_required_keys(config, {
            'fundamentals.simfin.api_key': 'SimFin API key'
        })
    elif provider == 'alpha_vantage':
        validate_required_keys(config, {
            'fundamentals.alpha_vantage.api_key': 'Alpha Vantage API key'
        })

    return config
