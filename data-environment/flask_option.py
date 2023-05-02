# pylint: disable=missing-docstring

import os

def start():
    """returns the right message"""
    env = os.getenv('FLASK_ENV')
    if env:
        return f"Starting in {env} mode..."
    return "Starting in empty mode..."

if __name__ == "__main__":
    print(start())
