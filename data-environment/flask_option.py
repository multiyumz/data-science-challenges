# pylint: disable=missing-docstring

import os

def start():
    """returns the right message"""
    env = os.getenv('FLASK_ENV')
    if env:
        return f"Starting in {env} mode..."
    return "Starting in empty mode..."

# def start():
#     """returns the right message"""
#     # $CHALLENGIFY_BEGIN
#     env = os.getenv(key = 'FLASK_ENV', default="empty")

#     return f"Starting in {env} mode..."

if __name__ == "__main__":
    print(start())
