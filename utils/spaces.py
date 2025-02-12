import functools

def GPU(func):
    """Decorator to run a function on the fake GPU
        to get comparable with HF Space"""
    @functools.wraps(func) # Preserves original function's metadata
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return wrapper