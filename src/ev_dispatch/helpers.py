
import cProfile
import pstats
import io
from functools import wraps

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(20)
        pr.dump_stats('profiler_out_3.prof')
        return result
    return wrapper