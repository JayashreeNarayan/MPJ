import os


allowed_profilers = {}
try:
    import line_profiler as lprof
except ImportError:
    pass
else:
    allowed_profilers['LINE_PROFILE'] = lprof.profile

try:
    import memory_profiler as mprof
except ImportError:
    pass
else:
    allowed_profilers['MEMORY_PROFILE'] = mprof.profile

for k,v in allowed_profilers.items():
    if k in os.environ:
        profile = v
        break
else:
    profile = lambda x: x