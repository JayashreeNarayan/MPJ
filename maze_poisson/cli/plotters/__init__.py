import re

rgx_N = re.compile(r'N(\d+)')

def get_N(filename):
    return int(rgx_N.search(filename).group(1))