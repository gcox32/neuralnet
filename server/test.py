"""
Usage: test.py --slug=<slug>
"""

import requests
from docopt import docopt

arguments = docopt(__doc__)

base = 'http://127.0.0.1:5000/'

slug = arguments['--slug']

if slug == None:
    slug = 'upload'

try:
    response = requests.get(base + slug)
    print(response.json())
except:
    print(f'404 : end point ({slug}) does not exist')

