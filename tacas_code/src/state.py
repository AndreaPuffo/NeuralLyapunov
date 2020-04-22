from collections import namedtuple


synthesizer_state = namedtuple('synthesizer_state', [
    'timed_out',
    'P',
    'A',
    'iter',
    'error',
    'elapsed_time',
    'learner',
    'verifier',
    'res',
])
