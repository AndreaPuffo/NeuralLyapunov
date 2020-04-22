from collections import namedtuple


instance_params = namedtuple('instance_params', [
    'learner',
    'verifier',
    'P',
    'A',
    'xs',
    'V',
    'Vd',
    'Vd_diag',
    #'V_list',
    #'Vd_list',
    'Vd_diag_list',
    'queue',
    'stop',
])
