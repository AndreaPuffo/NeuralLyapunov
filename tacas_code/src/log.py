from __future__ import print_function

import signal
import sys
import os
import datetime
import atexit

PRINT_FILE = False
PRINT_STDOUT = True
LOGS_DIR = None
DEBUG = True

def _write_to_file(s, f):
    f.write(s)
    f.flush()


def log(s, stdout=True, fileobj=None, filemode='a'):
    if stdout and PRINT_STDOUT:
        print(s)
        sys.stdout.flush()

    if fileobj is not None and PRINT_FILE:
        if isinstance(fileobj, str):
            with open(fileobj, filemode) as f:
                _write_to_file(s, f)
        elif isinstance(fileobj, file):
            _write_to_file(s, fileobj)


class Logger(object):
    def __init__(self):
        self._logs = []
        self._atexit_on = False

    @staticmethod
    def log(*args, **kwargs):
        return log(*args, **kwargs)

    def get_logs(self):
        return self._logs

    def register(self, logs, kwargs={}):
        self._logs = logs

        if not kwargs or 'fileobj' not in kwargs:
            calframe = sys._getframe(1)
            caller = calframe.f_code.co_name
            caller_args = ','.join("(%s=%s)" % (k, v) for k, v in calframe.f_locals.items())

            fname = str(caller) + str(datetime.datetime.now().isoformat())
        else:
            fname = kwargs['fileobj']
            kwargs = {k: v for k, v in kwargs.items() if k != 'fileobj'}
        fname = os.path.join(LOGS_DIR, fname)

        def handler(signum=0, *_):
            _fname = fname + ("_%d" % signum)
            try:
                _, cols = os.popen('stty size', 'r').read().split()
                cols = int(cols)
            except:
                cols = 10
            s = self._get_log(cols=cols, fname=_fname, caller=caller, caller_args=caller_args)
            log(s, fileobj=_fname, **kwargs)
            sys.exit(0)

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGABRT, handler)
        signal.signal(signal.SIGTERM, handler)

        if not self._atexit_on:
            self._atexit_on = True
            atexit.register(handler)

    def flush(self, fileobj, filemode='a'):
        s = self._get_log()
        log(s, fileobj=fileobj, filemode=filemode)

    def _get_log(self, cols=10, fname='', caller='', caller_args=''):
        log("\n" + "~" * (cols - 1))
        log("Printing logs by %s to %s" % (Logger.__name__, fname))
        if type(self._logs) == dict:
            s = '\n'.join(("%s %s" % (k, ' '.join(str(x) for x in v))) for k, v in self._logs.items())
        else:
            s = '\n'.join(str(x) for x in self._logs)
        s = ("%s(%s)" % (caller, caller_args)) + s
        return s
