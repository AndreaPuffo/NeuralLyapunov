from src.verifier import Z3Verifier
from src.iter_learner import IterativeLearner
from src.instance_params import instance_params
from src.state import synthesizer_state
from src.utils import gen_P
from src.z3_learner import IterativeZ3Learner
from src.consts import CEGIS_MAX_ITERS
from src.common import is_valid_matrix_A, diagonalize, check_with_timeout, x_dot, is_limit_matrix_A, diagonaliseVd
import timeit
import multiprocessing as mp
from z3 import *
import sympy as sp

from src.log import log


class Cegis(object):
    def __init__(self, A, max_iters=CEGIS_MAX_ITERS):
        assert is_valid_matrix_A(A)
        self.n = A.shape[0]

        self.A, self.U = self._diagonalize(A)

        self.max_iters = max_iters

        self.err = False
        self._stop_event = mp.Value('b', 0)

        self.learners = []
        self.verifiers = [
            Z3Verifier,
        ]

        self.xs = sp.Matrix(sp.symbols('x:%d' % self.n))
        self.P = None

        self.queue = None
        self.elapsed_time = 0
        self.iter = 0
        self.timeout, self.has_timedout = float('inf'), False

        self.postverification = False

    def __repr__(self):
        learner = "%s" % self.learners[0]
        timeout = ".timeout(%r)" % self.timeout if 0 < self.timeout < float('inf') else ""
        verifier = ".set_verifier(%r)" % self.verifiers[0]
        return "cegis(%s)%s%s%s" % (self.A, timeout, learner, verifier) if self.max_iters == CEGIS_MAX_ITERS \
            else "cegis(%s, %s)%s%s%s" % (self.A, self.max_iters, timeout, learner, verifier)

    def get_A(self):
        return self.A

    def get_n(self):
        return self.n

    def get_P(self):
        return {
            'P': self.P,
            'U': self.U
        }

    def set_max_iters(self, m):
        """

        :param m: number
        :return: self
        """
        assert m >= 0
        self.max_iters = m
        return self

    def error(self):
        """

        :return: True iff there was an error during solve()
        """
        return self.err

    def stop_early(self, b):
        """
        Stop as soon as one of the learners finds a valid P
        :param b: True iff cegis should stop early
        :return: self
        """
        self._stop_event = mp.Value('b', 0) if b is True else None
        return self

    def set_xs(self, xs):
        """
        Set x variables
        :param xs: sp.Matrix(sp.symbols('x:%d' % n))
        :return: self
        """
        self.xs = xs
        return self

    def solve(self, queue=None, exponent=1, nl=False, fx=None):
        """

        :param template_n: max number of P matrices (for template generation)
        :param queue: optional queue to get a synthesizer's state
        :param nl: True if computing nonlinear Vd
        :param fx: nonlinear sys fcn to compute nonlin Vd
        :return: True iff a solution has been found
        """
        assert exponent >= 1
        self.queue = queue

        res = False

        states = []
        ps = []
        qs = []

        Ps = []
        V_list = []
        Vd_list = []
        for e in range(1, exponent+1):
            xs_t = x_dot(self.xs, e)  # (x.)^t
            Ps = self._gen_P(e, Ps)
            # in case of eig == 0
            # assert Lyap fcn : V = x^T P x - epsi * x^T* eye * x
            # in this way the layp fcn is 'lifted'

            if True: # is_limit_matrix_A(self.A):  #TODO
                epsi = 0
                V_list += [xs_t.T* ( Ps[-1] - epsi * sp.eye(len(self.A[0,:])) ) * xs_t]
                if nl:
                    #Vd_list += [fx.T * (Ps[-1]-epsi*sp.eye(len(self.A[0,:]))) * xs_t + xs_t.T * (Ps[-1] - epsi * sp.eye(len(self.A[0,:]))) * fx]
                    # Vd = fx.T * (xs_t**2).diff(xs_t)
                    x2 = x_dot(xs_t, 2)
                    xd = sp.Matrix([ [ x2[l].diff(self.xs[l]) for l in range(len(xs_t))] ])
                    Vd_list += [ xd * Ps[-1] * fx ]
                else:
                    Vd_list += [xs_t.T*(self.A.T * (Ps[-1]- epsi * sp.eye(len(self.A[0,:]))) +  (Ps[-1] - epsi * sp.eye(len(self.A[0,:])))  *self.A)*xs_t]
            else:
                #V = sp.MatMul(xs_t.T, Ps[-1], xs_t)  # V(x) = x^T P x
                V_list += [xs_t.T* Ps[-1]* xs_t]
                # Vd = sp.MatMul(xs_t.T,           # Vd = x^T (A^T P + P A) x
                #                       sp.MatAdd(
                #                           sp.MatMul(self.A.T, Ps[-1]),
                #                           sp.MatMul(Ps[-1], self.A)
                #                       ),
                #                 xs_t
                # )
                Vd_list += [xs_t.T*(self.A.T * Ps[-1] + Ps[-1]*self.A)*xs_t]

        # divide Vd-diag and Vd-offdiag
        Vd_diag_list, _ = diagonaliseVd(Vd_list, self.xs)

        V = reduce(lambda x, y: x + y, V_list, sp.zeros(1))
        Vd = reduce(lambda x, y: x + y, Vd_list, sp.zeros(1))
        Vd_diag = reduce(lambda x, y: x + y, Vd_diag_list, sp.zeros(1))

        for learner in self.learners:
            for verifier in self.verifiers:
                queue = mp.Queue()
                qs.append(queue)
                ip = instance_params(
                    learner=learner,
                    verifier=verifier,
                    P=Ps,
                    A=self.A,
                    xs=self.xs,
                    V=V, Vd=Vd, Vd_diag=Vd_diag,
                    # V_list=V_list, Vd_list=Vd_list,
                    Vd_diag_list=Vd_diag_list,
                    queue=queue,
                    stop=self._stop_event,
                )
                ps.append(mp.Process(target=self._find_P, args=(ip,)))

        for p in ps:
            p.start()

        for queue in qs:
            s = queue.get()
            states.append(s)
            res = res or s.res

        for p in ps:
            p.join()

        best_state = sorted(states,
                            key=lambda state:
                            int(not state.error) *
                            int(not state.timed_out) *
                            int(state.iter < self.max_iters) *
                            int(bool(state.P)) *  # if one finished early
                            state.elapsed_time)[-1]

        s = synthesizer_state(
            timed_out=best_state.timed_out,
            P=best_state.P,
            iter=best_state.iter,
            A=self.get_A(),
            error=best_state.error,
            elapsed_time=best_state.elapsed_time,
            verifier=("%s" % best_state.verifier),
            learner=("%s" % best_state.learner),
            res=best_state.res
        )

        self.has_timedout = best_state.timed_out
        self.P = best_state.P
        self.iter = best_state.iter
        self.err = best_state.error
        self.elapsed_time = best_state.elapsed_time

        self._send_info(s)
        #log("\nall states:\n")
        #for _s in states:
        #    log("%s" % (_s,))
        log("\nbest state: %s" % (s,))
        return res

    def set_timeout(self, tSec):
        """

        :param tSec: time in seconds
        :return: self
        """
        self.timeout = tSec
        return self

    def set_learners(self, ls):
        """

        :param ls: list of learners' classes
        :return: self
        """
        self.learners = ls

        return self

    def set_verifiers(self, vs):
        """

        :param vs: list of verifiers' classes
        :return: self
        """
        self.verifiers = vs
        return self

    def _find_P(self, params):
        start_t = timeit.default_timer()

        i = 0
        _has_timedout = False
        err = False
        elapsed_time = 0
        stop = False

        stop_event = params.stop
        learner = params.learner(params)
        verifier = params.verifier(params)

        while not stop:
            i += 1

            learner.timeout(self.timeout - elapsed_time)
            learner_result = learner.learn()

            if stop_event and stop_event.value > 0:
                stop = True
            # no candidate
            elif learner_result == unsat or learner.error() or learner.has_timedout():  # needs to be the first test
                #log("unsat")
                elapsed_time = timeit.default_timer() - start_t
                params.queue.put(synthesizer_state(
                    timed_out=(_has_timedout or learner.has_timedout()),
                    P=None,
                    iter=i,
                    A=self.get_A(),
                    error=err,
                    elapsed_time=elapsed_time,
                    verifier=("%s" % verifier),
                    learner=("%s" % learner),
                    res=False,
                ))
                return False
            # candidate
            elif learner_result == sat:
                sol = learner.get_solution()

                verifier.timeout(self.timeout - elapsed_time)
                ver, _has_timedout = check_with_timeout(verifier, self.timeout - elapsed_time, args=(sol,))

                # counterexample
                if ver == sat:
                    n = 0
                    for counterexample in verifier.get_unseen_counterexamples():
                        #log("CEGIS received counterexample %s" % counterexample)
                        learner.add_counterexample(counterexample)
                        n += 1
                    if n == 0:
                        # no more counterexamples, P invalid
                        stop = True
                        err = True
                    # verifier._clear_counterexamples()
                # solution
                elif ver == unsat:
                    elapsed_time = timeit.default_timer() - start_t

                    # assert valid solution P
                    err = err or not self._verify(sol, self.A) if self.postverification else err

                    #log("iter %d P %s" % (i, sol))
                    #log("verifier %s - learner %s" % (verifier, learner))

                    params.queue.put(synthesizer_state(
                        timed_out=_has_timedout,
                        P=sol,
                        iter=i,
                        A=self.get_A(),
                        error=err,
                        elapsed_time=elapsed_time,
                        verifier=("%s" % verifier),
                        learner=("%s" % learner),
                        res=True,
                    ))
                    if stop_event:
                        stop_event.value = 1

                    return True
                else:
                    err = True
            else:
                err = True

            elapsed_time = timeit.default_timer() - start_t
            _has_timedout = elapsed_time >= self.timeout

            err = err or verifier.error() or learner.error()

            stop = stop or err or i >= self.max_iters or _has_timedout
            if stop_event:
                stop = stop or stop_event.value > 0
                if _has_timedout:
                    stop_event.value = 1

        params.queue.put(synthesizer_state(
            timed_out=_has_timedout,
            P=None,
            iter=i,
            A=self.get_A(),
            error=err,
            elapsed_time=elapsed_time,
            verifier=("%s" % verifier),
            learner=("%s" % learner),
            res=False,
        ))
        return False

    def _send_info(self, state):
        if self.queue is not None:
            self.queue.put(state)

    def _diagonalize(self, A):
        diag = diagonalize(A)
        if diag is None:
            return A, sp.eye(self.n)
        return diag

    def _reset_verifier(self, verifier_class, learner_class):
        raise NotImplementedError

    def _reset_learner(self, learner_class):
        raise NotImplementedError

    @staticmethod
    def _verify(P, A):
        # eg. for dReal
        import z3
        from src.sympy_converter import sympy_converter
        from src.linear import f0, f1
        from src.common import OrNotZero
        eigvals = sp.Matrix(P).eigenvals()
        #log("eig %s" % eigvals)
        try:
            if all(sp.re(l) > 0 for (l, m) in eigvals.items()):  # eig P > 0
                return True
            else:
                return False
        except:
            pass
        x = sp.Matrix([sp.Symbol('x%s' % i) for i in range(P.shape[0])])
        x3 = [z3.Real('x%s' % i) for i in range(P.shape[0])]
        _, _, V = sympy_converter(f0(x, x.T, P))
        _, _, Vd = sympy_converter(f1(x, x.T, P, A))
        s = z3.Solver()
        s.add(z3.Not(z3.Implies(OrNotZero(x3), z3.And(V > 0, Vd < 0))))
        r = s.check()
        #if r == z3.sat:
            #log("_verify has model %s\n\t against V = %s ; Vd = %s" % (s.model(), V, Vd))
        return r == z3.unsat

    def _gen_P(self, t, prev_P=[]):
        P = gen_P('p%d' % t, self.n, symmetric=True)
        return prev_P + [P]
