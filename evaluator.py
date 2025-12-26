import numpy as np

class CEC_Evaluator:
    """
    Wrapper around a CEC objective function that tracks the number of
    evaluations and enforces a strict evaluation budget.
    """
    def __init__(self, func, max_fes, lb, ub):
        self.func = func
        self.max_fes = int(max_fes)
        self.calls = 0
        self.lb = lb
        self.ub = ub
        self.stop_flag = False

    def evaluate(self, x):
        if self.calls >= self.max_fes:
            self.stop_flag = True
            # Large penalty value once the evaluation budget is exhausted
            return 1e15
        
        # Enforce box constraints before calling the underlying function
        x_clipped = np.clip(x, self.lb, self.ub)
        
        val = self.func(x_clipped)
        self.calls += 1
        return val

    def __call__(self, x):
        return self.evaluate(x)