import math
import random
import numpy as np
from typing import Tuple
from scipy.optimize import minimize
from scipy.stats import skewnorm
from src.evaluator import CEC_Evaluator

class IWOA_Strict:
    def __init__(
        self,
        evaluator: CEC_Evaluator,
        dim: int,
        bounds: np.ndarray,
        *,
        pop_size: int = 100,       
        min_pop_size: int = 20,    
        spiral_shape_const: float = 1.0,
        archive_size: int = 10,
    ) -> None:
        self.eval = evaluator
        self.dim = dim
        self.bounds = bounds
        self.lb, self.ub = bounds[0], bounds[1]
        self.pop_size = pop_size
        self.initial_pop_size = pop_size
        self.min_pop_size = min_pop_size
        self.spiral_shape_const = spiral_shape_const
        self.archive_size = archive_size

    # --- Initialization ---
    def gauss_map(self, x):
        x = np.where(x == 0, 1e-9, x)
        return (1.0 / x) % 1.0

    def logistic_map(self, x):
        return 4 * x * (1 - x)

    def _initial_candidates(self, size=None) -> Tuple[np.ndarray, np.ndarray]:
        n = size if size is not None else self.pop_size
        half = n // 2
        if n < 2: half = 0
        
        X_gauss = np.random.rand(half, self.dim)
        X_logis = np.random.rand(n - half, self.dim)
        
        for _ in range(10):
            X_gauss = self.gauss_map(X_gauss)
            X_logis = self.logistic_map(X_logis)
        
        if half > 0:
            X = np.vstack((X_gauss, X_logis)) 
        else:
            X = X_logis
            
        X = X * (self.ub - self.lb) + self.lb
        return X, self.lb + self.ub - X

    def initialize_population(self):
        X, X_opp = self._initial_candidates()
        f_X = np.array([self.eval(x) for x in X])
        
        f_Xopp = []
        for x in X_opp:
            if self.eval.calls < self.eval.max_fes:
                f_Xopp.append(self.eval(x))
            else:
                f_Xopp.append(1e15)
        f_Xopp = np.array(f_Xopp)
        
        take_opp = f_Xopp < f_X
        pop = np.where(take_opp[:, None], X_opp, X)
        fits = np.where(take_opp, f_Xopp, f_X)
        return pop, fits

    # --- Components ---
    def levy_flight(self, beta: float = 1.5) -> np.ndarray:
        sigma_u = (math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        u = np.random.randn(self.dim) * sigma_u
        v = np.random.randn(self.dim)
        return u / (np.abs(v)**(1/beta) + 1e-9)

    def get_quasi_reflection(self, x, curr_min, curr_max):
        center = (curr_min + curr_max) / 2.0
        r = np.random.rand(*x.shape)
        return center + (x - center) * r

    def apply_crossover(self, x, pop, idx):
        if self.pop_size < 4: return x 
        r_best = np.random.randint(0, max(1, int(self.pop_size * 0.2)))
        candidates = [i for i in range(self.pop_size) if i != idx and i != r_best]
        if len(candidates) < 2: return x
        
        r1, r2 = np.random.choice(candidates, 2, replace=False)
        F = 0.5 + 0.4 * random.random()
        mutant = x + F * (pop[r_best] - x) + F * (pop[r1] - pop[r2])
        mask = np.random.rand(self.dim) < 0.9
        return np.where(mask, mutant, x)

    # --- Diversity-based restart ---
    def check_and_restart(self, pop, fits, progress):
        if progress > 0.90: return pop, fits, False 
        if self.eval.stop_flag: return pop, fits, False

        std_pos = np.std(pop, axis=0)
        avg_diversity = np.mean(std_pos)
        
        domain_range = np.mean(self.ub - self.lb)
        threshold = 1e-4 * domain_range

        if avg_diversity < threshold:
            num_keep = max(1, int(self.pop_size * 0.1))
            sorted_idx = np.argsort(fits)
            
            num_replace = self.pop_size - num_keep
            if num_replace < 1: return pop, fits, False
            
            X_new, _ = self._initial_candidates(size=num_replace)
            replace_indices = sorted_idx[num_keep:]
            
            for i, idx in enumerate(replace_indices):
                if self.eval.stop_flag: break
                pop[idx] = X_new[i]
                fits[idx] = self.eval(pop[idx])
            
            return pop, fits, True 
        
        return pop, fits, False

    # --- Main Run ---
    def run(self) -> Tuple[np.ndarray, float]:
        pop, fits = self.initialize_population()
        
        idx_best = int(np.argmin(fits))
        global_best, global_best_fit = pop[idx_best].copy(), float(fits[idx_best])
        archive = [(global_best.copy(), global_best_fit)]
        
        stagnation_counter = 0
        dist_neg = skewnorm(a=-8)
        dist_pos = skewnorm(a=2)
        gen = 0

        while self.eval.calls < self.eval.max_fes:
            gen += 1
            progress = self.eval.calls / self.eval.max_fes
            
            # 1. LPSR
            plan_pop = int(np.round((self.min_pop_size - self.initial_pop_size) * progress + self.initial_pop_size))
            if self.pop_size > plan_pop:
                sorted_idx = np.argsort(fits)
                pop = pop[sorted_idx[:plan_pop]]
                fits = fits[sorted_idx[:plan_pop]]
                self.pop_size = plan_pop

            # 2. Check Restart
            pop, fits, restarted = self.check_and_restart(pop, fits, progress)
            if restarted:
                idx_best = int(np.argmin(fits))
                if fits[idx_best] < global_best_fit:
                    global_best, global_best_fit = pop[idx_best].copy(), float(fits[idx_best])
                stagnation_counter = 0
                continue

            # Parameters
            a = 2.0 * (1.0 - progress**1.5) 
            spiral_c = self.spiral_shape_const * (1 - progress)
            curr_min, curr_max = np.min(pop, axis=0), np.max(pop, axis=0)
            xs, x_opps = [], []
            crossover_prob = 0.8 if 0.2 < progress < 0.8 else 0.1

            # --- Whale Optimization Cycle ---
            for i in range(self.pop_size):
                x = pop[i].copy()
                r1, r2 = random.random(), random.random()
                A, C = 2 * a * r1 - a, 2 * r2
                
                if progress <= 0.25: p = dist_neg.rvs() 
                elif progress <= 0.5: p = random.random()
                else: p = dist_pos.rvs()
                p = float(np.clip(p, 0.0, 1.0))

                if p < 0.5:
                    if abs(A) < 1: # Encircling
                        D = np.abs(C * global_best - x)
                        omega = 0.8 * math.cos(math.pi * progress) + 0.2
                        x_new = global_best - A * D * omega
                    else: # Search
                        rand_idx = random.randint(0, self.pop_size - 1)
                        D = np.abs(C * pop[rand_idx] - x)
                        x_new = pop[rand_idx] - A * D
                else: # Spiral
                    D = np.abs(global_best - x)
                    l = random.uniform(-1, 1)
                    x_new = D * math.exp(spiral_c * l) * math.cos(2*math.pi*l) + global_best

                if stagnation_counter > 10:
                    if random.random() < 0.5:
                        x_new += self.levy_flight() * (x_new - global_best) * 0.5
                    else:
                        x_new += np.random.normal(0, 1.0, self.dim) * (self.ub-self.lb) * 0.01

                if random.random() < crossover_prob:
                    x_new = self.apply_crossover(x, pop, i)

                x_new = np.clip(x_new, self.lb, self.ub)
                xs.append(x_new)
                x_qr = self.get_quasi_reflection(x_new, curr_min, curr_max)
                x_opps.append(np.clip(x_qr, self.lb, self.ub))

            # --- Batch Evaluation ---
            fits_x, fits_opp = [], []
            for cand in xs:
                if self.eval.stop_flag: break
                fits_x.append(self.eval(cand))
            for cand in x_opps:
                if self.eval.stop_flag: break
                fits_opp.append(self.eval(cand))
            if self.eval.stop_flag: break

            # Selection
            updated_any = False
            for i in range(len(fits_x)):
                if fits_opp[i] < fits_x[i]:
                    cand_x, cand_fit = x_opps[i], fits_opp[i]
                else:
                    cand_x, cand_fit = xs[i], fits_x[i]
                
                if cand_fit < fits[i]:
                    pop[i], fits[i] = cand_x, cand_fit
                
                if fits[i] < global_best_fit:
                    global_best, global_best_fit = pop[i].copy(), float(fits[i])
                    updated_any = True
                    stagnation_counter = 0
                    archive.append((global_best.copy(), global_best_fit))
                    archive = sorted(archive, key=lambda z: z[1])[:self.archive_size]

            if not updated_any: stagnation_counter += 1

            # --- Chaotic Local Search ---
            if progress > 0.9 and not updated_any and not self.eval.stop_flag:
                chaos_val = np.random.rand(self.dim)
                for _ in range(5): chaos_val = 4.0 * chaos_val * (1 - chaos_val)
                epsilon = 1e-3 * (1 - progress)
                x_chaos = global_best + (chaos_val - 0.5) * epsilon * (self.ub - self.lb)
                fit_chaos = self.eval(x_chaos)
                
                if fit_chaos < global_best_fit:
                    global_best, global_best_fit = x_chaos, fit_chaos
                    pop[np.argmin(fits)] = x_chaos
                    fits[np.argmin(fits)] = fit_chaos
                    stagnation_counter = 0

            # --- Reactive Nelder-Mead (Boosted) ---
            is_stuck = (stagnation_counter > 15 and stagnation_counter % 5 == 0)
            is_final = (progress > 0.95 and gen % 10 == 0)
            
            if (is_stuck or is_final) and len(archive) > 0 and not self.eval.stop_flag:
                rem_budget = self.eval.max_fes - self.eval.calls
                nm_budget = min(500, rem_budget) 
                
                if nm_budget > self.dim * 2:
                    res = minimize(
                        self.eval, 
                        global_best, 
                        method="Nelder-Mead", 
                        bounds=list(zip(self.lb, self.ub)), 
                        options={"maxfev": nm_budget, "xatol": 1e-8}
                    )
                    
                    if res.fun < global_best_fit:
                        global_best, global_best_fit = res.x, res.fun
                        pop[np.argmin(fits)] = global_best
                        fits[np.argmin(fits)] = global_best_fit
                        stagnation_counter = 0

        return global_best, global_best_fit