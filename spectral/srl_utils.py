# srl_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from tqdm import trange

# -------------------- helpers --------------------
def _ensure_csr(X) -> sp.csr_matrix:
    if sp.isspmatrix_csr(X): return X
    return X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)

def _astype(X, dtype):
    return X.astype(dtype, copy=False)

def normalized_adjacency_with_self_loops(A: sp.csr_matrix, dtype=np.float32) -> sp.csr_matrix:
    A = _astype(_ensure_csr(A), dtype)
    n = A.shape[0]
    I = sp.eye(n, format="csr", dtype=dtype)
    A_hat = A + I
    deg = np.asarray(A_hat.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(deg, dtype=dtype)
    nz = deg > 0
    inv_sqrt[nz] = deg[nz] ** (-0.5)
    D = sp.diags(inv_sqrt, format="csr", dtype=dtype)
    return D @ A_hat @ D

def build_role_basis(R: sp.csr_matrix, dtype=np.float32) -> sp.csr_matrix:
    R = _astype(_ensure_csr(R), dtype)
    diag = np.array(R.power(2).sum(axis=0)).ravel()
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(diag > 0, diag**-0.5, 0.0).astype(dtype, copy=False)
    return R @ sp.diags(inv_sqrt, format="csr", dtype=dtype)

def rayleighs_on_columns(S: sp.csr_matrix, X: sp.csr_matrix) -> np.ndarray:
    S = _ensure_csr(S); X = _ensure_csr(X)
    SX = S @ X
    num = np.array(X.multiply(SX).sum(axis=0)).ravel()
    den = np.array(X.power(2).sum(axis=0)).ravel()
    den = np.where(den > 0, den, 1.0)
    return (num / den).astype(S.dtype, copy=False)

# -------------------- augmented normalized blocks --------------------
def _block_degrees(A: sp.csr_matrix, R: sp.csr_matrix, Q: sp.csr_matrix, dtype) -> Tuple[np.ndarray, np.ndarray]:
    A = _astype(_ensure_csr(A), dtype); R = _astype(_ensure_csr(R), dtype); Q = _astype(_ensure_csr(Q), dtype)
    n, k = R.shape
    ones_o = np.ones((n,1), dtype=dtype)
    ones_v = np.ones((k,1), dtype=dtype)
    deg_o = np.asarray(((A @ ones_o) + ones_o + (R @ ones_v))).ravel()
    deg_v = np.asarray(((Q @ ones_v) + ones_v + (R.T @ ones_o))).ravel()
    return deg_o, deg_v

def s_blocks(A: sp.csr_matrix, R: sp.csr_matrix, Q: sp.csr_matrix, dtype=np.float32) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    A = _astype(_ensure_csr(A), dtype); R = _astype(_ensure_csr(R), dtype); Q = _astype(_ensure_csr(Q), dtype)
    n, k = R.shape
    deg_o, deg_v = _block_degrees(A, R, Q, dtype)
    inv_o = np.where(deg_o > 0, deg_o**-0.5, 0.0).astype(dtype, copy=False)
    inv_v = np.where(deg_v > 0, deg_v**-0.5, 0.0).astype(dtype, copy=False)
    Dlo = sp.diags(inv_o, format="csr", dtype=dtype)
    Dlv = sp.diags(inv_v, format="csr", dtype=dtype)
    I_o = sp.eye(n, format="csr", dtype=dtype)
    I_v = sp.eye(k, format="csr", dtype=dtype)
    S_oo = Dlo @ (A + I_o) @ Dlo
    S_ov = Dlo @ R @ Dlv
    S_vv = Dlv @ (Q + I_v) @ Dlv
    return S_oo, S_ov, S_vv

def build_S_aug_csr(A: sp.csr_matrix, R: sp.csr_matrix, Q: sp.csr_matrix, dtype=np.float32) -> sp.csr_matrix:
    S_oo, S_ov, S_vv = s_blocks(A, R, Q, dtype=dtype)
    top = sp.hstack([S_oo, S_ov], format="csr", dtype=dtype)
    bot = sp.hstack([S_ov.T, S_vv], format="csr", dtype=dtype)
    return sp.vstack([top, bot], format="csr", dtype=dtype)

# -------------------- LinearOperator for S and (S - sigma I) --------------------
def make_S_aug_linop(A: sp.csr_matrix, R: sp.csr_matrix, Q: sp.csr_matrix, dtype=np.float32):
    A = _astype(_ensure_csr(A), dtype); R = _astype(_ensure_csr(R), dtype); Q = _astype(_ensure_csr(Q), dtype)
    n, k = R.shape
    deg_o, deg_v = _block_degrees(A, R, Q, dtype)
    do = np.where(deg_o > 0, deg_o**-0.5, 0.0).astype(dtype, copy=False)
    dv = np.where(deg_v > 0, deg_v**-0.5, 0.0).astype(dtype, copy=False)

    def matvec(z):
        xo = z[:n]; xv = z[n:]
        y_o = do * ( (A @ (do*xo)) + (do*xo) ) + do * ( R @ (dv*xv) )
        y_v = dv * ( R.T @ (do*xo) ) + dv * ( (Q @ (dv*xv)) + (dv*xv) )
        return np.concatenate([y_o, y_v]).astype(dtype, copy=False)

    return sla.LinearOperator(shape=(n+k, n+k), matvec=matvec, dtype=dtype)

def make_shift_matrix_op(Sop: sla.LinearOperator, sigma: float):
    # (S - sigma I) * x
    n = Sop.shape[0]
    def matvec(z):
        return Sop @ z - sigma * z
    return sla.LinearOperator(shape=Sop.shape, matvec=matvec, dtype=Sop.dtype)

# -------------------- 2x2 exact EP --------------------
@dataclass
class SRLComponents:
    mu_obs: np.ndarray
    mu_aug: np.ndarray
    tau: np.ndarray
    nu: np.ndarray
    lam_plus: np.ndarray
    delta_plus: np.ndarray
    lam_neg: np.ndarray
    delta_neg: np.ndarray
    delta_total: np.ndarray
    S_RAwR: np.ndarray

def per_role_2x2(A: sp.csr_matrix, R: sp.csr_matrix, Q: sp.csr_matrix, dtype=np.float32) -> SRLComponents:
    S_obs = normalized_adjacency_with_self_loops(A, dtype=dtype)
    C = build_role_basis(R, dtype=dtype)
    mu_obs = rayleighs_on_columns(S_obs, C)
    S_oo, S_ov, S_vv = s_blocks(A, R, Q, dtype=dtype)
    mu_aug = rayleighs_on_columns(S_oo, C)
    # coupling ruolo↔virtuale
    # T has shape (k_virtual, k_roles)
    T = (S_ov.T @ C).toarray()                      # dense (k × k_roles)
    tau = np.linalg.norm(T, axis=0).astype(dtype)   # ||T[:,j]||_2

    # S_RAwR =np.block([[S_oo,S_ov],[S_ov.T, S_vv]])
    S_RAwR = sp.bmat([[S_oo, S_ov],
                 [S_ov.T, S_vv]], format="csr")[:S_obs.shape[0]:,:S_obs.shape[0]]

    nu = np.zeros_like(tau, dtype=dtype)
    for j in range(T.shape[1]):
        tj = T[:, j]
        if tau[j] > 0:
            vhat = (tj / tau[j]).astype(dtype, copy=False)   # 1D (k,)
            Sv = S_vv.dot(vhat)                              # 1D (k,)
            nu[j] = float(np.dot(vhat, Sv))                  # scalar
        else:
            nu[j] = 0.0

    lam_plus = 0.5 * ((mu_aug + nu) + np.sqrt((mu_aug - nu)**2 + 4.0*(tau**2)))
    lam_neg = 0.5 * ((mu_aug + nu) - np.sqrt((mu_aug - nu)**2 + 4.0*(tau**2)))
    delta_plus = (lam_plus - mu_obs).astype(dtype, copy=False)
    delta_neg = (lam_neg - mu_obs).astype(dtype, copy=False)
    s = (mu_aug-nu)/np.sqrt((mu_aug-nu)**2 + 4*tau**2)
    delta_total = 0.5*( (1+s)*delta_plus + (1-s)*delta_neg )
    return SRLComponents(mu_obs=mu_obs, mu_aug=mu_aug, tau=tau, nu=nu, lam_plus=lam_plus, lam_neg=lam_neg, delta_plus=delta_plus, delta_neg=delta_neg, delta_total=delta_total, S_RAwR=S_RAwR)

# -------------------- labels + SRL scalari --------------------
def class_one_vs_rest(y: np.ndarray) -> Tuple[sp.csr_matrix, int]:
    y = np.asarray(y)
    # Tratta -1 (o qualunque sentinel) come unlabeled
    # Se usi un altro sentinel, aggiungilo in questo mask.
    unlabeled_mask = np.isnan(y)
    if np.issubdtype(y.dtype, np.integer):
        unlabeled_mask = unlabeled_mask | (y < 0)
    labeled = y[~unlabeled_mask]
    classes = np.unique(labeled).astype(int)
    # Mappa solo le classi labeled
    mapping = {c: i for i, c in enumerate(sorted(classes.tolist()))}
    rows_all = np.arange(y.shape[0])
    # Colonne: None per unlabeled
    cols_all = np.array([mapping.get(int(val)) if (not np.isnan(val) and (not (isinstance(val, (np.floating, float)) and np.isnan(val)))) and (int(val) in mapping) else -1
                         for val in y], dtype=int)
    keep = cols_all >= 0  # solo labeled
    rows = rows_all[keep]
    cols = cols_all[keep]
    data = np.ones_like(rows, dtype=np.float32)
    Y = sp.csr_matrix((data, (rows, cols)), shape=(y.shape[0], len(mapping)), dtype=np.float32)
    return Y, Y.shape[1]


def centered_unitnorm_columns(Y: sp.csr_matrix) -> sp.csr_matrix:
    Y = _ensure_csr(Y).astype(np.float32, copy=False)
    n, C = Y.shape
    mu = np.array(Y.sum(axis=0)).ravel() / max(n, 1)
    ones = sp.csr_matrix(np.ones((n,1), dtype=np.float32))
    Yc = Y - ones @ sp.csr_matrix(mu.reshape(1,-1))
    nrm2 = np.sqrt(np.array(Yc.multiply(Yc).sum(axis=0)).ravel())
    inv = np.where(nrm2 > 0, 1.0 / nrm2, 0.0).astype(np.float32, copy=False)
    return Yc @ sp.diags(inv, format="csr", dtype=np.float32)

def rho_omega_Robs(S_obs: sp.csr_matrix,
                   C: sp.csr_matrix,
                   Yc: sp.csr_matrix,
                   class_weights: Optional[np.ndarray] = None
                  ) -> Tuple[float, float, np.ndarray, float]:
    n, Cc = Yc.shape
    k = C.shape[1]

    if class_weights is None:
        w = np.ones(Cc, dtype=np.float32) / max(Cc, 1)
    else:
        w = np.asarray(class_weights, dtype=np.float32)
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w)/len(w)

    # rho (as you compute it now)
    beta_obs = np.sum(C.T @ Yc)**2
    PUy = C @ (C.T @ Yc)
    rho_c = np.array(PUy.multiply(PUy).sum(axis=0)).ravel().astype(np.float32, copy=False)
    rho = float(w @ rho_c)

    # rho_corr: dimension-corrected
    kn = float(k) / float(n) if n > 0 else 0.0
    denom_corr = 1.0 - kn
    if denom_corr > 1e-12:
        rho_corr = (rho - kn) / denom_corr
        # optional: clip to [0,1] for interpretability
        rho_corr = float(np.clip(rho_corr, 0.0, 1.0))
    else:
        # k ~ n: correction ill-conditioned; by convention set to 0 (or rho)
        rho_corr = 0.0

    # omega
    CY = C.T @ Yc
    if sp.issparse(CY):
        omega_num = np.array((CY.multiply(CY)) @ w.reshape(-1, 1)).ravel()
    else:
        omega_num = (CY**2) @ w
    denom = omega_num.sum()
    omega = (omega_num / denom if denom > 0 else omega_num).astype(np.float32, copy=False)

    # R_obs
    SY = S_obs @ Yc
    R_c = np.array(Yc.multiply(SY).sum(axis=0)).ravel().astype(np.float32, copy=False)
    R_obs = float(w @ R_c)

    return rho, rho_corr, omega, R_obs, beta_obs



import numpy as np

# ----------------- util -----------------
def _gs_orthonormalize(B, v, reorth=True, eps=1e-12):
    """
    Ortonormalizza v rispetto alle colonne ortonormali di B.
    Ritorna (v_hat, added). Se added=False, v era nello span(B).
    """
    if B is None or B.size == 0:
        w = v.copy()
    else:
        w = v - B @ (B.T @ v)
        if reorth and B.size > 0:
            w = w - B @ (B.T @ w)  # re-orth
    nrm = np.linalg.norm(w)
    if nrm < eps:
        return None, False
    return (w / nrm), True

def _project_operator(S_mul, B):
    """
    T = B^T S B usando solo moltiplicazioni per vettore.
    Ritorna (T, SB) con SB = S B (serve per il residuo).
    """
    SB = np.column_stack([S_mul(B[:, i]) for i in range(B.shape[1])])
    T = B.T @ SB
    return T, SB

# ----------------- nuovo approx_eigs -----------------
def approx_eigs_role_with_leakage(
    S_aug,
    C, j,
    n_obs, n_virt,
    S_oo=None, C_full=None,   # opzionali: per costruire il leakage "mirato"
    tol=1e-6, max_dim=6, max_krylov=2, reorth=True
):
    """
    Rayleigh–Ritz per-ruolo con vettore di leakage.
    - Parte da base 2D: {(C_j,0), (0,e_j)}.
    - Se residuo > tol, aggiunge ℓ_j = ((I - C C^T) S_oo C_j, 0) se S_oo e C_full dati;
      altrimenti usa il residuo proiettato sugli osservati come surrogato del leakage.
    - Se ancora > tol, aggiunge fino a 'max_krylov' vettori Krylov (S_aug x).

    Parametri:
        S_aug: (n x n) ndarray o oggetto con @ / .dot per matvec
        C: (n_obs x k) base dei ruoli (colonne ortonormali)
        j: indice ruolo
        n_obs, n_virt: dimensioni blocchi
        S_oo: (n_obs x n_obs) blocco osservati-osservati dell'augmented (opzionale)
        C_full: (n_obs x k) (uguale a C; serve per CC^T) (opzionale)
        tol: soglia residuo
        max_dim: dimensione massima del sottospazio
        max_krylov: num. massimo di arricchimenti Krylov
        reorth: Gram–Schmidt con re-orth

    Ritorna: dict con chiavi
        - lambda_ritz: float
        - x: Ritz vector (n,)
        - residual: float
        - dim: dimensione del sottospazio usato
        - added_leakage: bool
        - iters: int
        - warn: (opzionale) string
    """
    n = n_obs + n_virt
    assert C.shape[0] == n_obs
    assert 0 <= j < C.shape[1]

    # funzione matvec
    if hasattr(S_aug, "__matmul__"):
        S_mul = lambda x: S_aug @ x
    elif hasattr(S_aug, "dot"):
        S_mul = lambda x: S_aug.dot(x)
    else:
        raise TypeError("S_aug must support '@' or .dot() for matvec.")

    # seed 2D: (C_j, 0) e (0, e_j)
    v_obs = np.zeros(n, dtype=float); v_obs[:n_obs] = C[:, j]
    e = np.zeros(n_virt, dtype=float); e[j] = 1.0
    v_virt = np.zeros(n, dtype=float); v_virt[n_obs:] = e

    # base iniziale
    B = None
    v1, ok = _gs_orthonormalize(B, v_obs, reorth=reorth); 
    if ok: B = v1[:, None]
    v2, ok = _gs_orthonormalize(B, v_virt, reorth=reorth)
    if ok: B = np.column_stack([B, v2]) if B is not None else v2[:, None]
    if B is None or B.shape[1] < 1:
        raise ValueError("Initial basis is empty; check C_j / e_j.")

    added_leakage = False
    iters = 0

    # ciclo: proietta, risolvi, residuo, arricchisci
    while True:
        iters += 1

        # proiezione
        T, SB = _project_operator(S_mul, B)
        evals, evecs = np.linalg.eigh(T)
        idx = np.argmax(evals)
        lam = float(evals[idx])
        u = evecs[:, idx]
        x = B @ u
        Sx = SB @ u
        r = Sx - lam * x
        r_norm = np.linalg.norm(r)

        if r_norm <= tol:
            return dict(lambda_ritz=lam, x=x, residual=r_norm, dim=B.shape[1],
                        added_leakage=added_leakage, iters=iters)

        if B.shape[1] >= max_dim:
            return dict(lambda_ritz=lam, x=x, residual=r_norm, dim=B.shape[1],
                        added_leakage=added_leakage, iters=iters,
                        warn="Reached max_dim before hitting tol.")

        # prova ad aggiungere il leakage (una sola volta, se disponibile)
        grew = False
        if (not added_leakage) and (S_oo is not None) and (C_full is not None):
            # ℓ_j = ((I - C C^T) S_oo C_j, 0)
            Cj = C[:, j]
            # Proiezione su U: C(C^T S_oo Cj), poi (I - CC^T)
            proj_U = C_full @ (C_full.T @ (S_oo @ Cj))
            leak_obs = (S_oo @ Cj) - proj_U
            ell = np.zeros(n, dtype=float)
            ell[:n_obs] = leak_obs
            ell_hat, added = _gs_orthonormalize(B, ell, reorth=reorth)
            if added:
                B = np.column_stack([B, ell_hat]); grew = True; added_leakage = True

        # se non è cresciuto con leakage (o leakage non disponibile), usa Krylov
        if not grew:
            # fino a max_krylov arricchimenti con S_aug x
            added_k = 0
            for _ in range(max_krylov):
                cand = S_mul(x)
                cand_hat, added = _gs_orthonormalize(B, cand, reorth=reorth)
                if added:
                    B = np.column_stack([B, cand_hat]); added_k += 1
                    if B.shape[1] >= max_dim:
                        break
            if added_k == 0:
                # prova almeno con i due seed separati
                for seed in (v_obs, v_virt):
                    cand = S_mul(seed)
                    cand_hat, added = _gs_orthonormalize(B, cand, reorth=reorth)
                    if added:
                        B = np.column_stack([B, cand_hat]); added_k += 1
                        break
                if added_k == 0:
                    return dict(lambda_ritz=lam, x=x, residual=r_norm, dim=B.shape[1],
                                added_leakage=added_leakage, iters=iters,
                                warn="Could not enrich subspace (direction collapsed).")


def ritz_role_pair_with_tracking(S_aug, C, j, n_obs, n_virt, S_oo=None, C_full=None,
                                 tol=1e-6, max_dim=6, max_krylov=2, reorth=True):
    #print(f"Role {j}")
    # Matvec
    if hasattr(S_aug, "__matmul__"):
        S_mul = lambda x,y: S_aug**y @ x
    elif hasattr(S_aug, "dot"):
        S_mul = lambda x,y: (S_aug**y).dot(x)
    else:
        raise TypeError("S_aug must support '@' or .dot() for matvec.")
    n = n_obs + n_virt

    # Seed 2D
    v_obs = np.zeros(n); v_obs[:n_obs] = C[:, j].toarray().ravel().astype(float)
    e = np.zeros(n_virt); e[j] = 1.0
    v_virt = np.zeros(n); v_virt[n_obs:] = e

    # Ortho base B2
    def _gs(B, v):
        w = v if B is None or B.size==0 else v - B @ (B.T @ v)
        nrm = np.linalg.norm(w); 
        return (w/nrm if nrm>1e-12 else None)
    b1 = _gs(None, v_obs); B = b1[:,None]
    b2 = _gs(B, v_virt);  B = np.column_stack([B,b2])

    # 2x2 init
    SB = np.column_stack([S_mul(B[:, i],1) for i in range(B.shape[1])])
    T2 = B.T @ SB
    evals2, evecs2 = np.linalg.eigh(T2)
    idxs = np.argsort(evals2)[::-1]  # desc
    lam2_plus, lam2_minus = float(evals2[idxs[0]]), float(evals2[idxs[1]])
    u2_plus, u2_minus = evecs2[:, idxs[0]], evecs2[:, idxs[1]]
    x2_plus  = B @ u2_plus
    x2_minus = B @ u2_minus

    r = (SB @ u2_plus) - lam2_plus * x2_plus
    r_norm = float(np.linalg.norm(r))
    #print(f"Eigenvalues residuals: {r_norm}")
    if r_norm <= tol:
        return dict(lambda_plus=lam2_plus, lambda_minus=lam2_minus,
                        residual=r_norm, dim=B.shape[1])
    
    # Enrichment loop (come nella funzione precedente, ma tracking)
    added_leakage = False
    while True:
        iteration=1
        # proietta su B
        SB = np.column_stack([S_mul(B[:, i],1) for i in range(B.shape[1])])
        T = B.T @ SB
        evals, evecs = np.linalg.eigh(T)
        idxs = np.argsort(evals)[::-1]
        lambdas = evals[idxs]; U = evecs[:, idxs]
        Xritz = B @ U  # n x m

        # overlap tracking
        overlaps_plus  = np.abs(Xritz.T @ x2_plus)   # m
        overlaps_minus = np.abs(Xritz.T @ x2_minus)  # m
        i_plus = int(np.argmax(overlaps_plus))
        # per la minus, scegli il migliore tra i rimanenti
        mask = np.ones(len(lambdas), dtype=bool); mask[i_plus] = False
        i_minus = int(np.argmax(overlaps_minus[mask])); 
        i_minus = np.arange(len(lambdas))[mask][i_minus]

        lam_plus  = float(lambdas[i_plus]);  x_plus  = Xritz[:, i_plus]
        lam_minus = float(lambdas[i_minus]); x_minus = Xritz[:, i_minus]

        # residuo del "plus"
        r = (SB @ U[:, i_plus]) - lam_plus * x_plus
        r_norm = float(np.linalg.norm(r))
        #print(f"Eigenvalues residuals: {r_norm}")
        if r_norm <= tol:
            return dict(lambda_plus=lam_plus, lambda_minus=lam_minus,
                        residual=r_norm, dim=B.shape[1])

        if B.shape[1] >= max_dim:
            return dict(lambda_plus=lam_plus, lambda_minus=lam_minus,
                        residual=r_norm, dim=B.shape[1],
                        warn="Reached max_dim before hitting tol.")

        # leakage (se disponibile)
        grew = False
        if (not added_leakage) and (S_oo is not None) and (C is not None):
            Cj = C[:, j]
            proj_U = C @ (C.T @ (S_oo @ Cj))
            leak_obs = (S_oo @ Cj) - proj_U
            ell = np.zeros(n); ell[:n_obs] = leak_obs.toarray().ravel().astype(float)
            # ortho
            w = ell - B @ (B.T @ ell); nrm = np.linalg.norm(w)
            if nrm > 1e-12:
                B = np.column_stack([B, w/nrm]); grew=True; added_leakage=True

        # Krylov se serve
        if not grew:
            new_S_mul = S_mul(x_plus, iteration)
            w = new_S_mul - B @ (B.T @ new_S_mul); nrm = np.linalg.norm(w)
            iteration+=1
            if nrm > 1e-12:
                B = np.column_stack([B, w/nrm])
            else:
                #print("STOP")
                return dict(lambda_plus=lam_plus, lambda_minus=lam_minus,
                            residual=r_norm, dim=B.shape[1],
                            warn="Could not enrich subspace.")




# -------------------- slicing (no materialization) --------------------
@dataclass
class SliceParams:
    P: int = 24
    overlap: float = 0.01
    tol_eig: float = 1e-6
    lin_solve_tol: float = 1e-4
    lin_solve_maxiter: int = 500
    dtype: type = np.float32
    materialize: bool = False  # True = usa CSR + LU; False = LinearOperator + MINRES

def _slice_shift_invert_operator(A, R, Q, a, b, params: SliceParams):
    # costruisci operatori
    Sop = make_S_aug_linop(A, R, Q, dtype=params.dtype)
    sigma = 0.5*(a+b)
    Mop = make_shift_matrix_op(Sop, sigma)

    # define OPinv via MINRES solve: (S - sigma I) z = v
    def solve(v):
        z, info = sla.minres(Mop, v, tol=params.lin_solve_tol, maxiter=params.lin_solve_maxiter)
        if info != 0:
            # fallback: rilassa la tolleranza una volta
            z, _ = sla.minres(Mop, v, tol=max(params.lin_solve_tol*10, 1e-3), maxiter=params.lin_solve_maxiter*2)
        return z.astype(params.dtype, copy=False)

    OPinv = sla.LinearOperator(shape=Sop.shape, matvec=solve, dtype=params.dtype)

    # stima k_hint grossolana
    n = Sop.shape[0]
    width = b - a
    frac = width / 1.999  # range ~(-0.999,1.0]
    k = int(max(8, min(n-1, np.ceil(frac * n) + 16)))

    # ARPACK con OPinv (no LU)
    vals, vecs = sla.eigsh(Sop, k=k, OPinv=OPinv, sigma=sigma, which='LM', tol=params.tol_eig)
    mask = (vals >= a - 1e-10) & (vals <= b + 1e-10)
    return vals[mask], vecs[:, mask]

def slice_all_eigs_adaptive(A, R, Q, params: SliceParams, save_vecs: bool = False):
    # finestratura su (-1,1]
    left, right = -0.999, 1.0
    edges = np.linspace(left, right, params.P + 1)
    base_w = (edges[1]-edges[0])

    all_vals = []
    all_vecs = [] if save_vecs else None

    for i in range(params.P):
        a, b = edges[i], edges[i+1]
        if i > 0:       a -= params.overlap * base_w
        if i < params.P - 1: b += params.overlap * base_w

        if params.materialize:
            # CSR + LU path (memoria alta): sconsigliato su grafi grandi
            S = build_S_aug_csr(A, R, Q, dtype=params.dtype)
            n = S.shape[0]
            sigma = 0.5*(a+b)
            M = S - sigma*sp.eye(n, format="csr", dtype=S.dtype)
            lu = sla.splu(M.tocsc())
            Minv = sla.LinearOperator(shape=S.shape, matvec=lambda v: lu.solve(v), dtype=S.dtype)
            width = b - a; frac = width / (right-left)
            k = int(max(8, min(n-1, np.ceil(frac * n) + 16)))
            vals, vecs = sla.eigsh(S, k=k, OPinv=Minv, sigma=sigma, which='LM', tol=params.tol_eig)
            mask = (vals >= a - 1e-10) & (vals <= b + 1e-10)
            v_in = vals[mask]; U_in = vecs[:, mask]
        else:
            v_in, U_in = _slice_shift_invert_operator(A, R, Q, a, b, params)

        if v_in.size:
            all_vals.append(v_in)
            if save_vecs:
                all_vecs.append(U_in)

    if not all_vals:
        n_tot = A.shape[0] + R.shape[1]
        return np.array([], dtype=params.dtype), (None if not save_vecs else np.zeros((n_tot,0), dtype=params.dtype))

    vals = np.concatenate(all_vals, axis=0)
    if save_vecs:
        V = np.concatenate(all_vecs, axis=1)
    else:
        V = None

    # deduplica
    idx = np.argsort(vals)
    vals = vals[idx]
    if save_vecs:
        V = V[:, idx]
    keep = [0]
    for j in range(1, vals.size):
        if abs(vals[j]-vals[keep[-1]]) > 1e-8:
            keep.append(j)
    vals = vals[keep]
    if save_vecs:
        V = V[:, keep]
    return vals, V

# -------------------- SRL wrapper --------------------
def compute_srl(A: sp.csr_matrix,
                y: np.ndarray,
                R: sp.csr_matrix,
                Q: sp.csr_matrix,
                class_weights: Optional[np.ndarray] = None,
                mode: str = "exact2x2",
                eig_windows: int = 24,
                eig_overlap: float = 0.01,
                eig_tol: float = 1e-6,
                dtype=np.float32) -> Dict[str, np.ndarray | float]:

    Y, _ = class_one_vs_rest(y)
    Yc = centered_unitnorm_columns(Y)

    # --- riallineamento osservati vs virtuali ---
    nA = A.shape[0]
    nR = R.shape[0]          # numero di nodi osservati (righe di R)
    nRT = R.shape[1]
    nY = Yc.shape[0]         # righe della matrice label

    # scegliamo n_obs dal dato più affidabile: R (se presente), altrimenti min(nA, nY)
    n_obs = nR if nR > 0 else min(nA, nY)

    # 1) A_obs = blocco "osservato" di A
    #    Assunzione: i nodi osservati sono in testa (ordine [osservati | virtuali]).
    #    Se non è vero, serve una permutazione esplicita (mapping) esterna.
    A_obs = A[:n_obs, :n_obs] if nA != n_obs else A

    # 2) Yc deve avere n_obs righe: taglia o pad a zeri
    if nY > n_obs:
        Yc = Yc[:n_obs, :]
    elif nY < n_obs:
        pad = sp.csr_matrix((n_obs - nY, Yc.shape[1]), dtype=Yc.dtype)
        Yc = sp.vstack([Yc, pad], format="csr")

    # 3) R deve avere n_obs righe (di norma già vero); se è più lungo, taglia
    if R.shape[0] > n_obs:
        R = R[:n_obs, :]

    # ora tutto è compatibile: S_obs è costruita solo sugli osservati
    S_obs = normalized_adjacency_with_self_loops(A_obs, dtype=dtype)
    C     = build_role_basis(R, dtype=dtype)

    # (debug opzionale)
    print(f"[dbg-aligned] A_obs:{A_obs.shape}, R:{R.shape}, Yc:{Yc.shape}", flush=True)

    print("Costruiti vettor/matrici utili. Si procede col calcolo di rho, omega e R_obs")
    rho, rho_corr, omega, R_obs, beta_obs = rho_omega_Robs(S_obs, C, Yc, class_weights=class_weights)
    print("Calcolo eseguito. Procediamo con la eigendecomposition")
    
    if mode == "exact2x2":
        comps = per_role_2x2(A_obs, R, Q, dtype=dtype)
        # Stimo b_c assumendo un GCN lineare con due layers
        # b_c = Yc[:nRT, :]/comps.lam_plus[:,None]
        # print(b_c.shape)
        # print(C.shape)
        # alphas = C @ b_c
        # print(alphas.shape)
        # print(S_obs.shape)
        # print(comps.S_RAwR.shape)
        # print(C.shape)
        role_base_S_obs = C.T @ S_obs @ C
        role_base_S_RAwR = C.T @ comps.S_RAwR @ C
        # print(f"C_S_obs shape: {role_base_S_obs.shape}")
        # print(f"C_S_RAwR shape: {role_base_S_RAwR.shape}")
        commutator = role_base_S_obs @ role_base_S_RAwR - role_base_S_RAwR @ role_base_S_obs
        comm_norm = sp.linalg.norm(commutator)/(sp.linalg.norm(role_base_S_obs)*sp.linalg.norm(role_base_S_RAwR))
        srl_plus = float(rho * (omega @ comps.delta_plus[comps.lam_plus!=0]**2))
        srl_plus_corr = float(rho_corr * (omega @ comps.delta_plus[comps.lam_plus!=0]**2))
        srl_neg = float(rho * (omega @ comps.delta_neg**2))
        srl_neg_corr = float(rho_corr * (omega @ comps.delta_neg**2))
        srl_total = float(rho * (omega @ comps.delta_total**2))
        srl_total_corr = float(rho_corr * (omega @ comps.delta_total**2))
        return {
            "rho": float(rho), "R_obs": float(R_obs), "rho_corr": float(rho_corr),
            "srl_plus": srl_plus, "srl_plus_corr": srl_plus_corr,
            "srl_neg": srl_neg, "srl_neg_corr": srl_neg_corr,
            "srl_total": srl_total, "srl_total_corr": srl_total_corr,
            "mu_obs": comps.mu_obs, "mu_aug": comps.mu_aug,
            "tau": comps.tau, "nu": comps.nu,
            "lam_plus": comps.lam_plus, "delta_plus": np.mean(comps.delta_plus),
            "lam_neg": comps.lam_neg, "delta_neg": np.mean(comps.delta_neg),
            "omega": omega,
            "comm_norm": comm_norm
        }

    elif mode == "approx_eigs":
        # ---- per-ruolo: mu_obs e lambda_plus (Ritz con leakage) ----
        S_oo, S_ov, S_vv = s_blocks(A_obs, R, Q, dtype=dtype)
        S_aug = sp.bmat([[S_oo, S_ov],[S_ov.T, S_vv]])
        k = C.shape[1]
        mus_obs = np.empty(k); lambdas_plus = np.empty(k); lambdas_neg = np.empty(k); deltas_plus = np.empty(k); deltas_neg = np.empty(k)
        mus_obs = rayleighs_on_columns(S_obs, C)
        for j in trange(k):
            # Cj = C[:, j]
            # mus_obs[j] = float(Cj @ (S_obs @ Cj))  # Rayleigh su S_obs
            

            out = ritz_role_pair_with_tracking(
                S_aug=S_aug, C=C, j=j, n_obs=n_obs, n_virt=k,
                S_oo=S_oo, C_full=C, tol=5*1e-2, max_dim=20, max_krylov=1
            )
            lambdas_plus[j] = out["lambda_plus"]
            lambdas_neg[j] = out["lambda_minus"]

        deltas_plus[j] = lambdas_plus[j] - mus_obs[j]
        deltas_neg[j] = lambdas_neg[j] - mus_obs[j]

        srl_plus = float(rho * np.dot(omega, deltas_plus**2))
        srl_plus_norm = srl_plus / (1.0 - R_obs) if (1.0 - R_obs) > 1e-12 else 0.0

        srl_neg = float(rho * np.dot(omega, deltas_plus**2))
        srl_neg_norm = srl_neg / (1.0 - R_obs) if (1.0 - R_obs) > 1e-12 else 0.0

        return {
            "rho": float(rho), "R_obs": float(R_obs),
            "srl_plus": srl_plus, "srl_plus_norm": srl_plus_norm,
            "srl_neg": srl_neg, "srl_neg_norm": srl_neg_norm,
            "omega": omega,
        }

    else:
        raise ValueError(f"Unknown mode={mode!r}")
