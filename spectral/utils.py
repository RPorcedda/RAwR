import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# ---------- partition caching ----------
def _load_cached_partition(filename: str, num_nodes: int, eps: int):
    """Load cached partition if available: returns (num_blocks, block_sizes, part, partition) or None."""
    pfile = Path(f"partitions/{filename}P{eps}")
    if not pfile.exists():
        return None
    part = np.loadtxt(pfile, dtype=int).reshape(-1)
    # if part.size != num_nodes:
    #     # Defensive: if file exists but size mismatches, ignore
    #     return None
    unique_blocks = np.unique(part)
    partition = [np.where(part == b)[0].tolist() for b in unique_blocks]
    block_sizes = np.array([len(bl) for bl in partition], dtype=int)
    return unique_blocks.size, block_sizes, part, partition

def _get_or_compute_partition(filename: str, num_nodes: int, eps: int):
    """Return BE partition, creating it with epsBE.jar if missing."""
    cached = _load_cached_partition(filename, num_nodes, eps)
    if cached is not None:
        return cached
    # Fall back to jar call; computeBEpartition already writes partitions/{filename}P{eps}
    return computeBEpartition(filename, num_nodes, eps)

# ---------- reduced network caching ----------
def _get_or_compute_R(filename: str, G: nx.Graph, partition, eps: int):
    """Return reduced network R, cached on disk."""
    out = Path(f"reduced/{filename}_R_eps{eps}.npy")
    if out.exists():
        return np.load(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    R = get_R(G, partition)
    np.save(out, R)
    return R


def read_edge_file(edge_path: str | Path) -> pd.DataFrame:
    edge_path = Path(edge_path).expanduser()
    edges = []
    with edge_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, v, *_ = line.split()
            edges.append((u, v))
    df = pd.DataFrame(edges, columns=["source", "target"])
    return df


def read_labels(label_path):
    labels = pd.read_csv(label_path, header=None)
    return labels.to_numpy()


def get_one_hot(y: np.ndarray):
    """
    Always returns shape (n, C).
    Works even if labels are not contigous by remapping to {0,1,2}.
    """
    y = np.asarray(y).reshape(-1)                 # flatten to (n,)
    classes, y_mapped = np.unique(y, return_inverse=True)  # map to 0..C-1
    Y = np.eye(classes.size, dtype=float)[y_mapped]        # (n, C)
    return Y


def get_B(part: np.ndarray) -> np.ndarray:
    """
    Build the node-to-block incidence matrix.
    part: array of length n where part[i] is the block id of node i (0..k-1)
    Returns: B of shape (n, k) with B[i, part[i]] = 1
    """
    part = np.asarray(part, dtype=int)
    n = part.shape[0]
    unique_blocks = np.unique(part)
    k = unique_blocks.size
    B = np.zeros((n, k), dtype=float)
    B[np.arange(n), part] = 1.0
    return B


def computeBEpartition(filename, num_nodes, eps):
    """
    PARAM
    filename: network dataset name (it is expected a directory format like dataset/dataset.edgelist)
    num_nodes: number of nodes in network
    eps : tolerance from 0 (exact EP) to max degree (only one block. The same partition could be obtained also for a value less than max degree)

    OUTPUT
    num_blocks: number of blocks in partition
    block_sizes: number of nodes in each block
    part: array of length num_nodes which specifies to what block in the partition each node belongs to
    partition: list with num_blocks rows, each row contains nodes indices (starting from 0) of nodes in the block
    """
    
    print(f"eps-BE started on {filename} with epsilon={eps}")
    # set the parameters for one-shot eps-BE (not the iterative version)
    eps_0=eps
    delta=eps
    delta_max=eps
    if eps==0:
        delta=1

    result = subprocess.run(
        ["java", "-jar", "epsBE.jar",
            filename+"/"+filename+".edgelist", 
            str(num_nodes), 
            str(eps_0),
            str(delta_max),
            str(delta),
            "partitions/"+filename+"BE"+str(eps),
            "false",
            "false"],
        capture_output=True,
        text=True,
        check=True
    )

    f = open("partitions/"+filename+"BE"+str(eps),"r")

    line=f.readline()
    line=f.readline()
    f.close()
    line = line.split(",")
    part = np.zeros(num_nodes, dtype=int)
    block_sizes = []

    blockIndex = 0
    partition = []
    for elem in line:
        block = []
        elem = elem.split(" ")
        elem = elem[1:len(elem)-1]
        tot = 0
        inn = False
        for el in elem:
            inde = int(el.replace("x",""))
            part[inde-1] = blockIndex
            block.append(inde-1)
            tot = tot+1
            inn = True
        if(inn):
            partition.append(block)
            block_sizes.append(tot)
        blockIndex = blockIndex+1
    num_blocks = len(block_sizes)
    block_sizes = np.array(block_sizes)

    print("eps-BE concluded")
    print(f"Number of blocks in partition: {num_blocks}")
    #print(f"Block sizes:\n{block_sizes}")
    with open(f"partitions/{filename}P{eps}", "w") as f:
        for line in part:
            f.write("%s\n" % line)
        f.close()

    return num_blocks, block_sizes, part, partition


def reduceModelColumn(A,partition,N):
    dim = len(partition)
    res = np.zeros((N,dim))
    index = 0
    for elem in partition:
        for el in elem:
            # partition indices are already 0-based
            res[:,index] = res[:,index] +  A[:,el]
        index=index+1
    return res


def get_R(G, partition):
    ordered_nodes = sorted(G.nodes())
    #Extract adjacency matrix with nodelist=ordered_nodes to ensure that first row corresponds to node 0 and so on 
    A = nx.to_numpy_array(G, nodelist=ordered_nodes, dtype=int)

    redModelA= reduceModelColumn(A,partition,G.number_of_nodes())

    # partition is 0-based
    Peta = partition
    k = len(Peta)
    # Reduced model has dimension equals to the number of blocks in the partition
    # The row and column correspond to the block of the partition in the order they are stored.
    R = np.zeros((k, k))
    # Sum of all the embedding for the rows corresponding to nodes in the same block
    for i, block in enumerate(Peta):
        R[i, :] = redModelA[block, :].sum(axis=0)
    # Make the reduced network unweighted
    R[R > 0] = 1
    return R


def get_adjacency(filename, augmentation=None, eps=None, return_blocks=False):
    original_edgelist = read_edge_file(f"{filename}/{filename}.edgelist")
    original_edgelist["source"] = original_edgelist["source"].astype(int)
    original_edgelist["target"] = original_edgelist["target"].astype(int)
    G = nx.from_pandas_edgelist(original_edgelist, create_using=nx.Graph)
    #A = nx.to_numpy_array(G)

    # --- leggi labels (tutti i nodi, anche isolati) ---
    labels = read_labels(f"{filename}/{filename}.y")

    # Aggiungi nodi isolati mancanti
    all_nodes = [int(i) for i in range(len(labels))]
    
    G.add_nodes_from(all_nodes)

    # --- crea matrice di adiacenza completa ---
    # ordina per ID numerico per coerenza con labels
    #A = nx.to_numpy_array(G)
    A = nx.to_numpy_array(G, nodelist=all_nodes, dtype=int)

    if augmentation is not None:
        assert augmentation in ["RepNodes","RepEdges"], "augmentation must be either RepNodes or RepEdges"
        assert eps is not None, "eps must be specified for the augmentation"

        nodes = list(map(int, G.nodes))
        num_nodes = max(nodes) + 1

        num_blocks, _, part, partition = _get_or_compute_partition(filename, num_nodes, eps)
        B = get_B(part)

        if augmentation == "RepNodes":
            R = np.zeros((num_blocks, num_blocks), dtype=float)
        else:  # RepEdges
            R = _get_or_compute_R(filename, G, partition, eps)

        A = np.block([
            [A, B],
            [B.T, R]
        ])

        if return_blocks:
            return A, B, R
    return A



def get_shift_operator(filename, augmentation=None, eps=None):
    """
    Normalized adjacency (a.k.a. GCN 'shift' operator):
        S = D_hat^{-1/2} A_hat D_hat^{-1/2}
    with A_hat = A + I to avoid zero degrees.
    """
    A = get_adjacency(filename, augmentation, eps)
    n = A.shape[0]
    # ensure float and add self-loops
    A_hat = A.astype(float) + np.eye(n, dtype=float)
    # degree vector
    deg = A_hat.sum(axis=1)
    # make sure it's a 1D ndarray
    deg = np.asarray(deg).reshape(-1)
    # safe inverse square root: zero where degree == 0 (shouldn’t happen after adding I)
    inv_sqrt = np.where(deg > 0, deg**-0.5, 0.0)
    # shift operator
    S = (inv_sqrt[:, None]) * A_hat * (inv_sqrt[None, :])
    return S


def compute_etas(S: np.ndarray, y: np.ndarray, use_density=True):
    # Ensure y is one-hot encoded
    Y = get_one_hot(y)
    # eigendecomposition
    lambdas, V = np.linalg.eigh(S)  # eigh for symmetric matrices
    # projections (overlaps with Y)
    etas = np.mean(V[:y.shape[0],:].T @ Y,axis=1)
    if use_density:
        etas = etas**2
        etas = etas/np.sum(etas)
    return etas, lambdas


def plot_etas(etas, lambdas):
    plt.figure(figsize=(8,4))
    plt.stem(lambdas, etas)
    plt.xlabel("Eigenvalue λ")
    plt.ylabel("Weighted coefficient η")
    plt.title("Weighted spectrum p(λ)")
    plt.tight_layout()
    plt.show()


# ====== NEW: role basis, partition wrapper, and alignment measures ======
def get_partition(filename: str, n_nodes: int, eps: int):
    """
    Public wrapper for cached partition (EP/eps-BE).
    Returns (num_blocks, block_sizes, part, partition)
    """
    return _get_or_compute_partition(filename, n_nodes, eps)  # uses existing cache

def get_role_basis(part: np.ndarray) -> np.ndarray:
    """
    Orthonormal block-constant basis Q in R^{n x k} for U = span{block indicators}.
    Column j is 1/sqrt(|B_j|) on block j and 0 elsewhere.
    """
    part = np.asarray(part, dtype=int).reshape(-1)
    n = part.size
    blocks = np.unique(part)
    k = blocks.size
    Q = np.zeros((n, k), dtype=float)
    for j, b in enumerate(blocks):
        idx = (part == b)
        sz = idx.sum()
        if sz > 0:
            Q[idx, j] = 1.0 / np.sqrt(float(sz))
    return Q

def compute_role_alignment_measures(
    S_obs: np.ndarray,
    S_aug: np.ndarray,
    part: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    EP esatta o approssimata (k_v può essere diverso da k).
    Calcola:
      - rho: role-SNR (media pesata per classe, one-vs-rest centrato/norm.)
      - overlineDelta = E_ω[ λ_+ - μ ] con ω_j indotti dalle label sui ruoli
      - lb_rayleigh_gain = rho * overlineDelta
      - R_obs: Rayleigh osservato (media pesata per classe)
      - diagnostiche per-ruolo
    """
    n = S_obs.shape[0]

    # --- LABEL MATRIX: one-vs-rest centrata/norm. e pesi classe ---
    if y.ndim == 1:
        Y, classes = one_vs_rest_matrix(y, n=n)   # (n x C), mean-zero per colonna
        cw = class_weights(y, mode="micro")       # (C,)
    else:
        Y = y
        # fallback: pesi uniformi
        cw = np.ones(Y.shape[1], dtype=float) / Y.shape[1]

    # --- Base dei ruoli su nodi osservati ---
    Q = get_role_basis(part)          # (n x k)
    k = Q.shape[1]

    # --- Blocchi dell'augmented (nessuna assunzione su k_v) ---
    if S_aug.shape[0] < n:
        raise ValueError("S_aug ha dimensione < n: controlla l'augmented.")
    S_oo = S_aug[:n, :n]
    k_v = S_aug.shape[0] - n
    S_ov = S_aug[:n, n:n+k_v] if k_v > 0 else np.zeros((n, 0))
    S_vv = S_aug[n:n+k_v, n:n+k_v]    if k_v > 0 else np.zeros((0, 0))

    # --- μ_j = diag(Q^T S_oo Q) ---
    B = S_oo @ Q                      # (n x k)
    M = Q.T @ B                       # (k x k)
    mu = np.diag(M).astype(float)     # (k,)

    # --- τ_j e ν_j con best-direction virtuale accoppiata ---
    # tmat = S_ov^T Q  -> (k_v x k); col j = t_j
    if k_v > 0:
        tmat = S_ov.T @ Q             # (k_v x k)
        tau = np.linalg.norm(tmat, axis=0)  # (k,)
        nu = np.zeros(k, dtype=float)
        # ν_j = (t_j^T S_vv t_j) / ||t_j||^2 se τ_j>0, altrimenti = μ_j (nessun guadagno)
        if S_vv.size > 0:
            Sv_t = S_vv @ tmat        # (k_v x k)
            num = (tmat * Sv_t).sum(axis=0)   # k_v dot per colonna
            # Evita divisioni per zero
            nz = tau > 1e-15
            nu[~nz] = mu[~nz]
            nu[nz] = (num[nz] / (tau[nz]**2)).astype(float)
        else:
            # Non ci sono virtuali (k_v=0) -> τ=0 e ν irrilevante
            tau[:] = 0.0
            nu = mu.copy()
    else:
        tau = np.zeros(k, dtype=float)
        nu = mu.copy()

    # --- λ_+ per-ruolo ---
    rad = np.sqrt((mu - nu)**2 + 4.0 * (tau**2))
    lam_plus = 0.5 * (mu + nu + rad)

    # --- Proiezione label su U e pesi ω_j indotti dalle label ---
    Uy = Q @ (Q.T @ Y)                      # (n x C)
    # ρ: media pesata per classe (||P_U y_c||^2 / ||y_c||^2), con Y già normalizzato per colonna
    num = (Uy**2).sum(axis=0)
    den = (Y**2).sum(axis=0) + 1e-15
    rho_per_class = num / den
    rho = float((cw * rho_per_class).sum())

    # ω_j: energia di y sulle colonne di Q, media pesata sulle classi
    coeff = (Q.T @ Y)**2                    # (k x C)
    w = coeff @ cw                          # (k,)
    w_sum = float(w.sum()) + 1e-15
    omega = (w / w_sum).astype(float)

    # --- overlineDelta e lower bound ---
    delta_j = lam_plus - mu
    overlineDelta = float((omega * delta_j).sum())
    lb_rayleigh_gain = float(rho * overlineDelta)

    # --- Rayleigh osservato medio pesato ---
    SY = S_obs @ Y
    num_r = (Y * SY).sum(axis=0)
    den_r = (Y * Y).sum(axis=0) + 1e-15
    Ry = (num_r / den_r).astype(float)
    R_obs = float((cw * Ry).sum())

        # ---------- NEW: gain in W (intra-blocco) ----------
    # Proiezione in W: Yw = (I - Q Q^T) Y = Y - UY (abbiamo già UY)
    Yw = Y - Uy                             # (n x C)

    # Rayleigh in W sul grafo osservato (media pesata per classe)
    SYw_obs = S_obs @ Yw
    num_w_obs = (Yw * SYw_obs).sum(axis=0)              # y_w^T S_obs y_w
    den_w = (Yw * Yw).sum(axis=0) + 1e-15               # ||y_w||^2
    R_obs_Wc = (num_w_obs / den_w).astype(float)        # per classe
    R_obs_W  = float((cw * R_obs_Wc).sum())             # media pesata

    # Rayleigh in W "visto dall'augmented": usa il blocco top-left S_oo
    SYw_oo = S_oo @ Yw
    num_w_oo = (Yw * SYw_oo).sum(axis=0)                # y_w^T S_oo y_w
    R_oo_Wc  = (num_w_oo / den_w).astype(float)
    R_oo_W   = float((cw * R_oo_Wc).sum())

    # Gain in W (può essere negativo -> teniamo anche la versione truncated a 0)
    lb_gain_W = float(R_oo_W - R_obs_W)
    lb_gain_W_pos = max(0.0, lb_gain_W)

    # Normalizzazione sul margine residuo in W
    margin_W = max(1.0 - R_obs_W, 1e-6)
    lb_gain_W_norm = lb_gain_W / margin_W
    lb_gain_W_pos_norm = lb_gain_W_pos / margin_W

    # (facoltativo) quanta "fuga" da W verso i virtuali nel normalizzato:
    # coupling epsilon: norma dell'accoppiamento S_ov^T Yw rispetto a ||Yw||
    if k_v > 0:
        coupW = S_ov.T @ Yw                                 # (k_v x C)
        coupW_norm = (coupW * coupW).sum(axis=0)**0.5 / (den_w**0.5 + 1e-15)
        coupW_strength = float((cw * coupW_norm).sum())     # media pesata
    else:
        coupW_strength = 0.0

    # ---------- NEW: gain in W con completamento virtuale ottimo (stile 2x2) ----------
    # Per ciascuna classe c, calcoliamo mu, tau, nu e lambda_plus_w, poi aggreghiamo con cw.

    # Precompute utili
    Yw = Y - Uy                               # (n x C), già calcolato sopra
    x_norm2 = (Yw * Yw).sum(axis=0) + 1e-15   # ||x||^2 per colonna
    # mu_c = (x^T S_oo x) / ||x||^2
    mu_c = ((Yw * (S_oo @ Yw)).sum(axis=0) / x_norm2).astype(float)   # (C,)

    if k_v > 0:
        # t = S_ov^T x
        t = S_ov.T @ Yw                        # (k_v x C)
        t_norm = np.sqrt((t * t).sum(axis=0))  # ||t|| per classe (C,)
        tau_c = (t_norm / np.sqrt(x_norm2)).astype(float)

        # nu_c = (t^T S_vv t)/||t||^2 se ||t||>0, altrimenti = mu_c (nessun accoppiamento)
        Sv_t = S_vv @ t                        # (k_v x C)
        num_nu = (t * Sv_t).sum(axis=0)        # (C,)
        nu_c = mu_c.copy()
        nz = t_norm > 1e-15
        nu_c[nz] = (num_nu[nz] / (t_norm[nz]**2 + 1e-15)).astype(float)
    else:
        tau_c = np.zeros_like(mu_c)
        nu_c  = mu_c.copy()

    # lambda_plus_w per classe
    rad_c = np.sqrt((mu_c - nu_c)**2 + 4.0 * (tau_c**2))
    lam_plus_w_c = 0.5 * (mu_c + nu_c + rad_c)

    # Gain per-classe e aggregazione pesata
    gain_w_c = (lam_plus_w_c - mu_c).astype(float)
    lb_gain_W_star = float((cw * gain_w_c).sum())

    # Normalizzazione sul margine residuo in W (come prima, R_obs_W già calcolato sopra)
    margin_W = max(1.0 - R_obs_W, 1e-6)
    lb_gain_W_star_norm = lb_gain_W_star / margin_W



    # sanity
    if not np.isfinite(lb_rayleigh_gain):
        raise ValueError("lb_rayleigh_gain non finito: controlla S_aug/S_obs/part/Y")

    return {
        "rho": rho,
        "overlineDelta": overlineDelta,
        "lb_rayleigh_gain": lb_rayleigh_gain,
        "R_obs": R_obs,
        # diagnostiche utili
        "mu_vec": mu,
        "nu_vec": nu,
        "tau_vec": tau,
        "lambda_plus_vec": lam_plus,
        "omega_vec": omega,
        "delta_vec": delta_j,
        "k_roles": int(k),
        "k_virtual": int(k_v),
        "R_obs_W": R_obs_W,
        "R_oo_W": R_oo_W,
        "lb_gain_W": lb_gain_W,
        "lb_gain_W_pos": lb_gain_W_pos,
        "lb_gain_W_norm": lb_gain_W_norm,
        "lb_gain_W_pos_norm": lb_gain_W_pos_norm,
        "lb_gain_W_star": lb_gain_W_star,
        "lb_gain_W_star_norm": lb_gain_W_star_norm,
        "coupW_strength": coupW_strength,     # diagnostica: 0 con EP esatta

    }



def one_vs_rest_matrix(y: np.ndarray, n: int | None = None) -> np.ndarray:
    """Ritorna Y (n x C) one-vs-rest in {+1,-1}, già centrata per colonna."""
    if y.ndim != 1:
        raise ValueError("y deve essere (n,) con id di classe")
    classes = np.unique(y)
    C = len(classes)
    n = y.size if n is None else n
    Y = np.empty((n, C), dtype=float)
    for j, c in enumerate(classes):
        col = np.where(y == c, 1.0, -1.0)   # +1/-1
        col = col - col.mean()              # centering
        # opzionale: normalizzazione a varianza unitaria per stabilità
        s = np.linalg.norm(col) + 1e-15
        Y[:, j] = col / s
    return Y, classes

def class_weights(y: np.ndarray, mode: str = "micro") -> np.ndarray:
    """Restituisce pesi per classe: 'macro' = uniformi; 'micro' = proporzionali alla frequenza."""
    classes, counts = np.unique(y, return_counts=True)
    if mode == "macro":
        w = np.ones_like(counts, dtype=float) / len(classes)
    elif mode == "micro":
        w = counts.astype(float) / counts.sum()
    else:
        raise ValueError("mode deve essere 'macro' o 'micro'")
    return w


def infer_num_classes(y) -> int:
    """
    Restituisce il numero di classi.
    - Se y è (n, C) one-vs-rest / one-hot, ritorna C.
    - Se y è (n,), usa i valori unici ignorando marker di unlabeled (-1, -100).
    """
    y = np.asarray(y)
    if y.ndim == 2:
        return int(y.shape[1])
    y_flat = y.reshape(-1)
    mask = (y_flat != -1) & (y_flat != -100)
    vals = np.unique(y_flat[mask]) if mask.any() else np.unique(y_flat)
    return int(vals.size)
