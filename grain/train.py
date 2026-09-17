import argparse
import csv
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from TD3_agent import ReplayBuffer, TD3_Agent
from env.gcn import gcn_env
from utils import evaluate_policy, str2bool

RAWR_EPSILONS = {
    "Actor": [0, 2, 4, 8, 1303],
    "Caterpillar": [0, 1, 2, 9],
    "Chameleon": [0, 6, 12, 29, 732],
    "Cora": [0, 2, 3, 5, 168],
    "Cornell": [0, 1, 2, 4, 94],
    "Citeseer": [0, 1, 2, 3, 99],
    "Grid": [0, 3, 4],
    "Ladder": [0, 2, 3, 4],
    "Line": [0, 2, 4],
    "Lobster": [0, 1, 2, 51],
    "PubMed": [0, 1, 2, 4, 171],
    "Squirrel": [0, 7, 17, 166, 1905],
    "Texas": [0, 1, 2, 3, 104],
    "Tree": [0, 3],
    "Wisconsin": [0, 1, 2, 4, 122],
}


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_run(
    dataset: str,
    seed: int,
    args: argparse.Namespace,
    rewiring_epsilon: int | None = None,
    partition_path: Path | None = None,
    partition_edge_path: Path | None = None,
) -> dict:
    set_seed(seed)

    env = gcn_env(
        dataset=dataset,
        max_layer=args.layers,
        data_source=args.data_source,
        rawr_root=args.rawr_root,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=seed,
        use_features=args.use_features,
        rewiring_mode=args.rewiring_mode,
        rewiring_epsilon=rewiring_epsilon,
        partition_path=partition_path,
        partition_edge_path=partition_edge_path,
        partition_id_offset=args.partition_id_offset,
    )
    env.seed(seed)

    model = TD3_Agent(
        env_with_dw=True,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_num,
        max_action=args.max_action,
    )
    env.policy = model
    replay_buffer = ReplayBuffer(
        env.observation_space.shape[0],
        env.action_num,
        max(
            1,
            args.replay_multiplier * (len(env.train_indexes) - 1),
            (args.max_episodes + 1) * env.batch_size,
        ),
    )

    last_val = float("-inf")
    best_policy = deepcopy(model)
    total_steps = 0
    expl_noise = args.expl_noise

    while total_steps < args.max_episodes:
        s, done, steps = env.reset2(), False, 0

        while not done and total_steps < args.max_episodes:
            steps += 1
            if total_steps < args.start_steps:
                a = env.action_space(s.shape[0], 1)
            else:
                a = (
                    model.select_action(s)
                    + np.random.normal(
                        0,
                        args.max_action * expl_noise,
                        size=env.action_num,
                    )
                ).clip(0, args.max_action)
                a = a.reshape(-1, 1)

            s_prime, r, done, val = env.step2(a)
            done_flag = bool(done)
            # Keep the original repository's termination handling.
            dw = not (done_flag and steps != 3)

            replay_buffer.add(s, a, r, s_prime, dw)
            s = s_prime

            if total_steps >= 2 and total_steps % 3 == 0:
                for _ in range(3):
                    model.train(replay_buffer)

            if total_steps % args.log_every == 0:
                expl_noise *= args.expl_noise_decay
                score = evaluate_policy(env, model, False)
                print(
                    f"[train] dataset={dataset} seed={seed} "
                    f"step={total_steps} score={score:.6f}"
                )

            total_steps += 1
            if val > last_val:
                last_val = val
                best_policy = deepcopy(model)

    print(f"[eval] Training GNNs with learned meta-policy: dataset={dataset} seed={seed}")
    eval_env = gcn_env(
        dataset=dataset,
        max_layer=args.layers,
        data_source=args.data_source,
        rawr_root=args.rawr_root,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=seed,
        use_features=args.use_features,
        rewiring_mode=args.rewiring_mode,
        rewiring_epsilon=rewiring_epsilon,
        partition_path=partition_path,
        partition_edge_path=partition_edge_path,
        partition_id_offset=args.partition_id_offset,
    )
    eval_env.seed(seed)
    eval_env.policy = best_policy

    state = eval_env.reset2()
    best_test_acc = 0.0
    for i_episode in range(1, args.eval_episodes + 1):
        action = best_policy.select_action(state).reshape(-1, 1)
        state, _, _, val_acc = eval_env.step2(action)
        test_acc = eval_env.test_batch()
        print(
            f"[eval] dataset={dataset} seed={seed} episode={i_episode} "
            f"val_acc={val_acc:.6f} test_acc={test_acc:.6f}"
        )
        if best_test_acc < test_acc:
            best_test_acc = test_acc

    return {
        "dataset": dataset,
        "seed": seed,
        "layers": args.layers,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "data_source": args.data_source,
        "use_features": args.use_features,
        "rewiring_mode": args.rewiring_mode,
        "rewiring_epsilon": rewiring_epsilon,
        "original_graph_only": args.rewiring_mode == "none",
        "rewiring": args.rewiring_mode != "none",
        "best_test_accuracy": best_test_acc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GRAIN with RAwR-compatible settings (split ratios, 2 layers), "
            "on either original or RAwR-rewired graphs."
        )
    )
    parser.add_argument("--datasets", type=str, default="Cora,Citeseer,PubMed")
    parser.add_argument("--seeds", type=str, default="111,222,333,444,555")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--data_source", choices=["rawr", "planetoid"], default="rawr")
    parser.add_argument(
        "--rawr_root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to the RAwR repository root.",
    )
    parser.add_argument("--use_features", type=str2bool, default=True)
    parser.add_argument(
        "--rewiring_mode",
        choices=["none", "rep_nodes", "rep_edges"],
        default="none",
        help=(
            "RAwR rewiring mode: none (original graph), rep_nodes (partition stars), "
            "rep_edges (partition stars + partition edges)."
        ),
    )
    parser.add_argument(
        "--rewiring_epsilon",
        type=int,
        default=None,
        help=(
            "Epsilon used to auto-resolve partition files from RAwR folders: "
            "partitions/{dataset}P{epsilon} and reducedNetworks/{dataset}BE{epsilon}.edgelist."
        ),
    )
    parser.add_argument(
        "--rewiring_epsilons",
        type=str,
        default=None,
        help="Optional comma-separated epsilon list (e.g. 0,2,3,5,168).",
    )
    parser.add_argument(
        "--use_rawr_epsilons",
        action="store_true",
        help=(
            "Use the epsilon sets explicitly defined in RAwR run_experiments.py: "
            "Cora=[0,2,3,5,168], Citeseer=[0,1,2,3,99], PubMed=[0,1,2,4,171]."
        ),
    )
    parser.add_argument(
        "--partition_path",
        type=Path,
        default=None,
        help="Optional explicit partition file path. Overrides epsilon-based lookup.",
    )
    parser.add_argument(
        "--partition_edge_path",
        type=Path,
        default=None,
        help="Optional explicit partition-edge file path (for rep_edges).",
    )
    parser.add_argument(
        "--partition_id_offset",
        type=int,
        default=-1,
        help="Offset applied to IDs read from partition files (RAwR defaults use -1).",
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--max_episodes", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--start_steps", type=int, default=5)
    parser.add_argument("--max_action", type=float, default=5.0)
    parser.add_argument("--expl_noise", type=float, default=0.15)
    parser.add_argument("--expl_noise_decay", type=float, default=0.998)
    parser.add_argument("--replay_multiplier", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("grain_rawr_results.csv"),
        help="CSV file for run results.",
    )
    return parser.parse_args()


def resolve_rewiring_epsilons(dataset: str, args: argparse.Namespace) -> list[int | None]:
    if args.rewiring_mode == "none":
        return [None]

    if args.partition_path is not None:
        # Explicit file path -> single run configuration.
        return [args.rewiring_epsilon]

    if args.use_rawr_epsilons:
        if dataset not in RAWR_EPSILONS:
            available = ", ".join(sorted(RAWR_EPSILONS.keys()))
            raise ValueError(
                f"Dataset '{dataset}' has no RAwR epsilon list in run_experiments.py. "
                f"Available: {available}. "
                "Use --rewiring_epsilons for custom datasets."
            )
        return RAWR_EPSILONS[dataset]

    if args.rewiring_epsilons is not None:
        eps = parse_int_list(args.rewiring_epsilons)
        if not eps:
            raise ValueError("--rewiring_epsilons was provided but no values were parsed.")
        return eps

    if args.rewiring_epsilon is not None:
        return [args.rewiring_epsilon]

    raise ValueError(
        "For rewired runs, provide one of: --use_rawr_epsilons, "
        "--rewiring_epsilons, --rewiring_epsilon, or --partition_path."
    )


def main():
    args = parse_args()
    if args.layers != 2:
        raise ValueError(
            "This comparison script is configured for a fair two-layer setup. "
            "Use --layers 2."
        )

    datasets = parse_str_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    args.rawr_root = args.rawr_root.resolve()

    if args.data_source == "rawr" and not args.rawr_root.exists():
        raise FileNotFoundError(f"RAwR root not found: {args.rawr_root}")
    if args.rewiring_mode != "none" and args.data_source != "rawr":
        raise ValueError("Rewiring is only supported with --data_source rawr.")
    if (
        args.rewiring_mode != "none"
        and args.rewiring_epsilon is None
        and args.rewiring_epsilons is None
        and not args.use_rawr_epsilons
        and args.partition_path is None
    ):
        raise ValueError(
            "For rewired runs, provide --use_rawr_epsilons, --rewiring_epsilons, "
            "--rewiring_epsilon, or --partition_path."
        )
    if args.partition_path is not None and (
        args.use_rawr_epsilons or args.rewiring_epsilons is not None
    ):
        raise ValueError(
            "Do not combine --partition_path with --use_rawr_epsilons/--rewiring_epsilons. "
            "Use one strategy."
        )
    if args.partition_path is not None:
        args.partition_path = args.partition_path.resolve()
    if args.partition_edge_path is not None:
        args.partition_edge_path = args.partition_edge_path.resolve()

    all_rows = []
    for dataset in datasets:
        rewiring_epsilons = resolve_rewiring_epsilons(dataset, args)
        for seed in seeds:
            for epsilon in rewiring_epsilons:
                print(
                    f"\n[run] dataset={dataset} seed={seed} layers={args.layers} "
                    f"original_graph_only={args.rewiring_mode == 'none'} "
                    f"rewiring_mode={args.rewiring_mode} epsilon={epsilon}"
                )
                row = train_one_run(
                    dataset=dataset,
                    seed=seed,
                    args=args,
                    rewiring_epsilon=epsilon,
                    partition_path=args.partition_path,
                    partition_edge_path=args.partition_edge_path,
                )
                all_rows.append(row)
                print(
                    f"[done] dataset={dataset} seed={seed} epsilon={epsilon} "
                    f"best_test_accuracy={row['best_test_accuracy']:.6f}"
                )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[results] wrote {args.out_csv.resolve()}")


if __name__ == "__main__":
    main()
