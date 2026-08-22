#!/usr/bin/env python
"""
Train ECA-FedNet under Protocol B (Sec. IV-A, Table IX).

A fixed stratified 80/20 split (seed 42) removes 200 test records before any
client is formed; the remaining 800 records are partitioned across K = 4
simulated institutions with a per-class Dir(alpha = 0.4) draw.  The server runs
T = 25 communication rounds of FedAvg, each client performing E = 2 local
epochs.  Only model parameters are exchanged.

Produces the ECA-FedNet column of Table XI, Table V, and Figs. 7(h) and 9.

    python scripts/train_ecafednet.py --data-root data/SSMCH-ECG
    python scripts/train_ecafednet.py --data-root data/SSMCH-ECG --centralized-baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecanet.config import add_common_args, config_from_args
from ecanet.data import (build_index, build_transforms, build_weighted_sampler,
                         describe_clients, make_loader, partition_clients,
                         protocol_b_split)
from ecanet.engine import predict, train_model
from ecanet.federated import (client_update, communication_cost_mb, fedavg,
                              select_participants, to_cpu_state)
from ecanet.losses import build_loss, class_counts_from_targets
from ecanet.metrics import format_summary, summarize
from ecanet.models import ECANet
from ecanet.utils import ensure_dir, save_json, seed_everything, setup_logger
from ecanet.visualization import (plot_client_distribution, plot_confusion_matrix,
                                  plot_federated_convergence)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECA-FedNet — FedAvg over non-IID clients.")
    add_common_args(parser)
    parser.add_argument("--clients", type=int, default=4, help="K (Table IX).")
    parser.add_argument("--rounds", type=int, default=25, help="T (Table IX).")
    parser.add_argument("--local-epochs", type=int, default=2, help="E (Table IX).")
    parser.add_argument("--alpha", type=float, default=0.4, help="Dirichlet concentration.")
    parser.add_argument("--participation", type=float, default=1.0)
    parser.add_argument("--partition", default="dirichlet", choices=["dirichlet", "site"])
    parser.add_argument("--local-lr", type=float, default=1e-3,
                        help="Table IX gives 1e-3; Algorithm 3 gives 1e-4.")
    parser.add_argument("--centralized-baseline", action="store_true",
                        help="Also train a centralized model on the same 800/200 split "
                             "for a matched comparison (Sec. V-C).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    cfg.federated.num_clients = args.clients
    cfg.federated.rounds = args.rounds
    cfg.federated.local_epochs = args.local_epochs
    cfg.federated.dirichlet_alpha = args.alpha
    cfg.federated.participation = args.participation
    cfg.federated.partition = args.partition
    cfg.federated.local_lr = args.local_lr
    cfg.federated.batch_size = cfg.train.batch_size
    # Test-time augmentation is defined for centralized evaluation only (Sec. IV-A)
    cfg.eval.use_tta = False

    out_dir = ensure_dir(Path(cfg.output_dir) / f"ecafednet_{cfg.data.mode}")
    logger = setup_logger(out_dir)
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    index = build_index(cfg.data)
    logger.info(f"Loaded {index.size} records | classes {index.class_names} | "
                f"counts {index.counts()}")

    train_tf, eval_tf = build_transforms(cfg.model.input_size, cfg.augment)

    # ---- Protocol B: hold out the test set before forming any client ----
    train_idx, test_idx = protocol_b_split(index, cfg.eval, cfg.seed)
    logger.info(f"Protocol B split: {len(train_idx)} client-side records, "
                f"{len(test_idx)} held-out test records")

    clients = partition_clients(index, train_idx, cfg.federated, cfg.seed)
    clients = [c for c in clients if len(c) > 0]
    table_v = describe_clients(index, clients)
    logger.info("Client-wise distribution (Table V):")
    for row in table_v:
        detail = "  ".join(f"{name}={row.get(name, 0)}" for name in index.class_names)
        logger.info(f"  Client {row['Client']}: N_k={row['N_k']}  {detail}")
    sizes = [len(c) for c in clients]
    if min(sizes) > 0:
        logger.info(f"  size range {min(sizes)}-{max(sizes)} "
                    f"({max(sizes)/max(1, min(sizes)):.1f}-fold difference)")

    test_loader = make_loader(index, test_idx, eval_tf, cfg.train.batch_size,
                              False, cfg.data)

    client_loaders = []
    client_counts = []
    for ci in clients:
        sampler = build_weighted_sampler(index.targets, ci, index.num_classes, cfg.seed)
        client_loaders.append(make_loader(index, ci, train_tf, cfg.federated.batch_size,
                                          True, cfg.data, sampler=sampler))
        client_counts.append(class_counts_from_targets(index.targets, ci,
                                                       index.num_classes))

    def model_factory() -> ECANet:
        return ECANet(index.num_classes,
                      attention=cfg.model.attention,
                      reduction=cfg.model.reduction_ratio,
                      spatial_kernel=cfg.model.spatial_kernel,
                      dropout=cfg.model.dropout,
                      pretrained=cfg.model.pretrained)

    # ---- Algorithm 2: server-side federated optimisation ----
    global_model = model_factory().to(device)
    global_state = to_cpu_state(global_model)
    payload_mb = communication_cost_mb(global_state)
    logger.info(f"Per-client upload per round: {payload_mb:.1f} MB "
                f"({sum(v.numel() for v in global_state.values())/1e6:.2f} M values)")

    convergence = []
    rng = np.random.default_rng(cfg.seed)

    for rnd in range(1, cfg.federated.rounds + 1):
        participants = select_participants(len(clients), cfg.federated.participation, rng)
        states, sizes_used = [], []
        for k in participants:
            state, n_k = client_update(model_factory, global_state, client_loaders[k],
                                       client_counts[k], device, cfg.federated,
                                       num_samples=len(clients[k]),
                                       amp=cfg.train.amp)
            states.append(state)
            sizes_used.append(n_k)

        global_state = fedavg(
            states, sizes_used,
            integer_from_first_client=cfg.federated.aggregate_integer_buffers_from_first_client)
        global_model.load_state_dict({k: v.to(device) for k, v in global_state.items()})

        y_true, y_prob = predict(global_model, test_loader, device, use_tta=False)
        round_summary = summarize(y_true, y_prob.argmax(1), y_prob, index.class_names,
                                  index.positive_index, cfg.data.mode,
                                  cfg.eval.ci_method, cfg.eval.ci_level)
        convergence.append(round_summary["balanced_accuracy"])
        if rnd == 1 or rnd % 5 == 0 or rnd == cfg.federated.rounds:
            logger.info(f"  round {rnd:02d}/{cfg.federated.rounds}: "
                        f"raw {round_summary['accuracy']*100:.2f}% | "
                        f"balanced {round_summary['balanced_accuracy']*100:.2f}% | "
                        f"{index.class_names[index.positive_index]} sensitivity "
                        f"{round_summary['positive_sensitivity']*100:.2f}%")

    # ---- final evaluation on the held-out test set ----
    y_true, y_prob = predict(global_model, test_loader, device, use_tta=False)
    y_pred = y_prob.argmax(1)
    summary = summarize(y_true, y_pred, y_prob, index.class_names,
                        index.positive_index, cfg.data.mode,
                        cfg.eval.ci_method, cfg.eval.ci_level,
                        cfg.eval.n_bootstrap, cfg.seed)
    summary["protocol"] = "B (fixed stratified 80/20 split, held-out test set)"
    summary["convergence_balanced_accuracy"] = convergence
    summary["clients"] = table_v
    summary["payload_mb_per_client_per_round"] = payload_mb

    logger.info(format_summary(summary, f"ECA-FedNet ({cfg.data.mode}) — held-out test, "
                                        f"n = {len(y_true)}"))

    torch.save(global_state, out_dir / "ecafednet_global.pth")
    plot_confusion_matrix(y_true, y_pred, index.class_names,
                          f"ECA-FedNet ({cfg.data.mode}) — held-out test",
                          out_dir / "confusion_matrix.png")
    plot_client_distribution(table_v, index.class_names, out_dir / "client_distribution.png")

    # ---- optional matched centralized baseline (Sec. V-C, V-E) ----
    centralized_balanced = None
    if args.centralized_baseline:
        logger.info("Training a centralized baseline on the same 800/200 split "
                    "for a matched comparison.")
        seed_everything(cfg.seed)
        sampler = build_weighted_sampler(index.targets, train_idx, index.num_classes,
                                         cfg.seed)
        central_train = make_loader(index, train_idx, train_tf, cfg.train.batch_size,
                                    True, cfg.data, sampler=sampler)
        central_model = model_factory()
        backbone_params, head_params = central_model.parameter_groups()
        counts = class_counts_from_targets(index.targets, train_idx, index.num_classes)
        criterion = build_loss(cfg.train.loss, counts, device,
                               beta=cfg.train.cb_beta,
                               gamma=cfg.train.focal_gamma,
                               focal_label_smoothing=cfg.train.focal_label_smoothing,
                               ce_label_smoothing=cfg.train.ce_label_smoothing)
        # Match the federated compute budget: T x E local epochs
        cfg.train.epochs = cfg.federated.rounds * cfg.federated.local_epochs
        central_model, _ = train_model(central_model, central_train, test_loader,
                                       criterion, device, cfg.train, cfg.eval,
                                       head_params=head_params,
                                       backbone_params=backbone_params,
                                       log_fn=logger.info, tag="centralized")
        ct, cp = predict(central_model, test_loader, device, use_tta=False)
        central_summary = summarize(ct, cp.argmax(1), cp, index.class_names,
                                    index.positive_index, cfg.data.mode,
                                    cfg.eval.ci_method, cfg.eval.ci_level)
        centralized_balanced = central_summary["balanced_accuracy"]
        summary["centralized_baseline"] = central_summary
        logger.info(format_summary(central_summary,
                                   "Centralized baseline — same held-out test set"))
        gap = (centralized_balanced - summary["balanced_accuracy"]) * 100
        logger.info(f"  Matched comparison: centralized - federated = {gap:+.2f} "
                    f"balanced-accuracy points")

    plot_federated_convergence(convergence, out_dir / "convergence.png",
                               centralized_reference=centralized_balanced)
    save_json({"config": cfg.to_dict(), "summary": summary}, out_dir / "results.json")
    np.savez(out_dir / "test_predictions.npz", y_true=y_true, y_prob=y_prob,
             test_index=test_idx)
    logger.info(f"Artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
