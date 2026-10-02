\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
   

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import pathlib
import pickle
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np
import optax

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
APPENDIX_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = APPENDIX_DIR.parent
DEEPONET_DIR = PROJECT_ROOT / "deeponet_comparisons"
sys.path.insert(0, str(DEEPONET_DIR))

from deeponet_model_jax import (              
    composite_forward,
    init_composite_params,
    lf_branch_weight_norm_sq,
    modified_deeponet_grid_predict,
    nonlinear_branch_weight_norm_sq,
)
from deeponet_utils import rel_l2_batch, u_of              

DATA_DIR = APPENDIX_DIR / "data"
DEFAULT_RESULT_DIR = DATA_DIR / "results_accuracy_compute_codex"

LAMBDA_HF = 0.1
LAMBDA_LF = 1.0
LAMBDA_NL_REG = 1.0e-1
LAMBDA_LF_REG = 1.0e-4
GRAD_CLIP_NORM = 1.0

ACTIVE_PROGRESS_LOG: pathlib.Path | None = None
ACTIVE_STATUS_FILE: pathlib.Path | None = None
ACTIVE_LOCK_FILE: pathlib.Path | None = None
ACTIVE_LOCK_STREAM = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--niter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, nargs="+", default=[100] * 4)
    parser.add_argument("--out-width", type=int, default=100)
    parser.add_argument(
        "--bfkle-results",
        type=pathlib.Path,
        default=DEFAULT_RESULT_DIR / "bfkle_accuracy_compute_codex.npz",
        help="BF-KLE timing file that defines the matched wall-clock budget.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=DEFAULT_RESULT_DIR / "deeponet_accuracy_compute_codex.npz",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=pathlib.Path,
        help="Per-replication result/state directory (derived from --out by default).",
    )
    parser.add_argument(
        "--progress-log",
        type=pathlib.Path,
        help="Append-only JSONL progress log (derived from --out by default).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume checkpoints and skip compatible completed replications.",
    )
    return parser.parse_args()


def make_optimizer(learning_rate: float):
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=1000,
        decay_rate=0.9,
        staircase=True,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_NORM),
        optax.adam(schedule),
    )
    return optimizer


def loss_function(params, u_lf, y_lf, u_hf, y_hf, x_grid):
    predicted_lf = modified_deeponet_grid_predict(
        params["lf_net"], u_lf, x_grid, jnp.tanh
    )
    _, _, _, predicted_hf = composite_forward(params, u_hf, x_grid)

    loss_lf = jnp.mean((predicted_lf - y_lf) ** 2)
    loss_hf = jnp.mean((predicted_hf - y_hf) ** 2)
    return (
        LAMBDA_HF * loss_hf
        + LAMBDA_LF * loss_lf
        + LAMBDA_NL_REG * nonlinear_branch_weight_norm_sq(params)
        + LAMBDA_LF_REG * lf_branch_weight_norm_sq(params)
    )


def make_training_step(optimizer):
    @jax.jit
    def step(params, optimizer_state, u_lf, y_lf, u_hf, y_hf, x_grid):
        loss, gradients = jax.value_and_grad(loss_function)(
            params, u_lf, y_lf, u_hf, y_hf, x_grid
        )
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, params
        )
        params = optax.apply_updates(params, updates)
        return params, optimizer_state, loss

    return step


def load_replication(rep: int):
    data = np.load(DATA_DIR / f"jump_bifi_rep{rep}.npz")
    x = data["x"].astype(np.float64)
    xi_lf = data["xi_LF"][:, 0]
    xi_hf = data["xi_HF"][:, 0]

    arrays = {
        "x": jnp.asarray(x.reshape(-1, 1)),
        "u_lf": jnp.asarray(u_of(xi_lf, x)),
        "y_lf": jnp.asarray(data["LF_data"].T),
        "u_hf": jnp.asarray(u_of(xi_hf, x)),
        "y_hf": jnp.asarray(data["HF_data"].T),
        "u_test": jnp.asarray(u_of(data["a_grid"], x)),
        "y_test": data["HF_oracle"].T,
    }
    return arrays


def copy_params_to_host(params):
                                                                        
    return jax.tree.map(lambda value: np.array(value), params)


def evaluate_error(params, arrays) -> float:
    device_params = jax.tree.map(jnp.asarray, params)
    _, _, _, prediction = composite_forward(
        device_params, arrays["u_test"], arrays["x"]
    )
    return rel_l2_batch(np.asarray(prediction), arrays["y_test"])


def checkpoint_steps(niter: int) -> list[int]:
    requested = [1, 2, 5, 10, 50, 100, 250, 500, 1000, 2500, 5000]
    return sorted({step for step in requested if step <= niter} | {niter})


def atomic_savez(path: pathlib.Path, **arrays) -> None:
                                                                          
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def atomic_pickle(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def append_progress(path: pathlib.Path, event: str, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        "pid": os.getpid(),
        **values,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def write_status(path: pathlib.Path, event: str, **values) -> None:
    record = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        "pid": os.getpid(),
        **values,
    }
    if ACTIVE_LOCK_FILE is not None:
        record["lock_file"] = str(ACTIVE_LOCK_FILE)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def rep_result_path(checkpoint_dir: pathlib.Path, rep: int) -> pathlib.Path:
    return checkpoint_dir / f"rep_{rep:03d}.npz"


def rep_state_path(checkpoint_dir: pathlib.Path, rep: int) -> pathlib.Path:
    return checkpoint_dir / f"rep_{rep:03d}_checkpoint.pkl"


def validate_config(stored: dict, current: dict, path: pathlib.Path) -> None:
    mismatches = {
        key: (stored.get(key), value)
        for key, value in current.items()
        if stored.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Incompatible resumable result {path}: {mismatches}. "
            "Use a different --out/--checkpoint-dir for a different recipe."
        )


def save_replication_result(
    path: pathlib.Path,
    config: dict,
    rep: int,
    compile_seconds: float,
    fit_seconds: float,
    matched_seconds: float,
    matched_step: int,
    matched_error: float,
    curve_steps: list[int],
    curve_seconds: dict[int, float],
    curve_errors: dict[int, float],
) -> None:
    errors = np.asarray([curve_errors[step] for step in curve_steps])
    atomic_savez(
        path,
        config_json=json.dumps(config, sort_keys=True),
        replication=rep,
        fit_seconds=fit_seconds,
        compile_seconds=compile_seconds,
        final_error=float(errors[-1]),
        matched_budget_seconds=config["matched_budget_seconds"],
        matched_seconds=matched_seconds,
        matched_step=matched_step,
        matched_error=matched_error,
        curve_steps=np.asarray(curve_steps),
        curve_seconds=np.asarray([curve_seconds[step] for step in curve_steps]),
        curve_errors=errors,
        precision_bits=64,
        cpu_affinity_count=len(os.sched_getaffinity(0)),
    )


def load_replication_result(path: pathlib.Path, config: dict) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        validate_config(json.loads(str(archive["config_json"].item())), config, path)
        return {key: np.array(archive[key]) for key in archive.files}


def combine_replications(paths: list[pathlib.Path], out: pathlib.Path, config: dict) -> None:
    results = [load_replication_result(path, config) for path in paths]
    curve_steps = results[0]["curve_steps"]
    if any(not np.array_equal(item["curve_steps"], curve_steps) for item in results):
        raise ValueError("Per-replication curve_steps disagree; refusing to combine")
    atomic_savez(
        out,
        config_json=json.dumps(config, sort_keys=True),
        fit_seconds=np.asarray([float(item["fit_seconds"]) for item in results]),
        compile_seconds=max(float(item["compile_seconds"]) for item in results),
        final_errors=np.asarray([float(item["final_error"]) for item in results]),
        matched_budget_seconds=config["matched_budget_seconds"],
        matched_seconds=np.asarray([float(item["matched_seconds"]) for item in results]),
        matched_steps=np.asarray([int(item["matched_step"]) for item in results]),
        matched_errors=np.asarray([float(item["matched_error"]) for item in results]),
        curve_steps=curve_steps,
        curve_seconds=np.stack([item["curve_seconds"] for item in results]),
        curve_errors=np.stack([item["curve_errors"] for item in results]),
        n_reps=len(results),
        niter=config["niter"],
        seed=config["seed"],
        hidden=np.asarray(config["hidden"]),
        out_width=config["out_width"],
        precision_bits=64,
        cpu_affinity_count=np.asarray(
            [int(item["cpu_affinity_count"]) for item in results]
        ),
    )


def _legacy_main_unresumable() -> None:
    args = parse_args()
    if args.reps <= 0 or args.niter <= 0:
        raise ValueError("--reps and --niter must be positive")
    if not args.bfkle_results.exists():
        raise FileNotFoundError(
            f"Run benchmark_bfkle_accuracy_compute_codex.jl first: "
            f"{args.bfkle_results} does not exist"
        )

    bfkle = np.load(args.bfkle_results)
    matched_budget = float(np.median(bfkle["fit_seconds"]))
    curve_steps = checkpoint_steps(args.niter)

    optimizer = make_optimizer(1.0e-3)
    training_step = make_training_step(optimizer)
    random_key = jax.random.PRNGKey(args.seed)

    fit_seconds = np.zeros(args.reps)
    final_errors = np.zeros(args.reps)
    matched_seconds = np.zeros(args.reps)
    matched_steps = np.zeros(args.reps, dtype=int)
    matched_errors = np.zeros(args.reps)
    curve_seconds = np.zeros((args.reps, len(curve_steps)))
    curve_errors = np.zeros((args.reps, len(curve_steps)))

    compiled_step = None
    compile_seconds = 0.0

    for rep in range(1, args.reps + 1):
        arrays = load_replication(rep)
        nx = arrays["x"].shape[0]
        random_key, parameter_key = jax.random.split(random_key)
        params = init_composite_params(
            parameter_key,
            nx,
            hidden_layers=args.hidden,
            out_width=args.out_width,
        )
        optimizer_state = optimizer.init(params)

        step_args = (
            params,
            optimizer_state,
            arrays["u_lf"],
            arrays["y_lf"],
            arrays["u_hf"],
            arrays["y_hf"],
            arrays["x"],
        )
        if compiled_step is None:
            start_compile = time.perf_counter()
            compiled_step = training_step.lower(*step_args).compile()
            compile_seconds = time.perf_counter() - start_compile

        snapshots: dict[int, object] = {}
        matched_snapshot = None
        accumulated = 0.0
        segment_start = time.perf_counter()

        for iteration in range(1, args.niter + 1):
            params, optimizer_state, loss = compiled_step(
                params,
                optimizer_state,
                arrays["u_lf"],
                arrays["y_lf"],
                arrays["u_hf"],
                arrays["y_hf"],
                arrays["x"],
            )

            must_sync = iteration in curve_steps or matched_snapshot is None
            if must_sync:
                loss.block_until_ready()
                accumulated += time.perf_counter() - segment_start

                if matched_snapshot is None and accumulated >= matched_budget:
                    matched_snapshot = copy_params_to_host(params)
                    matched_steps[rep - 1] = iteration
                    matched_seconds[rep - 1] = accumulated

                if iteration in curve_steps:
                    snapshots[iteration] = copy_params_to_host(params)
                    curve_seconds[rep - 1, curve_steps.index(iteration)] = accumulated

                segment_start = time.perf_counter()

                                                                            
        fit_seconds[rep - 1] = accumulated
        if matched_snapshot is None:
            matched_snapshot = copy_params_to_host(params)
            matched_steps[rep - 1] = args.niter
            matched_seconds[rep - 1] = accumulated

        for column, iteration in enumerate(curve_steps):
            curve_errors[rep - 1, column] = evaluate_error(
                snapshots[iteration], arrays
            )
        final_errors[rep - 1] = curve_errors[rep - 1, -1]
        matched_errors[rep - 1] = evaluate_error(matched_snapshot, arrays)

        print(
            f"rep {rep:2d}/{args.reps}  fit {fit_seconds[rep - 1]:.3f} s  "
            f"matched step {matched_steps[rep - 1]} "
            f"({matched_seconds[rep - 1]:.6f} s)  "
            f"matched error {matched_errors[rep - 1]:.6e}  "
            f"final error {final_errors[rep - 1]:.6e}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        fit_seconds=fit_seconds,
        compile_seconds=compile_seconds,
        final_errors=final_errors,
        matched_budget_seconds=matched_budget,
        matched_seconds=matched_seconds,
        matched_steps=matched_steps,
        matched_errors=matched_errors,
        curve_steps=np.asarray(curve_steps),
        curve_seconds=curve_seconds,
        curve_errors=curve_errors,
        n_reps=args.reps,
        niter=args.niter,
        precision_bits=64,
        cpu_affinity_count=len(os.sched_getaffinity(0)),
    )

    print(f"DeepONet warm fit median: {np.median(fit_seconds):.3f} s")
    print(f"DeepONet XLA compilation: {compile_seconds:.3f} s")
    print(f"Matched BF-KLE budget:    {matched_budget:.6f} s")
    print(f"Saved {args.out}")


def main() -> None:
    global ACTIVE_PROGRESS_LOG, ACTIVE_STATUS_FILE, ACTIVE_LOCK_FILE
    global ACTIVE_LOCK_STREAM

    args = parse_args()
    if args.reps <= 0 or args.niter <= 0:
        raise ValueError("--reps and --niter must be positive")
    if not args.bfkle_results.exists():
        raise FileNotFoundError(
            "Run benchmark_bfkle_accuracy_compute_codex.jl first: "
            f"{args.bfkle_results} does not exist"
        )

    args.out = args.out.resolve()
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir
        else args.out.parent / f"{args.out.stem}_reps"
    )
    progress_log = (
        args.progress_log.resolve()
        if args.progress_log
        else args.out.parent / f"{args.out.stem}.progress.jsonl"
    )
    status_file = progress_log.with_suffix(".status.json")
    lock_file = progress_log.with_suffix(".lock")
    ACTIVE_PROGRESS_LOG = progress_log
    ACTIVE_STATUS_FILE = status_file
    ACTIVE_LOCK_FILE = lock_file
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_LOCK_STREAM = lock_file.open("a+", encoding="utf-8")
    try:
        fcntl.flock(ACTIVE_LOCK_STREAM, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            f"Another benchmark already holds the run lock {lock_file}"
        ) from error
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.bfkle_results, allow_pickle=False) as bfkle:
        matched_budget = float(np.median(bfkle["fit_seconds"]))
    curve_steps = checkpoint_steps(args.niter)
    config = {
        "schema_version": 2,
        "niter": args.niter,
        "seed": args.seed,
        "hidden": list(args.hidden),
        "out_width": args.out_width,
        "precision_bits": 64,
        "matched_budget_seconds": matched_budget,
    }
    paths = [rep_result_path(checkpoint_dir, rep) for rep in range(1, args.reps + 1)]

    print(f"Progress log: {progress_log}", flush=True)
    print(f"Current status: {status_file}", flush=True)
    print(f"Per-replication results: {checkpoint_dir}", flush=True)
    append_progress(
        progress_log,
        "run_started",
        reps=args.reps,
        niter=args.niter,
        resume=args.resume,
        output=str(args.out),
    )

                                                                              
                                                                              
                                                 
    materialized = 0
    if (
        args.resume
        and args.out.exists()
        and args.seed == 0
        and list(args.hidden) == [100] * 4
        and args.out_width == 100
    ):
        with np.load(args.out, allow_pickle=False) as aggregate:
            if (
                "config_json" not in aggregate.files
                and int(aggregate["niter"]) == args.niter
                and int(aggregate["n_reps"]) >= args.reps
                and np.array_equal(aggregate["curve_steps"], np.asarray(curve_steps))
            ):
                for rep, path in enumerate(paths, start=1):
                    if path.exists():
                        continue
                    index = rep - 1
                    atomic_savez(
                        path,
                        config_json=json.dumps(config, sort_keys=True),
                        replication=rep,
                        fit_seconds=float(aggregate["fit_seconds"][index]),
                        compile_seconds=float(aggregate["compile_seconds"]),
                        final_error=float(aggregate["final_errors"][index]),
                        matched_budget_seconds=matched_budget,
                        matched_seconds=float(aggregate["matched_seconds"][index]),
                        matched_step=int(aggregate["matched_steps"][index]),
                        matched_error=float(aggregate["matched_errors"][index]),
                        curve_steps=np.array(aggregate["curve_steps"]),
                        curve_seconds=np.array(aggregate["curve_seconds"][index]),
                        curve_errors=np.array(aggregate["curve_errors"][index]),
                        precision_bits=int(aggregate["precision_bits"]),
                        cpu_affinity_count=int(aggregate["cpu_affinity_count"]),
                    )
                    materialized += 1
        if materialized:
            append_progress(
                progress_log,
                "legacy_aggregate_materialized",
                replications_created=materialized,
                aggregate=str(args.out),
            )

    completed = {}
    if args.resume:
        for rep, path in enumerate(paths, start=1):
            if path.exists():
                completed[rep] = load_replication_result(path, config)
    if len(completed) == args.reps:
        print(
            f"All {args.reps} compatible replications already complete; "
            "no training was launched.",
            flush=True,
        )
        if materialized:
            print(
                f"Materialized {materialized} per-replication files without "
                "rewriting the existing aggregate.",
                flush=True,
            )
        append_progress(progress_log, "run_complete", reused_replications=args.reps)
        write_status(
            status_file,
            "complete",
            reps=args.reps,
            niter=args.niter,
            aggregate=str(args.out),
            reused_replications=args.reps,
        )
        return

    optimizer = make_optimizer(1.0e-3)
    training_step = make_training_step(optimizer)
    random_key = jax.random.PRNGKey(args.seed)
    compiled_step = None
    compile_seconds = 0.0

    for rep in range(1, args.reps + 1):
        random_key, parameter_key = jax.random.split(random_key)
        if rep in completed:
            print(f"rep {rep:2d}/{args.reps} already complete; skipping", flush=True)
            continue

        arrays = load_replication(rep)
        nx = arrays["x"].shape[0]
        initial_params = init_composite_params(
            parameter_key,
            nx,
            hidden_layers=args.hidden,
            out_width=args.out_width,
        )
        initial_optimizer_state = optimizer.init(initial_params)
        step_args = (
            initial_params,
            initial_optimizer_state,
            arrays["u_lf"],
            arrays["y_lf"],
            arrays["u_hf"],
            arrays["y_hf"],
            arrays["x"],
        )
        if compiled_step is None:
            start_compile = time.perf_counter()
            compiled_step = training_step.lower(*step_args).compile()
            compile_seconds = time.perf_counter() - start_compile
            append_progress(
                progress_log, "xla_compiled", compile_seconds=compile_seconds
            )

        checkpoint = rep_state_path(checkpoint_dir, rep)
        if args.resume and checkpoint.exists():
            with checkpoint.open("rb") as stream:
                state = pickle.load(stream)
            validate_config(state["config"], config, checkpoint)
            params = jax.tree.map(jnp.asarray, state["params"])
            optimizer_state = jax.tree.map(jnp.asarray, state["optimizer_state"])
            start_iteration = int(state["iteration"]) + 1
            accumulated = float(state["accumulated"])
            matched_snapshot = state["matched_snapshot"]
            matched_steps_value = int(state["matched_step"])
            matched_seconds_value = float(state["matched_seconds"])
            saved_curve_seconds = state["curve_seconds"]
            saved_curve_errors = state["curve_errors"]
            print(
                f"rep {rep:2d}/{args.reps} resuming at iteration "
                f"{start_iteration} after {accumulated:.3f} s of fitting",
                flush=True,
            )
            append_progress(
                progress_log,
                "rep_resumed",
                replication=rep,
                next_iteration=start_iteration,
                fit_seconds=accumulated,
            )
        else:
            params = initial_params
            optimizer_state = initial_optimizer_state
            start_iteration = 1
            accumulated = 0.0
            matched_snapshot = None
            matched_steps_value = 0
            matched_seconds_value = 0.0
            saved_curve_seconds = {}
            saved_curve_errors = {}

        segment_start = time.perf_counter()
        for iteration in range(start_iteration, args.niter + 1):
            params, optimizer_state, loss = compiled_step(
                params,
                optimizer_state,
                arrays["u_lf"],
                arrays["y_lf"],
                arrays["u_hf"],
                arrays["y_hf"],
                arrays["x"],
            )

            must_sync = iteration in curve_steps or matched_snapshot is None
            if not must_sync:
                continue
            loss.block_until_ready()
            accumulated += time.perf_counter() - segment_start
            host_params = copy_params_to_host(params)

            if matched_snapshot is None and accumulated >= matched_budget:
                matched_snapshot = host_params
                matched_steps_value = iteration
                matched_seconds_value = accumulated

            if iteration in curve_steps:
                saved_curve_seconds[iteration] = accumulated
                saved_curve_errors[iteration] = evaluate_error(host_params, arrays)
                rate = iteration / accumulated
                remaining_seconds = max(args.niter - iteration, 0) / rate
                eta = dt.datetime.now().astimezone() + dt.timedelta(
                    seconds=remaining_seconds
                )
                loss_value = float(loss)
                error_value = saved_curve_errors[iteration]
                print(
                    f"rep {rep:2d}/{args.reps}  iter {iteration:5d}/{args.niter}  "
                    f"fit {accumulated:9.3f} s  loss {loss_value:.6e}  "
                    f"rel-L2 {error_value:.6e}  ETA {eta:%H:%M:%S %Z}",
                    flush=True,
                )
                progress = {
                    "replication": rep,
                    "total_replications": args.reps,
                    "iteration": iteration,
                    "niter": args.niter,
                    "fit_seconds": accumulated,
                    "loss": loss_value,
                    "relative_l2_error": error_value,
                    "estimated_remaining_seconds": remaining_seconds,
                    "estimated_completion": eta.isoformat(timespec="seconds"),
                }
                append_progress(progress_log, "checkpoint", **progress)
                write_status(
                    status_file,
                    "running",
                    **progress,
                    progress_log=str(progress_log),
                )
                atomic_pickle(
                    checkpoint,
                    {
                        "config": config,
                        "replication": rep,
                        "iteration": iteration,
                        "params": host_params,
                        "optimizer_state": copy_params_to_host(optimizer_state),
                        "accumulated": accumulated,
                        "matched_snapshot": matched_snapshot,
                        "matched_step": matched_steps_value,
                        "matched_seconds": matched_seconds_value,
                        "curve_seconds": saved_curve_seconds,
                        "curve_errors": saved_curve_errors,
                    },
                )
                                                                                 
            segment_start = time.perf_counter()

        if matched_snapshot is None:
            matched_snapshot = copy_params_to_host(params)
            matched_steps_value = args.niter
            matched_seconds_value = accumulated
        matched_error_value = evaluate_error(matched_snapshot, arrays)
        save_replication_result(
            paths[rep - 1],
            config,
            rep,
            compile_seconds,
            accumulated,
            matched_seconds_value,
            matched_steps_value,
            matched_error_value,
            curve_steps,
            saved_curve_seconds,
            saved_curve_errors,
        )
        completed[rep] = load_replication_result(paths[rep - 1], config)
        append_progress(
            progress_log,
            "rep_complete",
            replication=rep,
            fit_seconds=accumulated,
            matched_step=matched_steps_value,
            matched_seconds=matched_seconds_value,
            matched_error=matched_error_value,
            final_error=saved_curve_errors[curve_steps[-1]],
            result=str(paths[rep - 1]),
        )
        combine_replications(paths[:rep], args.out, config)
        print(
            f"rep {rep:2d}/{args.reps} complete; saved {paths[rep - 1]} "
            f"and refreshed {args.out}",
            flush=True,
        )

    combine_replications(paths, args.out, config)
    results = [load_replication_result(path, config) for path in paths]
    fit_seconds = np.asarray([float(item["fit_seconds"]) for item in results])
    append_progress(
        progress_log,
        "run_complete",
        fit_seconds=fit_seconds.tolist(),
        aggregate=str(args.out),
    )
    write_status(
        status_file,
        "complete",
        reps=args.reps,
        niter=args.niter,
        fit_seconds=fit_seconds.tolist(),
        aggregate=str(args.out),
        progress_log=str(progress_log),
    )
    print(f"DeepONet warm fit median: {np.median(fit_seconds):.3f} s")
    print(f"DeepONet XLA compilation: {compile_seconds:.3f} s")
    print(f"Matched BF-KLE budget:    {matched_budget:.6f} s")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if ACTIVE_PROGRESS_LOG is not None and ACTIVE_STATUS_FILE is not None:
            append_progress(ACTIVE_PROGRESS_LOG, "run_interrupted")
            write_status(
                ACTIVE_STATUS_FILE,
                "interrupted",
                progress_log=str(ACTIVE_PROGRESS_LOG),
                message="Restart the identical command to resume the last checkpoint.",
            )
        raise
    except Exception as error:
        if ACTIVE_PROGRESS_LOG is not None and ACTIVE_STATUS_FILE is not None:
            append_progress(
                ACTIVE_PROGRESS_LOG,
                "run_failed",
                error_type=type(error).__name__,
                message=str(error),
            )
            write_status(
                ACTIVE_STATUS_FILE,
                "failed",
                progress_log=str(ACTIVE_PROGRESS_LOG),
                error_type=type(error).__name__,
                message=str(error),
            )
        raise
