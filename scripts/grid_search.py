r"""Grid search script for GenRec framework.

Arguments:
    --template: Path to the template config yaml file (required unless reusing an experiment).
    --search: Path to the search config yaml file (required unless reusing an experiment).
    --main: Python module to run, e.g., genrec.main_seqrec
    --gpu_groups: Optional list of comma-separated GPU IDs per worker (e.g., --gpu_groups "0,1" --gpu_groups "2,3").
    --gpu_num: Total number of GPUs available (used when gpu_groups is omitted).
    --per_trial_gpu_num: Number of GPUs per trial when deriving groupings automatically.
    --dryrun: Existing exp_id whose metrics should be aggregated without rerunning.
    --rerun: Existing exp_id whose failed trials should be rerun.
    --output_root: Root output directory for all experiments.

This script performs grid search over hyperparameters specified in the search YAML file,
using the template YAML as the base configuration. It schedules trials across available GPUs,
runs them, and collects results into a CSV file.

The experimental results are organized in the following structure:
    output_root/
        exp_{:03d}/
            base.yaml
            search.yaml
            test_metrics.csv
            trials/
                trial_{:03d}/
                    trail_config.yaml
                    run.log
                    test_metrics.json
Each trial's configuration is saved in its own directory, along with logs, metrics, and checkpoints.

You may also choose to reuse an existing experiment by specifying --dryrun or --rerun with the exp_id,
e.g., exp_000. In dryrun mode, no trials are executed; metrics are simply aggregated. In rerun mode,
only failed trials are re-executed.
"""

import itertools
import json
import multiprocessing as mp
import os
import subprocess
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


class Trial:
    """Represents a single trial in the grid search process.

    Attributes:
        idx (int): Index of the trial.
        config (Dict[str, Any]): Full YAML configuration for the trial, including overrides
            and current search values.
        search_params (Dict[str, Any]): Search parameters used in the trial (excluding overrides).
        output_dir (str): Output directory for the trial.
        success (bool): Whether the trial completed successfully (determined by test_metrics.json).
        metrics (Dict[str, Any]): Metrics dictionary read upon successful completion.
    """

    def __init__(
        self,
        idx: int,
        config: Dict[str, Any],
        search_params: Dict[str, Any],
        output_dir: str,
        total_trials: int,
    ) -> None:
        self.idx = idx
        self.config = config
        self.search_params = search_params
        self.output_dir = output_dir
        self.success = False
        self.metrics: Dict[str, Any] = {}
        self.total_trials = total_trials

    def save_config(self) -> str:
        """Saves the trial's configuration to its own output directory.

        Returns:
            str: Path to the saved config file for use in execution commands.
            The config is saved as 'trail_config.yaml' in the trial's output directory.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        config_path = os.path.join(self.output_dir, "trail_config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(self.config, f)
        return config_path


def extract_search_params(
    search_cfg: Dict[str, Any],
    prefix: str = "",
) -> Tuple[Dict[str, list[Any]], Dict[str, Any]]:
    """Extracts search items and overrides from the search configuration YAML.

    Search items are distinguished by the 'search__' prefix, with values being lists
    (indicating possible parameter values). Overrides have no prefix and single values.

    Args:
        search_cfg (Dict[str, Any]): The search configuration dictionary.
        prefix (str, optional): Prefix for nested parameter paths. Defaults to "".
            The nested keys are concatenated with '.'.

    Returns:
        tuple[Dict[str, list[Any]], Dict[str, Any]]:
            - search_items: {parameter_path: [list_of_values]}
            - overrides: {parameter_path: single_value}
    """
    search_items: Dict[str, list[Any]] = {}
    overrides: Dict[str, Any] = {}
    for k, v in search_cfg.items():
        # If the value is a dict, recurse into it
        if isinstance(v, dict):
            sub_search, sub_overrides = extract_search_params(v, prefix + k + ".")
            search_items.update(sub_search)
            overrides.update(sub_overrides)
        else:
            if isinstance(k, str) and k.startswith("search__"):
                # Remove the prefix to get the actual parameter name
                search_items[prefix + k[len("search__") :]] = v
            else:
                overrides[prefix + k] = v
    return search_items, overrides


def set_by_path(
    cfg: Dict[str, Any],
    path: str,
    value: Any,
) -> None:
    """Sets a value in a nested dictionary based on a path string.

    Args:
        cfg (Dict[str, Any]): The nested dictionary to modify.
        path (str): The path string (e.g., "trainer.config.learning_rate").
        value (Any): The value to set at the specified path.
    """
    keys = path.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value


def update_trial_statuses(trials: List["Trial"]) -> None:
    """Refreshes each trial's success flag and metrics from disk outputs."""

    for trial in trials:
        metrics_path = os.path.join(trial.output_dir, "test_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                trial.metrics = json.load(f)
            trial.success = True
        else:
            trial.metrics = {}
            trial.success = False


def resolve_gpu_groups(
    gpu_group_args: Optional[List[str]],
    gpu_num: Optional[int],
    per_trial_gpu_num: int,
) -> List[List[int]]:
    """Determines which GPU groupings to use for scheduling trials."""

    if gpu_group_args:
        groups: List[List[int]] = []
        for entry in gpu_group_args:
            tokens = [tok.strip() for tok in entry.split(",") if tok.strip()]
            if not tokens:
                raise ValueError("Encountered empty GPU group entry.")
            groups.append([int(tok) for tok in tokens])
        return groups

    if gpu_num is None:
        raise ValueError("Either --gpu_groups or --gpu_num must be provided.")
    if gpu_num % per_trial_gpu_num != 0:
        raise ValueError("gpu_num must be divisible by per_trial_gpu_num.")
    return [list(range(i, i + per_trial_gpu_num)) for i in range(0, gpu_num, per_trial_gpu_num)]


def load_yaml_config(path: str, label: str) -> Dict[str, Any]:
    """Loads a YAML config file, raising a descriptive error if missing."""

    if not path:
        raise ValueError(f"{label} path is required.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at {path}.")
    with open(path) as f:
        return yaml.safe_load(f)


def _throttle_launch(
    launch_lock: mp.Lock,
    next_launch_ts: mp.Value,
    launch_stagger: float,
) -> None:
    """Ensures accelerate launches are staggered across workers."""

    if launch_stagger <= 0:
        return

    while True:
        with launch_lock:
            now = time.time()
            earliest = next_launch_ts.value + launch_stagger
            if now >= earliest:
                next_launch_ts.value = now
                return
            wait_time = earliest - now
        time.sleep(min(wait_time, launch_stagger))


def run_trial(
    trial: Trial,
    main_module: str,
    gpu_ids: List[int],
    retry: int = 4,
    retry_delay: float = 0.0,
) -> None:
    """Runs a single trial by the following steps:

    1. Saves the config to the trial's own output directory.
    2. Constructs and runs the command to execute the main training module
        with the saved config on the specified GPU(s). Note that the log
        is redirected to the output directory / run.log file.
    3. Checks if 'test_metrics.json' is produced in the output directory:
       - If present, reads the metrics and marks success.
       - If absent, retries up to 'retry' times.

    .. note::
        Here, the presence of the result file is used to determine success.

    Args:
        trial (Trial): The trial to run.
        main_module (str): The main training module to execute (e.g., 'genrec.main_seqrec').
        gpu_ids (List[int]): List of GPU IDs to use for the trial.
        retry (int): Number of retries if the trial fails. Default is 4.
        retry_delay (float): Seconds to sleep between retry attempts so that previous
            accelerate launches release their networking resources.
    """
    config_path = trial.save_config()
    test_metrics_path = os.path.join(trial.output_dir, "test_metrics.json")

    cmd = [
        "poetry",
        "run",
        "accelerate",
        "launch",
        "-m",
        main_module,
        "--config",
        config_path,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    param_str = ", ".join(f"{k}={v}" for k, v in trial.search_params.items()) or "<empty search params>"
    trial_start_time = time.time()
    trial_label = f"[Trial {trial.idx + 1}/{trial.total_trials}]"

    for attempt in range(retry):
        attempt_start = time.time()
        attempt_start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(attempt_start))
        # Run the command and redirect output to log file
        print(
            f"{trial_label} Attempt {attempt+1} started at {attempt_start_str} using GPUs {gpu_ids} | "
            f"search params: {param_str}"
        )
        log_path = os.path.join(trial.output_dir, "run.log")
        with open(log_path, "w") as log_file:
            subprocess.run(cmd, env=env, stdout=log_file, stderr=log_file)
        # Check for the result file
        if os.path.exists(test_metrics_path):
            with open(test_metrics_path) as f:
                trial.metrics = json.load(f)
            trial.success = True
            elapsed = time.time() - trial_start_time
            finish_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"{trial_label} Success at {finish_str} (elapsed {elapsed:.2f}s)")
            return
        else:
            print(f"{trial_label} Failed attempt {attempt+1}")
            if retry_delay > 0 and attempt + 1 < retry:
                time.sleep(retry_delay)
    # set failure if all attempts exhausted
    trial.success = False


def worker(
    trial_queue: mp.JoinableQueue,
    main_module: str,
    gpu_ids: List[int],
    inter_trial_delay: float,
    retry_delay: float,
    launch_stagger: float,
    launch_lock: mp.Lock,
    next_launch_ts: mp.Value,
) -> None:
    """Worker process that continuously fetches and runs trials from the queue.

    Args:
        trial_queue (mp.JoinableQueue): Queue containing trials to run.
        main_module (str): The main training module to execute.
        gpu_ids (List[int]): List of GPU IDs to use for each trial.
        inter_trial_delay (float): Seconds to sleep after finishing each trial to prevent
            back-to-back accelerate launches on the same worker.
        retry_delay (float): Seconds to sleep between retry attempts (forwarded to run_trial).
        launch_stagger (float): Minimum seconds between launch start times across workers.
        launch_lock (mp.Lock): Shared lock protecting the launch timestamp.
        next_launch_ts (mp.Value): Shared timestamp of the most recent launch start.
    """
    while True:
        try:
            trial = trial_queue.get(timeout=10)  # Exit if no task in 10 seconds, but no influence on this script
        except:
            break
        _throttle_launch(launch_lock, next_launch_ts, launch_stagger)
        run_trial(trial, main_module, gpu_ids, retry_delay=retry_delay)
        trial_queue.task_done()
        if inter_trial_delay > 0:
            time.sleep(inter_trial_delay)


def schedule_trials(
    trials: List[Trial],
    main_module: str,
    gpu_groups: List[List[int]],
    inter_trial_delay: float,
    retry_delay: float,
    launch_stagger: float,
) -> None:
    """Schedules and runs all trials based on available GPU resources.

    Args:
        trials (List[Trial]): List of trials to run.
        main_module (str): The main training module to execute.
        gpu_groups (List[List[int]]): Explicit GPU groupings to assign per worker.
        inter_trial_delay (float): Seconds to sleep between consecutive trials per worker.
        retry_delay (float): Seconds to sleep between retry attempts inside run_trial.
        launch_stagger (float): Minimum spacing between accelerate launches started by different workers.
    """
    trial_queue = mp.JoinableQueue()

    # Enqueue all trials
    for t in trials:
        trial_queue.put(t)

    processes = []
    if not gpu_groups:
        raise ValueError("At least one GPU group must be specified.")

    launch_lock = mp.Lock()
    next_launch_ts = mp.Value("d", 0.0)

    for gpu_ids in gpu_groups:
        p = mp.Process(
            target=worker,
            args=(
                trial_queue,
                main_module,
                gpu_ids,
                inter_trial_delay,
                retry_delay,
                launch_stagger,
                launch_lock,
                next_launch_ts,
            ),
        )
        p.start()
        processes.append(p)

    trial_queue.join()

    for p in processes:
        p.join()


def get_exp_id(output_root: str) -> str:
    """Generates a new experiment ID based on existing experiments in the output directory.

    Args:
        output_root (str): Root output directory for all experiments.
            The experiment directory is structured as output_root/exp_{:03d}/.

    Returns:
        str: New experiment ID in the format 'exp_{:03d}'.
    """
    if not os.path.exists(output_root):
        return "exp_000"
    existing_exps = [
        d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d)) and d.startswith("exp_")
    ]
    existing_ids = [int(d.split("_")[1]) for d in existing_exps if d.split("_")[1].isdigit()]
    next_id = max(existing_ids, default=-1) + 1
    return f"exp_{next_id:03d}"


def main(
    template_path: Optional[str],
    search_path: Optional[str],
    main_module: str,
    gpu_num: int,
    per_trial_gpu_num: int,
    output_root: str,
    gpu_groups: Optional[List[str]] = None,
    dryrun_exp_id: Optional[str] = None,
    rerun_exp_id: Optional[str] = None,
    inter_trial_delay: float = 0.0,
    retry_delay: float = 0.0,
    launch_stagger: float = 0.0,
) -> None:
    """Overall workflow:
    1. Reads the template YAML and search YAML.
    2. Extracts search items and overrides.
    3. Expands all combinations to generate corresponding Trial objects.
    4. Depending on mode, runs all trials, reruns failed ones, or aggregates results only.

    Args:
        template_path (str | None): Path to the template YAML configuration file.
        search_path (str | None): Path to the search YAML configuration file.
        main_module (str): The main training module to execute.
        gpu_num (int): Total number of available GPUs (used if gpu_groups not provided).
        per_trial_gpu_num (int): Number of GPUs to allocate per trial when deriving groups.
        output_root (str): Root output directory for all experiments.
        gpu_groups (List[str] | None): Explicit GPU groups, e.g., ["0,1", "2,3"].
        dryrun_exp_id (str | None): Existing experiment ID to aggregate without rerunning.
        rerun_exp_id (str | None): Existing experiment ID whose failed trials should rerun.
        inter_trial_delay (float): Delay between trials executed sequentially on the same worker.
        retry_delay (float): Delay between retry attempts of a failed trial.
        launch_stagger (float): Minimum delay enforced between launch start times across workers.
    """
    mode = "standard"
    explicit_exp_id: Optional[str] = None
    if dryrun_exp_id:
        mode = "dryrun"
        explicit_exp_id = dryrun_exp_id
    elif rerun_exp_id:
        mode = "rerun"
        explicit_exp_id = rerun_exp_id

    if mode == "standard" and (template_path is None or search_path is None):
        raise ValueError("--template and --search must be provided for a new experiment run.")

    if explicit_exp_id:
        exp_id = explicit_exp_id
    else:
        exp_id = get_exp_id(output_root)

    exp_dir = os.path.join(output_root, exp_id)
    trials_root = os.path.join(exp_dir, "trials")

    if mode == "standard":
        assert (
            template_path is not None and search_path is not None
        ), "Template and search paths are required for new experiments."
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(trials_root, exist_ok=True)
        template = load_yaml_config(template_path, "Template config")
        search_cfg = load_yaml_config(search_path, "Search config")
        with open(os.path.join(exp_dir, "base.yaml"), "w") as f:
            yaml.safe_dump(template, f, sort_keys=False)
        with open(os.path.join(exp_dir, "search.yaml"), "w") as f:
            yaml.safe_dump(search_cfg, f, sort_keys=False)
    else:
        if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"Experiment directory {exp_dir} does not exist.")
        template_src = template_path or os.path.join(exp_dir, "base.yaml")
        search_src = search_path or os.path.join(exp_dir, "search.yaml")
        template = load_yaml_config(template_src, "Template config")
        search_cfg = load_yaml_config(search_src, "Search config")
        os.makedirs(trials_root, exist_ok=True)

    # extract search items and overrides
    search_items, overrides = extract_search_params(search_cfg)
    print("Search parameters (grid only):")
    if search_items:
        for path, values in search_items.items():
            print(f"  {path}: {values}")
    else:
        print("  <none>")
    # get all combinations in search space
    search_keys = list(search_items.keys())  # list of search parameter paths
    search_value_lists = list(search_items.values())  # list of value lists for each path
    search_combos = list(itertools.product(*search_value_lists))

    # generate a Trial for each combination
    trials = []
    total_trials = len(search_combos)
    for idx, combo in enumerate(search_combos):
        config = deepcopy(template)
        # first apply overrides
        for path, val in overrides.items():
            set_by_path(config, path, val)
        # then set current combination's search items
        search_params = {}
        for k, v in zip(search_keys, combo):
            set_by_path(config, k, v)
            search_params[k] = v
        trial_out_dir = os.path.join(trials_root, f"trial_{idx:03d}")
        config["output_dir"] = trial_out_dir
        trials.append(Trial(idx, config, search_params, trial_out_dir, total_trials))

    print(f"Total trials: {len(trials)}")

    if mode != "standard":
        update_trial_statuses(trials)

    if mode == "dryrun":
        completed = sum(1 for t in trials if t.success)
        print(f"[Dryrun] {completed}/{len(trials)} trials already have results. Skipping execution.")
    else:
        gpu_group_list = resolve_gpu_groups(gpu_groups, gpu_num, per_trial_gpu_num)
        if mode == "rerun":
            pending = [t for t in trials if not t.success]
            print(f"[Rerun] {len(pending)} pending trials out of {len(trials)} total.")
        else:
            pending = trials
        if pending:
            schedule_trials(
                pending,
                main_module,
                gpu_group_list,
                inter_trial_delay,
                retry_delay,
                launch_stagger,
            )
        else:
            print("No trials require execution.")
        update_trial_statuses(trials)

    # collect results into CSV
    update_trial_statuses(trials)
    result_rows = []
    for t in trials:
        row: Dict[str, Any] = {"trial_idx": t.idx}
        row.update(t.search_params)
        if t.success:
            # remove 'test_' prefix from metric keys
            for mk, mv in t.metrics.items():
                row[mk.replace("test_", "")] = mv
        else:
            # if failed, set metric columns to None
            for mk in t.metrics.keys():
                row[mk.replace("test_", "")] = None
        result_rows.append(row)
    df = pd.DataFrame(result_rows)
    csv_path = os.path.join(exp_dir, "test_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--template", help="Path to the template config yaml file.")
    parser.add_argument("--search", help="Path to the search config yaml file.")
    parser.add_argument("--main", required=True, help="Python module to run, e.g., genrec.main_seqrec")
    parser.add_argument("--gpu_num", type=int, default=4, help="Total number of GPUs available.")
    parser.add_argument("--per_trial_gpu_num", type=int, default=1, help="Number of GPUs per trial.")
    parser.add_argument(
        "--output_root",
        default="./outputs",
        help="Root output directory for all experiments.",
    )
    parser.add_argument(
        "--gpu_groups",
        action="append",
        help="Explicit GPU groups (comma-separated ids) to assign per worker. Repeat per group.",
    )
    parser.add_argument(
        "--inter_trial_delay",
        type=float,
        default=10.0,
        help="Seconds to sleep between consecutive trials per worker to avoid accelerate port conflicts.",
    )
    parser.add_argument(
        "--retry_delay",
        type=float,
        default=10.0,
        help="Seconds to sleep between retry attempts of the same trial.",
    )
    parser.add_argument(
        "--launch_stagger",
        type=float,
        default=10.0,
        help="Seconds of spacing enforced between accelerate launches started in parallel workers.",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dryrun",
        metavar="EXP_ID",
        help="Aggregate metrics for an existing exp without rerunning.",
    )
    mode_group.add_argument(
        "--rerun",
        metavar="EXP_ID",
        help="Rerun failed trials for an existing experiment.",
    )
    args = parser.parse_args()

    main(
        args.template,
        args.search,
        args.main,
        args.gpu_num,
        args.per_trial_gpu_num,
        args.output_root,
        args.gpu_groups,
        args.dryrun,
        args.rerun,
        args.inter_trial_delay,
        args.retry_delay,
        args.launch_stagger,
    )
