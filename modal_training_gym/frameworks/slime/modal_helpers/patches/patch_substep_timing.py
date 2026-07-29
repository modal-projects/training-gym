from pathlib import Path


ROOT = Path("/root/slime")
IMPORT = (
    "from modal_training_gym.frameworks.slime.substep_timing import "
    "begin_phase as _tg_begin_timing, "
    "create_driver_timing as _tg_create_timing, "
    "finish_phase as _tg_finish_timing, "
    "finish_role_timing as _tg_finish_role_timing, "
    "start_role_timing as _tg_start_role_timing, "
    "timed_await as _tg_timed_await\n"
)


def replace_once(source: str, old: str, new: str, path: Path) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}")
    return source.replace(old, new, 1)


def patch_entrypoint(path: Path, asynchronous: bool) -> None:
    source = path.read_text()
    if "_tg_create_timing" in source:
        return
    source = replace_once(source, "import ray\n", f"import ray\n\n{IMPORT}", path)
    source = replace_once(
        source,
        "    init_tracking(args)\n",
        "    init_tracking(args)\n    _tg_timing = _tg_create_timing(args)\n",
        path,
    )
    source = replace_once(
        source,
        "    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)\n",
        "    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)\n"
        "    _tg_timing.configure(args.start_rollout_id, args.num_rollout)\n"
        "    _tg_timing.configure_rollout_manager(rollout_manager)\n",
        path,
    )
    source = replace_once(
        source,
        "    for rollout_id in range(args.start_rollout_id, args.num_rollout):\n",
        "    for rollout_id in range(args.start_rollout_id, args.num_rollout):\n"
        "        _tg_timing.start_step()\n",
        path,
    )
    step_boundary = (
        "        _tg_report('weight_sync', args, rollout_id, 'substep_finish')\n"
    )
    publish = (
        step_boundary
        + "\n"
        + "        _tg_timing.publish_step(\n"
        + "            rollout_id,\n"
        + "            actor_ran=not args.debug_rollout_only and "
        "((not args.use_critic) or rollout_id >= args.num_critic_only_steps),\n"
        + "            critic_ran=not args.debug_rollout_only and args.use_critic,\n"
        + "            rollout_ran=not args.debug_train_only,\n"
        + "        )\n"
    )
    source = replace_once(source, step_boundary, publish, path)
    if asynchronous:
        source = replace_once(
            source,
            "            rollout_data_curr_ref = ray.get(rollout_data_next_future)\n",
            '            _tg_wait_timing = _tg_begin_timing("wait_for_rollout")\n'
            "            rollout_data_curr_ref = ray.get(rollout_data_next_future)\n"
            "            _tg_finish_timing(_tg_wait_timing)\n",
            path,
        )
        source = replace_once(
            source,
            "        if args.use_critic:\n"
            "            actor_trains_this_step = "
            "rollout_id >= args.num_critic_only_steps\n",
            '        _tg_training_timing = _tg_begin_timing("train_models")\n'
            "        if args.use_critic:\n"
            "            actor_trains_this_step = "
            "rollout_id >= args.num_critic_only_steps\n",
            path,
        )
        source = replace_once(
            source,
            "        # PATCHED_TRAINING_GYM_STEP_FINISH: training step finish\n",
            "        _tg_finish_timing(_tg_training_timing)\n\n"
            "        # PATCHED_TRAINING_GYM_STEP_FINISH: training step finish\n",
            path,
        )
        source = replace_once(
            source,
            "        if should_run_periodic_action(rollout_id, args.save_interval, "
            "num_rollout_per_epoch, args.num_rollout):\n",
            "        _tg_checkpoint_timing = None\n"
            "        if should_run_periodic_action(rollout_id, args.save_interval, "
            "num_rollout_per_epoch, args.num_rollout):\n"
            '            _tg_checkpoint_timing = _tg_begin_timing("checkpoint_save")\n'
            "            _tg_report('checkpoint_save', args, rollout_id)\n",
            path,
        )
        source = replace_once(
            source,
            "        if (rollout_id + 1) % args.update_weights_interval == 0:\n",
            "        _tg_finish_timing(_tg_checkpoint_timing)\n"
            "        _tg_report('weight_sync', args, rollout_id)\n\n"
            "        if (rollout_id + 1) % args.update_weights_interval == 0:\n",
            path,
        )
        source = replace_once(
            source,
            "            # sync generate before update weights to prevent update "
            "weight in the middle of generation\n"
            "            rollout_data_curr_ref = ray.get(x) if "
            "(x := rollout_data_next_future) is not None else None\n",
            "            # sync generate before update weights to prevent update "
            "weight in the middle of generation\n"
            '            _tg_timing.transition("wait_for_next_rollout")\n'
            "            rollout_data_curr_ref = ray.get(x) if "
            "(x := rollout_data_next_future) is not None else None\n"
            '            _tg_timing.transition("weight_sync")\n',
            path,
        )
    path.write_text(source)


def patch_actor(path: Path) -> None:
    source = path.read_text()
    if "_tg_start_role_timing" in source:
        return
    source = replace_once(source, "import torch\n", f"import torch\n\n{IMPORT}", path)
    source = replace_once(
        source,
        "    def train(self, rollout_id: int, rollout_data_ref: Box, "
        "external_data=None):\n"
        "        if self.args.debug_rollout_only:\n"
        "            return None\n\n"
        "        if self.args.offload_train:\n"
        "            self.wake_up()\n",
        "    def train(self, rollout_id: int, rollout_data_ref: Box, "
        "external_data=None):\n"
        "        if self.args.debug_rollout_only:\n"
        "            return None\n\n"
        "        _tg_role_timing = (\n"
        "            _tg_start_role_timing(self.args, rollout_id, self.role)\n"
        "            if is_megatron_main_rank()\n"
        "            else None\n"
        "        )\n\n"
        "        if self.args.offload_train:\n"
        "            self.wake_up()\n",
        path,
    )
    source = replace_once(
        source,
        "        return result\n\n    def train_critic",
        "        _tg_finish_role_timing(self.args, rollout_id, _tg_role_timing)\n"
        "        return result\n\n"
        "    def train_critic",
        path,
    )
    path.write_text(source)


def patch_model(path: Path) -> None:
    source = path.read_text()
    if "_tg_forward_backward_timing" in source:
        return
    source = replace_once(source, "import torch\n", f"import torch\n\n{IMPORT}", path)
    source = replace_once(
        source,
        "    losses_reduced = forward_backward_func(\n",
        '    _tg_forward_backward_timing = _tg_begin_timing("forward_backward")\n'
        "    losses_reduced = forward_backward_func(\n",
        path,
    )
    source = replace_once(
        source,
        "        forward_only=False,\n    )\n\n    valid_step = True",
        "        forward_only=False,\n"
        "    )\n"
        "    _tg_finish_timing(_tg_forward_backward_timing)\n\n"
        "    valid_step = True",
        path,
    )
    source = replace_once(
        source,
        "        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()\n",
        '        _tg_optimizer_timing = _tg_begin_timing("optimizer_step")\n'
        "        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()\n"
        "        _tg_finish_timing(_tg_optimizer_timing)\n",
        path,
    )
    path.write_text(source)


def patch_rollout(path: Path) -> None:
    source = path.read_text()
    if "_tg_rollout_timing" in source:
        return
    source = replace_once(source, "import torch\n", f"import torch\n\n{IMPORT}", path)
    source = replace_once(
        source,
        "    def generate(self, rollout_id):\n",
        "    def configure_training_gym_timing(\n"
        "        self, first_rollout_id, rollout_id_stop_exclusive\n"
        "    ):\n"
        "        self.args.first_rollout_id = first_rollout_id\n"
        "        self.args.rollout_id_stop_exclusive = rollout_id_stop_exclusive\n"
        "        self.args.training_gym_timing_boundary_ready = True\n\n"
        "    def generate(self, rollout_id):\n",
        path,
    )
    source = replace_once(
        source,
        "    def generate(self, rollout_id):\n        start_time = time.time()\n",
        "    def generate(self, rollout_id):\n"
        "        _tg_rollout_timing = _tg_start_role_timing("
        'self.args, rollout_id, "rollout")\n'
        "        start_time = time.time()\n",
        path,
    )
    source = replace_once(
        source,
        "        data, metrics = self._get_rollout_data(rollout_id=rollout_id)\n",
        '        _tg_generate_timing = _tg_begin_timing("generate_rollouts")\n'
        "        data, metrics = self._get_rollout_data(rollout_id=rollout_id)\n"
        "        _tg_finish_timing(_tg_generate_timing)\n",
        path,
    )
    source = replace_once(
        source,
        "            return\n        data = self._convert_samples_to_train_data(data)\n",
        "            _tg_finish_role_timing("
        "self.args, rollout_id, _tg_rollout_timing)\n"
        "            return\n"
        "        data = self._convert_samples_to_train_data(data)\n",
        path,
    )
    source = replace_once(
        source,
        "        return self._split_train_data_by_dp(data)\n",
        "        result = self._split_train_data_by_dp(data)\n"
        "        _tg_finish_role_timing(self.args, rollout_id, _tg_rollout_timing)\n"
        "        return result\n",
        path,
    )
    source = replace_once(
        source,
        "            return self.custom_reward_post_process_func(self.args, samples)\n",
        "            _tg_reward_post_process_timing = _tg_begin_timing("
        '"custom_reward_post_process")\n'
        "            result = self.custom_reward_post_process_func(self.args, samples)\n"
        "            _tg_finish_timing(_tg_reward_post_process_timing)\n"
        "            return result\n",
        path,
    )
    path.write_text(source)


def patch_reward(path: Path) -> None:
    source = path.read_text()
    if "_tg_timed_await" in source:
        return
    source = replace_once(source, "import random\n", f"import random\n\n{IMPORT}", path)
    source = source.replace(
        "return await rm_function(args, sample, **kwargs)",
        "return await _tg_timed_await("
        '"custom_reward", rm_function(args, sample, **kwargs))',
    )
    source = source.replace(
        "return await rm_function(args, samples, **kwargs)",
        "return await _tg_timed_await("
        '"custom_reward", rm_function(args, samples, **kwargs))',
    )
    path.write_text(source)


def main() -> None:
    patch_entrypoint(ROOT / "train.py", asynchronous=False)
    patch_entrypoint(ROOT / "train_async.py", asynchronous=True)
    patch_actor(ROOT / "slime/backends/megatron_utils/actor.py")
    patch_model(ROOT / "slime/backends/megatron_utils/model.py")
    patch_rollout(ROOT / "slime/ray/rollout.py")
    patch_reward(ROOT / "slime/rollout/rm_hub/__init__.py")


if __name__ == "__main__":
    main()
