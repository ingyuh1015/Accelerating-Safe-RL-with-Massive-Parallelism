from __future__ import annotations
import argparse
from collections.abc import Sequence
from archieve.utils import available_presets, available_training_algorithms, backend

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='ARCHIEVE safe reinforcement learning')
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='Train a policy')
    training_algorithms = available_training_algorithms()

    def load_choice_value(value: str) -> str:
        normalized = str(value).strip().lower()
        choice_map = {choice.lower(): choice for choice in training_algorithms}
        if normalized in choice_map:
            return choice_map[normalized]
        raise argparse.ArgumentTypeError(f"invalid choice: {value!r} (choose from {', '.join((repr(choice) for choice in training_algorithms))})")

    train.add_argument('--algo', type=load_choice_value, choices=training_algorithms, required=True, help='Training algorithm: CPO, TRPOLag, or CRPO. Matching is case-insensitive.')
    train.add_argument('--task', default='SafetyPointGoal1')
    train.add_argument('--preset', choices=available_presets(), default=None, help='Rollout and return-estimation preset. Individual flags override.')
    train.add_argument('--backend', choices=['auto', 'tpu', 'gpu', 'cpu'], default='auto')
    train.add_argument('--wandb', choices=['true', 'false'], default=None)
    train.add_argument('--seed', type=int, default=0)
    train.add_argument('--seeds', default=None)
    train.add_argument('--distributed', action='store_true', help='Initialize Cloud TPU multi-host execution and divide the same global seed list across hosts.')
    train.add_argument('--total-steps', type=int, default=None)
    train.add_argument('--num-envs', type=int, default=None)
    train.add_argument('--rollout-horizon', dest='rollout_horizon_override', type=int, default=None, help='Override per-env rollout horizon. Sets rollout_steps_per_update = num_envs * rollout_horizon before scale.')
    train.add_argument('--logging', choices=['true', 'false'], default=None)
    train.add_argument('--scale', type=int, default=None, help='Scale factor (>=2): num_envs, rollout_steps_per_update, update_batch_size all multiplied by N.')
    train.add_argument('--staggered-scale-axis', dest='staggered_scale_axis', choices=['env', 'horizon'], default=None, help="How --scale is spent for staggered rollouts. 'env' (default) scales both num_envs and rollout steps so per-env horizon is unchanged. 'horizon' keeps num_envs fixed and spends scale on a longer per-env rollout horizon.")
    train.add_argument('--redun', type=int, default=None, help='Constraint redundancy (>=1, default 1): replicate every task constraint N times. Env emits N identical cost copies, cost critic gets N output heads, Lagrangian/projection solver sees N copies of the same gradient.')
    train.add_argument('--seeds-per-device', type=int, default=None, help='Seeds per device via vmap. None=pmap only. E.g. 4 with 16 seeds on 4 devices = pmap(4)×vmap(4).')
    train.add_argument('--lagrange-lr', type=float, default=None, help='Override lagrange_lr.')
    train.add_argument('--lagrange-multiplier-init', type=float, default=None, help='Override lagrange_multiplier_init.')
    train.add_argument('--update-epochs', type=int, default=None, help='Override update_epochs (critic/actor minibatch epochs per update).')
    train.add_argument('--update-batch-size', type=int, default=None, help='Override update_batch_size (critic/actor minibatch size).')
    train.add_argument('--critic-hidden-sizes', dest='critic_hidden_sizes', default=None, help='Comma-separated critic widths, e.g. 64,64. Defaults to the shared hidden_sizes configuration.')
    train.add_argument('--critic-lr', dest='critic_lr', type=float, default=None, help='Override critic_lr.')
    train.add_argument('--critic-activation', dest='critic_activation', choices=['swish', 'tanh', 'relu', 'leaky_relu'], default=None, help="Critic hidden activation. Defaults to the shared network configuration.")
    train.add_argument('--critic-weight-init', dest='critic_weight_init', choices=['lecun_normal', 'kaiming_uniform', 'kaiming_standard', 'orthogonal'], default=None, help='Critic kernel initialization. Defaults to the shared network configuration.')
    train.add_argument('--critic-layer-norm', dest='critic_layer_norm', choices=['true', 'false'], default=None, help='Critic LayerNorm. Defaults to the shared network configuration.')
    train.add_argument('--use-critic-norm', dest='use_critic_norm', choices=['true', 'false'], default=None, help="Add critic_norm_coef * ||theta||^2 to the critic loss.")
    train.add_argument('--critic-norm-coef', dest='critic_norm_coef', type=float, default=None, help='Coefficient on the critic weight decay (default 1e-3).')
    train.add_argument('--tighten', dest='tighten', choices=['on', 'off'], default='off', help="PIDR-phase constraint tightening: 'off' (default, plain training) or 'on' (density-ratio phase-init tightening of the cost signal).")
    train.add_argument('--tighten-terms', dest='tighten_terms', choices=['both', 'beta', 'epsilon'], default='both', help="Which PIDR tightening terms are applied to the cost signal: 'both' (surrogate+beta+epsilon), 'beta' (surrogate+beta only), 'epsilon' (surrogate+epsilon only). beta and epsilon are still both computed/logged regardless.")
    train.add_argument('--sequential', action='store_true', help='Compile once and run training seeds sequentially.')
    evaluate = commands.add_parser('eval', help='Evaluate a saved policy')
    evaluate.add_argument('--run-dir', required=True, help='Training run directory containing config.yaml and model/.')
    evaluate.add_argument('--episodes', type=int, default=10)
    evaluate.add_argument('--num-envs', type=int, default=8)
    evaluate.add_argument('--backend', choices=['auto', 'cpu', 'gpu', 'tpu'], default='auto')
    evaluate.add_argument('--seed', type=int, default=0)
    evaluate.add_argument('--stochastic', action='store_true', help='Sample actions instead of using tanh(mean).')
    evaluate.add_argument('--output', default=None, help='Optional JSON result path.')
    return parser

def main(argv: Sequence[str] | None=None) -> None:
    parser = get_parser()
    args = parser.parse_args(argv)
    if args.command == 'eval' and (args.episodes < 1 or args.num_envs < 1):
        parser.error('--episodes and --num-envs must be positive')
    selected = backend.bootstrap_backend(args.backend)
    if args.command == 'train':
        args.seed_list = tuple(int(token.strip()) for token in args.seeds.split(',') if token.strip()) if args.seeds is not None else (args.seed,)
        if not args.seed_list:
            parser.error('--seeds provided but empty')
        if len(set(args.seed_list)) != len(args.seed_list):
            parser.error('--seeds must contain distinct values')
        if args.distributed:
            if args.sequential:
                parser.error('--distributed requires parallel seed execution')
            backend.initialize_distributed()
        else:
            backend.configure_host_device_count(selected, 1 if args.sequential else len(args.seed_list))
        from archieve.train import build_algo_train_config, run_training
        config = build_algo_train_config(args, seed=args.seed_list[0])
        run_training(args, config)
    else:
        backend.configure_host_device_count(selected, 1)
        from archieve.eval import run_evaluation
        run_evaluation(args)

if __name__ == '__main__':
    main()
