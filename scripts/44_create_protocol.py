#!/usr/bin/env python
"""Create the immutable Experiment 44 protocol; never overwrite an existing one."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results' / 'scarcity' / 'confirmatory_34' / 'protocol.json'
TARGET = ROOT / 'results' / 'robustness_multisystem_44' / 'protocol.json'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if TARGET.exists():
        raise FileExistsError(f'Protocol already exists; refusing overwrite: {TARGET}')
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    old = json.loads(SOURCE.read_text(encoding='utf-8'))
    for name, expected in old['source_sha256'].items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f'Frozen source integrity check failed: {name}')
    seeds = old['confirmatory_seeds']
    systems = old['systems']
    if len(seeds) != 15 or len(systems) != 3:
        raise RuntimeError('Unexpected frozen protocol dimensions')
    noise = [0.0, 0.01, 0.05, 0.10, 0.20]
    protocol = {
        'experiment': 44,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'preregistered_before_new_data_generation_and_training',
        'purpose': 'Multisystem observational-noise robustness extension of experiments 29-42',
        'independence_note': 'Reuses trajectories and seeds examined in experiment 34; not an independent replication. Lorenz exploratory noise results informed this design.',
        'source_protocol': str(SOURCE.relative_to(ROOT)).replace('\\', '/'),
        'source_protocol_sha256': sha256(SOURCE),
        'frozen_source_sha256': old['source_sha256'],
        'systems': systems,
        'seeds': seeds,
        'exploratory_seeds_excluded': old['exploratory_seeds_excluded_from_confirmatory_primary_test'],
        'observation_level': 'obs_100',
        'time_points': 160,
        'noise_standard_deviations_absolute': noise,
        'noise_labels': ['noise_000', 'noise_001', 'noise_005', 'noise_010', 'noise_020'],
        'noise_model': 'Additive independent zero-mean Gaussian noise per coordinate, trajectory, and observed time point; generate exclusively from x_clean, not x.',
        'noise_rng_policy': 'Separate deterministic RNG streams per system, seed, noise level and split; log generator identifiers and hashes in data manifests. Same noisy datasets for all methods. Noise=0 is exactly clean.',
        'source_data_root': 'data/scarcity',
        'data_root': 'data/robustness_multisystem_44',
        'results_root': 'results/robustness_multisystem_44',
        'methods': ['neural_ode', 'neural_distillation', 'direct_sindy'],
        'train_val_test_policy': 'Training and model selection use train and val only; test is opened only after all hyperparameters and models for a condition are fixed. Never use test to tune.',
        'training_policy': 'Preserve experiment 30 architecture, trajectory-only loss, normalization from noisy train only, 0.8-duration windows, optimization, validation selection and stopping. Pair initial weights and minibatch schedule within system/seed across noise levels.',
        'discovery_policy': 'Preserve experiment 31 polynomial degrees, thresholds, ridge, derivative estimators, 0.8 validation horizon and complexity penalty. Select from noisy validation only.',
        'primary_endpoint': 'Mean short-horizon test trajectory relative L2 over 5 clean test trajectories x 8 windows, per seed and model; use clean initial states.',
        'primary_contrast': 'neural_distillation minus direct_sindy; negative favors distillation',
        'primary_hypotheses': [
            {'system': system, 'noise_std': 0.20, 'alternative': 'two-sided difference between neural_distillation and direct_sindy'}
            for system in systems
        ],
        'primary_inference': 'Paired seed-level comparisons at noise_std=0.20 for each of three systems; two-sided exact paired sign-flip permutation test where feasible, Holm adjustment across three hypotheses; report paired effect sizes and uncertainty intervals. No claim of superiority from significance alone.',
        'secondary_analyses': [
            'Clean-start trajectory error at other noise levels and noise-response curves',
            'Noisy-start trajectory error with independent prespecified test-start perturbations shared by all methods, evaluated against clean reference',
            'Vector-field relative L2 on identical clean test states',
            'Symbolic structural F1 using each system true support',
            'Integration failure rates and successful-window means (sensitivity only)',
        ],
        'noisy_start_policy': 'For each training-noise condition, perturb each clean test window initial state with independent zero-mean Gaussian noise of the same absolute std; fixed deterministic noise per system/seed/noise/window, shared across methods. Reference trajectory remains clean. Do not use these perturbations in selection.',
        'test_evaluation_policy': 'Preserve experiment 32 eight 0.8-horizon windows per five clean test trajectories, RK45 rtol=1e-5 atol=1e-7 max_step=0.08; fixed clean reference and same states for field metric.',
        'failure_policy': 'Record every failure. If any of 40 windows fails, model aggregate trajectory error is +Infinity; failed method ranks worse than finite method, both failed tied. Report failure-aware comparisons and finite-only sensitivity with denominators. Never silently drop/cap failures.',
        'inference_unit': 'seed; windows and trajectories are repeated measurements, not independent replicates',
        'planned_data_conditions': len(systems)*len(seeds)*len(noise),
        'planned_neural_models': len(systems)*len(seeds)*len(noise),
        'planned_symbolic_models': len(systems)*len(seeds)*len(noise)*2,
        'planned_model_evaluations_per_start_policy': len(systems)*len(seeds)*len(noise)*3,
        'planned_test_windows_per_start_policy': len(systems)*len(seeds)*len(noise)*3*5*8,
        'integrity_policy': 'Never modify data/scarcity, results/scarcity or original scripts. Never overwrite a completed new artifact. Record source and output checksums. Pilot with one system/seed before full run.',
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with TARGET.open('x', encoding='utf-8') as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print('CREATED:', TARGET)
    print('SHA256:', sha256(TARGET))
    print('Conditions:', protocol['planned_data_conditions'])
    print('Neural models:', protocol['planned_neural_models'])
    print('Symbolic models:', protocol['planned_symbolic_models'])
    print('Test windows per start policy:', protocol['planned_test_windows_per_start_policy'])
    print('No training or data generation performed.')


if __name__ == '__main__':
    main()
