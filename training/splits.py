import numpy as np


def subject_session_split(labels, test_session, seed=42, val_trials=8, windows_per_trial=10):
    """Leave one session out and keep every trial intact in train/validation."""
    if labels.shape != (600, 2):
        raise ValueError(f"Expected 600 one-hot labels, got {labels.shape}")
    if test_session not in (0, 1, 2):
        raise ValueError(f"test_session must be 0, 1, or 2, got {test_session}")
    if val_trials <= 0 or val_trials >= 40 or val_trials % 2:
        raise ValueError("val_trials must be a positive even number below 40")

    all_trial_ids = np.arange(60)
    test_trial_ids = np.arange(test_session * 20, (test_session + 1) * 20)
    pool_trial_ids = np.setdiff1d(all_trial_ids, test_trial_ids)
    trial_class = np.argmax(labels[::windows_per_trial], axis=1)
    rng = np.random.RandomState(seed)

    val_trial_ids = []
    for class_id in (0, 1):
        candidates = pool_trial_ids[trial_class[pool_trial_ids] == class_id]
        selected = rng.choice(candidates, size=val_trials // 2, replace=False)
        val_trial_ids.extend(int(value) for value in selected)
    val_trial_ids = np.asarray(val_trial_ids, dtype=int)
    rng.shuffle(val_trial_ids)
    train_trial_ids = np.setdiff1d(pool_trial_ids, val_trial_ids)

    def expand(trials):
        return np.concatenate(
            [np.arange(trial * windows_per_trial, (trial + 1) * windows_per_trial) for trial in trials]
        ).astype(int)

    split = {
        "train_indices": expand(train_trial_ids),
        "val_indices": expand(val_trial_ids),
        "test_indices": expand(test_trial_ids),
        "train_trial_ids": train_trial_ids,
        "val_trial_ids": val_trial_ids,
        "test_trial_ids": test_trial_ids,
    }
    index_sets = [set(split[key].tolist()) for key in ("train_indices", "val_indices", "test_indices")]
    if index_sets[0] & index_sets[1] or index_sets[0] & index_sets[2] or index_sets[1] & index_sets[2]:
        raise AssertionError("Train/validation/test indices overlap")
    if sum(len(values) for values in index_sets) != 600:
        raise AssertionError("Split does not cover all 600 windows")
    return split
