"""
Concurrent session stress test — verifies SUPPORTS_CONCURRENT_SESSIONS = True.
Spins up 10 simultaneous sessions and confirms they don't cross-contaminate.
"""

import threading
from asha_sahayak.server.asha_environment import AshaEnvironment
from asha_sahayak.models import AshaAction


def run_episode(task_id: str, seed: int, results: dict, key: str):
    env = AshaEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    case_id = env._case.case_id

    action = AshaAction(
        referral_decision="PENDING",
        urgency="unknown",
        primary_concern="gathering_information",
        question="Does the patient have any danger signs like fast breathing or chest indrawing?",
        confidence=0.5,
    )
    obs = env.step(action)
    assert not obs.done, f"Episode ended too early for {key}"

    final_action = AshaAction(
        referral_decision=env._case.correct_referral,
        urgency=env._case.correct_urgency,
        primary_concern=env._case.correct_primary_concern,
        confidence=0.9,
    )
    obs = env.step(final_action)
    assert obs.done, f"Episode should be done for {key}"
    assert obs.reward > 0.0, f"Reward should be positive for correct answer in {key}"
    assert obs.reward_components is not None, f"reward_components missing for {key}"

    results[key] = {
        "case_id": case_id,
        "reward": obs.reward,
        "components": obs.reward_components,
    }


def test_concurrent_sessions():
    """10 concurrent sessions must not cross-contaminate."""
    results = {}
    threads = []

    configs = [
        ("easy", 0), ("easy", 1), ("easy", 2),
        ("medium", 10), ("medium", 11), ("medium", 12),
        ("hard", 20), ("hard", 21), ("hard", 22), ("hard", 23),
    ]

    for task_id, seed in configs:
        key = f"{task_id}_{seed}"
        t = threading.Thread(target=run_episode, args=(task_id, seed, results, key))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == len(configs), f"Expected {len(configs)} results, got {len(results)}"

    # Verify no cross-contamination: each session should have a unique case or same case
    # but independently computed reward
    for key, r in results.items():
        assert r["reward"] > 0.0, f"Zero reward for {key} — possible session contamination"
        assert "referral" in r["components"], f"Missing referral component for {key}"

    print(f"\nConcurrent session test passed — {len(results)} sessions ran cleanly")
    for key, r in results.items():
        print(f"  {key}: case={r['case_id']} reward={r['reward']:.3f} "
              f"referral={r['components']['referral']:.2f} "
              f"urgency={r['components']['urgency']:.2f}")


if __name__ == "__main__":
    test_concurrent_sessions()
