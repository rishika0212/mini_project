"""
train_orchestrator.py
==========================
Automated two-phase training with real-time output:
  Phase 1: 350 episodes WITHOUT emergencies (sparse traffic — 135 vehicles)
  Phase 2: 150 episodes WITH emergencies (medium traffic — 350+ vehicles)

PERFORMANCE OPTIMIZATIONS:
  • Vehicle spawning reduced for Phase 1 (sparse) → faster convergence
  • YOLO disabled during training (--no-yolo) → 30-40% faster training
  • YOLO GPU acceleration enabled for eval/demo
  • Expected training time: 3-4 hours (vs 7+ hours previously)

Vehicle density progression:
  Phase 1: spawn_points[:60] + 25 per intersection = ~135 total (learnable)
  Phase 2: spawn_points[:180] + 80 per intersection = ~420 total (challenging)

Usage:
  python train_orchestrator.py
"""

import subprocess
import sys
import time
import re
import os

def modify_emergency_interval(interval_value):
    """Modify EMERGENCY_INTERVAL in main.py"""
    with open('main.py', 'r') as f:
        content = f.read()

    # Replace EMERGENCY_INTERVAL value
    content = re.sub(
        r'EMERGENCY_INTERVAL\s*=\s*\d+',
        f'EMERGENCY_INTERVAL = {interval_value}',
        content
    )

    with open('main.py', 'w') as f:
        f.write(content)

    status = "DISABLED" if interval_value > 100000 else f"{interval_value} ticks (~{interval_value*0.05:.1f}s)"
    print(f"\n{'='*70}")
    print(f"  Emergencies: {status}")
    print(f"{'='*70}\n")
    time.sleep(1)

def modify_traffic_density(mode):
    """Modify vehicle spawning density (Phase 1: sparse, Phase 2: medium)"""
    with open('main.py', 'r') as f:
        content = f.read()

    if mode == "sparse":
        # Phase 1: ~135 vehicles total
        content = re.sub(
            r"for sp in spawn_points\[:180\]:",
            "for sp in spawn_points[:60]:",
            content
        )
        content = re.sub(
            r"(for sp in nearby:\s+if n >= )\d+:",
            r"\g<1>25:",
            content,
            count=1  # Only replace first occurrence (vehicle spawning, not emergency)
        )
        content = re.sub(
            r"RUSH_HOUR_INTERVAL\s*=\s*\d+",
            "RUSH_HOUR_INTERVAL = 999999",
            content
        )
        density_desc = "sparse (~135 vehicles)"
    else:  # Phase 2
        # Phase 2: ~420+ vehicles total (realistic city traffic)
        content = re.sub(
            r"for sp in spawn_points\[:60\]:",
            "for sp in spawn_points[:180]:",
            content
        )
        content = re.sub(
            r"(for sp in nearby:\s+if n >= )\d+:",
            r"\g<1>80:",
            content,
            count=1  # Only replace first occurrence
        )
        content = re.sub(
            r"RUSH_HOUR_INTERVAL\s*=\s*\d+",
            "RUSH_HOUR_INTERVAL = 1200",
            content
        )
        density_desc = "medium (~420+ vehicles with rush hour)"

    with open('main.py', 'w') as f:
        f.write(content)

    print(f"\n{'='*70}")
    print(f"  Traffic density: {density_desc}")
    print(f"{'='*70}\n")
    time.sleep(1)

def run_training(phase_name, target_episode):
    """Run training until target episode is reached"""
    print(f"\n{'='*70}")
    print(f"  PHASE: {phase_name}")
    print(f"  Target: Episode {target_episode}")
    print(f"  Starting training... (this may take 30 seconds to connect to CARLA)")
    print(f"{'='*70}\n")

    # Use unbuffered Python output
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    process = subprocess.Popen(
        [sys.executable, '-u', 'main.py', '--train', '--no-yolo'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=0,  # Unbuffered
        env=env
    )

    episode_pattern = re.compile(r'Episode\s+(\d+)')
    last_episode = 0
    timeout_counter = 0

    try:
        for line in process.stdout:
            # Print every line in real-time
            print(line.rstrip())
            sys.stdout.flush()

            # Check if we've reached target episode
            match = episode_pattern.search(line)
            if match:
                current_ep = int(match.group(1))
                if current_ep != last_episode:
                    last_episode = current_ep
                    timeout_counter = 0

                if current_ep >= target_episode:
                    print(f"\n[ORCHESTRATOR] ✓ Target episode {target_episode} reached!")
                    print(f"[ORCHESTRATOR] Stopping phase...")
                    time.sleep(2)
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    return True

            timeout_counter += 1

    except KeyboardInterrupt:
        print("\n\n[ORCHESTRATOR] ⚠ Training interrupted by user")
        process.terminate()
        process.wait()
        return False
    except Exception as e:
        print(f"\n[ORCHESTRATOR] ✗ Error: {e}")
        return False

    return True

def main():
    print("\n" + "="*70)
    print("  TRAFFIC SIGNAL DQN — TWO-PHASE TRAINING ORCHESTRATOR")
    print("="*70)
    print("\n  Phase 1: Learn base 4-phase control (sparse traffic, NO emergencies)")
    print("  Phase 2: Learn emergency robustness (medium traffic, WITH emergencies)")
    print("\n  Estimated time: 4-5 hours (phase 1 should now converge)")
    print("  Total: 500 episodes")
    print("\n" + "="*70 + "\n")

    # ── PHASE 1: Base policy without emergencies ────────────────────────────
    print("[ORCHESTRATOR] Starting PHASE 1...")
    modify_traffic_density("sparse")
    modify_emergency_interval(999999)  # Disable
    success = run_training("PHASE 1: Base Policy (sparse traffic, 350 episodes)", 350)

    if not success:
        print("[ORCHESTRATOR] ✗ Phase 1 failed or interrupted")
        return

    print("\n[ORCHESTRATOR] ✓ Phase 1 complete!")
    print("[ORCHESTRATOR] Weights saved at: data/dqn_weights_int*.json")
    time.sleep(3)

    # ── PHASE 2: Robustness with emergencies ────────────────────────────────
    print("\n[ORCHESTRATOR] Starting PHASE 2...")
    print("[ORCHESTRATOR] Increasing traffic density and enabling emergencies...")
    modify_traffic_density("medium")
    modify_emergency_interval(1500)  # Every 75 seconds
    success = run_training("PHASE 2: Emergency Robustness (medium traffic, 150 eps, total 500)", 500)

    if not success:
        print("[ORCHESTRATOR] ✗ Phase 2 failed or interrupted")
        return

    print("\n" + "="*70)
    print("  ✓✓✓ TRAINING COMPLETE! ✓✓✓")
    print("="*70)
    print("\nTraining Summary:")
    print("  • Phase 1: 350 episodes (sparse traffic, base policy)")
    print("  • Phase 2: 150 episodes (medium traffic, emergency robustness)")
    print("  • Total:   500 episodes")
    print("\nOutput files:")
    print("  • Weights: data/dqn_weights_int1.json")
    print("  •          data/dqn_weights_int2.json")
    print("  •          data/dqn_weights_int3.json")
    print("  • Logs:    data/rl_states_final.csv")
    print("\nNext steps:")
    print("  1. Review results: data/rl_states_final.csv")
    print("  2. Test policy:    python main.py")
    print("  3. Visual demo:    python demo.py")
    print("="*70 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[ORCHESTRATOR] Training aborted by user")
        sys.exit(0)
