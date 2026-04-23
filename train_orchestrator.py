"""
train_orchestrator.py
=====================
Two-phase training orchestrator for the demo.py single-arm DQN system.

Phase 1 — Base policy (no emergencies, sparse traffic)
  • 150 episodes
  • 100 vehicles
  • Emergencies disabled

Phase 2 — Emergency robustness (with emergencies, more traffic)
  • 100 episodes
  • 150 vehicles
  • Emergencies enabled

Usage:
    python train_orchestrator.py
"""

import subprocess
import sys
import time
import re
import os


def run_phase(label, episodes, vehicles, no_emg, timeout_hours=3.0):
    """
    Run main.py --train with given parameters.
    """
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  Episodes : {episodes}   Vehicles : {vehicles}   "
          f"Emergency : {'OFF' if no_emg else 'ON'}")
    print(f"  Expected time : up to {timeout_hours:.1f} hours")
    print(f"{'='*65}\n")
    time.sleep(2)

    cmd = [
        sys.executable, '-u', 'main.py',
        '--train',
        f'--episodes={episodes}',
        f'--vehicles={vehicles}',
    ]
    if no_emg:
        cmd.append('--no-emg')

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=0,
        env=env,
    )

    episode_re    = re.compile(r'Episode\s+(\d+)')
    last_episode  = 0
    reached       = False
    deadline      = time.time() + timeout_hours * 3600

    try:
        for line in process.stdout:
            print(line, end='', flush=True)

            m = episode_re.search(line)
            if m:
                ep = int(m.group(1))
                if ep != last_episode:
                    last_episode = ep

                if ep >= episodes:
                    reached = True
                    print(f"\n[ORCHESTRATOR] Target episode {episodes} reached — stopping phase.")
                    time.sleep(2)
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    return True

            if time.time() > deadline:
                print(f"\n[ORCHESTRATOR] Timeout after {timeout_hours:.1f} h — stopping phase.")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return False

    except KeyboardInterrupt:
        print("\n\n[ORCHESTRATOR] Interrupted by user.")
        process.terminate()
        process.wait()
        return False
    except Exception as exc:
        print(f"\n[ORCHESTRATOR] Error: {exc}")
        return False

    rc = process.wait()
    if reached:
        return True
    print(f"[ORCHESTRATOR] Process exited (code={rc}) before episode {episodes}. "
          f"Last seen: {last_episode}")
    return False


def check_weights():
    path = "data/dqn_weights_int1.json"
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"[ORCHESTRATOR] Weights found: {path} ({size_kb:.1f} KB)")
    else:
        print(f"[ORCHESTRATOR] No weights yet at {path} — fresh start.")


def main():
    print("\n" + "="*65)
    print("  TRAFFIC SIGNAL DQN — TWO-PHASE TRAINING ORCHESTRATOR")
    print("  System: single-arm pressure+DQN (matches demo.py exactly)")
    print("="*65)
    print("""
  Phase 1 : Learn basic arm switching (sparse, no emergencies)
            80 episodes · 80 vehicles · ~35–45 mins

  Phase 2 : Learn emergency robustness (more traffic, emergencies on)
            40 episodes · 120 vehicles · ~20–25 mins

  Total   : 120 episodes · ~1.0 hour
""")

    os.makedirs("data", exist_ok=True)
    check_weights()

    input("  Press ENTER to start training (make sure CARLA is running)...\n")

    # ── Phase 1 ───────────────────────────────────────────────────────────
    print("\n[ORCHESTRATOR] Starting PHASE 1 — base policy...")
    ok = run_phase(
        label         = "PHASE 1: Base policy — sparse traffic, no emergencies",
        episodes      = 80,
        vehicles      = 80,
        no_emg        = True,
        timeout_hours = 2.0,
    )

    if not ok:
        print("\n[ORCHESTRATOR] Phase 1 did not complete cleanly.")
        print("[ORCHESTRATOR] Weights up to the last checkpoint are still usable.")
        ans = input("[ORCHESTRATOR] Continue to Phase 2 anyway? [y/N]: ").strip().lower()
        if ans != 'y':
            print("[ORCHESTRATOR] Stopping. Run demo.py to test what was trained.")
            return

    print("\n[ORCHESTRATOR] Phase 1 complete.")
    check_weights()
    print("[ORCHESTRATOR] Pausing 10 s before Phase 2...")
    time.sleep(10)

    # ── Phase 2 ───────────────────────────────────────────────────────────
    print("\n[ORCHESTRATOR] Starting PHASE 2 — emergency robustness...")
    ok = run_phase(
        label         = "PHASE 2: Emergency robustness — medium traffic, emergencies enabled",
        episodes      = 40,   # 40 new episodes
        vehicles      = 120,
        no_emg        = False,
        timeout_hours = 2.0,
    )

    if not ok:
        print("\n[ORCHESTRATOR] Phase 2 did not complete cleanly.")
        print("[ORCHESTRATOR] Weights up to the last checkpoint are still usable.")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  TRAINING COMPLETE")
    print("="*65)
    check_weights()
    print("""
  Next steps:
    1. Evaluate:   python evaluate.py
    2. Plot:       python plot_results.py
    3. Demo:       python demo.py
""")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ORCHESTRATOR] Aborted.")
        sys.exit(0)
