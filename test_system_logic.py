#!/usr/bin/env python3
"""
test_system_logic.py
====================
Standalone test to validate system controller logic without CARLA.
"""

import numpy as np
from system_controller import TrafficSystemController, SystemMode
from dqn_agent import DQNAgent, build_state
from fallback import FallbackController, ControlMode


def test_system_initialization():
    """Test that system controller initializes correctly."""
    print("\n[TEST 1] System Controller Initialization")
    agent = DQNAgent()
    controller = TrafficSystemController(intersection_id=1, dqn_agent=agent, num_phases=4)

    assert controller.system_mode == SystemMode.NORMAL, "Should start in NORMAL mode"
    assert controller.intersection_id == 1, "Intersection ID should be set"
    assert controller.fallback is not None, "Fallback controller should exist"
    print("  PASS: Controller initialized correctly")


def test_normal_to_emergency_transition():
    """Test NORMAL -> GRACE -> PRE_CLEAR -> EMERGENCY transition."""
    print("\n[TEST 2] Emergency Detection Transition")
    agent = DQNAgent()
    controller = TrafficSystemController(intersection_id=1, dqn_agent=agent, num_phases=4)

    # Create dummy state
    state = np.zeros(12, dtype=np.float32)

    # Step 1: Detect emergency (should go to GRACE)
    action, mode, override = controller.update(
        state=state,
        yolo_confidence=0.8,
        yolo_count=5,
        emergency_flag=1,
        emg_vehicle=None
    )
    assert mode == SystemMode.GRACE, f"Should be in GRACE, got {mode}"
    assert controller.grace_counter == 40, "Grace counter should be 40"
    print("  PASS: Normal -> GRACE transition")

    # Step 2: Simulate grace period ticking down
    for _ in range(39):
        action, mode, override = controller.update(
            state=state,
            yolo_confidence=0.8,
            yolo_count=5,
            emergency_flag=1,
            emg_vehicle=None
        )
        assert mode == SystemMode.GRACE, "Should stay in GRACE"

    # Step 3: After grace expires, should transition to PRE_CLEAR
    action, mode, override = controller.update(
        state=state,
        yolo_confidence=0.8,
        yolo_count=5,
        emergency_flag=1,
        emg_vehicle=None
    )
    assert mode == SystemMode.PRE_CLEAR, f"Should be in PRE_CLEAR, got {mode}"
    print("  PASS: GRACE -> PRE_CLEAR transition")

    # Step 4: Simulate pre-clear period
    for _ in range(39):
        action, mode, override = controller.update(
            state=state,
            yolo_confidence=0.8,
            yolo_count=5,
            emergency_flag=1,
            emg_vehicle=None
        )
        assert mode == SystemMode.PRE_CLEAR, "Should stay in PRE_CLEAR"

    # Step 5: After pre-clear expires, should transition to EMERGENCY
    action, mode, override = controller.update(
        state=state,
        yolo_confidence=0.8,
        yolo_count=5,
        emergency_flag=1,
        emg_vehicle=None
    )
    assert mode == SystemMode.EMERGENCY, f"Should be in EMERGENCY, got {mode}"
    print("  PASS: PRE_CLEAR -> EMERGENCY transition")


def test_low_confidence_fallback():
    """Test NORMAL -> FIXED_TIME transition on low confidence."""
    print("\n[TEST 3] Low Confidence Fallback")
    agent = DQNAgent()
    controller = TrafficSystemController(intersection_id=1, dqn_agent=agent, num_phases=4)

    state = np.zeros(12, dtype=np.float32)

    # Inject low confidence for 4 ticks (not enough yet)
    for i in range(4):
        action, mode, override = controller.update(
            state=state,
            yolo_confidence=0.30,  # Below threshold 0.35
            yolo_count=0,
            emergency_flag=0,
            emg_vehicle=None
        )
        assert mode == SystemMode.NORMAL, f"Tick {i}: Should still be NORMAL (need 5 low-conf ticks)"

    # 5th tick should trigger FIXED_TIME
    action, mode, override = controller.update(
        state=state,
        yolo_confidence=0.30,
        yolo_count=0,
        emergency_flag=0,
        emg_vehicle=None
    )
    assert mode == SystemMode.FIXED_TIME, f"Should transition to FIXED_TIME, got {mode}"
    print("  PASS: Low confidence triggers FIXED_TIME after 5 ticks")


def test_recovery_to_normal():
    """Test RECOVERY -> NORMAL transition."""
    print("\n[TEST 4] Recovery Phase Completion")
    agent = DQNAgent()
    controller = TrafficSystemController(intersection_id=1, dqn_agent=agent, num_phases=4)

    state = np.zeros(12, dtype=np.float32)

    # Force into RECOVERY manually for testing
    controller.system_mode = SystemMode.RECOVERY
    controller.recovery_counter = 5  # Set to 5 ticks remaining

    # Tick down the recovery counter
    for i in range(4):
        action, mode, override = controller.update(
            state=state,
            yolo_confidence=0.8,
            yolo_count=5,
            emergency_flag=0,
            emg_vehicle=None
        )
        assert mode == SystemMode.RECOVERY, f"Tick {i}: Should stay in RECOVERY"
        assert controller.recovery_counter == 4 - i, f"Recovery counter should decrement"

    # Last tick should transition to NORMAL
    action, mode, override = controller.update(
        state=state,
        yolo_confidence=0.8,
        yolo_count=5,
        emergency_flag=0,
        emg_vehicle=None
    )
    assert mode == SystemMode.NORMAL, f"Should transition to NORMAL, got {mode}"
    print("  PASS: RECOVERY -> NORMAL after counter expires")


def test_emergency_clears():
    """Test emergency vehicle clearing (transition to RECOVERY)."""
    print("\n[TEST 5] Emergency Vehicle Clearing")
    agent = DQNAgent()
    controller = TrafficSystemController(intersection_id=1, dqn_agent=agent, num_phases=4)

    state = np.zeros(12, dtype=np.float32)

    # Go through GRACE -> PRE_CLEAR -> EMERGENCY
    # (same as test 2, abbreviated)
    for _ in range(40 + 40):
        action, mode, override = controller.update(
            state=state,
            yolo_confidence=0.8,
            yolo_count=5,
            emergency_flag=1,
            emg_vehicle=None
        )

    assert controller.system_mode == SystemMode.EMERGENCY, "Should be in EMERGENCY"

    # Now clear the emergency vehicle (emergency_flag=0)
    action, mode, override = controller.update(
        state=state,
        yolo_confidence=0.8,
        yolo_count=5,
        emergency_flag=0,  # Vehicle cleared
        emg_vehicle=None
    )
    assert mode == SystemMode.RECOVERY, f"Should transition to RECOVERY on vehicle clear, got {mode}"
    assert controller.recovery_counter == 200, "Recovery counter should be 200"
    print("  PASS: EMERGENCY -> RECOVERY on vehicle clear")


def test_diagnostic_tracking():
    """Test that diagnostics are properly tracked."""
    print("\n[TEST 6] Diagnostic Tracking")
    agent = DQNAgent()
    controller = TrafficSystemController(intersection_id=1, dqn_agent=agent, num_phases=4)

    state = np.zeros(12, dtype=np.float32)

    # Update once
    action, mode, override = controller.update(
        state=state,
        yolo_confidence=0.75,
        yolo_count=3,
        emergency_flag=0,
        emg_vehicle=None
    )

    # Get status
    status = controller.get_status()

    assert status['intersection_id'] == 1, "Intersection ID should be in status"
    assert status['mode'] == 'NORMAL', "Mode should be in status"
    assert status['yolo_confidence'] == 0.75, "Confidence should be tracked"
    assert status['yolo_count'] == 3, "YOLO count should be tracked"
    assert 'mode_history' in status, "Mode history should be in status"
    print("  PASS: Diagnostic tracking working")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("SYSTEM CONTROLLER TEST SUITE")
    print("="*60)

    try:
        test_system_initialization()
        test_normal_to_emergency_transition()
        test_low_confidence_fallback()
        test_recovery_to_normal()
        test_emergency_clears()
        test_diagnostic_tracking()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        return True
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        print("="*60)
        return False
    except Exception as e:
        print(f"\nERROR: {e}")
        print("="*60)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
