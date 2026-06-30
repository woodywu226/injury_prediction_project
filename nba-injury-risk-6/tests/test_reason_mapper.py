"""Tests for the reason-string mapping layer (Stage 1 core).

These pin the classifier's behavior on realistic prosportstransactions-style
strings. The strings here mirror the actual phrasing patterns of that source
(e.g. "(out for season)", "placed on IL", "sprained left ankle").
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_injury.reason_mapper import classify  # noqa: E402


def test_achilles_beats_generic_lower_limb():
    # "achilles" must win over generic lower-limb even though it's in the leg.
    lab = classify("torn left Achilles tendon (out for season)")
    assert lab.category == "achilles"
    assert lab.severe_tail is True
    assert lab.time_loss is True
    assert lab.ambiguous is False


def test_ankle_is_lower_limb_soft_tissue():
    lab = classify("sprained left ankle")
    assert lab.category == "lower_limb_soft_tissue"
    assert lab.time_loss is True


def test_acl_is_knee_ligament():
    lab = classify("torn ACL right knee")
    assert lab.category == "knee_ligament"


def test_rest_is_load_management_not_injury():
    lab = classify("rest (DNP)")
    assert lab.time_loss is False
    assert lab.ambiguous is False
    assert lab.category is None


def test_load_management_string():
    lab = classify("load management")
    assert lab.time_loss is False


def test_concussion():
    assert classify("placed in concussion protocol").category == "concussion"


def test_illness_covid():
    assert classify("health and safety protocols").category == "illness"
    assert classify("flu-like symptoms").category == "illness"


def test_back():
    assert classify("lower back spasms").category == "back"


def test_hand_finger():
    assert classify("fractured right thumb").category == "hand_finger"


def test_generic_injury_string_is_time_loss():
    lab = classify("placed on IL")
    assert lab.time_loss is True
    assert lab.ambiguous is False


def test_blank_is_ambiguous():
    lab = classify("")
    assert lab.ambiguous is True
    assert lab.time_loss is False


def test_truly_unknown_is_ambiguous():
    lab = classify("returned to competition under unspecified circumstances xyz")
    # contains no injury keyword and no load-mgmt keyword -> ambiguous
    assert lab.ambiguous is True


def test_priority_knee_ligament_over_generic_knee():
    # "meniscus" should land in knee_ligament, not knee_other
    assert classify("torn meniscus").category == "knee_ligament"
