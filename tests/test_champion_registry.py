import pytest
from core.champion_registry import (
    normalize_champion_name,
    champion_to_id,
    id_to_champion,
    NUM_CHAMPIONS,
    UNK_CHAMPION_ID,
    CANONICAL_CHAMPIONS,
)

def test_champion_registry_constants():
    assert NUM_CHAMPIONS >= 168
    assert UNK_CHAMPION_ID == 0
    assert len(CANONICAL_CHAMPIONS) == NUM_CHAMPIONS
    assert CANONICAL_CHAMPIONS[0] == "UNKNOWN"

def test_normalize_champion_name_aliases():
    assert normalize_champion_name("Wukong") == "MonkeyKing"
    assert normalize_champion_name("MonkeyKing") == "MonkeyKing"
    assert normalize_champion_name("Nunu & Willump") == "Nunu"
    assert normalize_champion_name("Nunu") == "Nunu"
    assert normalize_champion_name("Renata Glasc") == "Renata"
    assert normalize_champion_name("Renata") == "Renata"
    assert normalize_champion_name("  Aatrox  ") == "Aatrox"

def test_champion_to_id_mapping():
    aatrox_id = champion_to_id("Aatrox")
    ahri_id = champion_to_id("Ahri")
    assert aatrox_id > 0
    assert ahri_id > 0
    assert aatrox_id != ahri_id
    assert id_to_champion(aatrox_id) == "Aatrox"
    assert id_to_champion(ahri_id) == "Ahri"

def test_unknown_champion_fallback():
    assert champion_to_id("NonExistentChamp123") == UNK_CHAMPION_ID
    assert champion_to_id("") == UNK_CHAMPION_ID
    assert champion_to_id(None) == UNK_CHAMPION_ID
    assert id_to_champion(UNK_CHAMPION_ID) == "UNKNOWN"

def test_all_ids_are_contiguous():
    for cid in range(NUM_CHAMPIONS):
        name = id_to_champion(cid)
        assert isinstance(name, str)
        assert champion_to_id(name) == cid
