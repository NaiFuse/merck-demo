import pytest

from src.models.scope import LAB_WATER, LAB_WATER_POSSIBLE, is_lab_water, tag_lab_water
from src.models.winners import AVANTOR, IVD, MERCK, OTHER, THERMO, clean_name, winner_group


@pytest.mark.parametrize("name,group", [
    ("Thermo Fisher Scientific GmbH", THERMO),
    ("THERMOFISHER DIAGNOSTICS SAS", THERMO),
    ("Fisher Scientific SAS", THERMO),
    ("Life Technologies Europe B.V.", THERMO),
    ("Alfa Aesar GmbH & Co KG", THERMO),
    ("VWR International GmbH", AVANTOR),
    ("Avantor Performance Materials Poland S.A.", AVANTOR),
    ("J.T.Baker", AVANTOR),
    ("Merck Life Science AB", MERCK),
    ("MERCK CHIMIE SAS", MERCK),
    ("Sigma-Aldrich Chemie GmbH", MERCK),
    ("Merck Millipore Ltd", MERCK),
    ("Merck Sharp & Dohme B.V.", OTHER),   # US Merck & Co (MSD), not Merck KGaA
    ("MSD Animal Health", OTHER),
    ("Carl Roth GmbH + Co. KG", OTHER),
    ("Fishery Supplies Ltd", OTHER),       # word boundary: not 'fisher scientific'
    ("Roche Diagnostics Deutschland GmbH", IVD),
    ("SYSMEX FRANCE", IVD),
    ("Siemens Healthcare Diagnostics Products GmbH", IVD),
    ("Abbott Laboratories SA", IVD),
    ("Beckman Coulter France", IVD),
    ("bioMérieux SA", IVD),
    ("Werfen Nordic AB", IVD),
    ("Bio-Rad Laboratories Clinical Diagnostics", IVD),
    ("Bio-Rad Laboratories GmbH", OTHER),     # life-science and clinical share entities: not IVD by name
    ("ABL Diagnostics", IVD),                 # generic 'diagnost'
    ("Thermo Fisher Diagnostics SAS", THERMO),  # peer patterns win over generic IVD rule
    ("thermofischer", THERMO),                # common misspelling seen in TED
    ("FISHER", THERMO),
    ("Fisher Emergo BV", THERMO),             # Fisher Scientific's Dutch entity
    ("", OTHER),
    (None, OTHER),
])
def test_winner_group(name, group):
    assert winner_group(name) == group


def test_clean_name_strips_legal_forms_and_accents():
    assert clean_name("Dutscher S.A.S.") == "dutscher"
    assert clean_name("Carl Roth GmbH & Co. KG") == "carl roth"
    assert clean_name("Leica Microsystèmes SAS") == "leica microsystemes"


@pytest.mark.parametrize("text", ["Milli-Q IQ 7000", "Reinstwassersystem", "eau ultra pure", "Ultrapure water system",
                                  "ultrarent vatten", "Elix Essential"])
def test_lab_water_terms(text):
    assert is_lab_water(text)


def test_lab_water_only_tags_merck_and_distinguishes_possible():
    assert tag_lab_water("Merck Life Science AB", "Milli-Q service", None, []) == LAB_WATER
    assert tag_lab_water("Merck Life Science AB", "Reagents", None, ["Lot 3: Reinstwasser"]) == LAB_WATER_POSSIBLE
    assert tag_lab_water("Merck Life Science AB", "Reagents", "Lösungsmittel", []) is None
    assert tag_lab_water("Sartorius Lab Instruments", "Reinstwasseranlage", None, []) is None
    assert not is_lab_water("Laborreagenzien und Chemikalien")
