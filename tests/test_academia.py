"""Academia buyer filter: keep/exclude rules on real buyer names from TED and FTS."""

import pytest

import json

from src.models.academia import classify_buyer, notice_text


@pytest.mark.parametrize("name,legal,source,cat", [
    ("Southern Universities Purchasing Consortium (SUPC)", "BODY_PUBLIC;09", "uk_fts", "university_consortium"),
    ("North Eastern Universities Purchasing Consortium", None, "uk_fts", "university_consortium"),
    ("Education Procurement Service (EPS)", "body-pl", "ted", "university_consortium"),
    ("Queen's University Belfast", "BODY_PUBLIC;09", "uk_fts", "university"),
    ("Kungliga Tekniska högskolan", "la", "ted", "university"),          # name beats legal type 'la'
    ("Turun yliopisto", "body-pl", "ted", "university"),
    ("CHU BORDEAUX", "body-pl", "ted", "hospital"),
    ("Stadt Wien - Wiener Gesundheitsverbund", "body-pl", "ted", "hospital"),   # city-run hospital group
    ("Städtisches Klinikum Dessau", "body-pl-la", "ted", "hospital"),
    ("The Newcastle Upon Tyne Hospitals NHS Foundation Trust", "BODY_PUBLIC;07", "uk_fts", "hospital"),
    ("UNIHA-GCS", "body-pl", "ted", "hospital_purchasing"),
    ("Region Stockholm - Serviceförvaltningen", "ra", "ted", "health_region"),
    ("INSERM DELEGATION REGIONALE PARIS IDF CENTRE NORD", "body-pl", "ted", "research"),
    ("Deutsches Zentrum für Neurodegenerative Erkrankungen e.V.", "la", "ted", "research"),
    ("Sciensano", "cga", "ted", "gov_lab"),
    ("Landesamt für Verbraucherschutz Sachsen-Anhalt", "ra", "ted", "gov_lab"),
])
def test_kept(name, legal, source, cat):
    c = classify_buyer(name, legal, source)
    assert c.keep and c.category == cat, c


@pytest.mark.parametrize("name,legal,source,cat", [
    ("Scotland Excel", "BODY_PUBLIC;01", "uk_fts", "school_consortium"),
    ("YPO", "BODY_PUBLIC;01", "uk_fts", "school_consortium"),
    ("ESPO (Leicestershire County Council Trading as ESPO)", None, "uk_fts", "school_consortium"),
    ("Cork Education and Training Board", "body-pl", "ted", "school"),
    ("Freistaat Bayern vertreten durch das Markgräfin Wilhelmine Gymnasium Bayreuth", "ra", "ted", "school"),
    ("Universitätsstadt Tübingen - Zentrale Vergabestelle", "cga", "ted", "municipality_name"),
    ("NORTHUMBRIAN WATER GROUP LIMITED", "WATER", "uk_fts", "utility"),
    ("Régie Eau d'Azur (06)", "spec-rights-entity", "ted", "utility"),
    ("Ville de Lyon", "la", "ted", "municipality"),
    ("Glasgow City Council", "REGIONAL_AUTHORITY;01", "uk_fts", "municipality"),
    ("The Minister for the Cabinet Office acting through Crown Commercial Service", "MINISTRY;01", "uk_fts", "general_purchasing"),
    ("Adda Inköpscentral AB", "ra", "ted", "general_purchasing"),
    ("Some Unknown Body", "body-pl", "ted", "unclassified"),
    (None, None, "ted", "unclassified"),
])
def test_excluded(name, legal, source, cat):
    c = classify_buyer(name, legal, source)
    assert not c.keep and c.category == cat, c


def test_type_fallback_and_multi_buyer():
    assert classify_buyer("Unnamed water company", None, "uk_fts").category == "utility"
    assert classify_buyer("Some Body", "WATER", "uk_fts").category == "utility"
    assert classify_buyer("Some Trust", "BODY_PUBLIC;07", "uk_fts").category == "hospital"
    assert classify_buyer("Somewhere", "la", "ted").category == "municipality"
    assert classify_buyer("Somewhere", float("nan"), "ted").category == "unclassified"   # NaN legal type
    multi = classify_buyer("Hansel Oy; Pohjois-Pohjanmaan hyvinvointialue", "pub-undert;body-pl", "ted")
    assert multi.keep and multi.category == "health_region"


def test_departements_kept_only_for_departmental_lab():
    lab = "Fourniture de réactifs de biologie moléculaire pour le laboratoire Départemental d'analyses"
    fuel = "Fourniture et livraison de carburants en vrac | Secteur Brive"
    assert classify_buyer("Département de la Somme", "la", "ted", lab).category == "gov_lab_departmental"
    assert classify_buyer("Conseil Départemental des Bouches du Rhône", "la", "ted",
                          "FOURNITURE DE PRODUITS CHIMIQUES DESTINES AU LDA 13").keep
    c = classify_buyer("Conseil Départemental de la Corrèze", "la", "ted", fuel)
    assert not c.keep and c.category == "municipality"
    assert not classify_buyer("Ville de Lyon", "la", "ted", lab).keep        # condition is for Departements only


def test_defra_kept_as_gov_lab():
    c = classify_buyer("DEPARTMENT OF ENVIRONMENT, FOOD AND RURAL AFFAIRS (Defra Network eTendering Portal)",
                       "MINISTRY;05", "uk_fts")
    assert c.keep and c.category == "gov_lab"


def test_notice_text_reads_lot_titles():
    ted = json.dumps({"title-proc": {"fra": "Consommables"}, "title-lot": {"fra": ["Lot 1 laboratoire", "Lot 2"]}})
    assert "Lot 1 laboratoire" in notice_text(ted, "ted", "Titre")
    fts = json.dumps({"tender": {"title": "T", "lots": [{"title": "Lab lot", "description": "analysis"}]}})
    assert "analysis" in notice_text(fts, "uk_fts")
    assert notice_text("not json", "ted", "only title") == "only title"
