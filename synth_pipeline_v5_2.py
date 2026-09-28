# -*- coding: utf-8 -*-
"""
synth_pipeline_v5_2.py  --  synthetic pension-portability training corpus (5.2.0).

Public API -- the whole of it:

    from synth_pipeline_v5_2 import generate_synth_data   # or generate_example
    example = generate_synth_data()

Both names are the same zero-argument callable and return the same dict:

    {"instruction": str, "schema": dict, "input": str, "output": dict}

which is the shape the training and inference notebooks format into
"### Instruction: / ### Schema: / ### Input: / ### Response:" blocks.

Also exported: configure() to change generation defaults, generate_dataset()
to write a .jsonl directly, and run() for the self-test.

REQUIRED NOTEBOOK SETTING
    max_seq_length = 1024

    Examples are generated to a 1024-token budget, which is what it takes for
    arrays of objects and draft-07 schemas to appear in the corpus at a useful
    rate rather than being squeezed out. Left at 512, the trainer truncates the
    "### Response:" JSON off roughly half the corpus and trains on prompts with
    no target. To stay at 512 instead, call configure(max_tokens=512) and
    accept the narrower feature mix.

What this generator guarantees:

* Labels are grounded in the document. Every non-null label is either present
  in the document text, or is the normalised form of something present (dates
  to YYYY-MM-DD, phones to the local 10-digit form, amounts to numbers), or is
  the OCR-corrected form of a noised value the document does contain. No label
  is invented. This is what stops a model trained on the corpus from answering
  with a fabricated identifier when a field is genuinely absent.
* Old-versus-current employer is an explicit generated scenario, not an
  accident of sampling, so the "use the most recent value" instruction has
  supporting examples to learn from.
* Ghanaian names, employers, trustees, regions and identifiers throughout.
* Multi-level schemas, in shorthand and JSON Schema draft-07, including arrays
  of objects.
* A noise model applied so that it never destroys the characters a label
  depends on.

generate_example() defaults to compact mode with a 1024-token budget. Compact
stays on at 1024 because it measurably beats full mode there on every axis --
more arrays, twice the draft-07 schemas, deeper nesting, shorter examples --
since full mode spends its budget on a long instruction and 16-field schemas
rather than on structure. To change the defaults:

    import synth_pipeline_v5_2 as sp
    sp.configure(max_tokens=512)    # if you keep max_seq_length = 512
    sp.configure(seed=42)           # reproducible corpus

Supersedes synth_pipeline_v5_1.py. Generation behaviour is byte-identical for a
given seed; what changed is that the module no longer carries references to the
superseded 4.1 generator, and the public surface is declared in __all__.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import re
import statistics
import sys
import unicodedata
from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Optional

__all__ = [
    "generate_synth_data",   # what the notebooks import
    "generate_example",      # same callable, other name
    "configure",             # change generation defaults
    "generate_dataset",      # write n examples straight to .jsonl
    "run",                   # self-test / groundedness audit
]




# ==========================================================================
# from synth_function/ghana.py
# ==========================================================================

# -*- coding: utf-8 -*-



# ---------------------------------------------------------------- names
# Grouped by tradition so a record hangs together: an Akan given name pairs
# with an Akan surname rather than being drawn at random across the country.

AKAN_MALE = [
    "Kwasi", "Kwesi", "Kwadwo", "Kojo", "Kwabena", "Kobina", "Kwaku", "Yaw",
    "Kofi", "Kwame", "Fiifi", "Ato", "Ekow", "Akwasi", "Kwakye", "Boakye",
    "Amoako", "Nana Kwame", "Papa Yaw",
]
AKAN_FEMALE = [
    "Akosua", "Esi", "Adwoa", "Abenaa", "Araba", "Akua", "Ekua", "Yaa",
    "Aba", "Afua", "Efua", "Ama", "Abena", "Adjoa", "Afia", "Serwaa",
    "Konadu", "Nana Ama", "Maame Yaa",
]
AKAN_SURNAME = [
    "Mensah", "Osei", "Boateng", "Owusu", "Asante", "Agyeman", "Danso",
    "Frimpong", "Amoah", "Appiah", "Bediako", "Gyasi", "Sarpong",
    "Acheampong", "Nkrumah", "Ofori", "Antwi", "Adjei", "Baffour", "Yeboah",
    "Addo", "Amankwah", "Opoku", "Duah", "Nyarko", "Twum", "Wiredu",
    "Asamoah", "Agyapong", "Oduro", "Ntim", "Gyamfi", "Kusi", "Adomako",
    "Osei-Mensah", "Owusu-Ansah", "Boakye-Yiadom", "Agyei-Baffour",
]

EWE_MALE = [
    "Selorm", "Elikem", "Mawuli", "Senyo", "Edem", "Etornam", "Delali",
    "Kafui", "Sitsofe", "Dela", "Worlanyo", "Kwadzo", "Mawunyo", "Korku",
    "Kodzo", "Xolali",
]
EWE_FEMALE = [
    "Dzifa", "Sena", "Yayra", "Enyonam", "Mawusi", "Elorm", "Akpene",
    "Dzidzor", "Selikem", "Edinam", "Sedinam", "Mansa", "Adzo", "Afi",
    "Kekeli",
]
EWE_SURNAME = [
    "Agbeko", "Dogbe", "Tsikata", "Ahiabor", "Gbedemah", "Kpodo", "Nyaku",
    "Attipoe", "Akakpo", "Anku", "Fiadzo", "Zigah", "Ametefe", "Amenyo",
    "Adzoyi", "Doe", "Agbemava", "Sedode", "Klu", "Bansah", "Avornyo",
    "Dzamesi", "Hlordzi", "Wonyra", "Tettey-Enyo",
]

GA_MALE = [
    "Nii", "Ayikwei", "Tetteh", "Odoi", "Nortey", "Adjetey", "Laryea",
    "Okine", "Sowah", "Ashong", "Kotey", "Amartey", "Ayitey", "Teye",
    "Nii Armah", "Nii Ayi", "Aryeetey",
]
GA_FEMALE = [
    "Naa", "Dedei", "Korkor", "Adukwei", "Ayele", "Shormeh", "Kordai",
    "Lamiley", "Okailey", "Ayorkor", "Adjeley", "Naa Adjeley", "Naa Densua",
]
GA_SURNAME = [
    "Quartey", "Lartey", "Ankrah", "Quaye", "Aryee", "Clottey", "Tagoe",
    "Nortey", "Sowah", "Adjei-Mensah", "Bruce-Tagoe", "Okine", "Amarteifio",
    "Ashitey", "Nettey", "Odartey-Lamptey", "Lamptey", "Quaynor", "Hammond",
]

NORTHERN_MALE = [
    "Abdulai", "Mahama", "Iddrisu", "Alhassan", "Fuseini", "Yakubu",
    "Seidu", "Sulemana", "Mohammed", "Issahaku", "Abubakari", "Salifu",
    "Adamu", "Musah", "Zakaria", "Awudu", "Baba", "Osman", "Rafiu",
]
NORTHERN_FEMALE = [
    "Fati", "Amina", "Hawa", "Zeinab", "Rahama", "Salamatu", "Memuna",
    "Ayisha", "Sadia", "Damata", "Sanatu", "Hafsah", "Barikisu", "Mariama",
]
NORTHERN_SURNAME = [
    "Abdulai", "Mahama", "Iddrisu", "Alhassan", "Fuseini", "Yakubu",
    "Seidu", "Sulemana", "Mohammed", "Issahaku", "Abubakari", "Salifu",
    "Adam", "Zakaria", "Bawumia", "Dramani", "Tanko", "Wumbei", "Nashiru",
]

FANTE_MALE = ["Kobina", "Ekow", "Kwamena", "Ato", "Fiifi", "Egya", "Kow", "Paa Kwesi"]
FANTE_FEMALE = ["Araba", "Esi", "Ekua", "Aba", "Maame Efua", "Adwoa Mansa"]
FANTE_SURNAME = [
    "Aggrey", "Arthur", "Baidoo", "Bentum", "Ghartey", "Hayford", "Quansah",
    "Turkson", "Eshun", "Aidoo", "Essuman", "Koomson", "Otoo", "Panford",
    "Sam", "Abaka", "Enninful", "Micah", "Renner", "Blankson",
]

_TRADITIONS = {
    "akan": (AKAN_MALE, AKAN_FEMALE, AKAN_SURNAME),
    "ewe": (EWE_MALE, EWE_FEMALE, EWE_SURNAME),
    "ga": (GA_MALE, GA_FEMALE, GA_SURNAME),
    "northern": (NORTHERN_MALE, NORTHERN_FEMALE, NORTHERN_SURNAME),
    "fante": (FANTE_MALE, FANTE_FEMALE, FANTE_SURNAME),
}
_TRADITION_WEIGHTS = {"akan": 42, "ewe": 15, "ga": 14, "northern": 19, "fante": 10}

# Titles a Ghanaian pension form actually carries, including chieftaincy and
# Muslim honorifics a US-locale generator would never produce.
TITLES_MALE = ["Mr.", "Mr", "Dr.", "Rev.", "Prof.", "Nana", "Nii", "Alhaji", "Togbe"]
TITLES_FEMALE = ["Mrs.", "Mrs", "Miss", "Ms.", "Dr.", "Rev.", "Hajia", "Naa", "Maame"]

# ---------------------------------------------------------------- employers
EMPLOYERS = [
    "GCB Bank PLC", "Ecobank Ghana PLC", "Absa Bank Ghana Limited",
    "Stanbic Bank Ghana Limited", "Fidelity Bank Ghana Limited",
    "CalBank PLC", "Zenith Bank (Ghana) Limited", "Access Bank Ghana PLC",
    "Republic Bank (Ghana) PLC", "Societe Generale Ghana PLC",
    "Guinness Ghana Breweries PLC", "Unilever Ghana PLC",
    "Nestle Ghana Limited", "Fan Milk PLC", "Kasapreko Company Limited",
    "Accra Brewery PLC", "Tema Oil Refinery",
    "Ghana Ports and Harbours Authority", "Volta River Authority",
    "Electricity Company of Ghana", "Ghana Water Company Limited",
    "Ghana Cocoa Board", "Ghana Revenue Authority", "Ghana Health Service",
    "Ghana Education Service", "Ghana National Petroleum Corporation",
    "Ghana Airports Company Limited", "MTN Ghana", "Telecel Ghana",
    "AirtelTigo Ghana", "Vivo Energy Ghana", "TotalEnergies Marketing Ghana",
    "GOIL PLC", "Newmont Ghana Gold Limited", "Gold Fields Ghana Limited",
    "AngloGold Ashanti (Obuasi) Limited", "Tullow Ghana Limited",
    "Ghana Manganese Company Limited", "Zoomlion Ghana Limited",
    "Melcom Ghana Limited", "Social Security and National Insurance Trust",
    "University of Ghana", "Kwame Nkrumah University of Science and Technology",
    "Korle Bu Teaching Hospital", "Komfo Anokye Teaching Hospital",
    "Ghana Broadcasting Corporation", "Ghana Post Company Limited",
    "State Insurance Company", "Databank Financial Services",
]

SME_PATTERNS = [
    "{p} Ventures Limited", "{p} Enterprise", "{p} Company Limited",
    "{p} Trading Enterprise", "{p} Services Limited", "{p} and Sons Limited",
    "{p} Manufacturing Ltd", "{p} Construction Works",
    "{p} Agro Processing Ltd", "{p} Investments Limited", "{p} Logistics Ltd",
]
SME_PREFIX = [
    "Adom", "Nyame Ntie", "Obaatanpa", "Sunrise", "Dansoman", "Achimota",
    "Kaneshie", "Odawna", "Asafo", "Suame", "Bantama", "Nhyira",
    "Gye Nyame", "Ebenezer", "Divine Favour", "Alpha Omega", "Unity",
    "Golden Star", "Boafo Ye Na", "Onipa Nua", "Mawuli", "Akwaaba",
    "Rock City", "Trinity", "Nkosuo", "Sika",
]

# ---------------------------------------------------------------- trustees
# NPRA-licensed corporate trustees. A portability transfer moves a member from
# one of these to another, so both endpoints are drawn from this list.
TRUSTEES = [
    "Enterprise Trustees Limited", "Petra Trust Company Limited",
    "Glico Pensions Trustee Company Limited",
    "Metropolitan Pensions Trust Ghana Limited",
    "Old Mutual Pensions Trust Ghana Limited",
    "Stanbic Investment Management Services",
    "Axis Pension Trust Limited", "Peoples Pension Trust Ghana",
    "Secure Pensions Trust Limited", "Hedge Pensions Trust",
    "Best Pensions Trust Limited", "Daakye Pension Trust Limited",
    "NTHC Trustees Limited", "General Trust Company Limited",
    "First Merchant Trust Limited", "Kimpton Trust Limited",
    "United Pension Trustees Limited", "Pentrust Limited",
    "Negotiated Benefits Trust Company Limited", "QLAC Financial Trust",
    "Ideal Pension Trust Limited", "Apex Trust Services Limited",
]

SCHEME_TYPES = [
    "Tier 2 Occupational Pension Scheme",
    "Tier 3 Provident Fund Scheme",
    "Tier 3 Personal Pension Scheme",
    "Master Trust Occupational Pension Scheme",
    "Employer Sponsored Occupational Pension Scheme",
]

SCHEME_NAME_PATTERNS = [
    "{t} Master Trust Occupational Pension Scheme",
    "{e} Staff Provident Fund",
    "{e} Tier 2 Occupational Pension Scheme",
    "{t} Personal Pension Scheme",
    "{e} Employees Pension Scheme",
]

# ---------------------------------------------------------------- geography
# (city, region, GhanaPost GPS district prefix)
LOCALITIES = [
    ("Accra", "Greater Accra", "GA"), ("Tema", "Greater Accra", "GT"),
    ("Madina", "Greater Accra", "GM"), ("Adenta", "Greater Accra", "GD"),
    ("Ashaiman", "Greater Accra", "GS"), ("Dansoman", "Greater Accra", "GA"),
    ("Achimota", "Greater Accra", "GE"), ("East Legon", "Greater Accra", "GD"),
    ("Osu", "Greater Accra", "GA"), ("Teshie", "Greater Accra", "GZ"),
    ("Nungua", "Greater Accra", "GZ"), ("Spintex", "Greater Accra", "GT"),
    ("Kumasi", "Ashanti", "AK"), ("Obuasi", "Ashanti", "AO"),
    ("Ejisu", "Ashanti", "AE"), ("Konongo", "Ashanti", "AK"),
    ("Mampong", "Ashanti", "AM"), ("Bekwai", "Ashanti", "AB"),
    ("Takoradi", "Western", "WS"), ("Sekondi", "Western", "WS"),
    ("Tarkwa", "Western", "WT"), ("Axim", "Western", "WA"),
    ("Cape Coast", "Central", "CC"), ("Winneba", "Central", "CW"),
    ("Kasoa", "Central", "CK"), ("Elmina", "Central", "CE"),
    ("Koforidua", "Eastern", "EN"), ("Nkawkaw", "Eastern", "EK"),
    ("Akosombo", "Eastern", "EA"), ("Nsawam", "Eastern", "EW"),
    ("Ho", "Volta", "VH"), ("Keta", "Volta", "VK"), ("Hohoe", "Volta", "VH"),
    ("Aflao", "Volta", "VA"), ("Kpando", "Volta", "VP"),
    ("Tamale", "Northern", "NT"), ("Yendi", "Northern", "NY"),
    ("Savelugu", "Northern", "NS"), ("Bolgatanga", "Upper East", "UB"),
    ("Bawku", "Upper East", "UK"), ("Navrongo", "Upper East", "UN"),
    ("Wa", "Upper West", "XW"), ("Lawra", "Upper West", "XL"),
    ("Sunyani", "Bono", "BS"), ("Berekum", "Bono", "BB"),
    ("Techiman", "Bono East", "BT"), ("Goaso", "Ahafo", "HG"),
    ("Sefwi Wiawso", "Western North", "ZW"), ("Damongo", "Savannah", "SD"),
    ("Nalerigu", "North East", "MN"), ("Dambai", "Oti", "OD"),
]

STREET_WORDS = [
    "Ring Road", "Liberation Road", "Oxford Street", "Spintex Road",
    "Independence Avenue", "Castle Road", "Nsawam Road", "Graphic Road",
    "Kwame Nkrumah Avenue", "Boundary Road", "Lagos Avenue", "Otswe Street",
    "Palace Street", "Mango Street", "Adjiringanor Road",
    "Haatso-Atomic Road", "Ejisu-Kumasi Road", "Guggisberg Avenue",
    "Salaga Market Road",
]

OCCUPATIONS = [
    "Teacher", "Nurse", "Accountant", "Driver", "Security Officer",
    "Sales Executive", "Administrative Assistant", "Electrician", "Welder",
    "Banking Officer", "Field Supervisor", "Storekeeper", "Machine Operator",
    "Human Resource Officer", "Procurement Officer", "IT Support Officer",
    "Cashier", "Quality Control Officer", "Site Engineer", "Mason",
    "Seamstress", "Catering Officer", "Laboratory Technician", "Pharmacist",
    "Marketing Officer", "Internal Auditor", "Customer Service Officer",
    "Plant Operator", "Warehouse Assistant", "Records Officer",
]

RELATIONSHIPS = [
    "Spouse", "Wife", "Husband", "Son", "Daughter", "Mother", "Father",
    "Brother", "Sister", "Nephew", "Niece", "Uncle", "Aunt", "Cousin",
    "Guardian",
]

MARITAL_STATUS = ["Single", "Married", "Divorced", "Widowed", "Separated"]

# Mobile prefixes as issued by the NCA.
MOBILE_PREFIXES = {
    "MTN": ["024", "054", "055", "059", "025"],
    "Telecel": ["020", "050"],
    "AirtelTigo": ["026", "056", "027", "057"],
}
_ALL_PREFIXES = [p for v in MOBILE_PREFIXES.values() for p in v]


# ---------------------------------------------------------------- people
def person(rng, gender=None, tradition=None):
    """A name set that hangs together: given name, surname and title drawn from
    one tradition, with a consistent gender."""
    if tradition is None:
        names = list(_TRADITION_WEIGHTS)
        tradition = rng.choices(names, weights=[_TRADITION_WEIGHTS[n] for n in names])[0]
    if gender is None:
        gender = rng.choice(["male", "female"])

    male, female, surnames = _TRADITIONS[tradition]
    pool = male if gender == "male" else female

    return {
        "title": rng.choice(TITLES_MALE if gender == "male" else TITLES_FEMALE),
        "firstName": rng.choice(pool),
        "middleName": rng.choice(pool + surnames) if rng.random() < 0.45 else None,
        "lastName": rng.choice(surnames),
        "gender": "Male" if gender == "male" else "Female",
        "tradition": tradition,
    }


def full_name(p, include_middle=True):
    parts = [p["firstName"]]
    if include_middle and p.get("middleName"):
        parts.append(p["middleName"])
    parts.append(p["lastName"])
    return " ".join(parts)


# ---------------------------------------------------------------- employers
def employer(rng):
    if rng.random() < 0.55:
        return rng.choice(EMPLOYERS)
    return rng.choice(SME_PATTERNS).format(p=rng.choice(SME_PREFIX))


def trustee_pair(rng):
    """Current and receiving trustee, guaranteed distinct: a transfer to the
    same trustee is not a portability event."""
    a, b = rng.sample(TRUSTEES, 2)
    return a, b


def scheme_name(rng, employer_name, trustee_name):
    return rng.choice(SCHEME_NAME_PATTERNS).format(
        e=employer_name.replace(" Limited", "").replace(" PLC", ""),
        t=trustee_name.replace(" Limited", "").replace(" Company", ""),
    )


# ---------------------------------------------------------------- identifiers
def ghana_card(rng):
    """NIA Ghana Card PIN: GHA-<9 digits>-<check digit>."""
    return f"GHA-{rng.randint(100000000, 999999999)}-{rng.randint(0, 9)}"


def ssnit(rng):
    """SSNIT numbers appear in two shapes in live data: the biometric format
    (a letter followed by 12 digits) and older all-numeric registrations.
    Both are emitted, because a real ingestion pipeline meets both."""
    if rng.random() < 0.7:
        return f"{rng.choice('BCPHDS')}{rng.randint(10 ** 11, 10 ** 12 - 1)}"
    return str(rng.randint(10 ** 10, 10 ** 11 - 1))


def tin(rng):
    """Legacy GRA TIN. Since 2021 the Ghana Card PIN doubles as the TIN, so a
    minority of records carry the card PIN in the TIN field instead."""
    if rng.random() < 0.25:
        return ghana_card(rng)
    return f"{rng.choice('PCGV')}{rng.randint(10 ** 9, 10 ** 10 - 1)}"


def phone(rng):
    """Canonical local form: 10 digits beginning with 0. Surface formatting
    variants are applied later by the noise layer."""
    return rng.choice(_ALL_PREFIXES) + "".join(str(rng.randint(0, 9)) for _ in range(7))


def digital_address(rng, prefix=None):
    """GhanaPost GPS code, e.g. GA-543-0125."""
    if prefix is None:
        prefix = rng.choice(LOCALITIES)[2]
    return f"{prefix}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"


def staff_id(rng):
    return rng.choice([
        lambda: f"EMP{rng.randint(1000, 99999)}",
        lambda: f"STF/{rng.randint(2005, 2025)}/{rng.randint(100, 9999)}",
        lambda: f"{rng.randint(10000, 999999)}",
        lambda: f"HR-{rng.randint(100, 999)}-{rng.randint(10, 99)}",
    ])()


def member_number(rng):
    return rng.choice([
        lambda: f"PEN/{rng.randint(2010, 2025)}/{rng.randint(1000, 99999)}",
        lambda: f"MBR{rng.randint(100000, 999999)}",
        lambda: f"{rng.randint(2010, 2025)}-{rng.randint(10000, 99999)}",
    ])()


def email(rng, p):
    user = f"{p['firstName']}.{p['lastName']}".lower().replace(" ", "")
    domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "ymail.com"]
    if rng.random() < 0.25:
        return f"{user}@{rng.choice(domains)}"
    return f"{user}{rng.randint(1, 99)}@{rng.choice(domains)}"


def locality(rng):
    return rng.choice(LOCALITIES)


def street(rng):
    return f"{rng.randint(1, 120)} {rng.choice(STREET_WORDS)}"


# ---------------------------------------------------------------- values
def date_between(rng, start_year, end_year):
    start = _dt.date(start_year, 1, 1).toordinal()
    end = _dt.date(end_year, 12, 31).toordinal()
    return _dt.date.fromordinal(rng.randint(start, end))


def money(rng, low, high):
    return round(rng.uniform(low, high), 2)


# ==========================================================================
# from synth_function/records.py
# ==========================================================================

# -*- coding: utf-8 -*-





@dataclass(frozen=True)
class Field:
    key: str                    # stable identifier used across the package
    path: tuple                 # location in the canonical record
    type: str                   # JSON type: string | number | integer
    aliases: tuple              # plausible schema field names
    labels: tuple               # plausible document labels
    sensitive: bool = False     # identifier/amount: protect from char-level OCR
    group: str = "member"       # used to keep generated schemas coherent


def F(key, path, type_, aliases, labels, sensitive=False, group="member"):
    return Field(key, tuple(path), type_, tuple(aliases), tuple(labels), sensitive, group)


FIELDS = {f.key: f for f in [
    # ---- identity
    F("title", ["member", "title"], "string",
      ["title", "salutation", "memberTitle"],
      ["Title", "Salutation", "Title (Mr/Mrs/Miss)"], group="identity"),
    F("first_name", ["member", "firstName"], "string",
      ["firstName", "givenName", "first_name", "forename"],
      ["First Name", "Given Name", "Forename", "FIRST NAME"], group="identity"),
    F("middle_name", ["member", "middleName"], "string",
      ["middleName", "otherNames", "middle_name"],
      ["Middle Name", "Other Names", "Other Name(s)"], group="identity"),
    F("last_name", ["member", "lastName"], "string",
      ["lastName", "surname", "last_name", "familyName"],
      ["Surname", "Last Name", "Family Name", "SURNAME"], group="identity"),
    F("gender", ["member", "gender"], "string",
      ["gender", "sex"], ["Gender", "Sex"], group="identity"),
    F("date_of_birth", ["member", "dateOfBirth"], "string",
      ["dateOfBirth", "dob", "date_of_birth", "birthDate"],
      ["Date of Birth", "D.O.B", "DOB", "Birth Date"], group="identity"),
    F("marital_status", ["member", "maritalStatus"], "string",
      ["maritalStatus", "marital_status"],
      ["Marital Status", "Status"], group="identity"),
    F("nationality", ["member", "nationality"], "string",
      ["nationality", "citizenship"], ["Nationality", "Citizenship"],
      group="identity"),

    # ---- identifiers
    F("ghana_card", ["identification", "ghanaCard"], "string",
      ["ghanaCard", "ghanaCardNumber", "nationalId", "ghana_card_number"],
      ["Ghana Card", "Ghana Card No.", "Ghana Card Number", "National ID",
       "NIA Number", "Ghana Card PIN"],
      sensitive=True, group="identifiers"),
    F("ssnit_number", ["identification", "ssnitNumber"], "string",
      ["ssnitNumber", "ssnit", "socialSecurityNumber", "ssnit_number"],
      ["SSNIT Number", "SSNIT No.", "SSNIT", "Social Security Number",
       "SSNIT Biometric No."],
      sensitive=True, group="identifiers"),
    F("tin", ["identification", "tin"], "string",
      ["tin", "taxIdentificationNumber", "tinNumber"],
      ["TIN", "Tax Identification Number", "T.I.N"],
      sensitive=True, group="identifiers"),

    # ---- contact
    F("calling_code", ["contact", "phone", "callingCode"], "string",
      ["callingCode", "countryCode", "dialCode"],
      ["Country Code", "Calling Code"], group="contact"),
    F("phone_number", ["contact", "phone", "number"], "string",
      ["number", "phoneNumber", "mobile", "msisdn", "phone"],
      ["Phone", "Mobile Number", "Telephone", "Mobile No.", "Contact Number",
       "Phone No"],
      sensitive=True, group="contact"),
    F("alt_phone", ["contact", "altPhone", "number"], "string",
      ["alternatePhone", "altPhone", "secondaryPhone"],
      ["Alternate Phone", "Other Number", "Second Contact"],
      sensitive=True, group="contact"),
    F("email", ["contact", "email"], "string",
      ["email", "emailAddress", "email_address"],
      ["Email", "E-mail", "Email Address"],
      sensitive=True, group="contact"),
    F("digital_address", ["contact", "address", "digitalAddress"], "string",
      ["digitalAddress", "ghanaPostGps", "gpsAddress"],
      ["Digital Address", "GhanaPost GPS", "GPS Address", "Digital Addr."],
      sensitive=True, group="contact"),
    F("street", ["contact", "address", "street"], "string",
      ["street", "streetAddress", "addressLine1"],
      ["Address", "Street Address", "Residential Address"],
      sensitive=True, group="contact"),
    F("city", ["contact", "address", "city"], "string",
      ["city", "town", "locality"], ["City", "Town", "City/Town"],
      group="contact"),
    F("region", ["contact", "address", "region"], "string",
      ["region", "state", "province"], ["Region", "State"], group="contact"),

    # ---- employment
    F("employer_name", ["employment", "employerName"], "string",
      ["employerName", "employer", "companyName", "employer_name"],
      ["Employer", "Employer Name", "Company", "Name of Employer",
       "Current Employer"], group="employment"),
    F("staff_id", ["employment", "staffId"], "string",
      ["staffId", "employeeNumber", "staffNumber"],
      ["Staff ID", "Staff Number", "Employee No.", "Emp. ID"],
      sensitive=True, group="employment"),
    F("occupation", ["employment", "occupation"], "string",
      ["occupation", "jobTitle", "designation"],
      ["Occupation", "Job Title", "Designation", "Position"],
      group="employment"),
    F("date_of_engagement", ["employment", "dateOfEngagement"], "string",
      ["dateOfEngagement", "employmentDate", "hireDate", "dateEmployed"],
      ["Date of Engagement", "Employment Date", "Date Employed", "Date Joined"],
      group="employment"),
    F("monthly_salary", ["employment", "monthlySalary"], "number",
      ["monthlySalary", "basicSalary", "salary"],
      ["Monthly Salary", "Basic Salary", "Gross Salary"],
      sensitive=True, group="employment"),

    # ---- pension
    F("member_number", ["pension", "memberNumber"], "string",
      ["memberNumber", "memberId", "pensionMemberNumber", "member_id"],
      ["Member Number", "Member ID", "Membership No.", "Member No"],
      sensitive=True, group="pension"),
    F("scheme_name", ["pension", "schemeName"], "string",
      ["schemeName", "scheme", "pensionScheme"],
      ["Scheme", "Scheme Name", "Name of Scheme"],
      sensitive=True, group="pension"),
    F("scheme_type", ["pension", "schemeType"], "string",
      ["schemeType", "schemeCategory", "tierType"],
      ["Scheme Type", "Type of Scheme", "Category"],
      sensitive=True, group="pension"),
    F("tier", ["pension", "tier"], "string",
      ["tier", "pensionTier"], ["Tier", "Pension Tier"],
      sensitive=True, group="pension"),
    F("current_trustee", ["pension", "currentTrustee"], "string",
      ["currentTrustee", "transferringFrom", "cedingTrustee", "fromTrustee"],
      ["Current Trustee", "Transferring From", "Ceding Trustee",
       "Existing Trustee", "From"], group="pension"),
    F("receiving_trustee", ["pension", "receivingTrustee"], "string",
      ["receivingTrustee", "transferringTo", "newTrustee", "toTrustee"],
      ["Receiving Trustee", "Transferring To", "New Trustee", "To"],
      group="pension"),
    F("accrued_benefit", ["pension", "accruedBenefit"], "number",
      ["accruedBenefit", "totalContribution", "fundValue", "accountBalance"],
      ["Accrued Benefit", "Total Contribution", "Fund Value",
       "Account Balance", "Total Contributions to Date"],
      sensitive=True, group="pension"),
    F("last_contribution_date", ["pension", "lastContributionDate"], "string",
      ["lastContributionDate", "lastContribution"],
      ["Last Contribution", "Last Contribution Date", "Last Remittance"],
      group="pension"),

    # ---- next of kin
    F("nok_name", ["nextOfKin", "fullName"], "string",
      ["fullName", "name", "nextOfKinName"],
      ["Next of Kin", "Next of Kin Name", "NOK Name"], group="nok"),
    F("nok_relationship", ["nextOfKin", "relationship"], "string",
      ["relationship", "relationshipToMember"],
      ["Relationship", "Relationship to Member"], group="nok"),
    F("nok_phone", ["nextOfKin", "phone"], "string",
      ["phone", "phoneNumber", "contact"],
      ["NOK Phone", "Next of Kin Phone", "Contact"],
      sensitive=True, group="nok"),
]}

# Array-valued sections. These give the schema genuine depth: an array of
# objects nested inside an object is where flat extractors break down.
BENEFICIARY_FIELDS = {
    "fullName": ("string", ["fullName", "name", "beneficiaryName"], False),
    "relationship": ("string", ["relationship", "relation"], False),
    "phone": ("string", ["phone", "phoneNumber", "contact"], True),
    "allocationPercent": ("number", ["allocationPercent", "allocation", "sharePercent"], True),
}

CONTRIBUTION_FIELDS = {
    "period": ("string", ["period", "month", "contributionPeriod"], False),
    "employeeAmount": ("number", ["employeeAmount", "employeeContribution", "memberAmount"], True),
    "employerAmount": ("number", ["employerAmount", "employerContribution"], True),
}


# ---------------------------------------------------------------- accessors
def get_path(record, path):
    """Read a dotted path out of the canonical record, tolerating absent keys."""
    cur = record
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def path_str(path):
    return ".".join(path)


# ---------------------------------------------------------------- builder
def build_record(rng, missing_rate=0.18):
    """A complete canonical member record.

    `missing_rate` drops optional fields outright. Real trustee submissions are
    rarely complete, and a generator that always fills every field teaches the
    model that null never occurs.
    """

    def maybe(value, p=missing_rate):
        return None if rng.random() < p else value

    member = person(rng)
    city, region, gps_prefix = locality(rng)
    employer_name = employer(rng)
    current_trustee, receiving_trustee = trustee_pair(rng)

    dob = date_between(rng, 1962, 2005)
    engagement = date_between(rng, max(1985, dob.year + 18), 2025)
    salary = money(rng, 900, 18000)

    nok = person(rng)

    n_beneficiaries = rng.choices([0, 1, 2, 3, 4], weights=[15, 30, 30, 18, 7])[0]
    beneficiaries = []
    remaining = 100
    for i in range(n_beneficiaries):
        b = person(rng)
        last = i == n_beneficiaries - 1
        share = remaining if last else rng.choice([10, 15, 20, 25, 30, 40, 50])
        share = min(share, remaining)
        remaining -= share
        beneficiaries.append({
            "fullName": full_name(b, include_middle=False),
            "relationship": rng.choice(RELATIONSHIPS),
            "phone": phone(rng),
            "allocationPercent": share,
        })
    if beneficiaries and remaining > 0:
        beneficiaries[-1]["allocationPercent"] += remaining

    n_contributions = rng.choices([0, 3, 4, 5, 6], weights=[35, 18, 18, 16, 13])[0]
    contributions = []
    year = rng.randint(2022, 2025)
    month = rng.randint(1, 12)
    for _ in range(n_contributions):
        employee = money(rng, salary * 0.045, salary * 0.055)
        contributions.append({
            "period": f"{year}-{month:02d}",
            "employeeAmount": employee,
            "employerAmount": round(employee * 2.6, 2),
        })
        month += 1
        if month > 12:
            month, year = 1, year + 1

    return {
        "member": {
            "title": maybe(member["title"], 0.3),
            "firstName": member["firstName"],
            "middleName": member["middleName"],
            "lastName": member["lastName"],
            "gender": maybe(member["gender"], 0.25),
            "dateOfBirth": maybe(dob.isoformat(), 0.15),
            "maritalStatus": maybe(rng.choice(MARITAL_STATUS), 0.35),
            "nationality": maybe("Ghanaian", 0.3),
        },
        "identification": {
            "ghanaCard": maybe(ghana_card(rng), 0.12),
            "ssnitNumber": maybe(ssnit(rng), 0.12),
            "tin": maybe(tin(rng), 0.45),
        },
        "contact": {
            "phone": {"callingCode": "233", "number": phone(rng)},
            "altPhone": {"number": maybe(phone(rng), 0.55)},
            "email": maybe(email(rng, member), 0.4),
            "address": {
                "digitalAddress": maybe(digital_address(rng, gps_prefix), 0.35),
                "street": maybe(street(rng), 0.3),
                "city": city,
                "region": region,
            },
        },
        "employment": {
            "employerName": employer_name,
            "staffId": maybe(staff_id(rng), 0.3),
            "occupation": maybe(rng.choice(OCCUPATIONS), 0.25),
            "dateOfEngagement": maybe(engagement.isoformat(), 0.3),
            "monthlySalary": maybe(salary, 0.4),
        },
        "pension": {
            "memberNumber": member_number(rng),
            "schemeName": maybe(scheme_name(rng, employer_name, current_trustee), 0.25),
            "schemeType": maybe(rng.choice(SCHEME_TYPES), 0.3),
            "tier": maybe(rng.choice(["Tier 2", "Tier 3", "2", "3"]), 0.4),
            "currentTrustee": current_trustee,
            "receivingTrustee": receiving_trustee,
            "accruedBenefit": maybe(money(rng, 1500, 320000), 0.3),
            "lastContributionDate": maybe(
                date_between(rng, 2024, 2026).isoformat(), 0.4),
        },
        "nextOfKin": {
            "fullName": maybe(full_name(nok, include_middle=False), 0.35),
            "relationship": maybe(rng.choice(RELATIONSHIPS), 0.35),
            "phone": maybe(phone(rng), 0.45),
        },
        "beneficiaries": beneficiaries,
        "contributions": contributions,
    }


# ==========================================================================
# from synth_function/noise.py
# ==========================================================================

# -*- coding: utf-8 -*-



# ---------------------------------------------------------------- profile


@dataclass
class NoiseProfile:
    severity: int = 1           # 0 clean, 1 mild, 2 moderate, 3 heavy
    ocr: bool = True
    layout: bool = True
    furniture: bool = True
    mojibake: bool = False
    id_ocr_relabel: bool = False   # see module docstring
    compact: bool = False          # small-context mode; see documents.Doc

    @classmethod
    def sample(cls, rng, weights=(18, 34, 30, 18)):
        """Draw a profile. Most documents are messy; a minority are clean."""
        sev = rng.choices([0, 1, 2, 3], weights=list(weights))[0]
        return cls(
            severity=sev,
            ocr=sev > 0,
            layout=sev > 0 and rng.random() < 0.8,
            furniture=rng.random() < (0.15 + 0.2 * sev),
            mojibake=sev >= 2 and rng.random() < 0.3,
            id_ocr_relabel=False,
        )


# ---------------------------------------------------------------- segments

BOILERPLATE = "boilerplate"
LABEL = "label"
VALUE = "value"


@dataclass
class Segment:
    text: str
    kind: str = BOILERPLATE
    field_key: Optional[str] = None
    sensitive: bool = False


def B(text):
    return Segment(text, BOILERPLATE)


def L(text):
    return Segment(text, LABEL)


def V(text, field_key=None, sensitive=False):
    return Segment(str(text), VALUE, field_key, sensitive)


# ---------------------------------------------------------------- OCR
# Substitutions a real optical scanner makes. Ordered longest-first so the
# multi-character cases get a chance before the single-character ones.
_OCR_MULTI = [
    ("rn", "m"), ("m", "rn"), ("cl", "d"), ("d", "cl"),
    ("vv", "w"), ("w", "vv"), ("ll", "II"), ("fi", "n"),
]
_OCR_SINGLE = {
    "O": ["0", "Q", "D"], "0": ["O", "o"], "o": ["0"],
    "I": ["1", "l", "|"], "l": ["1", "I"], "1": ["l", "I"],
    "S": ["5", "$"], "5": ["S"], "s": ["5"],
    "B": ["8", "R"], "8": ["B"], "G": ["6", "C"], "6": ["G"],
    "Z": ["2"], "2": ["Z"], "g": ["9"], "9": ["g"],
    "t": ["f", "+"], "e": ["c"], "a": ["o"], "u": ["v"],
    "n": ["h"], "h": ["b"], "y": ["v"], "D": ["O"], "Q": ["O"],
    "-": ["~", "_"], ".": [","], ",": ["."],
}


def ocr_text(text, rng, rate, max_subs=None):
    """Apply scanner-style confusion. `rate` is per-character probability."""
    if rate <= 0 or not text:
        return text

    out = []
    i = 0
    subs = 0
    while i < len(text):
        if max_subs is not None and subs >= max_subs:
            out.append(text[i:])
            break

        matched = False
        if rng.random() < rate * 0.35:
            for src, dst in _OCR_MULTI:
                if text.startswith(src, i):
                    out.append(dst)
                    i += len(src)
                    subs += 1
                    matched = True
                    break
        if matched:
            continue

        ch = text[i]
        if ch in _OCR_SINGLE and rng.random() < rate:
            out.append(rng.choice(_OCR_SINGLE[ch]))
            subs += 1
        else:
            out.append(ch)
        i += 1

    return "".join(out)


def space_noise(text, rng, rate):
    """Scanners split words at arbitrary points: 'Member' becomes 'Memb er'."""
    if rate <= 0 or len(text) < 4:
        return text
    out = []
    for i, ch in enumerate(text):
        out.append(ch)
        if 0 < i < len(text) - 1 and ch != " " and text[i + 1] != " ":
            if rng.random() < rate:
                out.append(" ")
    return "".join(out)


# ---------------------------------------------------------------- formatters
# Surface variants. All are recoverable: the canonical value can be derived by
# normalisation, which is what the instruction asks the model to do.

_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]


def _ordinal(n):
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def date_display(rng, iso):
    """Render an ISO date the many ways a Ghanaian form actually writes it."""
    if not iso:
        return ""
    try:
        y, m, d = (int(x) for x in iso.split("-"))
    except (ValueError, AttributeError):
        return str(iso)

    return rng.choice([
        lambda: iso,
        lambda: f"{d:02d}/{m:02d}/{y}",
        lambda: f"{d:02d}-{m:02d}-{y}",
        lambda: f"{d:02d}.{m:02d}.{y}",
        lambda: f"{d}/{m}/{y}",
        lambda: f"{d}/{m}/{str(y)[2:]}",
        lambda: f"{_ordinal(d)} {_MONTHS[m - 1]}, {y}",
        lambda: f"{_MONTHS[m - 1]} {d}, {y}",
        lambda: f"{d} {_MONTHS[m - 1][:3]} {y}",
        lambda: f"{d:02d}{_MONTHS[m - 1][:3].upper()}{y}",
    ])()


def money_display(rng, value):
    """Cedi amounts, with every symbol variant seen in practice."""
    if value is None:
        return ""
    plain = f"{value:,.2f}"
    nocomma = f"{value:.2f}"
    return rng.choice([
        lambda: plain,
        lambda: nocomma,
        lambda: f"GHS {plain}",
        lambda: f"GHS{nocomma}",
        lambda: f"GH₵{plain}",
        lambda: f"GH¢ {plain}",
        lambda: f"GHC {nocomma}",
        lambda: f"¢{plain}",
        lambda: f"{plain} GHS",
        lambda: f"GHS {plain} only",
    ])()


def phone_display(rng, number):
    """Canonical form is the local 10-digit number; these are its surfaces."""
    if not number or len(number) < 10:
        return str(number or "")
    p, a, b = number[:3], number[3:6], number[6:]
    intl = "233" + number[1:]
    return rng.choice([
        lambda: number,
        lambda: f"{p}-{a}-{b}",
        lambda: f"{p} {a} {b}",
        lambda: f"{p} {number[3:]}",
        lambda: f"+233 {number[1:3]} {a} {b}",
        lambda: f"+{intl}",
        lambda: intl,
        lambda: f"({p}) {number[3:]}",
        lambda: f"{p}/{number[3:]}",
        lambda: f"0{number[1:3]}-{a}{b}",
    ])()


def id_display(rng, value, severity):
    """Spacing and separator drift in identifiers. Never character damage --
    the digits stay intact so the label remains recoverable."""
    if not value:
        return ""
    s = str(value)
    if severity <= 0:
        return s
    r = rng.random()
    if r < 0.15 and len(s) > 6:
        cut = rng.randint(3, len(s) - 3)
        return s[:cut] + " " + s[cut:]
    if r < 0.25:
        return s.replace("-", " - ")
    if r < 0.32:
        return s.replace("-", "/")
    if r < 0.38:
        return s.replace("-", "")
    return s


def name_display(rng, name, severity):
    """Case and ordering chaos. Recoverable by normalisation."""
    if not name:
        return ""
    if severity <= 0:
        return name
    r = rng.random()
    if r < 0.28:
        return name.upper()
    if r < 0.36:
        return name.lower()
    if r < 0.42:
        return "  ".join(name.split())
    return name


# ---------------------------------------------------------------- furniture
_HEADERS = [
    "NATIONAL PENSIONS REGULATORY AUTHORITY",
    "*** CONFIDENTIAL - FOR OFFICIAL USE ONLY ***",
    "Scanned by CamScanner",
    "Doc Ref: {ref}",
    "--- Page {p} of {n} ---",
    "RECEIVED  |  {d}  |  RECORDS UNIT",
    "[ TRUSTEE COPY ]",
]
_FOOTERS = [
    "Page {p} of {n}",
    "This document was generated electronically and requires no signature.",
    "...........................            ...........................",
    "Authorised Signatory                    Date",
    "Doc ID {ref} | printed {d}",
    "*** END OF DOCUMENT ***",
    "Scanned with CamScanner",
]
_STAMPS = [
    "[STAMP: RECEIVED]", "[SIGNATURE ILLEGIBLE]", "[OFFICIAL STAMP]",
    "<<smudged>>", "[ILLEGIBLE]", "[torn]",
]


def _ref(rng):
    return f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(10000, 99999)}"


def _stamp_date(rng):
    return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/202{rng.randint(3, 6)}"


def add_furniture(text, rng, profile):
    """Page headers, footers, stamps -- the debris a scanner leaves behind."""
    if not profile.furniture:
        return text

    page, total = 1, rng.randint(1, 3)
    ctx = {"p": page, "n": total, "ref": _ref(rng), "d": _stamp_date(rng)}

    lines = text.split("\n")
    if rng.random() < 0.7:
        lines.insert(0, rng.choice(_HEADERS).format(**ctx))
    if rng.random() < 0.6:
        lines.append("")
        lines.append(rng.choice(_FOOTERS).format(**ctx))
    if profile.severity >= 2 and rng.random() < 0.5 and len(lines) > 4:
        lines.insert(rng.randint(2, len(lines) - 1), rng.choice(_STAMPS))
    return "\n".join(lines)


# ---------------------------------------------------------------- layout
def layout_noise(text, rng, profile):
    """Whitespace and line-structure damage. Character-safe by construction:
    every transform here only moves or duplicates whitespace and lines."""
    if not profile.layout:
        return text
    sev = profile.severity

    if rng.random() < 0.3 * sev:
        text = text.replace(": ", ":\n")
    if rng.random() < 0.2 * sev:
        text = text.replace("\n\n", "\n")

    lines = text.split("\n")

    if rng.random() < 0.25 * sev:
        lines = [(" " * rng.randint(0, 6)) + ln for ln in lines]
    if rng.random() < 0.3 * sev:
        lines = [ln + (" " * rng.randint(0, 4)) for ln in lines]
    if rng.random() < 0.15 * sev:
        lines = [ln.replace("  ", "\t") if rng.random() < 0.3 else ln for ln in lines]

    # A line duplicated by a double feed, or swallowed by a paper jam.
    if sev >= 2 and len(lines) > 5:
        if rng.random() < 0.25:
            i = rng.randrange(len(lines))
            lines.insert(i, lines[i])
        if rng.random() < 0.25:
            blanks = [i for i, ln in enumerate(lines) if not ln.strip()]
            if blanks:
                del lines[rng.choice(blanks)]

    # Two columns collapsing into one another.
    if sev >= 3 and len(lines) > 6 and rng.random() < 0.35:
        i = rng.randrange(len(lines) - 1)
        merged = lines[i].rstrip() + "   " + lines[i + 1].lstrip()
        lines[i:i + 2] = [merged]

    return "\n".join(lines)


_MOJIBAKE = {
    "’": "â€™", "‘": "â€˜",
    "“": "â€œ", "”": "â€�",
    "–": "â€“", "—": "â€”",
    "¢": "Â¢", "₵": "â‚µ",
    "é": "Ã©",
}


def mojibake(text, rng, profile):
    """UTF-8 bytes decoded as latin-1: the classic legacy-export artefact."""
    if not profile.mojibake:
        return text
    for src, dst in _MOJIBAKE.items():
        if src in text and rng.random() < 0.7:
            text = text.replace(src, dst)
    return text


# ---------------------------------------------------------------- render
def render(segments, rng, profile):
    """Join segments into a document, applying noise according to segment kind.

    Returns (text, damaged, surfaces).

    `surfaces` maps field_key -> the string actually printed for it *after*
    noise, which is what label validation must be checked against; reading the
    pre-noise segment text instead silently passes damaged values.

    `damaged` is the subset of sensitive fields deliberately corrupted under
    `id_ocr_relabel`. The caller uses it to re-point the gold label.
    """
    sev = profile.severity
    damaged = {}
    surfaces = {}
    parts = []

    for seg in segments:
        text = seg.text

        if seg.kind in (BOILERPLATE, LABEL):
            if profile.ocr and sev > 0:
                text = ocr_text(text, rng, rate=0.012 * sev)
                if rng.random() < 0.10 * sev:
                    text = space_noise(text, rng, rate=0.05)

        elif seg.kind == VALUE:
            if seg.sensitive:
                if profile.id_ocr_relabel and profile.ocr and sev >= 2 \
                        and rng.random() < 0.25:
                    text = ocr_text(text, rng, rate=0.05, max_subs=2)
                    if seg.field_key:
                        damaged[seg.field_key] = text
                # otherwise: characters are left intact by design
            else:
                if profile.ocr and sev >= 2 and rng.random() < 0.3:
                    text = ocr_text(text, rng, rate=0.02, max_subs=1)
                if sev >= 2 and rng.random() < 0.15:
                    text = space_noise(text, rng, rate=0.04)

        if seg.kind == VALUE and seg.field_key:
            surfaces[seg.field_key] = text
        parts.append(text)

    text = "".join(parts)
    text = layout_noise(text, rng, profile)
    text = add_furniture(text, rng, profile)
    text = mojibake(text, rng, profile)
    return text, damaged, surfaces


# ---------------------------------------------------------------- wrappers
_DISCLAIMERS = [
    "This e-mail and any attachments are confidential and intended solely for "
    "the addressee. If you have received it in error please notify the sender "
    "and delete it from your system.",
    "CONFIDENTIALITY NOTICE: The information contained in this transmission is "
    "privileged. Unauthorised disclosure is prohibited.",
    "Please consider the environment before printing this email.",
]


def email_wrapper(rng, body_lines, sender_name, sender_org, subject, profile):
    """Wrap a body in email furniture: headers, quoting, signature, disclaimer.

    Informal email is one of the three ingestion vectors named in the proposal,
    and it carries noise that no form or spreadsheet does -- reply chains,
    quote markers, and long legal footers that dwarf the payload.
    """
    out = []
    if rng.random() < 0.5:
        out.append("-------- Forwarded message --------")
    out.append(f"From: {sender_name} <{sender_name.split()[0].lower()}@"
               f"{sender_org.split()[0].lower().replace(',', '')}.com.gh>")
    out.append(f"Sent: {_stamp_date(rng)} {rng.randint(7, 19)}:{rng.randint(0, 59):02d}")
    out.append("To: Member Services")
    out.append(f"Subject: {subject}")
    out.append("")

    quote = rng.random() < 0.35
    for ln in body_lines:
        out.append(("> " + ln) if quote and ln.strip() else ln)

    out.append("")
    out.append(rng.choice(["Regards,", "Best regards,", "Kind regards,",
                           "Thank you.", "Yours faithfully,", "Regards"]))
    out.append(sender_name)
    out.append(rng.choice(["HR Officer", "Human Resource Manager",
                           "Payroll Officer", "Admin Officer",
                           "Head, Human Resource"]))
    out.append(sender_org)
    if rng.random() < 0.55:
        out.append("")
        out.append(rng.choice(_DISCLAIMERS))
    return out


# ==========================================================================
# from synth_function/schemas.py
# ==========================================================================

# -*- coding: utf-8 -*-




# Fields a receiving trustee might ask for that a Ghanaian pension document
# essentially never carries. They exist to teach null rather than invention.
DISTRACTORS = {
    "passportNumber": "string",
    "votersId": "string",
    "driversLicence": "string",
    "bankAccountNumber": "string",
    "nhisNumber": "string",
    "previousSchemeName": "string",
    "employerContributionRate": "number",
}

CONTAINERS = {
    "identity": ["member", "personalDetails", "bioData", "individual", "person"],
    "identifiers": ["identification", "ids", "identifiers", "nationalIds"],
    "contact": ["contact", "contactDetails", "communication", "reachability"],
    "employment": ["employment", "employerDetails", "work", "employmentInfo"],
    "pension": ["pension", "schemeDetails", "pensionDetails", "portability"],
    "nok": ["nextOfKin", "kin", "emergencyContact"],
}


@dataclass
class Leaf:
    schema_path: tuple      # where the value goes in the output
    field_key: str          # catalogue key, or None for a distractor


@dataclass
class ArraySpec:
    schema_path: tuple
    source: str                 # "beneficiaries" | "contributions"
    item_map: dict              # schema item name -> (source key, json type)


@dataclass
class SchemaPlan:
    schema: dict
    leaves: list = _dc_field(default_factory=list)
    arrays: list = _dc_field(default_factory=list)
    notation: str = "shorthand"
    shape: str = "flat"

    def field_keys(self):
        return {lf.field_key for lf in self.leaves if lf.field_key}


# ---------------------------------------------------------------- helpers
def _set_path(tree, path, value):
    cur = tree
    for part in path[:-1]:
        cur = cur.setdefault(part, {})
    cur[path[-1]] = value


def _alias(rng, fld, vary=True):
    return rng.choice(fld.aliases) if vary else fld.aliases[0]


def _shorthand_type(json_type, optional):
    return f"{json_type}|null" if optional else json_type


# ---------------------------------------------------------------- selection
def _select_fields(rng, prefer=(), p_visible=0.55, p_hidden=0.16, max_fields=16):
    """Pick a coherent subset of fields for the schema.

    `prefer` is the set of field keys the document actually renders. A
    receiving trustee's intake schema broadly overlaps the sending trustee's
    form without matching it, so fields the document shows are included more
    often than fields it does not. The two probabilities set the null rate in
    the corpus: raising p_hidden trains harder on returning null, at the cost
    of examples that carry less extractable signal.
    """
    prefer = set(prefer)
    chosen = []

    # Names are the spine of any portability record.
    for key in ("first_name", "last_name"):
        chosen.append(FIELDS[key])

    for f in FIELDS.values():
        if f.key in ("first_name", "last_name"):
            continue
        p = p_visible if f.key in prefer else p_hidden
        if rng.random() < p:
            chosen.append(f)

    # Cap the schema size. This is the main lever on example length, and the
    # serialised schema plus its mirrored output is over half of every example.
    if len(chosen) > max_fields:
        head = chosen[:2]
        chosen = head + rng.sample(chosen[2:], max_fields - 2)

    return chosen


# ---------------------------------------------------------------- shapes
def _insert(tree, path, value):
    """Place a field in the schema tree, or report a structural collision.

    Field aliases and container names are drawn from overlapping vocabularies,
    so one field can land on a path that is a strict prefix of another's: a leaf
    at ("data", "contact") blocks ("data", "contact", "email"), and vice versa.
    Returns False in that case so the caller skips the field rather than
    crashing or silently overwriting a subtree.
    """
    cur = tree
    for part in path[:-1]:
        node = cur.get(part)
        if node is None:
            node = cur[part] = {}
        elif not isinstance(node, dict):
            return False          # a leaf already occupies this branch
        cur = node
    last = path[-1]
    if last in cur:
        return False              # occupied, by a leaf or a whole subtree
    cur[last] = value
    return True


def _build_flat(rng, fields, vary):
    leaves, props = [], {}
    used = set()
    for f in fields:
        name = _alias(rng, f, vary)
        while name in used:
            name = name + "2"
        used.add(name)
        props[name] = f
        leaves.append(Leaf((name,), f.key))
    return leaves, props


def _build_grouped(rng, fields, vary):
    """Two levels: one container per catalogue group."""
    leaves = []
    tree = {}
    for f in fields:
        container = rng.choice(CONTAINERS.get(f.group, [f.group]))
        path = (container, _alias(rng, f, vary))
        if _insert(tree, path, f):
            leaves.append(Leaf(path, f.key))
    return leaves, tree


def _build_canonical(rng, fields, vary):
    """Mirror the canonical record, giving genuine three-level nesting such as
    contact.address.city and contact.phone.number."""
    leaves = []
    tree = {}
    for f in fields:
        path = list(f.path)
        if vary:
            path[-1] = _alias(rng, f, vary)
        path = tuple(path)
        if _insert(tree, path, f):
            leaves.append(Leaf(path, f.key))
    return leaves, tree


def _build_custom(rng, fields, vary):
    """Invented containers at a random depth: the receiving trustee's schema
    need not resemble the sending trustee's."""
    roots = ["data", "record", "payload", "memberRecord", "portabilityProfile"]
    root = rng.choice(roots)
    leaves = []
    tree = {}
    for f in fields:
        depth = rng.choices([1, 2, 3], weights=[25, 50, 25])[0]
        path = [root]
        if depth >= 2:
            path.append(rng.choice(CONTAINERS.get(f.group, [f.group])))
        if depth >= 3:
            path.append(rng.choice(["details", "attributes", "values", "info"]))
        path.append(_alias(rng, f, vary))
        path = tuple(path)
        if _insert(tree, path, f):
            leaves.append(Leaf(path, f.key))
    return leaves, tree


_SHAPES = {
    "flat": _build_flat,
    "grouped": _build_grouped,
    "canonical": _build_canonical,
    "custom": _build_custom,
}


# ---------------------------------------------------------------- arrays
def _maybe_arrays(rng, root_tree, shape, prefer_arrays=()):
    """Arrays are requested far more often when the document actually carries
    the corresponding table, for the same reason as _select_fields: a schema
    that asks for beneficiaries against a document with no beneficiary section
    only ever teaches null, and the array structure itself is never exercised."""
    prefer_arrays = set(prefer_arrays)
    specs = []

    if rng.random() < (0.80 if "beneficiaries" in prefer_arrays else 0.10):
        name = rng.choice(["beneficiaries", "nominees", "beneficiaryList"])
        keys = rng.sample(list(BENEFICIARY_FIELDS),
                          rng.randint(2, len(BENEFICIARY_FIELDS)))
        item_map = {}
        for k in keys:
            jtype, aliases, _sens = BENEFICIARY_FIELDS[k]
            item_map[rng.choice(aliases)] = (k, jtype)
        specs.append(ArraySpec((name,), "beneficiaries", item_map))

    if rng.random() < (0.75 if "contributions" in prefer_arrays else 0.08):
        name = rng.choice(["contributions", "contributionHistory", "remittances"])
        keys = rng.sample(list(CONTRIBUTION_FIELDS),
                          rng.randint(2, len(CONTRIBUTION_FIELDS)))
        item_map = {}
        for k in keys:
            jtype, aliases, _sens = CONTRIBUTION_FIELDS[k]
            item_map[rng.choice(aliases)] = (k, jtype)
        specs.append(ArraySpec((name,), "contributions", item_map))

    return specs


# ---------------------------------------------------------------- emit
def _emit_shorthand(tree, rng, optional_rate=0.75):
    """Render the field tree in the shorthand notation, recursively."""
    out = {}
    for name, node in tree.items():
        if isinstance(node, dict):
            out[name] = _emit_shorthand(node, rng, optional_rate)
        else:
            out[name] = _shorthand_type(node.type, rng.random() < optional_rate)
    return out


def _emit_jsonschema(tree, rng, required_rate=0.35, compact=False):
    """Draft-07.

    The type stays a ["string", "null"] union in both modes: the output emits
    null for every field the document does not show, so a scalar type would
    make the schema reject its own gold output. Compact mode economises by
    dropping the optional descriptions instead, and by carrying fewer fields.
    """
    props = {}
    required = []
    for name, node in tree.items():
        if isinstance(node, dict):
            props[name] = _emit_jsonschema(node, rng, required_rate, compact)
        else:
            entry = {"type": [node.type, "null"]}
            if not compact and node.type == "string" and rng.random() < 0.15:
                entry["description"] = "As printed on the document"
            props[name] = entry
        if rng.random() < required_rate:
            required.append(name)
    schema = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def _array_schema_shorthand(spec):
    return [{name: _shorthand_type(jtype, True)
             for name, (_src, jtype) in spec.item_map.items()}]


def _array_schema_jsonschema(spec, compact=False):
    return {
        "type": ["array", "null"],
        "items": {
            "type": "object",
            "properties": {name: {"type": [jtype, "null"]}
                           for name, (_src, jtype) in spec.item_map.items()},
        },
    }


# ---------------------------------------------------------------- entry
def generate_schema(rng, notation=None, shape=None, alias_variation=True,
                    prefer=(), p_visible=0.55, p_hidden=0.16,
                    prefer_arrays=(), max_fields=16, n_distractors=None,
                    compact=False):
    fields = _select_fields(rng, prefer, p_visible, p_hidden, max_fields)

    if shape is None:
        shape = rng.choices(["flat", "grouped", "canonical", "custom"],
                            weights=[30, 28, 27, 15])[0]
    if notation is None:
        notation = rng.choices(["shorthand", "jsonschema"], weights=[62, 38])[0]

    leaves, tree = _SHAPES[shape](rng, fields, alias_variation)

    # Distractors: requested but absent from any document.
    n_distract = (rng.choices([0, 1, 2], weights=[55, 33, 12])[0]
                  if n_distractors is None else n_distractors)
    for name in rng.sample(list(DISTRACTORS), n_distract):
        if shape == "flat":
            path = (name,)
        else:
            top = next(iter(tree)) if tree else "data"
            if not isinstance(tree.get(top), dict):
                continue
            path = (top, name)
        if _insert(tree, path, _Distractor(DISTRACTORS[name])):
            leaves.append(Leaf(path, None))

    arrays = _maybe_arrays(rng, tree, shape, prefer_arrays)
    # An array name can collide with a field alias already in the tree; drop
    # the array rather than overwrite a scalar the projection expects.
    arrays = [a for a in arrays if a.schema_path[0] not in tree]

    if notation == "shorthand":
        schema = _emit_shorthand(tree, rng)
        for spec in arrays:
            schema[spec.schema_path[0]] = _array_schema_shorthand(spec)
    else:
        schema = _emit_jsonschema(tree, rng, compact=compact)
        for spec in arrays:
            schema["properties"][spec.schema_path[0]] =                 _array_schema_jsonschema(spec, compact)

    return SchemaPlan(schema=schema, leaves=leaves, arrays=arrays,
                      notation=notation, shape=shape)


class _Distractor:
    """Stands in for a catalogue Field so the emitters can treat it uniformly."""
    __slots__ = ("type",)

    def __init__(self, type_):
        self.type = type_


# ---------------------------------------------------------------- projection
def project(plan, record, visible, damaged=None, array_visible=()):
    """Build the gold output.

    `visible` is the set of catalogue field keys the document actually renders.
    Anything outside it becomes null -- that is the anti-hallucination signal.
    `damaged` re-points a label at the corrupted surface string under the
    id_ocr_relabel policy.
    """
    damaged = damaged or {}
    out = {}

    for lf in plan.leaves:
        if lf.field_key is None:
            _set_path(out, lf.schema_path, None)
            continue
        if lf.field_key not in visible:
            _set_path(out, lf.schema_path, None)
            continue
        if lf.field_key in damaged:
            _set_path(out, lf.schema_path, damaged[lf.field_key])
            continue
        _set_path(out, lf.schema_path, get_path(record, FIELDS[lf.field_key].path))

    for spec in plan.arrays:
        if spec.source not in array_visible:
            _set_path(out, spec.schema_path, None)
            continue
        items = []
        for src_item in record.get(spec.source, []):
            items.append({name: src_item.get(src_key)
                          for name, (src_key, _t) in spec.item_map.items()})
        _set_path(out, spec.schema_path, items or None)

    return out


# ==========================================================================
# from synth_function/documents.py
# ==========================================================================

# -*- coding: utf-8 -*-



_DATE_KEYS = {"date_of_birth", "date_of_engagement", "last_contribution_date"}
_MONEY_KEYS = {"monthly_salary", "accrued_benefit"}
_PHONE_KEYS = {"phone_number", "alt_phone", "nok_phone"}

_EMPTY_MARKERS = ["", "N/A", "n/a", "-", "--", "NIL", "Nil", "....", "____",
                  "Not provided", "TBA", "?"]


class Doc:
    """Accumulates segments and tracks which fields were actually rendered."""

    # In compact mode every renderer is shortened the same way -- by a budget
    # on how many labelled fields it may print -- rather than by excluding the
    # longer document types outright. Excluding them cost the corpus its two
    # richest layouts and skewed it towards the sparsest one.
    COMPACT_FIELD_BUDGET = 9
    COMPACT_TABLE_ROWS = 2

    def __init__(self, rng, record, profile):
        self.rng = rng
        self.record = record
        self.profile = profile
        self.compact = getattr(profile, "compact", False)
        self.field_budget = self.COMPACT_FIELD_BUDGET if self.compact else None
        self.segments = []
        self.visible = set()
        self.arrays = set()

    # -- raw output -----------------------------------------------------
    def line(self, text=""):
        self.segments.append(B(text + "\n"))
        return self

    def blank(self, n=1):
        self.segments.append(B("\n" * n))
        return self

    def rule(self, char="-", width=None):
        width = width or self.rng.choice([28, 36, 44, 52])
        return self.line(char * width)

    def heading(self, text):
        if self.rng.random() < 0.6:
            text = text.upper()
        return self.line(text)

    # -- values ---------------------------------------------------------
    def _display(self, key, value):
        rng, sev = self.rng, self.profile.severity
        if key in _DATE_KEYS:
            return date_display(rng, value)
        if key in _MONEY_KEYS:
            return money_display(rng, value)
        if key in _PHONE_KEYS:
            return phone_display(rng, value)
        fld = FIELDS[key]
        if fld.sensitive:
            return id_display(rng, value, sev)
        if key in ("first_name", "middle_name", "last_name", "nok_name",
                   "employer_name"):
            return name_display(rng, str(value), sev)
        return str(value)

    def field(self, key, label=None, sep=": ", show_empty=True):
        """Render one labelled field. Marks it visible only when a value is
        actually printed."""
        if self.field_budget is not None:
            if self.field_budget <= 0:
                return self
            self.field_budget -= 1

        fld = FIELDS[key]
        value = get_path(self.record, fld.path)
        lbl = label or self.rng.choice(fld.labels)

        self.segments.append(L(lbl))
        self.segments.append(B(sep))

        if value is None or value == "":
            if not show_empty:
                self.segments.pop()
                self.segments.pop()
                return self
            self.segments.append(B(self.rng.choice(_EMPTY_MARKERS)))
        else:
            self.segments.append(
                V(self._display(key, value), key, fld.sensitive))
            self.visible.add(key)

        self.segments.append(B("\n"))
        return self

    def name_line(self, label=None):
        """A combined name line, which is how most Ghanaian forms print it.
        Registers both components as visible."""
        rng = self.rng
        m = self.record["member"]
        first, middle, last = m["firstName"], m.get("middleName"), m["lastName"]

        style = rng.choices(["first_last", "surname_first", "full"],
                            weights=[45, 25, 30])[0]
        if style == "surname_first":
            text = f"{last.upper()}, {first}"
        elif style == "full" and middle:
            text = f"{first} {middle} {last}"
        else:
            text = f"{first} {last}"

        if rng.random() < 0.3 and m.get("title"):
            text = f"{m['title']} {text}"
            self.visible.add("title")

        self.segments.append(L(label or rng.choice(
            ["Name", "Member Name", "Full Name", "Name of Member",
             "Name of Contributor"])))
        self.segments.append(B(": "))
        self.segments.append(V(name_display(rng, text, self.profile.severity)))
        self.segments.append(B("\n"))

        self.visible.add("first_name")
        self.visible.add("last_name")
        if style == "full" and middle:
            self.visible.add("middle_name")
        return self

    # -- tables ---------------------------------------------------------
    def _cap(self, key, rows):
        """Shorten a table in compact mode, writing the truncation back to the
        record. The projection reads the record, so capping only the rendered
        rows would label rows the document never printed."""
        if self.compact and len(rows) > self.COMPACT_TABLE_ROWS:
            rows = rows[:self.COMPACT_TABLE_ROWS]
            self.record[key] = rows
        return rows

    def beneficiary_table(self):
        rows = self.record.get("beneficiaries") or []
        if not rows:
            return self
        rows = self._cap("beneficiaries", rows)
        rng = self.rng
        style = rng.choice(["pipe", "space", "broken"])

        self.line(rng.choice(["BENEFICIARY NOMINATION", "Nominated Beneficiaries",
                              "BENEFICIARIES"]))
        if style == "pipe":
            self.line("Name | Relationship | Phone | Allocation (%)")
            self.line("-----|--------------|-------|---------------")
        elif style == "space":
            self.line("Name                Relationship   Phone         %")

        for r in rows:
            name = name_display(rng, r["fullName"], self.profile.severity)
            phone = phone_display(rng, r["phone"])
            if style == "pipe":
                parts = [(name, False), (r["relationship"], False),
                         (phone, True), (f"{r['allocationPercent']}", True)]
                sep = " | " if rng.random() < 0.8 else "|"
            elif style == "space":
                parts = [(name.ljust(20), False), (r["relationship"].ljust(15), False),
                         (phone.ljust(14), True), (f"{r['allocationPercent']}", True)]
                sep = ""
            else:
                # Delimiters lost in transmission: the "fragmented text table"
                # vector from the proposal.
                parts = [(name, False), (r["relationship"], False),
                         (phone, True), (f"{r['allocationPercent']}%", True)]
                sep = rng.choice(["   ", "  ", " ,", "\t"])

            for i, (txt, sensitive) in enumerate(parts):
                if i:
                    self.segments.append(B(sep))
                self.segments.append(V(txt, None, sensitive))
            self.segments.append(B("\n"))

        self.arrays.add("beneficiaries")
        return self

    def contribution_table(self):
        rows = self.record.get("contributions") or []
        if not rows:
            return self
        rows = self._cap("contributions", rows)
        rng = self.rng

        self.line(rng.choice(["CONTRIBUTION HISTORY", "Remittance Schedule",
                              "Contribution Schedule"]))
        header = rng.choice([
            "Period | Employee | Employer",
            "Month      Employee Amt    Employer Amt",
            "PERIOD,EMPLOYEE,EMPLOYER",
        ])
        self.line(header)

        sep = " | " if "|" in header else ("," if "," in header else "    ")
        for r in rows:
            self.segments.append(V(r["period"], None, True))
            self.segments.append(B(sep))
            self.segments.append(V(money_display(rng, r["employeeAmount"]), None, True))
            self.segments.append(B(sep))
            self.segments.append(V(money_display(rng, r["employerAmount"]), None, True))
            self.segments.append(B("\n"))

        self.arrays.add("contributions")
        return self


# ---------------------------------------------------------------- renderers
def membership_form(rng, record, profile):
    d = Doc(rng, record, profile)
    d.heading(rng.choice(["Membership Registration Form",
                          "Scheme Member Enrolment Form",
                          "Pension Scheme Application Form"]))
    d.rule().blank()

    d.line("SECTION A - PERSONAL DETAILS")
    for key in ["title", "first_name", "middle_name", "last_name", "gender",
                "date_of_birth", "marital_status", "nationality"]:
        if rng.random() < 0.85:
            d.field(key)
    d.blank()

    d.line("SECTION B - IDENTIFICATION")
    for key in ["ghana_card", "ssnit_number", "tin"]:
        if rng.random() < 0.9:
            d.field(key)
    d.blank()

    d.line("SECTION C - CONTACT")
    for key in ["phone_number", "alt_phone", "email", "digital_address",
                "street", "city", "region"]:
        if rng.random() < 0.75:
            d.field(key)
    d.blank()

    d.line("SECTION D - EMPLOYMENT")
    for key in ["employer_name", "staff_id", "occupation",
                "date_of_engagement", "monthly_salary"]:
        if rng.random() < 0.8:
            d.field(key)

    if rng.random() < 0.5:
        d.blank().line("SECTION E - NEXT OF KIN")
        for key in ["nok_name", "nok_relationship", "nok_phone"]:
            d.field(key)

    if rng.random() < 0.35:
        d.blank()
        d.beneficiary_table()

    return d


def transfer_request_letter(rng, record, profile):
    d = Doc(rng, record, profile)
    cur = record["pension"]["currentTrustee"]
    rec = record["pension"]["receivingTrustee"]

    d.line(cur)
    d.line(f"{locality(rng)[0]}, Ghana")
    d.blank()
    d.line(f"Date: {date_display(rng, date_between(rng, 2025, 2026).isoformat())}")
    d.blank()
    d.line("The Head of Operations")
    d.line(rec)
    d.blank()
    d.line("Dear Sir/Madam,")
    d.blank()
    d.heading("RE: REQUEST FOR TRANSFER OF ACCRUED PENSION BENEFITS")
    d.blank()
    d.line("We write to request the transfer of the accrued benefits of the")
    d.line("under-mentioned member in accordance with the portability")
    d.line("provisions of the National Pensions Act, 2008 (Act 766).")
    d.blank()

    d.name_line()
    for key in ["member_number", "ssnit_number", "ghana_card",
                "employer_name", "scheme_name", "scheme_type", "tier",
                "accrued_benefit", "last_contribution_date"]:
        if rng.random() < 0.8:
            d.field(key)

    d.field("current_trustee", label=rng.choice(
        ["Current Trustee", "Transferring From", "Ceding Trustee"]))
    d.field("receiving_trustee", label=rng.choice(
        ["Receiving Trustee", "Transferring To", "New Trustee"]))

    d.blank()
    d.line("Kindly treat as urgent.")
    d.blank()
    d.line("Yours faithfully,")
    d.blank()
    d.line("....................................")
    d.line("Scheme Administrator")
    d.line(cur)
    return d


def hr_email(rng, record, profile):
    """Informal email from a corporate HR desk: loose prose, few labels."""
    d = Doc(rng, record, profile)
    m = record["member"]
    sender = person(rng)
    sender_name = full_name(sender, include_middle=False)
    org = record["employment"]["employerName"]

    body = Doc(rng, record, profile)
    body.line(rng.choice([
        "Good morning,", "Hello,", "Dear Team,", "Good afternoon,", "Hi,",
    ]))
    body.blank()
    body.line(rng.choice([
        "Please find below the details of our staff who is moving his pension",
        "Kindly effect the transfer of the under-listed staff member from our",
        "We wish to transfer the following employee's pension account from our",
    ]))
    body.line(rng.choice([
        "account to your outfit with effect from next month.",
        "current trustee to yourselves. Details are as follows.",
        "existing provider. See particulars below.",
    ]))
    body.blank()

    body.name_line()
    for key in ["ssnit_number", "ghana_card", "member_number", "phone_number",
                "date_of_birth", "occupation", "employer_name",
                "current_trustee", "receiving_trustee", "accrued_benefit"]:
        if rng.random() < 0.62:
            body.field(key, sep=rng.choice([": ", " - ", " : ", ":  "]))

    body.blank()
    body.line(rng.choice([
        "Let me know if you need anything else.",
        "Kindly confirm receipt.",
        "Please advise on next steps.",
        "Awaiting your feedback.",
    ]))

    # Rebuild as email: take the body's text lines but keep its segments.
    d.segments.extend(_email_segments(rng, body, sender_name, org, profile))
    d.visible |= body.visible
    d.arrays |= body.arrays
    return d


def _email_segments(rng, body, sender_name, org, profile):
    """Wrap the body segments in email furniture without disturbing them."""
    subject = rng.choice([
        "Pension Transfer Request", "Transfer of Staff Pension Account",
        "FW: Member Portability - Urgent", "Request for Fund Transfer",
        "Staff Pension Movement",
    ])
    head = email_wrapper(rng, [], sender_name, org, subject, profile)
    # email_wrapper returns header lines, sign-off and disclaimer around an
    # empty body; split it back apart at the blank line after Subject.
    cut = head.index("") + 1
    pre, post = head[:cut], head[cut:]

    segs = [B("\n".join(pre) + "\n")]
    segs += body.segments
    segs.append(B("\n".join(post) + "\n"))
    return segs


def payroll_schedule(rng, record, profile):
    """A spreadsheet exported to text with its alignment lost."""
    d = Doc(rng, record, profile)
    d.heading(rng.choice(["Monthly Contribution Schedule",
                          "Employer Remittance Schedule",
                          "TIER 2 CONTRIBUTION REPORT"]))
    d.field("employer_name")
    if rng.random() < 0.7:
        d.field("tin", label="Employer TIN")
    d.blank()

    header = rng.choice([
        "S/N | Staff ID | Name | SSNIT No | Basic Salary | Employee | Employer",
        "SN,STAFF ID,NAME,SSNIT,BASIC,EE,ER",
        "S/N  Staff ID   Name              SSNIT No      Basic",
    ])
    d.line(header)
    sep = " | " if "|" in header else ("," if "," in header else "   ")

    m = record["member"]
    salary = record["employment"]["monthlySalary"]
    row = [
        ("1", False),
        (str(record["employment"]["staffId"] or "-"), True),
        (f"{m['firstName']} {m['lastName']}", False),
        (str(record["identification"]["ssnitNumber"] or "-"), True),
        (money_display(rng, salary) if salary else "-", True),
    ]
    for i, (txt, sens) in enumerate(row):
        if i:
            d.segments.append(B(sep))
        d.segments.append(V(txt, None, sens))
    d.segments.append(B("\n"))

    if record["employment"]["staffId"]:
        d.visible.add("staff_id")
    if record["identification"]["ssnitNumber"]:
        d.visible.add("ssnit_number")
    if salary:
        d.visible.add("monthly_salary")
    d.visible.add("first_name")
    d.visible.add("last_name")
    d.visible.add("employer_name")

    # A few unrelated staff rows, so the model must pick the right one.
    for i in range(rng.randint(0, 3)):
        other = person(rng)
        cells = [str(i + 2), staff_id(rng),
                 f"{other['firstName']} {other['lastName']}",
                 ssnit(rng), money_display(rng, money(rng, 900, 9000))]
        d.line(sep.join(cells))

    if record.get("contributions") and rng.random() < 0.6:
        d.blank()
        d.contribution_table()
    return d


def correction_notice(rng, record, profile):
    """A wrong value followed by its correction. The gold is the corrected one."""
    d = Doc(rng, record, profile)
    m = record["member"]
    wrong = person(rng, tradition=m.get("tradition"))

    d.heading("Member Record Update")
    d.rule().blank()
    d.line("The following record was submitted in error:")
    d.blank()
    d.line(f"Name: {wrong['firstName']} {record['member']['lastName']}")
    if record["identification"]["ssnitNumber"] and rng.random() < 0.5:
        d.line(f"SSNIT Number: {ssnit(rng)}")
    d.blank()
    d.line(rng.choice(["CORRECTION", "*** CORRECTION NOTICE ***",
                       "Please note the corrected details below:",
                       "AMENDMENT"]))
    d.blank()

    d.name_line(label="Correct Name")
    for key in ["ssnit_number", "ghana_card", "member_number", "date_of_birth"]:
        if rng.random() < 0.7:
            d.field(key)

    d.blank()
    d.line("Kindly update your records accordingly.")
    return d


def employment_conflict(rng, record, profile):
    """Old and current employer both present. The gold is the current one."""
    d = Doc(rng, record, profile)
    old = employer(rng)

    d.heading("Employment Status Update")
    d.rule().blank()
    d.line(f"Previous Employer: {old}")
    d.line(f"Date of Exit: {date_display(rng, date_between(rng, 2018, 2024).isoformat())}")
    d.blank()
    d.field("employer_name", label=rng.choice(
        ["Current Employer", "Present Employer", "New Employer"]))
    d.field("date_of_engagement")
    if rng.random() < 0.7:
        d.field("occupation")
    d.blank()
    d.name_line()
    for key in ["ssnit_number", "member_number", "staff_id"]:
        if rng.random() < 0.75:
            d.field(key)
    return d


def beneficiary_nomination(rng, record, profile):
    d = Doc(rng, record, profile)
    d.heading("Beneficiary Nomination Form")
    d.rule().blank()
    d.name_line()
    for key in ["member_number", "ssnit_number", "scheme_name"]:
        if rng.random() < 0.8:
            d.field(key)
    d.blank()
    d.beneficiary_table()
    d.blank()
    if rng.random() < 0.6:
        d.line("I confirm the above nominations are correct.")
        d.line("Signature: ..............   Date: ..............")
    return d


def benefit_statement(rng, record, profile):
    d = Doc(rng, record, profile)
    d.heading(rng.choice(["Member Benefit Statement", "Statement of Account",
                          "ANNUAL BENEFIT STATEMENT"]))
    # Rendered as a value segment, not boilerplate: it is marked visible, so
    # its characters must be protected from the aggressive boilerplate OCR.
    d.segments.append(V(record["pension"]["currentTrustee"], "current_trustee"))
    d.segments.append(B("\n"))
    d.visible.add("current_trustee")
    d.rule().blank()
    d.name_line()
    for key in ["member_number", "ssnit_number", "scheme_name", "scheme_type",
                "tier", "employer_name", "accrued_benefit",
                "last_contribution_date"]:
        if rng.random() < 0.82:
            d.field(key)
    d.blank()
    if record.get("contributions"):
        d.contribution_table()
    return d


def handwritten_note(rng, record, profile):
    """A sparse, half-legible transcription. Few fields, lots of null."""
    d = Doc(rng, record, profile)
    d.line(rng.choice(["[transcribed from handwritten note]",
                       "-- handwritten request --",
                       "MEMO (handwritten)"]))
    d.blank()
    d.line(rng.choice([
        "pls transfer my pension to the new trustee. thank u",
        "I want to move my pension acct. details below",
        "Kindly transfer my funds as discussed.",
    ]))
    d.blank()
    d.name_line()
    keys = [k for k in ["ssnit_number", "phone_number", "employer_name",
                        "member_number", "ghana_card"]
            if rng.random() < 0.5]
    for key in keys[:3]:
        d.field(key, sep=rng.choice([": ", " ", " - "]))
    d.blank()
    if rng.random() < 0.5:
        d.line("[remainder illegible]")
    return d


def scanned_intake(rng, record, profile):
    """A form that has been through a bad scanner: labels on their own lines,
    values adrift below them."""
    d = Doc(rng, record, profile)
    d.heading("Member Data Capture Sheet")
    d.rule("=").blank()
    d.name_line()
    for key in ["date_of_birth", "gender", "ghana_card", "ssnit_number",
                "phone_number", "email", "digital_address", "city", "region",
                "employer_name", "occupation", "member_number"]:
        if rng.random() < 0.7:
            d.field(key, sep=":\n")
    return d


RENDERERS = [
    (membership_form, 16),
    (transfer_request_letter, 15),
    (hr_email, 15),
    (payroll_schedule, 11),
    (correction_notice, 10),
    (employment_conflict, 9),
    (beneficiary_nomination, 8),
    (benefit_statement, 8),
    (handwritten_note, 4),
    (scanned_intake, 4),
]


def render_document(rng, record, profile, compact=False):
    # The same weights in both modes. Compact documents are shortened by the
    # field budget in Doc, not by dropping whole document types -- the corpus
    # keeps its full range of layouts either way.
    fns = [f for f, _w in RENDERERS]
    weights = [w for _f, w in RENDERERS]
    fn = rng.choices(fns, weights=weights)[0]
    doc = fn(rng, record, profile)
    return doc, fn.__name__


# ==========================================================================
# from synth_function/pipeline.py
# ==========================================================================

# -*- coding: utf-8 -*-




# ---------------------------------------------------------------- instructions
# The normalisation contract is spelled out, because the gold labels demand it:
# a document showing "12/03/1985" is labelled "1985-03-12", and one showing
# "GHS 48,215.60" is labelled 48215.60. A model cannot be expected to infer
# those conventions from the schema alone.

_NORMALISATION = """Normalisation rules:
- Dates: return ISO format YYYY-MM-DD.
- Amounts: return a number, without currency symbols or thousand separators.
- Phone numbers: return the local 10-digit form beginning with 0.
- Names: return them in their normal written case, not block capitals.
- Identifiers: return them as printed, less any stray internal spaces."""

INSTRUCTIONS = [
    """You are a structured data extraction engine.

Extract information from the document according to the supplied schema.

Requirements:
- Follow the schema exactly, including its nesting.
- Return valid JSON only, with no commentary.
- Extract only information supported by the document.
- Return null for any field the document does not contain.
- Do not invent information.
- Use corrected values where a correction is present.
- Use the most recent value where several conflict.
- Correct obvious OCR errors in words; never guess at damaged digits.

""" + _NORMALISATION,

    """Extract the fields named in the schema from the document below.

Return one JSON object matching the schema's structure exactly. Any field that
does not appear in the document must be null. Do not infer, calculate or
invent values. Where the document corrects itself, use the corrected value.

""" + _NORMALISATION,

    """Read the document and populate the supplied JSON schema.

The document may be noisy: it may come from a scanner, an email body or a
broken spreadsheet export. Ignore page headers, footers, stamps, signatures
and email disclaimers. Return null rather than guessing. Return JSON only.

""" + _NORMALISATION,
]


# Compact instructions for small context windows. The standard instructions run
# to ~640 characters, roughly a quarter of a whole example -- constant overhead
# that buys little once the model has seen it a few hundred times. These keep
# the operative rules and drop the exposition.
TERSE_INSTRUCTIONS = [
    "Extract the schema fields from the document. Return JSON only. Use null "
    "for anything absent. Dates as YYYY-MM-DD, amounts as numbers, phones as "
    "10 digits from 0. Do not invent values.",

    "Populate the schema from the document. JSON only, null when missing, "
    "corrected values where corrected. Dates YYYY-MM-DD; amounts numeric; "
    "phones local 10-digit.",

    "Return the schema filled from the document as JSON. null for absent "
    "fields. Normalise dates to YYYY-MM-DD, amounts to numbers, phones to the "
    "local 10-digit form. Never guess.",
]


# ---------------------------------------------------------------- validation
_PUNCT = re.compile(r"[^0-9a-z]+")


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return _PUNCT.sub("", s.casefold())


def _within_one_edit(a, b):
    """True when a and b differ by at most one substitution, insertion or
    deletion. Used to accept a mildly OCR-damaged word as still recoverable."""
    if abs(len(a) - len(b)) > 1:
        return False
    if a == b:
        return True
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(long)):
        if short == long[:i] + long[i + 1:]:
            return True
    return False


def _digits(s):
    return re.sub(r"\D", "", str(s))


_MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december"]


def _recoverable(key, canonical, surface, doc_norm):
    """Can the gold value be derived from what the document shows?"""
    if surface is None:
        # Rendered inside a combined line (e.g. a name); fall back to the whole
        # document, with the same one-edit tolerance applied elsewhere so a
        # mildly OCR-damaged word is not discarded as unrecoverable.
        cn = _norm(canonical)
        if cn in doc_norm:
            return True
        return len(cn) >= 5 and any(
            _within_one_edit(cn, doc_norm[i:i + w])
            for w in (len(cn), len(cn) - 1, len(cn) + 1)
            for i in range(max(0, len(doc_norm) - w + 1))
        )

    cn, sn = _norm(canonical), _norm(surface)

    if key in ("date_of_birth", "date_of_engagement", "last_contribution_date"):
        y, m, d = str(canonical).split("-")
        sd = _digits(surface)
        # The month may be printed as a name ("12 Mar 1985"), in which case it
        # contributes no digits at all.
        name = _MONTH_NAMES[int(m) - 1]
        month_ok = str(int(m)) in sd or name[:3] in sn or name in sn
        return (y in sd or y[2:] in sd) and month_ok and str(int(d)) in sd

    if key in ("monthly_salary", "accrued_benefit"):
        return _digits(f"{canonical:.2f}") in _digits(surface)

    if key in ("phone_number", "alt_phone", "nok_phone"):
        return _digits(canonical)[-9:] in _digits(surface)

    if cn == sn or cn in sn:
        return True
    return len(cn) >= 5 and _within_one_edit(cn, sn)


def validate(plan, record, output, visible, surfaces, doc_text):
    """Return the set of field keys whose labels are not recoverable."""
    doc_norm = _norm(doc_text)
    bad = set()
    for lf in plan.leaves:
        if lf.field_key is None or lf.field_key not in visible:
            continue
        canonical = get_path(record, FIELDS[lf.field_key].path)
        if canonical is None:
            continue
        if not _recoverable(lf.field_key, canonical, surfaces.get(lf.field_key), doc_norm):
            bad.add(lf.field_key)
    return bad


# ---------------------------------------------------------------- assembly
def _generate_example_core(rng=None, profile=None, on_unrecoverable="drop",
                     alias_variation=True, notation=None, shape=None,
                     with_meta=False, compact=False, max_chars=None,
                     max_tokens=None):
    """Build one training example.

    on_unrecoverable:
      "drop"   the offending field is treated as not visible, so its label
               becomes null. Safe, and the default.
      "retry"  rebuild the example from scratch (bounded).
      "keep"   emit anyway. Only for inspecting the noise layer.

    compact:   target a small context window (~512 tokens). Uses the terse
               instructions, caps the schema at 8 fields, prefers the shorthand
               notation and the shorter document types, and drops page
               furniture. See the README for the measured effect.

    max_tokens: reject and regenerate examples estimated to exceed this many
               tokens. Prefer this over max_chars: heavily OCR-damaged text
               tokenises far worse per character than clean text, so a
               character budget cannot bound the token count.

    max_chars: a cheaper character budget, kept for callers that want it.
    """
    rng = rng or random.Random()
    best = None
    budgeted = max_chars is not None or max_tokens is not None

    for _attempt in range(12 if budgeted else 6):
        # Progressive tightening. Rejecting an over-long example and re-rolling
        # the same settings just reproduces it, so each failed attempt shrinks
        # the schema and, eventually, forces the terser notation. Without this
        # the combinations that are inherently large -- an array plus draft-07
        # plus a full field set -- never converge and fall through to the
        # over-budget fallback, which is what let arrays drop out of the corpus.
        squeeze = max(0, _attempt - 2) if budgeted else 0
        prof = profile or NoiseProfile.sample(rng)
        if compact:
            prof.furniture = False
            prof.compact = True
        record = build_record(rng)
        doc, doc_type = render_document(rng, record, prof, compact=compact)

        text, damaged, surfaces = render(doc.segments, rng, prof)

        plan = generate_schema(
            rng,
            notation=notation or (
                "shorthand" if (compact and squeeze >= 4) else
                rng.choices(["shorthand", "jsonschema"], weights=[65, 35])[0]
                if compact else None),
            shape=shape,
            alias_variation=alias_variation,
            prefer=doc.visible,
            prefer_arrays=doc.arrays,
            # An array section costs as much as several scalar fields, so a
            # schema carrying one gets a smaller scalar budget rather than
            # being discarded.
            max_fields=max(3, (5 if (compact and doc.arrays) else
                               8 if compact else 16) - squeeze),
            compact=compact,
            n_distractors=(rng.choices([0, 1], weights=[70, 30])[0]
                           if compact else None),
        )

        visible = set(doc.visible)
        bad = validate(plan, record, None, visible, surfaces, text)

        if bad and on_unrecoverable == "retry":
            continue
        if bad and on_unrecoverable == "drop":
            visible -= bad

        output = project(plan, record, visible, damaged, doc.arrays)

        example = {
            "instruction": rng.choice(
                TERSE_INSTRUCTIONS if compact else INSTRUCTIONS),
            "schema": plan.schema,
            "input": text.strip("\n"),
            "output": output,
        }
        # Attached before the budget check, so that the fallback below returns
        # a fully-formed example too: with_meta must always produce _meta.
        if with_meta:
            example["_meta"] = {
                "document_type": doc_type,
                "schema_shape": plan.shape,
                "schema_notation": plan.notation,
                "severity": prof.severity,
                "visible_fields": sorted(visible),
                "dropped_unrecoverable": sorted(bad),
                "arrays": sorted(doc.arrays),
                "null_leaves": sum(1 for lf in plan.leaves
                                   if lf.field_key not in visible),
                "total_leaves": len(plan.leaves),
                "over_budget": False,
            }

        if budgeted:
            over = False
            rank = 0
            if max_tokens is not None:
                est = estimate_tokens(example)
                rank = est / max_tokens
                over = over or est > max_tokens
            if max_chars is not None:
                size = serialised_length(example)
                rank = max(rank, size / max_chars)
                over = over or size > max_chars
            if best is None or rank < best[1]:
                best = (example, rank)
            if over:
                continue

        return example

    if best is not None:
        # The budget could not be met. Return the shortest candidate rather
        # than failing, so a long generation run never dies part-way through.
        example = best[0]
        if with_meta:
            example["_meta"]["over_budget"] = True
        return example

    raise RuntimeError("could not build a valid example")


def serialise(example):
    """The example as the training notebook lays it out, minus the EOS."""
    return (f"### Instruction:\n{example['instruction']}\n\n"
            f"### Schema:\n"
            f"{json.dumps(example['schema'], ensure_ascii=False, indent=2)}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n"
            f"{json.dumps(example['output'], ensure_ascii=False, indent=2)}")


def serialised_length(example):
    """Character length of the example as the training notebook serialises it."""
    return len(serialise(example))


# Counting word/digit/punctuation pieces tracks BPE far more closely than
# counting characters does, because it is insensitive to how long the words
# are -- and OCR noise mostly changes word length, not word count. Calibrated
# against the Qwen2.5 tokenizer over 2,500 examples, actual/estimate ranged
# 0.88-1.10; the 1.12 multiplier turns the estimate into a safe upper bound.
_PIECE = re.compile(r"[A-Za-z]+|\d|[^\sA-Za-z\d]|\n")
_TOKEN_SAFETY = 1.12


def estimate_tokens(example, safety=_TOKEN_SAFETY):
    """Upper-bound estimate of the tokenised length, tokenizer-free.

    Deliberately conservative: this exists so that generate_example() can hold
    a context-window budget when no tokenizer is available. When one is (see
    notebook.build_dataset), measure instead of estimating.
    """
    text = example if isinstance(example, str) else serialise(example)
    return int(len(_PIECE.findall(text)) * safety) + 2   # +2 for EOS and slack


# Both entry-point names are exported: the notebooks import `generate_synth_data`
# in one cell and `generate_example` in another.


# ---------------------------------------------------------------- dataset
def generate_dataset(n, seed=None, path=None, with_meta=False, **kwargs):
    """Generate n examples. With a seed, the corpus is exactly reproducible."""
    rng = random.Random(seed)
    items = [generate_example(rng, with_meta=with_meta, **kwargs) for _ in range(n)]

    if path:
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                # A real newline, so the file is one JSON object per line.
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return items


# ==========================================================================
# from synth_function/notebook.py
# ==========================================================================

# -*- coding: utf-8 -*-




PROMPT_FORMAT = """### Instruction:
{instruction}

### Schema:
{schema}

### Input:
{input}

### Response:
{output}{eos}"""


def format_example(example, eos_token=""):
    """Serialise one example exactly as the training notebook does."""
    return PROMPT_FORMAT.format(
        instruction=example["instruction"],
        schema=json.dumps(example["schema"], ensure_ascii=False, indent=2),
        input=example["input"],
        output=json.dumps(example["output"], ensure_ascii=False, indent=2),
        eos=eos_token,
    )


def _token_lengths(tokenizer, texts, batch=256):
    lens = []
    for i in range(0, len(texts), batch):
        enc = tokenizer(texts[i:i + batch], add_special_tokens=False)
        lens.extend(len(ids) for ids in enc["input_ids"])
    return lens


def build_dataset(
    tokenizer,
    n_samples,
    seed=42,
    test_size=0.1,
    max_seq_length=512,
    compact=None,
    save_dir=None,
    split_seed=42,
    max_attempts=6,
    verbose=True,
    **generate_kwargs,
):
    """Generate a training corpus that fits the model's context window.

    Returns (data, train_dataset, eval_dataset) where `data` is the list of
    {"text": ...} records and the two datasets are 🤗 `Dataset` objects, matching
    what the notebook's existing cells produce.

    tokenizer       the loaded tokenizer; used for EOS and for measuring length
    n_samples       number of examples to return
    seed            seeds generation, so the corpus is reproducible
    max_seq_length  hard budget; examples over it are regenerated, not truncated
    compact         defaults to True when max_seq_length <= 1024
    save_dir        if given, writes synthetic_dataset / train_dataset /
                    eval_dataset .jsonl there, as the notebook does today
    """
    from datasets import Dataset

    if compact is None:
        compact = max_seq_length <= 1024

    eos = getattr(tokenizer, "eos_token", "") or ""
    rng = random.Random(seed)

    # Character prefilter, so most candidates are already inside the budget
    # before the tokenizer is consulted. ~3.4 chars/token is conservative for
    # this mix of prose, JSON punctuation and identifiers.
    char_budget = generate_kwargs.pop("max_chars", None)
    if char_budget is None and compact:
        char_budget = int(max_seq_length * 3.4)

    kept, kept_lens = [], []
    generated = discarded = 0

    for attempt in range(max_attempts):
        need = n_samples - len(kept)
        if need <= 0:
            break

        # Over-generate to absorb the ones that come back too long.
        batch_n = int(need * (1.25 if attempt else 1.1)) + 8
        batch = []
        for _ in range(batch_n):
            ex = generate_example(rng, compact=compact,
                                  max_chars=char_budget, **generate_kwargs)
            batch.append(format_example(ex, eos))
        generated += len(batch)

        lens = _token_lengths(tokenizer, batch)
        for text, n_tok in zip(batch, lens):
            if n_tok <= max_seq_length:
                kept.append(text)
                kept_lens.append(n_tok)
            else:
                discarded += 1
            if len(kept) >= n_samples:
                break

    if len(kept) < n_samples and verbose:
        print(f"WARNING: produced {len(kept)} of {n_samples} requested examples "
              f"within {max_seq_length} tokens after {max_attempts} attempts. "
              f"Lower max_seq_length pressure by raising it, or set "
              f"compact=True.")

    kept, kept_lens = kept[:n_samples], kept_lens[:n_samples]
    data = [{"text": t} for t in kept]

    dataset = Dataset.from_list(data).train_test_split(
        test_size=test_size, seed=split_seed)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]

    if verbose and kept_lens:
        s = sorted(kept_lens)
        pct = lambda q: s[min(int(len(s) * q), len(s) - 1)]
        print(f"synth_function: {len(data)} examples "
              f"(seed={seed}, compact={compact})")
        print(f"  token length   p50 {pct(.50)}  p90 {pct(.90)}  "
              f"p99 {pct(.99)}  max {s[-1]}   budget {max_seq_length}")
        print(f"  generated {generated}, discarded {discarded} over budget "
              f"({discarded / max(generated, 1) * 100:.1f}%)")
        print(f"  train {len(train_dataset)}  eval {len(eval_dataset)}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        _write(save_dir / "synthetic_dataset.jsonl", data)
        _write(save_dir / "train_dataset.jsonl", train_dataset)
        _write(save_dir / "eval_dataset.jsonl", eval_dataset)
        if verbose:
            print(f"  wrote 3 .jsonl files to {save_dir}")

    return data, train_dataset, eval_dataset


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def check_budget(tokenizer, n=200, max_seq_length=512, compact=None, seed=0,
                 **generate_kwargs):
    """Measure the token-length distribution without building a dataset.

    Run this once before a long training job to see how much of the corpus
    would be truncated at the chosen `max_seq_length`.
    """
    if compact is None:
        compact = max_seq_length <= 1024
    eos = getattr(tokenizer, "eos_token", "") or ""
    rng = random.Random(seed)
    char_budget = generate_kwargs.pop(
        "max_chars", int(max_seq_length * 3.4) if compact else None)

    texts = [format_example(
        generate_example(rng, compact=compact, max_chars=char_budget,
                         **generate_kwargs), eos) for _ in range(n)]
    lens = sorted(_token_lengths(tokenizer, texts))
    over = sum(1 for x in lens if x > max_seq_length)
    pct = lambda q: lens[min(int(len(lens) * q), len(lens) - 1)]

    print(f"compact={compact}  n={n}  budget={max_seq_length}")
    print(f"  p50 {pct(.50)}   p90 {pct(.90)}   p99 {pct(.99)}   max {lens[-1]}")
    print(f"  over budget: {over}/{n} ({over / n * 100:.1f}%)  "
          f"median {int(statistics.median(lens))}")
    return lens


# ==========================================================================
# from synth_function/selftest.py
# ==========================================================================

# -*- coding: utf-8 -*-





def _leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def grounded(value, doc, doc_words=None):
    """Is `value` derivable from `doc`?

    Returns "exact", "ocr" or None.

      exact   the value is present as printed, modulo normalisation
      ocr     present but for at most one substituted character per word --
              the damage the design permits on free text and instructs the
              model to correct
      None    not present at all: a hallucinated label

    Digits are held to the exact standard throughout. Identifiers, amounts and
    dates are never character-damaged by the noise layer, so any drift in them
    is a defect rather than a tolerated hard case.
    """
    if value is None or isinstance(value, bool):
        return "exact"

    doc_norm = _norm(doc)
    doc_digits = _digits(doc)

    if isinstance(value, (int, float)):
        d = _digits(f"{float(value):.2f}")
        return "exact" if (d in doc_digits or _digits(str(value)) in doc_digits) else None

    s = str(value)

    # ISO date: components must be present, in whatever order. The month may
    # be printed as a name ("13 Nov 2019"), contributing no digits at all.
    parts = s.split("-")
    if len(parts) == 3 and all(p.isdigit() for p in parts) and len(parts[0]) == 4:
        y, m, d = parts
        name = _MONTH_NAMES[int(m) - 1]
        month_ok = (str(int(m)) in doc_digits
                    or name in doc_norm or name[:3] in doc_norm)
        ok = ((y in doc_digits or y[2:] in doc_digits)
              and month_ok and str(int(d)) in doc_digits)
        return "exact" if ok else None

    n = _norm(s)
    if not n:
        return "exact"
    if n in doc_norm:
        return "exact"

    # Phone numbers survive as their trailing digits.
    if s.isdigit() and len(s) >= 9:
        return "exact" if s[-9:] in doc_digits else None

    # Multi-word values may be split across a line break or column boundary.
    words = [_norm(w) for w in s.split() if len(_norm(w)) > 2]
    if words and all(w in doc_norm for w in words):
        return "exact"

    # Free text carrying permitted OCR damage. Anything with digits in it is
    # excluded: those are the fields the noise layer must never touch.
    if words and not any(ch.isdigit() for ch in s):
        doc_words = doc_words if doc_words is not None else _doc_words(doc)
        if all(any(_within_one_edit(w, dw) for dw in doc_words) for w in words):
            return "ocr"
        # The word-level check fails when the scanner also split a word in two.
        # Comparing the whole normalised value against the document as a
        # sliding window catches that, and stays tight enough that a genuinely
        # invented value cannot slip through: one edit across the entire string.
        if _fuzzy_in(n, doc_norm):
            return "ocr"

    return None


def _fuzzy_in(needle, haystack):
    """Does `needle` occur in `haystack` with at most one edit?"""
    k = len(needle)
    if k < 5 or k > len(haystack):
        return False
    for width in (k, k - 1, k + 1):
        if width < 1 or width > len(haystack):
            continue
        for i in range(len(haystack) - width + 1):
            if _within_one_edit(needle, haystack[i:i + width]):
                return True
    return False


def _doc_words(doc):
    return {_norm(w) for w in doc.split() if len(_norm(w)) > 2}


def run(n=500, seed=0, verbose=True, **kwargs):
    rng = random.Random(seed)
    total = nulls = checked = failures = tolerated = 0
    bad_examples = []

    for i in range(n):
        ex = generate_example(rng, with_meta=True, **kwargs)
        doc = ex["input"]
        words = _doc_words(doc)
        for path, value in _leaves(ex["output"]):
            total += 1
            if value is None:
                nulls += 1
                continue
            checked += 1
            verdict = grounded(value, doc, words)
            if verdict == "ocr":
                tolerated += 1
            elif verdict is None:
                failures += 1
                if len(bad_examples) < 8:
                    bad_examples.append((i, path, value, ex["_meta"]["document_type"]))

    if verbose:
        print(f"examples             {n}")
        print(f"leaves total         {total}")
        print(f"  null               {nulls}  ({nulls / max(total,1) * 100:.1f}%)")
        print(f"  non-null           {checked}")
        print(f"    exact in doc     {checked - tolerated - failures}")
        print(f"    within 1 OCR sub {tolerated}  "
              f"({tolerated / max(checked,1) * 100:.2f}%)  [permitted]")
        print(f"    UNGROUNDED       {failures}  "
              f"({failures / max(checked,1) * 100:.2f}%)  [defects]")
        for i, path, value, dt in bad_examples:
            print(f"      example {i} [{dt}] {path} = {value!r}")

    return failures, checked

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
# generate_example() takes no arguments and is driven by a module-level RNG, so
# the notebooks can call it in a bare loop. The defaults matter: without a token
# budget roughly 94% of examples would be long enough that the trainer truncates
# the "### Response:" JSON off the end, leaving nothing to learn from.

_DEFAULT_RNG = random.Random()

_OPTIONS = {
    "compact": True,      # terse instructions, <=8 schema fields, shorter docs
    "max_tokens": 1024,   # set max_seq_length = 1024 in the notebook to match
}


def configure(seed=None, **options):
    """Change generation defaults for subsequent generate_example() calls.

    configure(seed=42)                    make the corpus reproducible
    configure(max_tokens=512)             if max_seq_length stays at 512
    configure(compact=False)              full-size examples; measurably worse
                                          feature mix at any budget below ~2048
    """
    global _DEFAULT_RNG
    if seed is not None:
        _DEFAULT_RNG = random.Random(seed)
    unknown = set(options) - {"compact", "max_tokens", "max_chars",
                              "on_unrecoverable", "notation", "shape",
                              "alias_variation"}
    if unknown:
        raise TypeError(f"unknown option(s): {sorted(unknown)}")
    _OPTIONS.update(options)
    return dict(_OPTIONS)


def generate_example(rng=None, **kwargs):
    """Generate one training example: instruction, schema, input, output."""
    opts = dict(_OPTIONS)
    opts.update(kwargs)
    return _generate_example_core(rng if rng is not None else _DEFAULT_RNG,
                                  **opts)


# Alias. The notebooks import this name; it is the same callable.
def generate_synth_data(*args, **kwargs):
    return generate_example(*args, **kwargs)


def main():
    p = argparse.ArgumentParser(prog="synth_pipeline_v5_2")
    p.add_argument("--samples", type=int, default=8000)
    p.add_argument("--output", default="dataset_v5_2.jsonl")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--selftest", type=int, default=0)
    args = p.parse_args()

    if args.selftest:
        failures, _ = run(args.selftest, **_OPTIONS)
        sys.exit(1 if failures else 0)

    if args.seed is not None:
        configure(seed=args.seed)

    with open(args.output, "w", encoding="utf-8") as f:
        for _ in range(args.samples):
            # A real newline, so the file is one JSON object per line.
            f.write(json.dumps(generate_example(), ensure_ascii=False) + "\n")

    print(f"Generated {args.samples} samples -> {args.output}")


if __name__ == "__main__":
    main()
