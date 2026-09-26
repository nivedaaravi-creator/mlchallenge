#!/usr/bin/env python3
"""
Stage 1: Preprocessing Layer
ML-Only Business Entity Resolution Pipeline

Purpose:
Transform raw, noisy business records into standardized, comparable formats.
Handles country-specific variations in naming conventions and address formats.

Components:
1. Country Detection:
   - Reads the country field from each record
   - Routes to country-specific normalization rules
   - Handles US, India, France, and unseen countries with fallback rules

2. Name Normalization:
   - Legal suffix removal: Strips country-specific business suffixes
     US: INC, CORP, LLC, LTD, CO, COMPANY, PC, PLLC, etc.
     India: PVT, PRIVATE, LTD, OPC, LLP, PLC, Indic scripts
     France: SARL, SA, SAS, SASU, EURL, SCI, SNC, FILS, etc.
   - Abbreviation expansion: Converts common abbreviations to full forms
     & -> AND, + -> PLUS, PVT -> PRIVATE, CORP -> CORPORATION, etc.
   - Text cleanup: Uppercase conversion, accent normalization, consecutive duplicate removal,
     punctuation removal, whitespace normalization

3. Address Normalization:
   - Street type standardization: Rd -> ROAD, St -> STREET, Ave -> AVENUE, Blvd -> BOULEVARD, etc.
   - Component extraction: Parses address into structured fields (city, state, PIN/ZIP)
     Robust to component ordering (State-first, City-first, Street-first)
   - Geographic code extraction:
     US: First 3 digits of ZIP code (e.g. 902) or state-level default
     India: First 3 digits of PIN code (e.g. 600) or state-level default
     France: First 2 digits of postal code (department, e.g. 33, 59) or department default
"""

import sys
import re
import os
import unicodedata
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd


# =====================================================================
# 1. COUNTRY DETECTION & ROUTING
# =====================================================================

COUNTRY_SYNONYMS = {
    'US': 'US',
    'USA': 'US',
    'UNITED STATES': 'US',
    'UNITED STATES OF AMERICA': 'US',
    'U.S.': 'US',
    'U.S.A.': 'US',
    'INDIA': 'India',
    'IND': 'India',
    'IN': 'India',
    'BHARAT': 'India',
    'HINDUSTAN': 'India',
    'FRANCE': 'France',
    'FRA': 'France',
    'FR': 'France',
    'REPUBLIQUE FRANCAISE': 'France',
}


def detect_country(country_raw: Any) -> str:
    """Standardize and detect country code from raw input.
    
    Handles US, India, France, and fallback for unseen countries.
    """
    if country_raw is None or pd.isna(country_raw):
        return 'UNKNOWN'
    c_str = str(country_raw).strip().upper()
    c_clean = re.sub(r'[^\w\s]', '', c_str).strip()
    return COUNTRY_SYNONYMS.get(c_clean, COUNTRY_SYNONYMS.get(c_str, c_clean if c_clean else 'UNKNOWN'))


# =====================================================================
# 2. NAME NORMALIZATION
# =====================================================================

# Universal / Common Business Abbreviations
COMMON_ABBREVIATIONS = {
    r'\b&\b': 'AND',
    r'\b\+\b': 'PLUS',
    r'\b@\b': 'AT',
    r'\bPVT\.?\b': 'PRIVATE',
    r'\bCORP\.?\b': 'CORPORATION',
    r'\bCO\.?\b': 'COMPANY',
    r'\bLTD\.?\b': 'LIMITED',
    r'\bINC\.?\b': 'INCORPORATED',
    r'\bINTL\.?\b': 'INTERNATIONAL',
    r'\bINT\'L\b': 'INTERNATIONAL',
    r'\bMGMT\.?\b': 'MANAGEMENT',
    r'\bTECH\.?\b': 'TECHNOLOGY',
    r'\bSVCS\.?\b': 'SERVICES',
    r'\bSERV\.?\b': 'SERVICES',
    r'\bASSOC\.?\b': 'ASSOCIATES',
    r'\bASSN\.?\b': 'ASSOCIATION',
    r'\bGRP\.?\b': 'GROUP',
    r'\bDEPT\.?\b': 'DEPARTMENT',
    r'\bNATL\.?\b': 'NATIONAL',
    r'\bUNIV\.?\b': 'UNIVERSITY',
    r'\bHOSP\.?\b': 'HOSPITAL',
    r'\bCTR\.?\b': 'CENTER',
    r'\bCNTR\.?\b': 'CENTER',
    r'\bMFG\.?\b': 'MANUFACTURING',
    r'\bCOMM\.?\b': 'COMMUNICATION',
    r'\bSYS\.?\b': 'SYSTEMS',
    r'\bSOLN\.?\b': 'SOLUTIONS',
    r'\bSOLNS\.?\b': 'SOLUTIONS',
    r'\bENT\.?\b': 'ENTERPRISES',
    r'\bCONS\.?\b': 'CONSULTING',
    r'\bMED\.?\b': 'MEDICAL',
    r'\bPHARM\.?\b': 'PHARMACEUTICALS',
    r'\bFIN\.?\b': 'FINANCIAL',
}

# French Specific Abbreviations
FRANCE_ABBREVIATIONS = {
    r'\bSTE\.?\b': 'SOCIETE',
    r'\bCIE\.?\b': 'COMPAGNIE',
    r'\bETS\.?\b': 'ETABLISSEMENTS',
    r'\bENTR\.?\b': 'ENTREPRISE',
    r'\bDISTRIB\.?\b': 'DISTRIBUTION',
}

COMPILED_COMMON_ABBREVS = [(re.compile(k, re.IGNORECASE), v) for k, v in COMMON_ABBREVIATIONS.items()]
COMPILED_FR_ABBREVS = [(re.compile(k, re.IGNORECASE), v) for k, v in FRANCE_ABBREVIATIONS.items()]

# Country-Specific Legal Suffix Removal Patterns
LEGAL_SUFFIXES_RAW = {
    'US': [
        r'\bINCORPORATED\b', r'\bINC\b', r'\bI N C\b',
        r'\bCORPORATION\b', r'\bCORP\b', r'\bC O R P\b',
        r'\bLIMITED LIABILITY COMPANY\b', r'\bLLC\b', r'\bL L C\b',
        r'\bLIMITED LIABILITY CO\b', r'\bLIMITED LIABILITY PARTNERSHIP\b',
        r'\bLLP\b', r'\bL L P\b', r'\bLIMITED PARTNERSHIP\b', r'\bLP\b', r'\bL P\b',
        r'\bPROFESSIONAL CORPORATION\b', r'\bPC\b', r'\bP C\b',
        r'\bPLLC\b', r'\bP L L C\b', r'\bPA\b', r'\bP A\b',
        r'\bLIMITED\b', r'\bLTD\b', r'\bL T D\b',
        r'\bCOMPANY\b', r'\bCO\b', r'\bC O\b',
        r'\bCHARTERED\b',
    ],
    'India': [
        r'\bPRIVATE LIMITED\b', r'\bPVT LTD\b', r'\bPVT LIMITED\b', r'\bPRIVATE LTD\b',
        r'\bP LTD\b', r'\bP LIMITED\b', r'\(P\)\s*LTD\b', r'\(P\)\s*LIMITED\b',
        r'\bPRIVATE\b', r'\bPVT\b', r'\bP V T\b',
        r'\bLIMITED\b', r'\bLTD\b', r'\bL T D\b',
        r'\bONE PERSON COMPANY\b', r'\bOPC\b', r'\bO P C\b',
        r'\bLIMITED LIABILITY PARTNERSHIP\b', r'\bLLP\b', r'\bL L P\b',
        r'\bPUBLIC LIMITED\b', r'\bPLC\b', r'\bP L C\b',
        r'\bPROPRIETORSHIP\b',
        r'\(INDIA\)\b', r'\bINDIA\b',
        # Indic Script Legal Suffixes
        r'प्राइवेट\s+लिमिटेड', r'प्रा\s*लि', r'लिमिटेड', r'एलएलपी', r'प्राइवेट', r'कंपनी',
        r'பிரைவேட்\s+லிமிடெட்', r'லிமிடெட்', r'எல்எல்பி',
        r'ಪ್ರೈವೇಟ್\s+ನಿಯಮಿತ', r'ನಿಯಮಿತ',
        r'પ્રાઇવેટ\s+લિમિટેડ', r'લિમિટેડ',
        r'প্রাইভেট\s+লিমিটেড', r'লিমিটেড',
        r'പ്രൈവറ്റ്\s+ലിമിറ്റഡ്', r'ലിമിറ്റഡ്',
        r'ప్రైవేట్\s+లిమిటెడ్', r'పరిమిత',
    ],
    'France': [
        r'\bSOCIETE A RESPONSABILITE LIMITEE\b', r'\bSARL\b', r'\bS A R L\b',
        r'\bSOCIETE PAR ACTIONS SIMPLIFIEE UNIPERSONNELLE\b', r'\bSASU\b', r'\bS A S U\b',
        r'\bSOCIETE PAR ACTIONS SIMPLIFIEE\b', r'\bSAS\b', r'\bS A S\b',
        r'\bENTREPRISE UNIPERSONNELLE A RESPONSABILITE LIMITEE\b', r'\bEURL\b', r'\bE U R L\b',
        r'\bSOCIETE ANONYME\b', r'\bSA\b', r'\bS A\b',
        r'\bSOCIETE CIVILE IMMOBILIERE\b', r'\bSCI\b', r'\bS C I\b',
        r'\bSOCIETE EN NOM COLLECTIF\b', r'\bSNC\b', r'\bS N C\b',
        r'\bGROUPEMENT D\'INTERET ECONOMIQUE\b', r'\bGIE\b', r'\bG I E\b',
        r'\bET FILS\b', r'\bAND FILS\b', r'\bFILS\b',
        r'\bET FRERES\b', r'\bAND FRERES\b', r'\bFRERES\b',
        r'\bET COMPAGNIE\b', r'\bAND COMPAGNIE\b', r'\bAND CIE\b', r'\bCIE\b',
        r'\bSOCIETE\b', r'\bSTE\b',
    ],
    'Fallback': [
        r'\bINCORPORATED\b', r'\bINC\b', r'\bCORPORATION\b', r'\bCORP\b',
        r'\bLIMITED LIABILITY COMPANY\b', r'\bLLC\b', r'\bLIMITED\b', r'\bLTD\b',
        r'\bCOMPANY\b', r'\bCO\b', r'\bPRIVATE LIMITED\b', r'\bPVT LTD\b',
        r'\bPRIVATE\b', r'\bPVT\b', r'\bSARL\b', r'\bSA\b', r'\bSAS\b',
        r'\bEURL\b', r'\bSCI\b',
        r'\bGMBH\s+AND\s+CO\s+KG\b', r'\bGMBH\s+&\s+CO\s+KG\b', r'\bGMBH\b',
        r'\bKG\b', r'\bK G\b', r'\bAG\b', r'\bA G\b',
        r'\bBV\b', r'\bB V\b', r'\bNV\b', r'\bN V\b',
        r'\bPTY\s+LTD\b', r'\bPTY\b',
    ]
}

COMPILED_LEGAL_SUFFIXES = {
    c: [re.compile(pat, re.IGNORECASE) for pat in pats]
    for c, pats in LEGAL_SUFFIXES_RAW.items()
}


def strip_latin_accents(text: str) -> str:
    """Normalize unicode and strip Latin diacritics/accents while preserving Indic scripts."""
    if not isinstance(text, str):
        return ''
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not ('\u0300' <= c <= '\u036f'))


def clean_name_noise(text: str) -> str:
    """Remove URLs, domain extensions, email addresses, and decorative noise."""
    s = text.replace('&', ' AND ').replace('+', ' PLUS ').replace('@', ' AT ')
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s, flags=re.IGNORECASE)
    s = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', s)
    s = re.sub(r'\.(com|org|net|in|fr|co|io|biz|info|gov|edu)\b', ' ', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(null|none|nan|undefined)\b', ' ', s, flags=re.IGNORECASE)
    return s


def deduplicate_consecutive_words(text: str) -> str:
    """Remove repeated consecutive words (e.g. 'Unit Unit' -> 'Unit', 'AAA AAA' -> 'AAA')."""
    prev = None
    curr = text
    while prev != curr:
        prev = curr
        curr = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', curr, flags=re.IGNORECASE)
    return curr


def remove_punctuation_keep_indic(text: str) -> str:
    """Strip punctuation and symbols while preserving alphanumeric, Indic scripts and matras."""
    res = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith(('P', 'S')):
            res.append(' ')
        else:
            res.append(ch)
    return ''.join(res)


def trim_conjunctions_and_articles(text: str) -> str:
    """Trim dangling leading/trailing conjunctions and articles left after suffix removal."""
    s = re.sub(r'^(?:THE|LE|LA|LES|L|D|DE|DU|DES|AND|ET|AN|A)\s+', '', text)
    s = re.sub(r'\s+(?:AND|ET|OF|DE|DU|DES|THE|AND CO|AND COMPANY|ET CIE)$', '', s)
    return s.strip()


def normalize_name(raw_name: Any, country: str = 'US') -> Dict[str, Any]:
    """Perform comprehensive Name Normalization.
    
    Returns structured dictionary with:
      - clean_name: fully normalized name with legal suffixes removed
      - raw_clean_name: normalized name before legal suffix removal
      - legal_suffix: detected suffix if any
      - name_tokens: list of non-empty tokens
    """
    if raw_name is None or pd.isna(raw_name):
        return {
            'clean_name': '',
            'raw_clean_name': '',
            'legal_suffix': '',
            'name_tokens': []
        }

    # Step 1: Text cleanup & Unicode normalization
    s = str(raw_name).strip()
    s = strip_latin_accents(s)
    s = clean_name_noise(s)
    s = s.upper()

    # Step 2: Abbreviation expansion
    for pattern, replacement in COMPILED_COMMON_ABBREVS:
        s = pattern.sub(replacement, s)
    if country == 'France':
        for pattern, replacement in COMPILED_FR_ABBREVS:
            s = pattern.sub(replacement, s)

    # Step 3: Punctuation removal & consecutive word deduplication
    s = remove_punctuation_keep_indic(s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = deduplicate_consecutive_words(s)
    raw_clean_name = s

    # Step 4: Legal suffix removal
    suffix_list = COMPILED_LEGAL_SUFFIXES.get(country, COMPILED_LEGAL_SUFFIXES['Fallback'])
    detected_suffix = ''
    clean_s = s
    for suff_pat in suffix_list:
        match = suff_pat.search(clean_s)
        if match:
            cand = suff_pat.sub(' ', clean_s).strip()
            if cand:
                detected_suffix = match.group(0).strip()
                clean_s = cand

    clean_s = re.sub(r'\s+', ' ', clean_s).strip()
    clean_s = trim_conjunctions_and_articles(clean_s)

    if not clean_s:
        clean_s = raw_clean_name

    tokens = [t for t in clean_s.split(' ') if t]

    return {
        'clean_name': clean_s,
        'raw_clean_name': raw_clean_name,
        'legal_suffix': detected_suffix,
        'name_tokens': tokens
    }


# =====================================================================
# 3. ADDRESS NORMALIZATION & COMPONENT EXTRACTION
# =====================================================================

STREET_TYPE_MAP = {
    # US / English Street Types
    r'\bRD\.?\b': 'ROAD',
    r'\bST\.?\b': 'STREET',
    r'\bSTR\.?\b': 'STREET',
    r'\bAVE\.?\b': 'AVENUE',
    r'\bAV\.?\b': 'AVENUE',
    r'\bBLVD\.?\b': 'BOULEVARD',
    r'\bBOUL\.?\b': 'BOULEVARD',
    r'\bBD\.?\b': 'BOULEVARD',
    r'\bDR\.?\b': 'DRIVE',
    r'\bLN\.?\b': 'LANE',
    r'\bCT\.?\b': 'COURT',
    r'\bCRT\.?\b': 'COURT',
    r'\bPL\.?\b': 'PLACE',
    r'\bPKWY\.?\b': 'PARKWAY',
    r'\bPKY\.?\b': 'PARKWAY',
    r'\bHWY\.?\b': 'HIGHWAY',
    r'\bCIR\.?\b': 'CIRCLE',
    r'\bTRL\.?\b': 'TRAIL',
    r'\bTER\.?\b': 'TERRACE',
    r'\bTERR\.?\b': 'TERRACE',
    r'\bEXPY\.?\b': 'EXPRESSWAY',
    r'\bEXPWY\.?\b': 'EXPRESSWAY',
    r'\bSTE\.?\b': 'SUITE',
    r'\bAPT\.?\b': 'APARTMENT',
    r'\bBLDG\.?\b': 'BUILDING',
    r'\bFLR\.?\b': 'FLOOR',
    r'\bFL\.?\b': 'FLOOR',
    r'\bP\.?O\.?\s*BOX\b': 'PO BOX',
    r'\bPOBOX\b': 'PO BOX',

    # French Street Types
    r'\bR\.?\b': 'RUE',
    r'\bCHE\.?\b': 'CHEMIN',
    r'\bCH\.?\b': 'CHEMIN',
    r'\bIMP\.?\b': 'IMPASSE',
    r'\bALL\.?\b': 'ALLEE',
    r'\bRTE\.?\b': 'ROUTE',
    r'\bRT\.?\b': 'ROUTE',
    r'\bQU\.?\b': 'QUAI',
    r'\bSQ\.?\b': 'SQUARE',
    r'\bPASS\.?\b': 'PASSAGE',
    r'\bCRS\.?\b': 'COURS',
    r'\bRES\.?\b': 'RESIDENCE',

    # Indian Address Terms
    r'\bMARG\b': 'MARG',
    r'\bRASTA\b': 'RASTA',
    r'\bNAGAR\b': 'NAGAR',
    r'\bCOLONY\b': 'COLONY',
    r'\bENCLAVE\b': 'ENCLAVE',
    r'\bSOCIETY\b': 'SOCIETY',
    r'\bLAYOUT\b': 'LAYOUT',
    r'\bSECTOR\b': 'SECTOR',
    r'\bBLOCK\b': 'BLOCK',
    r'\bPHASE\b': 'PHASE',
    r'\bAPTS\.?\b': 'APARTMENT',
    r'\bH\.?NO\.?\b': 'HNO',
    r'\bHOUSE\s+NO\.?\b': 'HNO',
    r'\bPLOT\s+NO\.?\b': 'PLOT',
    r'\bOPP\.?\b': 'OPPOSITE',
    r'\bNR\.?\b': 'NEAR',
    r'\bB/?H\b': 'BEHIND',
}

COMPILED_STREET_TYPES = [(re.compile(k, re.IGNORECASE), v) for k, v in STREET_TYPE_MAP.items()]

# US Street Identification Keywords
US_STREET_KEYWORDS = {
    'ROAD', 'STREET', 'AVENUE', 'BOULEVARD', 'DRIVE', 'LANE', 'COURT', 'PLACE',
    'PARKWAY', 'HIGHWAY', 'CIRCLE', 'TRAIL', 'WAY', 'TERRACE', 'EXPRESSWAY',
    'SUITE', 'APARTMENT', 'BUILDING', 'FLOOR', 'UNIT', 'PO BOX', 'BOX', 'WALK'
}

# US States & Default Geographic ZIP Prefixes
US_STATES = {
    'AL': 'AL', 'AK': 'AK', 'AZ': 'AZ', 'AR': 'AR', 'CA': 'CA', 'CO': 'CO', 'CT': 'CT', 'DE': 'DE',
    'FL': 'FL', 'GA': 'GA', 'HI': 'HI', 'ID': 'ID', 'IL': 'IL', 'IN': 'IN', 'IA': 'IA', 'KS': 'KS',
    'KY': 'KY', 'LA': 'LA', 'ME': 'ME', 'MD': 'MD', 'MA': 'MA', 'MI': 'MI', 'MN': 'MN', 'MS': 'MS',
    'MO': 'MO', 'MT': 'MT', 'NE': 'NE', 'NV': 'NV', 'NH': 'NH', 'NJ': 'NJ', 'NM': 'NM', 'NY': 'NY',
    'NC': 'NC', 'ND': 'ND', 'OH': 'OH', 'OK': 'OK', 'OR': 'OR', 'PA': 'PA', 'RI': 'RI', 'SC': 'SC',
    'SD': 'SD', 'TN': 'TN', 'TX': 'TX', 'UT': 'UT', 'VT': 'VT', 'VA': 'VA', 'WA': 'WA', 'WV': 'WV',
    'WI': 'WI', 'WY': 'WY', 'DC': 'DC', 'PR': 'PR',
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 'CALIFORNIA': 'CA',
    'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 'FLORIDA': 'FL', 'GEORGIA': 'GA',
    'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
    'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD', 'MASSACHUSETTS': 'MA',
    'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 'MISSOURI': 'MO', 'MONTANA': 'MT',
    'NEBRASKA': 'NE', 'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM',
    'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
    'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT', 'VERMONT': 'VT',
    'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY'
}

US_STATE_GEO_PREFIX = {
    'AL': '350', 'AK': '995', 'AZ': '850', 'AR': '716', 'CA': '900', 'CO': '800', 'CT': '060', 'DE': '197',
    'FL': '320', 'GA': '300', 'HI': '967', 'ID': '832', 'IL': '600', 'IN': '460', 'IA': '500', 'KS': '660',
    'KY': '400', 'LA': '700', 'ME': '039', 'MD': '206', 'MA': '010', 'MI': '480', 'MN': '550', 'MS': '386',
    'MO': '630', 'MT': '590', 'NE': '680', 'NV': '890', 'NH': '030', 'NJ': '070', 'NM': '870', 'NY': '100',
    'NC': '270', 'ND': '580', 'OH': '430', 'OK': '730', 'OR': '970', 'PA': '150', 'RI': '028', 'SC': '290',
    'SD': '570', 'TN': '370', 'TX': '750', 'UT': '840', 'VT': '050', 'VA': '220', 'WA': '980', 'WV': '247',
    'WI': '530', 'WY': '820', 'DC': '200', 'PR': '006'
}

# India States & Default Geographic PIN Prefixes
INDIA_STATES = {
    'MAHARASHTRA': 'MH', 'MH': 'MH', 'महाराष्ट्र': 'MH',
    'DELHI': 'DL', 'DL': 'DL', 'NEW DELHI': 'DL', 'दिल्ली': 'DL',
    'KARNATAKA': 'KA', 'KA': 'KA', 'ಕರ್ನಾಟಕ': 'KA',
    'TAMIL NADU': 'TN', 'TN': 'TN', 'TAMILNADU': 'TN', 'தமிழ்நாடு': 'TN',
    'GUJARAT': 'GJ', 'GJ': 'GJ', 'ગુજરાત': 'GJ',
    'WEST BENGAL': 'WB', 'WB': 'WB', 'পশ্চিমবঙ্গ': 'WB',
    'TELANGANA': 'TG', 'TG': 'TG', 'TS': 'TG', 'తెలంగాణ': 'TG',
    'UTTAR PRADESH': 'UP', 'UP': 'UP', 'उत्तर प्रदेश': 'UP',
    'HARYANA': 'HR', 'HR': 'HR', 'हरियाणा': 'HR',
    'KERALA': 'KL', 'KL': 'KL', 'കേരളം': 'KL',
    'RAJASTHAN': 'RJ', 'RJ': 'RJ', 'राजस्थान': 'RJ',
    'BIHAR': 'BR', 'BR': 'BR', 'बिहार': 'BR',
    'MADHYA PRADESH': 'MP', 'MP': 'MP', 'मध्य प्रदेश': 'MP',
    'ANDHRA PRADESH': 'AP', 'AP': 'AP', 'ఆంధ్రప్రదేశ్': 'AP',
    'ORISSA': 'OD', 'ODISHA': 'OD', 'OD': 'OD', 'OR': 'OD', 'ଓଡ଼ିଶା': 'OD',
    'PUNJAB': 'PB', 'PB': 'PB', 'ਪੰਜਾਬ': 'PB',
    'ASSAM': 'AS', 'AS': 'AS', 'অসম': 'AS',
    'JHARKHAND': 'JH', 'JH': 'JH', 'झारखंड': 'JH',
    'CHHATTISGARH': 'CG', 'CG': 'CG', 'CT': 'CG', 'छत्तीसगढ़': 'CG',
    'GOA': 'GA', 'GA': 'GA',
    'HIMACHAL PRADESH': 'HP', 'HP': 'HP',
    'JAMMU AND KASHMIR': 'JK', 'JK': 'JK',
    'UTTARAKHAND': 'UK', 'UTTARANCHAL': 'UK', 'UK': 'UK', 'UT': 'UK',
    'PUDUCHERRY': 'PY', 'PONDICHERRY': 'PY', 'PY': 'PY',
    'CHANDIGARH': 'CH', 'CH': 'CH'
}

INDIA_STATE_GEO_PREFIX = {
    'DL': '110', 'MH': '400', 'KA': '560', 'TN': '600', 'WB': '700', 'TG': '500', 'GJ': '380',
    'UP': '201', 'HR': '122', 'KL': '682', 'RJ': '302', 'BR': '800', 'MP': '452', 'AP': '520',
    'OD': '751', 'PB': '141', 'AS': '781', 'JH': '834', 'CG': '492', 'GA': '403', 'HP': '171',
    'JK': '190', 'UK': '248', 'PY': '605', 'CH': '160'
}

INDIA_CITIES = [
    'MUMBAI', 'DELHI', 'NEW DELHI', 'BENGALURU', 'BANGALORE', 'HYDERABAD', 'AHMEDABAD',
    'CHENNAI', 'KOLKATA', 'SURAT', 'PUNE', 'JAIPUR', 'LUCKNOW', 'KANPUR', 'NAGPUR',
    'INDORE', 'THANE', 'BHOPAL', 'VISAKHAPATNAM', 'PATNA', 'VADODARA', 'GHAZIABAD',
    'LUDHIANA', 'AGRA', 'NASHIK', 'FARIDABAD', 'MEERUT', 'RAJKOT', 'VARANASI',
    'SRINAGAR', 'AURANGABAD', 'DHANBAD', 'AMRITSAR', 'NAVI MUMBAI', 'ALLAHABAD',
    'PRAYAGRAJ', 'HOWRAH', 'RANCHI', 'GWALIOR', 'JABALPUR', 'COIMBATORE', 'VIJAYAWADA',
    'JODHPUR', 'MADURAI', 'RAIPUR', 'KOTA', 'CHANDIGARH', 'GUWAHATI', 'SOLAPUR',
    'HUBLI', 'MYSORE', 'TIRUCHIRAPPALLI', 'BAREILLY', 'ALIGARH', 'TIRUPPUR', 'GURGAON',
    'GURUGRAM', 'MORADABAD', 'JALANDHAR', 'BHUBANESWAR', 'SALEM', 'WARANGAL',
    'THIRUVANANTHAPURAM', 'KOCHI', 'KOZHIKODE', 'KANNUR', 'AMBALA', 'SECUNDERABAD',
    'NOIDA', 'GREATER NOIDA', 'MIRZAPUR', 'KARAULI'
]

# France Regions, Departments & Geographic Codes
FRANCE_DEPTS = {
    'NOUVELLE-AQUITAINE': '33', 'NOUVELLE AQUITAINE': '33',
    'HAUTS-DE-FRANCE': '59', 'HAUTS DE FRANCE': '59',
    'PAYS DE LA LOIRE': '44', 'PAYS DE LOIRE': '44', 'ILE-DE-FRANCE': '75',
    'NORD': '59', 'PAS-DE-CALAIS': '62', 'PAS DE CALAIS': '62', 'GIRONDE': '33', 'LOIRE-ATLANTIQUE': '44',
    'PARIS': '75', 'RHONE': '69', 'BOUCHES-DU-RHONE': '13', 'HAUTE-GARONNE': '31'
}

FRANCE_CITIES = {
    'BORDEAUX': '33', 'LA TESTE-DE-BUCH': '33', 'LA TESTE DE BUCH': '33',
    'PESSAC': '33', 'MERIGNAC': '33', 'LEGE-CAP-FERRET': '33', 'LEGE CAP FERRET': '33',
    'LILLE': '59', 'DUNKERQUE': '59', 'TOURCOING': '59', 'ROUBAIX': '59', 'VILLENEUVE-D-ASCQ': '59',
    'CALAIS': '62', 'BOULOGNE-SUR-MER': '62', 'ARRAS': '62',
    'NANTES': '44', 'SAINT-NAZAIRE': '44', 'SAINT-HERBLAIN': '44', 'PORNIC': '44', 'REZE': '44',
    'PARIS': '75', 'LYON': '69', 'MARSEILLE': '13', 'TOULOUSE': '31', 'NICE': '06', 'STRASBOURG': '67',
    'MONTPELLIER': '34', 'RENNES': '35', 'REIMS': '51', 'TOULON': '83', 'ANGERS': '49', 'GRENOBLE': '38'
}


def clean_address_text(addr_raw: Any) -> str:
    """Standardize street abbreviations, unicode, consecutive word duplicates, and casing in raw address."""
    if addr_raw is None or pd.isna(addr_raw):
        return ''
    s = str(addr_raw).strip()
    if not s or s.lower() in ('nan', 'null', 'none', 'undefined'):
        return ''

    s = strip_latin_accents(s).upper()

    # Deduplicate consecutive words before punctuation replacement
    s = deduplicate_consecutive_words(s)

    # Standardize separator punctuation
    res = []
    for ch in s:
        cat = unicodedata.category(ch)
        if ch in (',', '-', '/', '#'):
            res.append(f' {ch} ')
        elif cat.startswith(('P', 'S')):
            res.append(' ')
        else:
            res.append(ch)
    s = ''.join(res)

    # Expand street type abbreviations
    for pattern, rep in COMPILED_STREET_TYPES:
        s = pattern.sub(rep, s)

    s = re.sub(r'\s+', ' ', s).strip()
    s = deduplicate_consecutive_words(s)
    return s


def extract_components_us(addr_text: str) -> Tuple[str, str, str, str]:
    """Parse US address into (city, state, postal_code, geo_code).

    ZIP extraction: strict 5-digit match at end / after state to avoid
    false positives on street numbers (e.g. '17560 Ellis Road').

    City extraction: 4-strategy multi-pass search anchored on the state part.
      Strategy 1 – adjacent parts (no digits at all)
      Strategy 2 – forward scan from state (state-first addresses)
      Strategy 3 – backward scan to state (state-last addresses)
      Strategy 4 – bidirectional scan (state in middle)
    Each strategy skips parts that start with a digit, contain a 3-digit+
    number, or match unit/suite indicators (UNIT, APT, LOT, BLDG, FLOOR …).

    Examples handled:
      "17560 Ellis Road, Tahlequah, OK"               → TAHLEQUAH / OK
      "OH, Columbus, 5559 Orville Avenue"              → COLUMBUS / OH
      "Unit BUILDING 3030, MD, 2701 Eastern Blvd, Middle River" → MIDDLE RIVER / MD
      "Charlotte, NC, 833 Reliance Street"             → CHARLOTTE / NC
      "TN, 27..., Jackson"                             → JACKSON / TN
    """
    # ── 1. Strict ZIP extraction ──────────────────────────────────────────────
    postal_code = ''
    m_zip = re.search(r'(?:[A-Z]{2}|[A-Za-z]+|\,)\s+(\d{5})(?:-\d{4})?(?:\s*,|\s*$)', addr_text)
    if not m_zip:
        m_zip = re.search(r',\s*(\d{5})(?:-\d{4})?\s*$', addr_text)
    if m_zip:
        postal_code = m_zip.group(1)

    # ── 2. Split and normalise parts ──────────────────────────────────────────
    parts = [p.strip() for p in addr_text.upper().split(',') if p.strip()]

    # ── 3. Locate the state part ──────────────────────────────────────────────
    found_state = ''
    state_idx = -1

    for idx, p in enumerate(parts):
        for w in p.split():
            if w in US_STATES:
                found_state = US_STATES[w]
                state_idx = idx
                break
        if found_state:
            break

    # Fallback: whole-string scan for full state name
    if not found_state:
        for k, v in US_STATES.items():
            if re.search(r'\b' + re.escape(k) + r'\b', addr_text.upper()):
                found_state = v
                break

    # ── 4. Multi-strategy city extraction ────────────────────────────────────
    _UNIT_PATTERNS = [
        r'\bUNIT\b', r'\bAPARTMENT\b', r'\bAPT\b', r'\bLOT\b',
        r'\bBLDG\b', r'\bBUILDING\b', r'\bFLOOR\b', r'\bSTE\b', r'\bSUITE\b',
    ]

    def _is_city_candidate(part: str) -> bool:
        """True when part is plausibly a city (not a street, ZIP, or unit label)."""
        if not part:
            return False
        # Starts with a digit → street number
        if re.match(r'^\d', part):
            return False
        # Contains 3+ consecutive digits → ZIP or building number
        if re.search(r'\d{3,}', part):
            return False
        # Unit / suite / floor indicator
        if any(re.search(pat, part) for pat in _UNIT_PATTERNS):
            return False
        # Pure state abbreviation
        if part in US_STATES:
            return False
        return True

    city = ''

    if state_idx >= 0:
        n = len(parts)

        # Strategy 1: immediate neighbours (no digits whatsoever)
        for offset in (-1, 1):
            ni = state_idx + offset
            if 0 <= ni < n and not re.search(r'\d', parts[ni]):
                candidate = parts[ni]
                if _is_city_candidate(candidate):
                    city = candidate
                    break

        # Strategy 2: forward scan (handles state-first)
        if not city and state_idx == 0:
            for i in range(1, n):
                if _is_city_candidate(parts[i]):
                    city = parts[i]
                    break

        # Strategy 3: backward scan (handles state-last)
        if not city and state_idx == n - 1:
            for i in range(state_idx - 1, -1, -1):
                if _is_city_candidate(parts[i]):
                    city = parts[i]
                    break

        # Strategy 4: bidirectional scan (state in middle)
        if not city:
            for i in range(state_idx - 1, -1, -1):
                if _is_city_candidate(parts[i]):
                    city = parts[i]
                    break
            if not city:
                for i in range(state_idx + 1, n):
                    if _is_city_candidate(parts[i]):
                        city = parts[i]
                        break

    # ── 5. Strip residual state/ZIP tokens from the city string ──────────────
    if city:
        clean_words = [
            w for w in city.split()
            if w not in US_STATES
            and not (w.isdigit() and len(w) == 5)
        ]
        city = ' '.join(clean_words)

    # ── 6. Geographic code ────────────────────────────────────────────────────
    if postal_code:
        geo_code = postal_code[:3]
    elif found_state and found_state in US_STATE_GEO_PREFIX:
        geo_code = US_STATE_GEO_PREFIX[found_state]
    else:
        geo_code = ''

    return city, found_state, postal_code, geo_code


# Sort state keys by length descending to match multi-word names first
SORTED_IN_STATES = sorted(INDIA_STATES.keys(), key=lambda x: len(x), reverse=True)
COMPILED_IN_STATES = [
    (re.compile(r'\b' + re.escape(st) + r'\b', re.IGNORECASE) if st.isascii() else re.compile(re.escape(st)), INDIA_STATES[st])
    for st in SORTED_IN_STATES
]
COMPILED_IN_CITIES = [
    re.compile(r'\b' + re.escape(c) + r'\b', re.IGNORECASE) for c in INDIA_CITIES
]


def extract_components_india(addr_text: str) -> Tuple[str, str, str, str]:
    """Parse Indian address into (city, state, postal_code, geo_code).
    
    Geo code: First 3 digits of PIN code if present, or state/city default prefix.
    """
    # Strict 6-digit PIN pattern (starts with 1-9, at end or separated)
    m_pin = re.search(r'(?:[A-Z]+|\,)\s*([1-9]\d{5})(?:\s*,|\s*$)', addr_text)
    if not m_pin:
        m_pin = re.search(r',\s*([1-9]\d{5})\s*$', addr_text)
    postal_code = m_pin.group(1) if m_pin else ''

    # Search for state across entire normalized address using compiled patterns
    found_state = ''
    for pat, code in COMPILED_IN_STATES:
        if pat.search(addr_text):
            found_state = code
            break

    # Search for city from predefined major Indian cities
    found_city = ''
    for c, pat in zip(INDIA_CITIES, COMPILED_IN_CITIES):
        if pat.search(addr_text):
            found_city = c
            break

    if not found_city:
        parts = [p.strip() for p in addr_text.split(',') if p.strip()]
        if parts:
            for p in parts:
                if p != found_state:
                    found_city = p
                    break

    if postal_code:
        geo_code = postal_code[:3]
    elif found_state and found_state in INDIA_STATE_GEO_PREFIX:
        geo_code = INDIA_STATE_GEO_PREFIX[found_state]
    else:
        geo_code = ''

    return found_city, found_state, postal_code, geo_code


def extract_components_france(addr_text: str) -> Tuple[str, str, str, str]:
    """Parse French address into (city, state/department, postal_code, geo_code).
    
    Geo code: First 2 digits of postal code (department code, e.g. 33, 59).
    """
    # Strict 5-digit French postal code (after comma or city/street, e.g. '75008 PARIS')
    m_post = re.search(r'(?:[A-Z]+|\,)\s*(\d{5})(?:\s*,|\s*$)', addr_text)
    if not m_post:
        m_post = re.search(r',\s*(\d{5})\s*$', addr_text)
    postal_code = m_post.group(1) if m_post else ''

    found_dept = ''
    found_city = ''

    compact_text = re.sub(r'\s*-\s*', '-', addr_text)
    spaced_text = re.sub(r'[-]', ' ', addr_text)

    for c, code in FRANCE_CITIES.items():
        c_compact = re.sub(r'\s*-\s*', '-', c)
        c_spaced = re.sub(r'[-]', ' ', c)
        if c_compact in compact_text or c_spaced in spaced_text or c in addr_text:
            found_city = c
            break

    for k, v in FRANCE_DEPTS.items():
        k_compact = re.sub(r'\s*-\s*', '-', k)
        k_spaced = re.sub(r'[-]', ' ', k)
        if k_compact in compact_text or k_spaced in spaced_text or k in addr_text:
            found_dept = k
            break

    if postal_code:
        geo_code = postal_code[:2]
    elif found_city and found_city in FRANCE_CITIES:
        geo_code = FRANCE_CITIES[found_city]
    elif found_dept and found_dept in FRANCE_DEPTS:
        geo_code = FRANCE_DEPTS[found_dept]
    else:
        geo_code = ''

    return found_city, found_dept, postal_code, geo_code


def extract_components_fallback(addr_text: str) -> Tuple[str, str, str, str]:
    """Parse address for unseen/fallback countries into (city, state, postal_code, geo_code)."""
    m_post = re.search(r'(?:[A-Z]+|\,)\s*(\d{4,6})(?:\s*,|\s*$)', addr_text)
    postal_code = m_post.group(1) if m_post else ''
    geo_code = postal_code[:3] if postal_code else ''

    parts = [p.strip() for p in addr_text.split(',') if p.strip()]
    city = parts[-2].strip() if len(parts) >= 2 else (parts[0].strip() if parts else '')
    state = parts[-1].strip() if len(parts) >= 1 else ''

    return city, state, postal_code, geo_code


def normalize_address(raw_address: Any, country: str = 'US') -> Dict[str, Any]:
    """Perform complete Address Normalization and Component Extraction.
    
    Returns structured dictionary with:
      - clean_address: standardized address string
      - city: extracted city name
      - state: extracted state/region code
      - postal_code: extracted ZIP/PIN/postal code
      - geo_code: geographic prefix (3 digits for US/India, 2 digits for France)
      - address_tokens: list of non-empty tokens
    """
    if raw_address is None or pd.isna(raw_address):
        return {
            'clean_address': '',
            'city': '',
            'state': '',
            'postal_code': '',
            'geo_code': '',
            'address_tokens': []
        }

    clean_addr = clean_address_text(raw_address)
    if not clean_addr:
        return {
            'clean_address': '',
            'city': '',
            'state': '',
            'postal_code': '',
            'geo_code': '',
            'address_tokens': []
        }

    if country == 'US':
        city, state, postal_code, geo_code = extract_components_us(clean_addr)
    elif country == 'India':
        city, state, postal_code, geo_code = extract_components_india(clean_addr)
    elif country == 'France':
        city, state, postal_code, geo_code = extract_components_france(clean_addr)
    else:
        city, state, postal_code, geo_code = extract_components_fallback(clean_addr)

    tokens = [t for t in re.sub(r'[^\w\s]', ' ', clean_addr).split() if t]

    return {
        'clean_address': clean_addr,
        'city': city,
        'state': state,
        'postal_code': postal_code,
        'geo_code': geo_code,
        'address_tokens': tokens
    }


# =====================================================================
# 4. RECORD PREPROCESSOR (PIPELINE INTERFACE)
# =====================================================================

def preprocess_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess a single business record dictionary."""
    raw_id = record.get('entity_id', '')
    raw_country = record.get('country', '')
    raw_name = record.get('business_name', '')
    raw_addr = record.get('business_address', '')

    country = detect_country(raw_country)
    name_info = normalize_name(raw_name, country=country)
    addr_info = normalize_address(raw_addr, country=country)

    return {
        'entity_id': str(raw_id),
        'country': country,
        'clean_name': name_info['clean_name'],
        'raw_clean_name': name_info['raw_clean_name'],
        'legal_suffix': name_info['legal_suffix'],
        'name_tokens': ' '.join(name_info['name_tokens']),
        'clean_address': addr_info['clean_address'],
        'city': addr_info['city'],
        'state': addr_info['state'],
        'postal_code': addr_info['postal_code'],
        'geo_code': addr_info['geo_code'],
        'address_tokens': ' '.join(addr_info['address_tokens']),
    }


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess an entire Pandas DataFrame of business records with high throughput."""
    records = df.to_dict('records')
    processed = [preprocess_record(r) for r in records]
    return pd.DataFrame(processed)


def preprocess_tsv_file(input_path: str, output_path: Optional[str] = None, chunksize: int = 200000) -> Optional[pd.DataFrame]:
    """Preprocess a TSV file in chunks and optionally save normalized TSV to output_path."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    chunks_out = []
    first_chunk = True

    for chunk in pd.read_csv(input_path, sep='\t', chunksize=chunksize, encoding='utf-8', dtype=str):
        clean_chunk = preprocess_dataframe(chunk)
        if output_path:
            clean_chunk.to_csv(
                output_path,
                sep='\t',
                index=False,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                encoding='utf-8'
            )
            first_chunk = False
        else:
            chunks_out.append(clean_chunk)

    if chunks_out:
        return pd.concat(chunks_out, ignore_index=True)
    return None


# =====================================================================
# 5. CLI & VERIFICATION RUNNER
# =====================================================================

if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Stage 1: Preprocessing Layer for Business Entity Resolution")
    parser.add_argument('--input', type=str, help='Path to input TSV file')
    parser.add_argument('--output', type=str, default=None, help='Path to output normalized TSV file')
    parser.add_argument('--sample', type=int, default=0, help='Process first N sample rows and display summary')
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding='utf-8')

    if args.input:
        print(f"[*] Preprocessing input file: {args.input}")
        t0 = time.time()
        if args.sample > 0:
            df = pd.read_csv(args.input, sep='\t', nrows=args.sample, encoding='utf-8', dtype=str)
            res = preprocess_dataframe(df)
            print(f"[+] Successfully preprocessed {len(res)} rows in {time.time() - t0:.2f}s")
            print(res.head(10)[['entity_id', 'country', 'clean_name', 'city', 'state', 'geo_code', 'clean_address']])
        else:
            preprocess_tsv_file(args.input, args.output)
            print(f"[+] Completed preprocessing in {time.time() - t0:.2f}s -> {args.output}")
    else:
        print("Stage 1: Preprocessing Layer - Self Test Demonstration:")
        demo_records = [
            {'entity_id': 'S1-773889195', 'country': 'US', 'business_name': 'Prime Money', 'business_address': '17560 Ellis Road, Tahlequah, OK'},
            {'entity_id': 'S1-851869949', 'country': 'US', 'business_name': 'Custom Wealth Services LLC', 'business_address': 'OH, Columbus, 5559 Orville Avenue'},
            {'entity_id': 'S1-22305073', 'country': 'US', 'business_name': 'Dermatology Green Medicine', 'business_address': '294 Meadowcreek Drive, Unit Unit 2, Village Of Pewaukee, WI'},
            {'entity_id': 'S1-626914593', 'country': 'US', 'business_name': 'Unified Choice Dynamix', 'business_address': 'Unit BUILDING 3030, MD, 2701 Eastern Boulevard, Middle River'},
            {'entity_id': 'S1-309349399', 'country': 'US', 'business_name': 'Callicoat & Dailey Inc', 'business_address': 'Charlotte, NC, 833 Reliance Street'},
            {'entity_id': 'S1-755362802', 'country': 'India', 'business_name': 'Prabhav Business Center', 'business_address': '797, Lake Town Block A, Kolkata, Howrah, West Bengal'},
            {'entity_id': 'S1-785847572', 'country': 'India', 'business_name': 'Consulting Nyasa Nursing Private Limited', 'business_address': '2505, Tower 1, Oakwood, Mulund Goreagon Link Road, Mumbai, Maharashtra'},
            {'entity_id': 'S1-006', 'country': 'India', 'business_name': 'राम मार्केटिंग प्राइवेट लिमिटेड', 'business_address': 'KH NO. -570/13, NEW DELHI, Delhi'},
            {'entity_id': 'S1-007', 'country': 'India', 'business_name': 'ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி', 'business_address': '6(29), C.I.T. Colony, Chennai, Tamil Nadu'},
            {'entity_id': 'S1-008', 'country': 'France', 'business_name': '<< Team Ecole', 'business_address': '175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine'},
            {'entity_id': 'S1-009', 'country': 'France', 'business_name': 'Thermal & Fils SASU', 'business_address': '20 Rue Parmentier, Dunkerque, Hauts-de-France'},
            {'entity_id': 'S1-010', 'country': 'France', 'business_name': 'Grain & Fils', 'business_address': 'Lille, 329 Avenue de Dunkerque, Hauts-de-France'},
        ]
        df_demo = pd.DataFrame(demo_records)
        df_clean = preprocess_dataframe(df_demo)
        print("\n--- Preprocessed Demonstration Results ---")
        for _, r in df_clean.iterrows():
            print(f"[{r.country}] ID: {r.entity_id}")
            print(f"   Clean Name:    '{r.clean_name}' (Raw clean: '{r.raw_clean_name}', Suffix: '{r.legal_suffix}')")
            print(f"   Clean Address: '{r.clean_address}'")
            print(f"   City: '{r.city}' | State: '{r.state}' | Geo Code: '{r.geo_code}' | Postal: '{r.postal_code}'\n")
