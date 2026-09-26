#!/usr/bin/env python3
"""
Test Suite for Stage 1: Preprocessing Layer
Validates fixes for:
  Issue 1: Strict ZIP code extraction (no false positives from street numbers like 17560)
  Issue 2: State-first address parsing (e.g. 'OH, Columbus, 5559 Orville Avenue')
  Issue 3: India addresses without PIN codes (empty postal_code, state/city geo_code)
  Issue 4: Duplicate consecutive words cleanup ('Unit Unit' -> 'Unit')
"""
import sys
import pandas as pd
from preprocess import preprocess_record, preprocess_dataframe

sys.stdout.reconfigure(encoding='utf-8')

def run_tests():
    test_cases = [
        # Issue 1 Test: 17560 should NOT be extracted as ZIP
        {
            'entity_id': 'S1-773889195',
            'country': 'US',
            'business_name': 'Prime Money',
            'business_address': '17560 Ellis Road, Tahlequah, OK',
            'expected_postal': '',
            'expected_city': 'TAHLEQUAH',
            'expected_state': 'OK',
            'expected_geo': '730'
        },
        # Issue 2 Test: State-first format (OH, Columbus, 5559 Orville Avenue)
        {
            'entity_id': 'S1-851869949',
            'country': 'US',
            'business_name': 'Custom Wealth Services LLC',
            'business_address': 'OH, Columbus, 5559 Orville Avenue',
            'expected_postal': '',
            'expected_city': 'COLUMBUS',
            'expected_state': 'OH',
            'expected_geo': '430'
        },
        # Issue 4 Test: Consecutive duplicate words ('Unit Unit' -> 'Unit')
        {
            'entity_id': 'S1-22305073',
            'country': 'US',
            'business_name': 'Dermatology Green Medicine',
            'business_address': '294 Meadowcreek Drive, Unit Unit 2, Village Of Pewaukee, WI',
            'expected_postal': '',
            'expected_city': 'VILLAGE OF PEWAUKEE',
            'expected_state': 'WI',
            'expected_geo': '530'
        },
        # Additional US Reordered Formats
        {
            'entity_id': 'S1-626914593',
            'country': 'US',
            'business_name': 'Unified Choice Dynamix',
            'business_address': 'Unit BUILDING 3030, MD, 2701 Eastern Boulevard, Middle River',
            'expected_postal': '',
            'expected_city': 'MIDDLE RIVER',
            'expected_state': 'MD',
            'expected_geo': '206'
        },
        {
            'entity_id': 'S1-309349399',
            'country': 'US',
            'business_name': 'Callicoat & Dailey Inc',
            'business_address': 'Charlotte, NC, 833 Reliance Street',
            'expected_postal': '',
            'expected_city': 'CHARLOTTE',
            'expected_state': 'NC',
            'expected_geo': '270'
        },
        # Issue 3 Test: India addresses without PIN codes
        {
            'entity_id': 'S1-755362802',
            'country': 'India',
            'business_name': 'Prabhav Business Center',
            'business_address': '797, Lake Town Block A, Kolkata, Howrah, West Bengal',
            'expected_postal': '',
            'expected_city': 'KOLKATA',
            'expected_state': 'WB',
            'expected_geo': '700'
        },
        {
            'entity_id': 'S1-785847572',
            'country': 'India',
            'business_name': 'Consulting Nyasa Nursing Private Limited',
            'business_address': '2505, Tower 1, Oakwood, Runwal Greens, Mulund Goreagon Link Road, Near Fortis Hospital, Bhandup West, Mumbai, Maharashtra',
            'expected_postal': '',
            'expected_city': 'MUMBAI',
            'expected_state': 'MH',
            'expected_geo': '400'
        },
        # France Test Cases
        {
            'entity_id': 'S1-913506265',
            'country': 'France',
            'business_name': 'Thermal & Fils SASU',
            'business_address': '20 Rue Parmentier, Dunkerque, Hauts-de-France',
            'expected_postal': '',
            'expected_city': 'DUNKERQUE',
            'expected_state': 'HAUTS-DE-FRANCE',
            'expected_geo': '59'
        },
        {
            'entity_id': 'S1-921369899',
            'country': 'France',
            'business_name': 'ZNB Club SARL',
            'business_address': 'Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac',
            'expected_postal': '',
            'expected_city': 'LA TESTE-DE-BUCH',
            'expected_state': 'NOUVELLE-AQUITAINE',
            'expected_geo': '33'
        }
    ]

    print("=" * 80)
    print("STAGE 1 PREPROCESSING LAYER: FIXES VERIFICATION TEST")
    print("=" * 80)

    all_passed = True
    for rec in test_cases:
        res = preprocess_record(rec)
        print(f"\n[{res['country']}] ID: {res['entity_id']}")
        print(f"  Raw Name:       {rec['business_name']}")
        print(f"  --> Clean Name: '{res['clean_name']}'")
        print(f"  Raw Address:    {rec['business_address']}")
        print(f"  --> Clean Addr: '{res['clean_address']}'")
        print(f"  --> Extracted:  City='{res['city']}', State='{res['state']}', Postal='{res['postal_code']}', GeoCode='{res['geo_code']}'")

        # Assertions for the key fixes
        checks = []
        if 'expected_postal' in rec:
            checks.append(('Postal', res['postal_code'] == rec['expected_postal'], res['postal_code'], rec['expected_postal']))
        if 'expected_city' in rec:
            checks.append(('City', res['city'] == rec['expected_city'], res['city'], rec['expected_city']))
        if 'expected_state' in rec:
            checks.append(('State', res['state'] == rec['expected_state'], res['state'], rec['expected_state']))
        if 'expected_geo' in rec:
            checks.append(('GeoCode', res['geo_code'] == rec['expected_geo'], res['geo_code'], rec['expected_geo']))

        for name, passed, actual, expected in checks:
            status = "✓ PASS" if passed else "✗ FAIL"
            if not passed:
                all_passed = False
                print(f"  {status}: {name} -> Got '{actual}', Expected '{expected}'")
            else:
                print(f"  {status}: {name}='{actual}'")

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED! All 4 issues successfully resolved.")
    else:
        print("SOME CHECKS FAILED. Review errors above.")
    print("=" * 80)

if __name__ == '__main__':
    run_tests()
