# Business Entity Resolution Pipeline

This repository contains the end-to-end Machine Learning pipeline for the Business Entity Resolution Challenge.

## Stage 1: Preprocessing Layer

### Purpose
Transform raw, noisy business records into standardized, comparable formats. This stage handles country-specific variations in naming conventions and address formats.

### Components
1. **Country Detection**
   - Reads the `country` field from each record.
   - Standardizes country codes (`US`, `India`, `France`).
   - Routes to country-specific normalization rules with fallback for unseen countries.

2. **Name Normalization**
   - **Legal Suffix Removal**: Strips country-specific business suffixes:
     - *US*: `INC`, `CORP`, `LLC`, `LTD`, `CO`, `COMPANY`, `PC`, `PLLC`, `LP`, etc.
     - *India*: `PVT`, `PRIVATE`, `LTD`, `LIMITED`, `OPC`, `LLP`, `PLC`, and Indic script variants.
     - *France*: `SARL`, `SA`, `SAS`, `SASU`, `EURL`, `SCI`, `SNC`, `FILS`, etc.
     - *Fallback*: Universal business suffixes.
   - **Abbreviation Expansion**: Converts common business abbreviations to full canonical forms:
     - `&` → `AND`, `+` → `PLUS`, `PVT` → `PRIVATE`, `CORP` → `CORPORATION`, `CO` → `COMPANY`, `LTD` → `LIMITED`, `INC` → `INCORPORATED`, `TECH` → `TECHNOLOGY`, `MGMT` → `MANAGEMENT`, `SVCS` → `SERVICES`, `STE` → `SOCIETE`, `CIE` → `COMPAGNIE`, etc.
   - **Text Cleanup**: Uppercase conversion, Latin accent stripping (preserving Indic scripts), punctuation removal, whitespace normalization.

3. **Address Normalization**
   - **Street Type Standardization**: `Rd` → `ROAD`, `St` → `STREET`, `Ave` → `AVENUE`, `Blvd` → `BOULEVARD`, `Dr` → `DRIVE`, `Ln` → `LANE`, `Ct` → `COURT`, `Pkwy` → `PARKWAY`, `R` / `Rue` → `RUE`, `Av` → `AVENUE`, `Bd` → `BOULEVARD`, `Marg` → `MARG`, `Nagar` → `NAGAR`, `HNo` → `HNO`, etc.
   - **Component Extraction**: Parses address into structured fields (`city`, `state`, `postal_code`, `street_address`).
   - **Geographic Code Extraction**:
     - *US*: First 3 digits of ZIP code (e.g., `902`, fallback to state prefix).
     - *India*: First 3 digits of PIN code (e.g., `600`, fallback to state/city prefix).
     - *France*: First 2 digits of postal code / department (e.g., `33`, `59`, `44`, fallback to department/city code).
     - *Fallback*: First 2-3 characters of extracted postal code.

### Usage
Run the standalone preprocessor CLI:
```bash
python src/preprocess.py --input ../../dataset/train/train_source1.tsv --sample 20
```

Or import as a module:
```python
from src.preprocess import preprocess_record, preprocess_dataframe, preprocess_tsv_file

# Preprocess a single record
clean_record = preprocess_record({
    "entity_id": "S1-00001",
    "country": "US",
    "business_name": "Acme Corp Inc.",
    "business_address": "123 Main St, New York, NY 10001"
})

# Preprocess a DataFrame
clean_df = preprocess_dataframe(df)
```
