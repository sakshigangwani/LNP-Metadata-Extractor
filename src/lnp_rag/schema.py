"""The LNP formulation metadata schema.

This Pydantic model is the single source of truth for the fields the model extracts.
Field descriptions are sent to the model (they become the Structured Outputs JSON
schema), so keep them accurate and curator-facing.

Every field is Optional — the model returns null when a value is not reported in
the paper, and records the missing/ambiguous field names in `uncertain_fields`.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class LNPMetadata(BaseModel):
    # ----- ID -----
    paper_doi: Optional[str] = Field(None, description="DOI of the paper, e.g. '10.1038/s41565-020-0669-6'.")
    paper_first_author: Optional[str] = Field(None, description="Surname of the first author.")
    formulation_id: Optional[str] = Field(None, description="Any label the paper uses to refer to this specific formulation or LNP sample — e.g. 'LNP-1', 'cKK-E12', 'Formulation A', a sample/batch code, OR a cargo- or target-based name such as 'Luc-LNP', 'mTERT-LNP', 'mRNA-LNP'. Use the paper's own name for the lead formulation. Only leave null if the paper truly uses no label at all.")
    curator_initials: Optional[str] = Field(None, description="Initials of the human curator. Usually null at extraction time.")
    curation_date: Optional[str] = Field(None, description="Curation date in ISO format (YYYY-MM-DD). Usually null at extraction time.")

    # ----- Ionizable lipid -----
    il_name: Optional[str] = Field(None, description="Name of the ionizable/cationic lipid (e.g. 'MC3', 'ALC-0315', 'SM-102').")
    il_smiles: Optional[str] = Field(None, description="SMILES string of the ionizable lipid, if given.")
    il_mol_pct: Optional[float] = Field(None, description="Ionizable lipid molar percentage of total lipid (mol %).")
    il_pka_measured: Optional[str] = Field(None, description="Measured apparent pKa of the ionizable lipid (not the formulation), EXACTLY as reported (e.g. '6.4', '6.0-6.5').")
    il_is_clinical: Optional[bool] = Field(None, description="True if the ionizable lipid is clinically approved/used (e.g. MC3, ALC-0315, SM-102).")

    # ----- Phospholipid -----
    pl_name: Optional[str] = Field(None, description="Helper phospholipid name (e.g. 'DSPC', 'DOPE').")
    pl_smiles: Optional[str] = Field(None, description="SMILES of the phospholipid, if given.")
    pl_mol_pct: Optional[float] = Field(None, description="Phospholipid molar percentage (mol %).")

    # ----- Cholesterol -----
    chol_type: Optional[str] = Field(None, description="Cholesterol or sterol type (e.g. 'cholesterol', 'beta-sitosterol').")
    chol_smiles: Optional[str] = Field(None, description="SMILES of the cholesterol/sterol, if given.")
    chol_mol_pct: Optional[float] = Field(None, description="Cholesterol molar percentage (mol %).")

    # ----- PEG / coating -----
    peg_lipid_name: Optional[str] = Field(None, description="PEG-lipid name (e.g. 'DMG-PEG2000', 'ALC-0159').")
    peg_lipid_smiles: Optional[str] = Field(None, description="SMILES of the PEG-lipid, if given.")
    peg_molecular_weight: Optional[float] = Field(None, description="Molecular weight of the PEG block in Da (e.g. 2000).")
    peg_lipid_mol_pct: Optional[float] = Field(None, description="PEG-lipid molar percentage (mol %).")
    alternative_coating: Optional[str] = Field(None, description="Alternative surface coating instead of/with PEG (e.g. 'polysarcosine', 'none').")

    # ----- Formulation -----
    np_ratio: Optional[str] = Field(None, description="N/P ratio (ionizable amine nitrogen to nucleic-acid phosphate), EXACTLY as reported (e.g. '6', '3:1').")
    total_lipid_concentration: Optional[str] = Field(None, description="Total lipid concentration EXACTLY as reported, including units (e.g. '0.5 mg/mL', '~1 mg/mL').")
    manufacturing_route: Optional[str] = Field(None, description="Manufacturing method (e.g. 'microfluidic mixing', 'ethanol injection', 'T-junction').")
    formulation_buffer: Optional[str] = Field(None, description="Buffer used during formulation/mixing (e.g. 'citrate pH 4', 'acetate').")
    formulation_ph: Optional[str] = Field(None, description="pH of the formulation buffer during mixing, EXACTLY as reported (e.g. '4', '4.0-4.5').")
    dispersion_buffer: Optional[str] = Field(None, description="Final dispersion/storage buffer (e.g. 'PBS', 'Tris sucrose').")

    # ----- Particle (values recorded verbatim — papers report ranges/inequalities/qualitative) -----
    hydrodynamic_diameter_nm: Optional[str] = Field(None, description="Z-average hydrodynamic diameter (DLS) in nm, EXACTLY as reported — preserve ranges/approximations (e.g. '~90-110 nm', '85 nm', '<100 nm'). Do not round to one number.")
    pdi: Optional[str] = Field(None, description="Polydispersity index EXACTLY as reported — preserve inequalities/ranges (e.g. '<0.2', '0.12', '0.1-0.2').")
    zeta_potential_mv: Optional[str] = Field(None, description="Zeta potential EXACTLY as reported, INCLUDING qualitative descriptors (e.g. 'slightly negative', 'near neutral', '-2.3 mV', '-5 to -10 mV').")
    encapsulation_efficiency_pct: Optional[str] = Field(None, description="Encapsulation efficiency EXACTLY as reported — preserve inequalities/ranges (e.g. '>95%', '94%', '90-95%').")
    apparent_pka_formulation: Optional[str] = Field(None, description="Apparent pKa of the assembled LNP (e.g. by TNS assay), EXACTLY as reported (e.g. '6.2', '6.0-6.5').")
    morphology: Optional[str] = Field(None, description="Reported particle morphology (e.g. 'spherical', 'electron-dense core').")

    # ----- Cargo -----
    cargo_copies_per_particle: Optional[str] = Field(None, description="Estimated cargo copies per particle, EXACTLY as reported (e.g. '~10', '5-15'), if given.")
    cargo_type: Optional[str] = Field(None, description="Cargo type (e.g. 'mRNA', 'siRNA', 'saRNA', 'pDNA', 'ASO').")
    cargo_length_nt: Optional[int] = Field(None, description="Cargo length in nucleotides.")
    cargo_modifications: Optional[str] = Field(None, description="Chemical modifications of the cargo (e.g. 'N1-methylpseudouridine', '2'-OMe').")
    cap_structure: Optional[str] = Field(None, description="mRNA cap structure (e.g. 'Cap1', 'ARCA').")
    poly_a_tail_length: Optional[int] = Field(None, description="Poly(A) tail length in nucleotides.")
    cargo_lipid_mass_ratio: Optional[str] = Field(None, description="Cargo-to-total-lipid mass ratio EXACTLY as reported (e.g. '1:10 wt/wt', '0.05').")

    # ----- Biology -----
    study_type: Optional[str] = Field(None, description="Study type: 'in vitro', 'in vivo', 'ex vivo', or 'clinical'.")
    species: Optional[str] = Field(None, description="Species studied (e.g. 'mouse', 'human', 'non-human primate').")
    strain_or_line: Optional[str] = Field(None, description="Animal strain or cell line (e.g. 'C57BL/6', 'HeLa', 'HEK293').")
    cell_type_or_tissue: Optional[str] = Field(None, description="Cell type or target tissue/organ (e.g. 'hepatocyte', 'liver', 'spleen').")
    disease_state: Optional[str] = Field(None, description="Disease model or condition, if any (e.g. 'B16F10 melanoma', 'healthy').")
    route_of_administration: Optional[str] = Field(None, description="Route of administration (e.g. 'IV', 'IM', 'SC', 'intratumoral').")
    dose_mg_per_kg: Optional[str] = Field(None, description="In vivo dose of cargo in mg/kg, EXACTLY as reported (e.g. '0.5 mg/kg', '0.1-1 mg/kg').")
    dose_ng_per_well: Optional[str] = Field(None, description="In vitro dose of cargo in ng/well, EXACTLY as reported (e.g. '100 ng', '50-200 ng').")
    time_post_admin_hours: Optional[str] = Field(None, description="Time point of the main readout post-administration, EXACTLY as reported (e.g. '6 h', '24-48 h').")

    # ----- Immunogenicity -----
    pre_anti_peg_titer_measured: Optional[bool] = Field(None, description="True if a pre-existing anti-PEG antibody titer was measured.")
    pre_anti_peg_titer_value: Optional[str] = Field(None, description="Pre-existing anti-PEG titer value EXACTLY as reported (e.g. '1:160', '<1:40'), if reported.")
    post_anti_peg_titer_measured: Optional[bool] = Field(None, description="True if a post-dose anti-PEG titer was measured.")
    post_anti_peg_igm_titer: Optional[str] = Field(None, description="Post-dose anti-PEG IgM titer value EXACTLY as reported (e.g. '1:1280', '2-fold increase').")
    post_anti_peg_igg_titer: Optional[str] = Field(None, description="Post-dose anti-PEG IgG titer value EXACTLY as reported.")
    cytokine_panel_measured: Optional[bool] = Field(None, description="True if a cytokine panel was measured.")
    cytokines_reported: Optional[List[str]] = Field(None, description="List of cytokines reported (e.g. ['IL-6', 'TNF-alpha', 'IFN-gamma']).")
    complement_measured: Optional[bool] = Field(None, description="True if complement activation was measured.")
    complement_markers: Optional[List[str]] = Field(None, description="Complement markers reported (e.g. ['C3a', 'C5a', 'SC5b-9']).")
    abc_observed: Optional[bool] = Field(None, description="True if accelerated blood clearance (ABC) was observed.")
    abc_fold_change: Optional[str] = Field(None, description="ABC fold-change in clearance EXACTLY as reported (e.g. '2-fold', '~3x'), if quantified.")
    repeat_dose_study: Optional[bool] = Field(None, description="True if the study used repeat dosing.")
    dose_count: Optional[int] = Field(None, description="Number of doses administered.")
    inter_dose_interval_days: Optional[str] = Field(None, description="Interval between doses EXACTLY as reported (e.g. '21 days', '3 weeks').")

    # ----- Notes -----
    notes: Optional[str] = Field(None, description="Free-text notes: caveats, units, anything that didn't fit a field.")
    uncertain_fields: Optional[List[str]] = Field(None, description="Names of fields you were uncertain about or had to infer. Helps the curator prioritize review.")


# Curator-facing grouping for display (label -> ordered field names).
FIELD_GROUPS: dict[str, list[str]] = {
    "ID": ["paper_doi", "paper_first_author", "formulation_id", "curator_initials", "curation_date"],
    "Ionizable Lipid": ["il_name", "il_smiles", "il_mol_pct", "il_pka_measured", "il_is_clinical"],
    "Phospholipid": ["pl_name", "pl_smiles", "pl_mol_pct"],
    "Cholesterol": ["chol_type", "chol_smiles", "chol_mol_pct"],
    "PEG / Coating": ["peg_lipid_name", "peg_lipid_smiles", "peg_molecular_weight", "peg_lipid_mol_pct", "alternative_coating"],
    "Formulation": ["np_ratio", "total_lipid_concentration", "manufacturing_route", "formulation_buffer", "formulation_ph", "dispersion_buffer"],
    "Particle": ["hydrodynamic_diameter_nm", "pdi", "zeta_potential_mv", "encapsulation_efficiency_pct", "apparent_pka_formulation", "morphology"],
    "Cargo": ["cargo_copies_per_particle", "cargo_type", "cargo_length_nt", "cargo_modifications", "cap_structure", "poly_a_tail_length", "cargo_lipid_mass_ratio"],
    "Biology": ["study_type", "species", "strain_or_line", "cell_type_or_tissue", "disease_state", "route_of_administration", "dose_mg_per_kg", "dose_ng_per_well", "time_post_admin_hours"],
    "Immunogenicity": [
        "pre_anti_peg_titer_measured", "pre_anti_peg_titer_value", "post_anti_peg_titer_measured",
        "post_anti_peg_igm_titer", "post_anti_peg_igg_titer", "cytokine_panel_measured", "cytokines_reported",
        "complement_measured", "complement_markers", "abc_observed", "abc_fold_change",
        "repeat_dose_study", "dose_count", "inter_dose_interval_days",
    ],
    "Notes": ["notes", "uncertain_fields"],
}

# Ordered flat list of all field names (matches the requested column order).
ALL_FIELDS: list[str] = [f for fields in FIELD_GROUPS.values() for f in fields]
