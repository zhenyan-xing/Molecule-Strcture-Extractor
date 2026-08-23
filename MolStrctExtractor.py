#!/usr/bin/env python3
"""Extract one or more RNA-binding protein targets from each PDB/mmCIF in a folder.

The default automatic rule is deliberately conservative:

1. Find the protein chain with the most residue-level contacts to nucleic acid.
2. Select its biological assembly and keep one representative of every distinct
   protein entity in that assembly.
3. Keep multiple copies of the same protein entity only when they form a
   substantial protein-protein interface.
4. Exclude RNA, DNA, water, ions, and non-protein ligands.

Every decision is recorded in ``extraction_report.json``. Pass a JSON override
for a known target, or a nested list for multiple independent targets from one
complex, for example
``{"1URN": [["A"], ["B"]], "6CF2": ["A", "B", "F"]}``.
Chain IDs are author chain IDs.

Example:
    python MolStrctExtractor.py ./complexes -o ./protein_structures
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from Bio.PDB import MMCIFParser, NeighborSearch, PDBIO, PDBParser, Select
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Polypeptide import (
    is_aa,
    is_nucleic,
    protein_letters_3to1_extended,
)


STRUCTURE_SUFFIXES = {".cif", ".mmcif", ".pdb", ".ent"}


@dataclass(frozen=True)
class ChainSummary:
    chain_id: str
    label_chain_ids: tuple[str, ...]
    assembly_ids: tuple[str, ...]
    entity_id: str | None
    polymer_type: str | None
    kind: str
    residue_count: int
    sequence: str


@dataclass(frozen=True)
class SelectionResult:
    selected_chain_ids: tuple[str, ...]
    primary_chain_id: str | None
    method: str
    reason: str
    warnings: tuple[str, ...]


class ProteinOnlySelect(Select):
    """PDBIO selector that keeps amino-acid residues in selected chains."""

    def __init__(self, chain_ids: Iterable[str]):
        self.chain_ids = frozenset(chain_ids)

    def accept_chain(self, chain) -> int:
        return int(chain.id in self.chain_ids)

    def accept_residue(self, residue) -> int:
        chain = residue.get_parent()
        return int(chain.id in self.chain_ids and _is_protein_residue(residue))


def _is_protein_residue(residue) -> bool:
    if is_aa(residue, standard=False):
        return True
    atom_names = set(residue.child_dict)
    return {"N", "CA", "C"}.issubset(atom_names)


def _is_nucleic_residue(residue) -> bool:
    residue_name = residue.get_resname().strip().upper()
    if is_nucleic(residue_name.ljust(3), standard=False):
        return True
    atom_names = set(residue.child_dict)
    has_sugar = any(name in atom_names for name in ("C4'", "C4*"))
    has_base = any(name in atom_names for name in ("N1", "N9"))
    return has_sugar and has_base


def _is_water(residue) -> bool:
    return residue.id[0] == "W" or residue.get_resname().strip() in {"HOH", "WAT"}


def _as_list(value) -> list[str]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _clean_cif_value(value: str | None) -> str | None:
    if value is None or value in {".", "?", ""}:
        return None
    return str(value)


def _mmcif_chain_metadata(path: Path) -> dict[str, dict[str, object]]:
    """Map author chain IDs used by Biopython to mmCIF entity metadata."""

    mmcif = MMCIF2Dict(str(path))
    entity_types = {
        str(entity): str(polymer_type)
        for entity, polymer_type in zip(
            _as_list(mmcif.get("_entity_poly.entity_id")),
            _as_list(mmcif.get("_entity_poly.type")),
        )
    }
    label_entities = {
        str(label): str(entity)
        for label, entity in zip(
            _as_list(mmcif.get("_struct_asym.id")),
            _as_list(mmcif.get("_struct_asym.entity_id")),
        )
    }

    label_to_auth: dict[str, str] = {}
    for label, auth in zip(
        _as_list(mmcif.get("_atom_site.label_asym_id")),
        _as_list(mmcif.get("_atom_site.auth_asym_id")),
    ):
        cleaned_auth = _clean_cif_value(auth)
        label_to_auth.setdefault(str(label), cleaned_auth or str(label))

    label_assemblies: dict[str, set[str]] = defaultdict(set)
    for assembly_id, asym_id_list in zip(
        _as_list(mmcif.get("_pdbx_struct_assembly_gen.assembly_id")),
        _as_list(mmcif.get("_pdbx_struct_assembly_gen.asym_id_list")),
    ):
        for label in str(asym_id_list).replace(" ", "").split(","):
            if label:
                label_assemblies[label].add(str(assembly_id))

    by_auth: dict[str, dict[str, object]] = {}
    for label, entity_id in label_entities.items():
        auth = label_to_auth.get(label, label)
        polymer_type = entity_types.get(entity_id)
        record = by_auth.setdefault(
            auth,
            {
                "entity_id": None,
                "polymer_type": None,
                "label_chain_ids": [],
                "assembly_ids": set(),
            },
        )
        record["label_chain_ids"].append(label)
        record["assembly_ids"].update(label_assemblies.get(label, set()))
        if polymer_type is not None:
            record["entity_id"] = entity_id
            record["polymer_type"] = polymer_type

    return by_auth


def _kind_from_polymer_type(polymer_type: str | None) -> str | None:
    if not polymer_type:
        return None
    lowered = polymer_type.lower()
    if "polypeptide" in lowered:
        return "protein"
    if "ribonucleotide" in lowered or "deoxyribonucleotide" in lowered:
        return "nucleic_acid"
    return None


def _residues_for_kind(chain, kind: str) -> list:
    residues = list(chain.get_residues())
    if kind == "protein":
        return [residue for residue in residues if _is_protein_residue(residue)]
    if kind == "nucleic_acid":
        recognized = [residue for residue in residues if _is_nucleic_residue(residue)]
        if recognized:
            return recognized
        return [
            residue
            for residue in residues
            if not _is_water(residue) and not _is_protein_residue(residue)
        ]
    return []


def _protein_sequence(residues: Sequence) -> str:
    return "".join(
        protein_letters_3to1_extended.get(
            residue.get_resname().strip().upper(), "X"
        )
        for residue in residues
    )


def load_structure(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
        metadata = _mmcif_chain_metadata(path)
    else:
        parser = PDBParser(QUIET=True)
        metadata = {}
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models())
    return structure, model, metadata


def summarize_chains(model, metadata: Mapping[str, Mapping[str, object]]):
    summaries: dict[str, ChainSummary] = {}
    for chain in model:
        record = metadata.get(chain.id, {})
        polymer_type = record.get("polymer_type")
        kind = _kind_from_polymer_type(
            str(polymer_type) if polymer_type is not None else None
        )

        if kind is None:
            protein_count = sum(_is_protein_residue(r) for r in chain.get_residues())
            nucleic_count = sum(_is_nucleic_residue(r) for r in chain.get_residues())
            if protein_count > nucleic_count and protein_count:
                kind = "protein"
            elif nucleic_count:
                kind = "nucleic_acid"
            else:
                kind = "other"

        residues = _residues_for_kind(chain, kind)
        entity_id = _clean_cif_value(record.get("entity_id"))
        labels = tuple(str(label) for label in record.get("label_chain_ids", ()))
        assembly_ids = tuple(
            sorted(str(value) for value in record.get("assembly_ids", ()))
        )
        summaries[chain.id] = ChainSummary(
            chain_id=chain.id,
            label_chain_ids=labels,
            assembly_ids=assembly_ids,
            entity_id=entity_id,
            polymer_type=str(polymer_type) if polymer_type is not None else None,
            kind=kind,
            residue_count=len(residues),
            sequence=_protein_sequence(residues) if kind == "protein" else "",
        )
    return summaries


def _residue_key(residue) -> tuple[str, int, str, str]:
    hetero_flag, sequence_number, insertion_code = residue.id
    return (
        str(hetero_flag),
        int(sequence_number),
        str(insertion_code),
        residue.get_resname().strip(),
    )


def calculate_contacts(model, summaries: Mapping[str, ChainSummary], cutoff: float):
    residue_owner: dict[int, tuple[str, str]] = {}
    atoms = []

    for chain in model:
        summary = summaries.get(chain.id)
        if summary is None or summary.kind not in {"protein", "nucleic_acid"}:
            continue
        for residue in _residues_for_kind(chain, summary.kind):
            residue_owner[id(residue)] = (chain.id, summary.kind)
            atoms.extend(
                atom
                for atom in residue.get_atoms()
                if (atom.element or "").strip().upper() not in {"H", "D"}
            )

    nucleic_contacts: dict[str, set[tuple]] = defaultdict(set)
    protein_interfaces: dict[tuple[str, str], set[tuple]] = defaultdict(set)
    if not atoms:
        return {}, {}

    neighbors = NeighborSearch(atoms)
    for left, right in neighbors.search_all(cutoff, level="R"):
        left_owner = residue_owner.get(id(left))
        right_owner = residue_owner.get(id(right))
        if left_owner is None or right_owner is None:
            continue
        left_chain, left_kind = left_owner
        right_chain, right_kind = right_owner
        if left_chain == right_chain:
            continue

        if {left_kind, right_kind} == {"protein", "nucleic_acid"}:
            if left_kind == "protein":
                protein_chain, protein_residue = left_chain, left
                nucleic_chain, nucleic_residue = right_chain, right
            else:
                protein_chain, protein_residue = right_chain, right
                nucleic_chain, nucleic_residue = left_chain, left
            nucleic_contacts[protein_chain].add(
                (
                    _residue_key(protein_residue),
                    nucleic_chain,
                    _residue_key(nucleic_residue),
                )
            )
        elif left_kind == right_kind == "protein":
            chain_pair = tuple(sorted((left_chain, right_chain)))
            if left_chain == chain_pair[0]:
                residue_pair = (_residue_key(left), _residue_key(right))
            else:
                residue_pair = (_residue_key(right), _residue_key(left))
            protein_interfaces[chain_pair].add(residue_pair)

    return (
        {chain: len(pairs) for chain, pairs in nucleic_contacts.items()},
        {pair: len(pairs) for pair, pairs in protein_interfaces.items()},
    )


def _same_protein(left: ChainSummary, right: ChainSummary) -> bool:
    if left.entity_id is not None and right.entity_id is not None:
        return left.entity_id == right.entity_id
    return bool(left.sequence and left.sequence == right.sequence)


def _protein_entity_groups(
    chain_ids: Iterable[str], proteins: Mapping[str, ChainSummary]
) -> list[set[str]]:
    remaining = set(chain_ids)
    groups = []
    while remaining:
        seed = min(remaining)
        group = {
            chain
            for chain in remaining
            if _same_protein(proteins[seed], proteins[chain])
        }
        groups.append(group)
        remaining -= group
    return groups


def choose_protein_chains(
    proteins: Mapping[str, ChainSummary],
    nucleic_contact_pairs: Mapping[str, int],
    protein_interface_pairs: Mapping[tuple[str, str], int],
    min_interface_pairs: int,
    override: Sequence[str] | None = None,
    keep_all: bool = False,
) -> SelectionResult:
    if not proteins:
        raise ValueError("No protein chains were found")

    if override is not None:
        selected = tuple(dict.fromkeys(str(chain) for chain in override))
        missing = [chain for chain in selected if chain not in proteins]
        if missing:
            raise ValueError(f"Override contains non-protein chain IDs: {missing}")
        if not selected:
            raise ValueError("Override must contain at least one protein chain ID")
        return SelectionResult(
            selected_chain_ids=selected,
            primary_chain_id=selected[0],
            method="override",
            reason="Selected by the user-provided override file.",
            warnings=(),
        )

    if keep_all:
        selected = tuple(sorted(proteins))
        return SelectionResult(
            selected_chain_ids=selected,
            primary_chain_id=selected[0],
            method="keep_all",
            reason="Kept every detected protein chain by request.",
            warnings=(),
        )

    ranked = sorted(
        proteins,
        key=lambda chain: (
            -nucleic_contact_pairs.get(chain, 0),
            -proteins[chain].residue_count,
            chain,
        ),
    )
    primary = ranked[0]
    primary_assemblies = proteins[primary].assembly_ids
    selected_assembly = min(primary_assemblies) if primary_assemblies else None
    if selected_assembly is None:
        in_scope = set(proteins)
        assembly_description = "the coordinate model"
    else:
        in_scope = {
            chain
            for chain, summary in proteins.items()
            if selected_assembly in summary.assembly_ids
        }
        assembly_description = f"biological assembly {selected_assembly}"

    selected: set[str] = set()
    for group in _protein_entity_groups(in_scope, proteins):
        if primary in group:
            representative = primary
        else:
            representative = sorted(
                group,
                key=lambda chain: (
                    -sum(
                        protein_interface_pairs.get(
                            tuple(sorted((chain, other))), 0
                        )
                        for other in in_scope
                        if other not in group
                    ),
                    -nucleic_contact_pairs.get(chain, 0),
                    -proteins[chain].residue_count,
                    chain,
                ),
            )[0]

        entity_selection = {representative}
        queue = deque([representative])
        while queue:
            current = queue.popleft()
            for candidate in group:
                if candidate in entity_selection:
                    continue
                pair = tuple(sorted((current, candidate)))
                if protein_interface_pairs.get(pair, 0) >= min_interface_pairs:
                    entity_selection.add(candidate)
                    queue.append(candidate)
        selected.update(entity_selection)

    warnings = []
    primary_contacts = nucleic_contact_pairs.get(primary, 0)
    if primary_contacts == 0:
        warnings.append(
            "No protein-nucleic-acid contacts were detected; selected the largest "
            "protein chain."
        )

    selected_ids = tuple(sorted(selected))
    reason = (
        f"Chain {primary} had the strongest nucleic-acid interface "
        f"({primary_contacts} residue pairs). Retained one representative of "
        f"every protein entity in {assembly_description}; interacting copies "
        f"of the same entity were also retained ({', '.join(selected_ids)})."
    )

    return SelectionResult(
        selected_chain_ids=selected_ids,
        primary_chain_id=primary,
        method="auto",
        reason=reason,
        warnings=tuple(warnings),
    )


def output_name_for(
    input_name: str | Path, chain_ids: Sequence[str] | None = None
) -> str:
    stem = f"{Path(input_name).stem.lower()}protein"
    suffix = (
        "-" + "-".join(chain.lower() for chain in chain_ids)
        if chain_ids
        else ""
    )
    return f"{stem}{suffix}.pdb"


def _write_protein(model, output_path: Path, chain_ids: Sequence[str]) -> None:
    invalid = [chain for chain in chain_ids if len(chain) != 1]
    if invalid:
        raise ValueError(
            "PDB output requires one-character author chain IDs; unsupported IDs: "
            + ", ".join(invalid)
        )
    writer = PDBIO()
    writer.set_structure(model)
    writer.save(str(output_path), ProteinOnlySelect(chain_ids))


def _verify_output(path: Path, expected_chain_ids: Sequence[str]) -> dict[str, int]:
    structure = PDBParser(QUIET=True).get_structure(path.stem, str(path))
    model = next(structure.get_models())
    observed_chains = {chain.id for chain in model}
    expected = set(expected_chain_ids)
    if observed_chains != expected:
        raise RuntimeError(
            f"Output verification failed for {path.name}: expected chains "
            f"{sorted(expected)}, found {sorted(observed_chains)}"
        )

    residues = list(model.get_residues())
    non_protein = [residue for residue in residues if not _is_protein_residue(residue)]
    if non_protein:
        raise RuntimeError(
            f"Output verification failed for {path.name}: found non-protein residues"
        )
    return {
        "chain_count": len(observed_chains),
        "residue_count": len(residues),
        "atom_count": sum(1 for _ in model.get_atoms()),
    }


def _normalize_override_groups(value, name: str) -> tuple[tuple[str, ...], ...]:
    if isinstance(value, str):
        groups = ((value,),)
    elif isinstance(value, (list, tuple)) and value and all(
        isinstance(chain, str) for chain in value
    ):
        groups = (tuple(value),)
    elif isinstance(value, (list, tuple)) and value:
        if not all(
            isinstance(group, (list, tuple))
            and group
            and all(isinstance(chain, str) for chain in group)
            for group in value
        ):
            raise ValueError(
                f"Override for {name!r} must contain non-empty chain groups"
            )
        groups = tuple(tuple(group) for group in value)
    else:
        raise ValueError(
            f"Override for {name!r} must be a chain, chain list, or nested chain lists"
        )
    return groups


def _load_overrides(path: Path | None) -> dict[str, tuple[tuple[str, ...], ...]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Override JSON must be an object mapping structure names to chains")
    overrides = {}
    for name, chains in raw.items():
        normalized_name = Path(str(name)).stem.lower()
        overrides[normalized_name] = _normalize_override_groups(chains, str(name))
    return overrides


def find_structure_files(input_dir: Path) -> list[Path]:
    files = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in STRUCTURE_SUFFIXES
        and not path.stem.lower().endswith("protein")
    ]
    return sorted(files, key=lambda path: path.name.lower())


def process_folder(
    input_dir: Path,
    output_dir: Path,
    *,
    contact_cutoff: float = 5.0,
    min_interface_pairs: int = 10,
    overrides: Mapping[str, object] | None = None,
    keep_all: bool = False,
    overwrite: bool = False,
) -> dict:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    inputs = find_structure_files(input_dir)
    if not inputs:
        raise FileNotFoundError(f"No PDB/mmCIF files found directly under {input_dir}")

    normalized_overrides = {
        Path(str(name)).stem.lower(): _normalize_override_groups(chains, str(name))
        for name, chains in (overrides or {}).items()
    }
    jobs = []
    for input_path in inputs:
        target_groups = normalized_overrides.get(input_path.stem.lower())
        if target_groups is None:
            jobs.append((input_path, output_dir / output_name_for(input_path), None))
            continue
        multiple_targets = len(target_groups) > 1
        for group in target_groups:
            output_name = output_name_for(
                input_path, group if multiple_targets else None
            )
            jobs.append((input_path, output_dir / output_name, group))

    output_paths = [output_path for _, output_path, _ in jobs]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("Multiple input files would produce the same output filename")
    existing = [path for path in output_paths if path.exists()]
    report_path = output_dir / "extraction_report.json"
    if report_path.exists():
        existing.append(report_path)
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(path.name for path in existing)
            + ". Pass --overwrite to replace files created by an earlier run."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []

    for input_path, output_path, target_override in jobs:
        _, model, metadata = load_structure(input_path)
        summaries = summarize_chains(model, metadata)
        proteins = {
            chain: summary
            for chain, summary in summaries.items()
            if summary.kind == "protein" and summary.residue_count > 0
        }
        nucleic_contacts, protein_interfaces = calculate_contacts(
            model, summaries, contact_cutoff
        )
        selection = choose_protein_chains(
            proteins,
            nucleic_contacts,
            protein_interfaces,
            min_interface_pairs,
            override=target_override,
            keep_all=keep_all,
        )
        _write_protein(model, output_path, selection.selected_chain_ids)
        verification = _verify_output(output_path, selection.selected_chain_ids)

        entries.append(
            {
                "input_file": str(input_path),
                "output_file": str(output_path),
                "selection": asdict(selection),
                "chains": [asdict(summaries[chain]) for chain in sorted(summaries)],
                "nucleic_contact_residue_pairs": {
                    chain: nucleic_contacts.get(chain, 0) for chain in sorted(proteins)
                },
                "protein_interface_residue_pairs": {
                    f"{left}--{right}": count
                    for (left, right), count in sorted(protein_interfaces.items())
                },
                "verification": verification,
            }
        )

    report = {
        "schema_version": 2,
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "settings": {
            "contact_cutoff_angstrom": contact_cutoff,
            "min_interface_residue_pairs": min_interface_pairs,
            "keep_all_proteins": keep_all,
        },
        "structures": entries,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Folder containing PDB/mmCIF complexes")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output folder (default: INPUT_DIR/extracted-proteins)",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        help='JSON chain groups, e.g. {"1URN": [["A"], ["B"]]}',
    )
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=5.0,
        help="Heavy-atom contact cutoff in angstroms (default: 5.0)",
    )
    parser.add_argument(
        "--min-interface-pairs",
        type=int,
        default=10,
        help="Residue-pair threshold for retaining interacting symmetric copies (default: 10)",
    )
    parser.add_argument(
        "--keep-all-proteins",
        action="store_true",
        help="Keep all protein chains instead of selecting one target assembly",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files from an earlier extraction run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or args.input_dir / "extracted-proteins"
    report = process_folder(
        args.input_dir,
        output_dir,
        contact_cutoff=args.contact_cutoff,
        min_interface_pairs=args.min_interface_pairs,
        overrides=_load_overrides(args.overrides),
        keep_all=args.keep_all_proteins,
        overwrite=args.overwrite,
    )
    for entry in report["structures"]:
        selection = entry["selection"]
        print(
            f"{Path(entry['input_file']).name} -> "
            f"{Path(entry['output_file']).name}: "
            f"chains {','.join(selection['selected_chain_ids'])} "
            f"({selection['method']})"
        )
        for warning in selection["warnings"]:
            print(f"  WARNING: {warning}")
    print(f"Report: {Path(report['output_directory']) / 'extraction_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
