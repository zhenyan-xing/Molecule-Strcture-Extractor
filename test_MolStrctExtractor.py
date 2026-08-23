import json
import tempfile
import unittest
from pathlib import Path

from Bio.PDB import Chain, Residue

from MolStrctExtractor import (
    ChainSummary,
    ProteinOnlySelect,
    _load_overrides,
    choose_protein_chains,
    output_name_for,
)


def protein(
    chain_id: str,
    entity_id: str,
    length: int = 100,
    assembly_ids: tuple[str, ...] = ("1",),
) -> ChainSummary:
    return ChainSummary(
        chain_id=chain_id,
        label_chain_ids=(chain_id,),
        assembly_ids=assembly_ids,
        entity_id=entity_id,
        polymer_type="polypeptide(L)",
        kind="protein",
        residue_count=length,
        sequence="A" * length,
    )


class AutomaticSelectionTests(unittest.TestCase):
    def test_prefers_the_protein_with_most_nucleic_acid_contacts(self):
        proteins = {
            "A": protein("A", "1"),
            "B": protein("B", "1"),
        }

        result = choose_protein_chains(
            proteins,
            nucleic_contact_pairs={"A": 4, "B": 23},
            protein_interface_pairs={},
            min_interface_pairs=10,
        )

        self.assertEqual(result.selected_chain_ids, ("B",))

    def test_keeps_interacting_copies_of_the_same_protein(self):
        proteins = {
            "A": protein("A", "1"),
            "B": protein("B", "1"),
        }

        result = choose_protein_chains(
            proteins,
            nucleic_contact_pairs={"A": 20, "B": 9},
            protein_interface_pairs={("A", "B"): 18},
            min_interface_pairs=10,
        )

        self.assertEqual(result.selected_chain_ids, ("A", "B"))

    def test_drops_noninteracting_symmetric_copies(self):
        proteins = {
            "A": protein("A", "1"),
            "B": protein("B", "1"),
        }

        result = choose_protein_chains(
            proteins,
            nucleic_contact_pairs={"A": 20, "B": 19},
            protein_interface_pairs={("A", "B"): 3},
            min_interface_pairs=10,
        )

        self.assertEqual(result.selected_chain_ids, ("A",))

    def test_drops_crystal_contacts_from_different_biological_assemblies(self):
        proteins = {
            "A": protein("A", "1", assembly_ids=("1",)),
            "B": protein("B", "1", assembly_ids=("2",)),
        }

        result = choose_protein_chains(
            proteins,
            nucleic_contact_pairs={"A": 20, "B": 20},
            protein_interface_pairs={("A", "B"): 40},
            min_interface_pairs=10,
        )

        self.assertEqual(result.selected_chain_ids, ("A",))

    def test_keeps_all_distinct_protein_entities_in_the_selected_assembly(self):
        proteins = {
            "A": protein("A", "heavy_chain"),
            "B": protein("B", "light_chain"),
            "F": protein("F", "rna_binder", length=57),
        }

        result = choose_protein_chains(
            proteins,
            nucleic_contact_pairs={"A": 0, "B": 0, "F": 36},
            protein_interface_pairs={
                ("A", "B"): 64,
                ("A", "F"): 19,
                ("B", "F"): 22,
            },
            min_interface_pairs=10,
        )

        self.assertEqual(result.selected_chain_ids, ("A", "B", "F"))

    def test_does_not_pull_proteins_from_an_alternative_assembly(self):
        proteins = {
            "A": protein("A", "protein_a", assembly_ids=("1",)),
            "B": protein("B", "protein_b", assembly_ids=("2",)),
        }

        result = choose_protein_chains(
            proteins,
            nucleic_contact_pairs={"A": 20, "B": 19},
            protein_interface_pairs={("A", "B"): 30},
            min_interface_pairs=10,
        )

        self.assertEqual(result.selected_chain_ids, ("A",))

    def test_explicit_override_takes_precedence(self):
        proteins = {
            "A": protein("A", "1"),
            "B": protein("B", "2"),
        }

        result = choose_protein_chains(
            proteins,
            nucleic_contact_pairs={"A": 50, "B": 1},
            protein_interface_pairs={},
            min_interface_pairs=10,
            override=("B",),
        )

        self.assertEqual(result.selected_chain_ids, ("B",))
        self.assertEqual(result.method, "override")


class OutputTests(unittest.TestCase):
    def test_output_name_matches_benchmark_convention(self):
        self.assertEqual(output_name_for("8TG4.cif"), "8tg4protein.pdb")

    def test_multi_target_output_name_includes_chain_suffix(self):
        self.assertEqual(
            output_name_for("1URN.cif", ("B",)), "1urnprotein-b.pdb"
        )

    def test_override_file_can_define_multiple_independent_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overrides.json"
            path.write_text(json.dumps({"1URN": [["A"], ["B"]]}))

            overrides = _load_overrides(path)

        self.assertEqual(overrides["1urn"], (("A",), ("B",)))

    def test_writer_select_excludes_water_and_ligands(self):
        chain = Chain.Chain("A")
        alanine = Residue.Residue((" ", 1, " "), "ALA", " ")
        water = Residue.Residue(("W", 2, " "), "HOH", " ")
        ligand = Residue.Residue(("H_ATP", 3, " "), "ATP", " ")
        chain.add(alanine)
        chain.add(water)
        chain.add(ligand)
        selector = ProteinOnlySelect(("A",))

        self.assertEqual(selector.accept_residue(alanine), 1)
        self.assertEqual(selector.accept_residue(water), 0)
        self.assertEqual(selector.accept_residue(ligand), 0)


if __name__ == "__main__":
    unittest.main()
