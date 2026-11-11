import logging
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent

SPACE_GROUPS = yaml.safe_load(open(ROOT / "data" / "space_groups.yaml", "r"))
WYCKOFF_DATA = yaml.safe_load(open(ROOT / "data" / "wyckoff.yaml", "r"))

LOG = logging.getLogger("crys_former")

def sg_tokenizer(sg, vocab):
    """
    Expands the given space group into its space group symbol, crystal-lattice
    system, point symmetry, crystal class, and point group, and converts this
    information into tokens.

    Parameters
    ----------
    sg: int
        The space group to be tokenized.
    vocab: dict
        A dictionary mapping each token in the vocab to an index.

    Returns a list of indices corresponding to tokens representing the space
    group information of the given space group.
    """
    sg = SPACE_GROUPS[sg]
    token_indices = [
        vocab[sg["full_symbol"]],
        vocab[sg["crystal_system"] + "-" + sg["lattice_system"]],
        vocab[sg["point_symmetry"]],
        vocab[sg["crystal_class"]],
        vocab[sg["point_group"]],
    ]
    return token_indices


# def wyckoff_tokenizer(
#     wyckoff_string: str, comp_vocab: Dict[str, int], return_multiplicity: bool = False
# ) -> tuple[list[list] | list, ...]:
#     """
#     Parameters
#     ----------

#     # wyckoff_tokenizer example
#     comp_vocab = {"O": 0, "Mn": 1}
#     el_indices, wyckoff_indices = wyckoff_tokenizer("(O,10)-(Mn,5)", comp_vocab)
#     # Output: ([0, 1], [10, 5])

#     wyckoff_string: str
#         A string containing the Wyckoff position information.
#         The element-wyckoff pairs to be tokenized.
#         Expected: "({element},{wyckoff})-({element},{wyckoff})"
#         Example: "(O,10)-(Mn,10)"
#     comp_vocab: dict
#         The vocabulary used to tokenize composition.

#     Returns a tuple containing:
#     - A list of indices corresponding to the elements in the composition (index 0).
#     - A list of wyckoff indices (index 1).
#     - A list of multiplicities (index 2).
#     """
#     wks = defaultdict(list)
#     # wyckoff_string = wyckoff_string.strip()
#     pairs = wyckoff_string.split("-")
#     for pair in pairs:
#         el, w_pos = pair.strip("()").split(",")
#         wks[el].append( (int(w_pos), WYCKOFF_DATA[int(w_pos)]["multiplicity"]) )
#     res = wks.values()
#     wk_pos = [[ a[0] for a in b] for b in res]
#     wk_mult = [[ a[1] for a in b] for b in res]
#     elms = list(map(lambda el: comp_vocab[el], wks.keys()))
#     if return_multiplicity:
#         return elms, wk_pos, wk_mult
#     return elms, wk_pos


# def _has_scalar_value(value) -> bool:
#     if value is None:
#         return False
#     if isinstance(value, (pd.Series, pd.DataFrame, np.ndarray, list, tuple, dict)):
#         return False
#     try:
#         if pd.isna(value):
#             return False
#     except (TypeError, ValueError):
#         return False
#     if isinstance(value, str):
#         return bool(value.strip())
#     return True


# def _has_sequence_value(value) -> bool:
#     if value is None:
#         return False
#     if isinstance(value, (pd.Series, pd.DataFrame)):
#         return False
#     try:
#         if pd.isna(value):
#             return False
#     except (TypeError, ValueError):
#         pass
#     try:
#         return len(value) > 0
#     except TypeError:
#         return False


# def geometry_tokenizer(
#     k: int,
#     vocab: Dict[str, int],
#     cif_path: str = "",
#     cif_string: str = "",
#     formulae: str = "",
#     lattice_parameters: List[float] = [],
#     coords: Union[List[List[float]], np.ndarray] = [],
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Tokenizer for geometry-based  descriptors using Point Distance Distribution.
#     PDD features are computed using the AMD library, which computes a distribution of distances from each atom to its k nearest neighbors.
#     It gives an invariant representation of the local atomic environment/geometry, which can be used as a proxy for atomic positions in a way that is invariant to permutations and symmetries of the structure.

#     Parameters
#     ----------
#     # geometry_tokenizer example
#     pdd_features, token_indices = geometry_tokenizer(k=12, vocab=vocab, cif_path="path/to/cif/file.cif")
#     pdd_features: np.ndarray
#         A 2D array of PDD features representing the atomic positions in the structure.
#     token_indices: List[int]
#         A list of indices corresponding to the rows in the pdd_features, obtained by mapping the element symbols to their corresponding indices in the provided vocabulary.

#     entry: dict
#         A dictionary containing the crystal structure information.
#     k: int
#         The number of nearest neighbors to consider when computing PDD features.

#     Returns a tuple containing:
#     - A 2D array of PDD features representing the atomic positions in the structure.
#     - A list of indices corresponding to the rows in the pdd_features.
#     """

#     periodic_set = None
#     p_structure = None
#     if _has_scalar_value(cif_path):
#         periodic_set = amd.CifReader(cif_path).read()
#         p_structure = Structure.from_file(cif_path)
#     elif _has_scalar_value(cif_string):
#         p_structure = Structure.from_str(cif_string, fmt="cif")
#         periodic_set = amd.periodicset_from_pymatgen_structure(p_structure)
#     elif (
#         _has_scalar_value(formulae)
#         and _has_sequence_value(lattice_parameters)
#         and _has_sequence_value(coords)
#     ):
#         # create a pymatgen structure from the composition and lattice parameters
#         comp = Composition(formulae)
#         counts = [int(comp[el]) for el in comp.elements]
#         species = list(
#             chain.from_iterable(
#                 [[el.symbol] * c for el, c in zip(comp.elements, counts)]
#             )
#         )
#         lattice = Lattice.from_parameters(*lattice_parameters)
#         p_structure = Structure(
#             lattice, species, coords if isinstance(coords, List) else coords.tolist()
#         )
#         periodic_set = amd.periodicset_from_pymatgen_structure(p_structure)
#     else:
#         raise ValueError(
#             "Invalid input parameters, must contains either cif_path or formulae, lattice_parameters and coords."
#         )
#     pdd, el_indices = amd.PDD(periodic_set, k, collapse=False, return_row_data=True)  # type: ignore
#     el_indices = [el[0] if el is not None else None for el in el_indices]
#     known_el_indices = [el for el in el_indices if el is not None]
#     if known_el_indices:
#         max_el_index = max(known_el_indices)
#         assert max_el_index <= len(p_structure.sites) - 1, (
#             f"Element index {max_el_index} exceeds number of sites "
#             f"{len(p_structure.sites)}"
#         )
#     missing_token = vocab.get("[MISSING]", vocab.get("[PAD]", 0))
#     el_species: list = [
#         p_structure.sites[i].species_string if i is not None else None
#         for i in el_indices
#     ]
#     if None in el_species:
#         LOG.debug('Will fill in missing/pad token {missing_token} in the stead of missing')
#     token_indices = np.array(
#         [vocab.get(el, missing_token) for el in el_species], dtype=np.int64
#     )

#     # Aggregate rows by element token. The first PDD column is the row weight and
#     # is summed per element; the remaining k columns are averaged per element.
#     unique_tokens, inverse_indices = np.unique(token_indices, return_inverse=True)
#     pdd_per_elements = np.zeros((len(unique_tokens), pdd.shape[1]), dtype=np.float32)
#     np.add.at(pdd_per_elements[:, 0], inverse_indices, pdd[:, 0])
#     np.add.at(pdd_per_elements[:, 1:], inverse_indices, pdd[:, 1:])
#     counts = np.bincount(inverse_indices).astype(np.float32)[:, None]
#     pdd_per_elements[:, 1:] = pdd_per_elements[:, 1:] / counts

#     return pdd_per_elements, unique_tokens.tolist()
