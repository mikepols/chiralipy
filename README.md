# ChiraliPy

A Python tool to analyze the structural chirality of 2D halide perovskite structures, both static and dynamic. Various components of the crystal lattice can be analyzed, including: octahedral distortions, chiral distortions, hydrogen bond asymmetry, as well as cations and headgroup orientations. ChiraliPy was developed to study the temperature-dependent chirality in 2D halide perovskites [1], the manuscript and supporting information (SI) contain an extensive overview of its inner workings.

### Structural descriptors

The following categories of structural descriptors can be analyzed with ChiraliPy:
- **Intraoctahedral distortions**: angle- and distance-based distortions of metal halide octahedra [2].
- **Interoctahedral distortions**: distortions between the octahedral cages of the inorganic framework [3].
- **Planar distortions**: distortions of the inorganic planes from their average position [4].
- **Hydrogen bond asymmetry**: asymmetry in hydrogen bonds lengths above and below an inorganic layer.
- **Structural chirality**: chirality of various structural features, using bond-based equivalent of *vector chirality* which has been applied to various spin systems [5]:
    1. Organic cations: chirality of the packing of the organic cations.
    2. In-plane framework: chirality in the various in-plane directions of the inorganic framework.
    3. Out-of-plane framework: chirality of the inorganic framework in the direction perpendicular to the planes.

###  Orientation vectors

Orientation vectors of various bonds can be assessed as a function of time, providing insights into the temperature-dependent persistence of bond and molecular orientations. The following vectors, defined through atomic fingerprints, can be studied:
1.  **Headgroup orientation**: bond vectors defining the ammonium headgroup orientation: N-H bonds.
2.  **Cation orientation**: vectors spanning arbitrary interatomic vectors within the organic cations.

## Requirements

ChiraliPy makes use of the following Python packages: ase (3.22.1), numpy (1.20.0), scipy (1.7.1), and tqdm (4.62.3). The numbers behind the packages indicate the version with which ChiraliPy was developed. These packages can be installed through the `requirements.txt` file, using
```
pip install -r requirements.txt
```

Some of the descriptors make use of the atomic fingerprints that were introduced with the [FingerprintiPy](https://github.com/mikepols/fingerprintipy). It is therefore recommended to also install FingerprintiPy to make full use of the functionality of ChiraliPy.

## Installation

The tool can be automatically installed through
```
./install.sh
```
which adds `chirali.py` to the `$PATH`.

## Usage

To use ChiraliPy, it has to be supplied with a `json`-file specifying all input parameters as
```
chirali.py -s settings.json
```

For a static crystal structure the analysis should progress fairly quickly, whereas molecular dynamics trajectories, as a result of their many frames, may involve relatively expensive and repeated calculations.

## Examples

The [examples](examples) folder contains demonstrations of ChiraliPy for static structures ([examples/static](examples/static)) and dynamic trajectories ([examples/trajectory](examples/trajectory)). These examples can be run by navigating into the respective folder and running the script with the respective `json`-files. In the [examples/settings](examples/settings) folder, commented files are found to explain all of the options.

These examples will output the following files:
- `static`: `log`-file containing the values of the computed descriptors
- `trajectory`: `log`-file containing the averaged values, `vasp`-file with the averaged structure, and `pickle`-files with the various descriptors

## Data

ChiraliPy was used to analyze a variety of structures, experimental and DFT-optimized, from various earlier studies. These analyses can be found in the [data](data) folder, which contains the following folders:
1. `2017_inorg_chem_two_dimensional` [6]: PEA<sub>2</sub>PbI<sub>4</sub>.
2. `2020_j_am_chem_soc_highly_distorted` [7]: (*S*-MBA)<sub>2</sub>SnI<sub>4</sub>, (*R*-MBA)<sub>2</sub>SnI<sub>4</sub>, and (*rac*-MBA)<sub>2</sub>SnI<sub>4</sub>.
3. `2020_j_phys_chem_lett_bulk_chiral` [8]: (*rac*-MBA)<sub>2</sub>PbI<sub>4</sub>.
4. `2020_nat_commun_organic_inorganic` [9]: (*S*-MBA)<sub>2</sub>PbI<sub>4</sub>, (*R*-MBA)<sub>2</sub>PbI<sub>4</sub>, (*S*-1NEA)<sub>2</sub>PbBr<sub>4</sub>, (*R*-1NEA)<sub>2</sub>PbBr<sub>4</sub>, and (*rac*-1NEA)<sub>2</sub>PbBr<sub>4</sub>.
5. `2021_acs_nano_strongly_anharmonic` [10]: BA<sub>2</sub>PbI<sub>4</sub>.
6. `2021_nat_commun_structual_descriptor` [3]: (*R*-4-Cl-MBA)<sub>2</sub>PbBr<sub>4</sub>, (*S*-1-Me-HA)<sub>2</sub>PbI<sub>4</sub>, and (*S*-2-Me-BuA)<sub>2</sub>PbBr<sub>4</sub>.
7. `2023_nat_commun_unraveling_chirality` [11]: (*S*-2NEA)<sub>2</sub>PbBr<sub>4</sub>, (*R*-2NEA)<sub>2</sub>PbBr<sub>4</sub>, and (*rac*-2NEA)<sub>2</sub>PbBr<sub>4</sub>.
8. `fingerprints`: atomic fingerprints for the various cations as computed with FingerprintiPy.

## Notes

- To compute the descriptors along a trajectory of the perovskites, the descriptors have to be computed repeatedly for every frame. Such computations are computationally demanding and benefit from the use of workstations or HPC.
- Inorganic layers, i.e. metal cations and halide anions, should not be on the boundary of the unit cell. It is advised to shift the atoms in a cell to prevent the script from encountering such situations.
- The script can run into problems when species in the organic cations are also found in the inorganic layers. To be able to analyze these, adaptations have to be made to the script.

##  Use cases

ChiraliPy has been used in the following publications:
- Pols *et al.*, *J. Phys. Chem. Lett.*, 15, 8057-8064 (2024), DOI:  [`10.1021/acs.jpclett.4c01629`](https://doi.org/10.1021/acs.jpclett.4c01629); code development, analysis of a variety of static 2D perovskites, and analysis of dynamics of  and dynamic MBA<sub>2</sub>PbI<sub>4</sub>.
- Nurdillayeva *et al.*, *ACS Nano*, 15, 8057-8064 (2025), DOI:  [`10.1021/acsnano.5c00480`](https://doi.org/10.1021/acsnano.5c00480); investigation of the effect of water adsorption on the structural chirality of MBA-based chiral perovskites.
- Pols *et al.*, *arXiv:2508.00158*, 1-8 (2025), DOI:  [`arXiv:2508.00158`](https://arxiv.org/abs/2508.00158); analysis of the effect of metal cation substitutions on structural and dynamic chirality in chiral 2D perovskites.

## References & Citing

1. Pols *et al.*, *J. Phys. Chem. Lett.*, 15, 8057-8064 (2024), DOI:  [`10.1021/acs.jpclett.4c01629`](https://doi.org/10.1021/acs.jpclett.4c01629).
2. Robinson *et al.*, *Science*, 172, 567-570 (1971), DOI:  [`10.1126/science.172.3983.567`](https://doi.org/10.1126/science.172.3983.567).
3. Jana *et al.*, *Nat. Commun.*, 12, 4982 (2021), DOI:  [`10.1038/s41467-021-25149-7`](https://doi.org/10.1038/s41467-021-25149-7).
4. Apergi *et al.*, *J. Phys. Chem. Lett.*, 14, 11565-11572 (2023), DOI:  [`10.1021/acs.jpclett.3c02705`](https://doi.org/10.1021/acs.jpclett.3c02705).
5. Ding *et al.*, *Nat. Commun.*, 12, 5339 (2023), DOI:  [`10.1038/s41467-021-25657-6`](https://doi.org/10.1038/s41467-021-25657-6).
6. Du *et al.*, *Inorg. Chem.*, 56, 9291-9302 (2017), DOI:  [`10.1021/acs.inorgchem.7b01094`](https://doi.org/10.1021/acs.inorgchem.7b01094).
7. Lu *et al.*, *J. Am. Chem. Soc.*, 142, 13030-13040 (2020), DOI:  [`10.1021/jacs.0c03899`](https://doi.org/10.1021/jacs.0c03899).
8. Dang *et al.*, *J. Phys. Chem. Lett.*, 11, 1689-1696 (2020), DOI:  [`10.1021/acs.jpclett.9b03718`](https://doi.org/10.1021/acs.jpclett.9b03718).
9. Jana *et al.*, *Nat. Commun.*, 11, 4699 (2020), DOI:  [`10.1038/s41467-020-18485-7`](https://doi.org/10.1038/s41467-020-18485-7).
10. Menahem *et al.*, *ACS Nano*, 15, 10153-10162 (2021), DOI:  [`10.1021/acsnano.1c02022`](https://doi.org/10.1021/acsnano.1c02022).
11. Son *et al.*, *Nat. Commun.*, 14, 3124 (2023), DOI:  [`10.1038/s41467-023-38927-2`](https://doi.org/10.1038/s41467-023-38927-2).

Please consider citing the relevant works upon using ChiraliPy [1], the various structural descriptors [2-5], or the provided structures [3, 6-11]. 

## License

The code is available as open source under the terms of the [MIT License](LICENSE).
