import argparse

def merge_gro(protein_gro, ligand_gro, complex_gro):
    with open(protein_gro, 'r') as f:
        protein = f.readlines()
    with open(ligand_gro, 'r') as f:
        ligand = f.readlines()
    natoms_complex = int(protein[1].strip()) + int(ligand[1].strip())
    with open(complex_gro, 'w') as f:
        f.write(protein[0])
        f.write(str(natoms_complex) + "\n")
        for ii in range(2, len(protein) - 1):
            f.write(protein[ii])
        for ii in range(2, len(ligand) - 1):
            f.write(ligand[ii])
        f.write(protein[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge protein.gro and ligand.gro to complex.gro")
    parser.add_argument("-p", "--protein", default="protein.gro", help="The gro file of the protein")
    parser.add_argument("-l", "--ligand", default="ligand.gro", help="The gro file of the ligand")
    parser.add_argument("-c", "--complex", default="complex.gro", help="The gro file of the complex (output)")
    args = parser.parse_args()

    protein_gro = args.protein
    ligand_gro = args.ligand
    complex_gro = args.complex
    merge_gro(protein_gro, ligand_gro, complex_gro)