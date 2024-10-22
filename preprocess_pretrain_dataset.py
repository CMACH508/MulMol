import sys
sys.path.append("..")

import numpy as np
from multiprocessing import Pool
from rdkit import Chem
from scipy import sparse as sp
import argparse 

from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import AllChem
from rdkit import RDConfig
import os
import numpy as np
fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)

from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Generate
sigFactory = SigFactory(featFactory, minPointCount=2, maxPointCount=3)
sigFactory.skipFeats=['PosIonizable', 'NegIonizable', 'Aromatic', 'Acceptor', 'Donor', 'ZnBinder']
sigFactory.SetBins([(0,10),(10,20)])
sigFactory.Init()

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()
    return args

def preprocess_dataset(args):
    with open(f"{args.data_path}/smiles.smi", 'r') as f:
            lines = f.readlines()
            smiless = [line.strip('\n') for line in lines]

    print('extracting fingerprints')
    FP_list = []
    for smiles in smiless:
        mol = Chem.MolFromSmiles(smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = np.array(FP_list)
    FP_sp_mat = sp.csc_matrix(FP_arr)
    print('saving fingerprints')
    sp.save_npz(f"{args.data_path}/rdkfp1-7_512.npz", FP_sp_mat)

    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(args.n_jobs).imap(generator.process, smiless)
    arr = np.array(list(features_map))
    np.savez_compressed(f"{args.data_path}/molecular_descriptors.npz",md=arr[:,1:])

    print('extracting chemical features')
    cf_list = []
    idx = 0
    for smiles in smiless:
        idx += 1
        print(idx)
        mol = Chem.MolFromSmiles(smiles)
        cf_list.append(list(list(Generate.Gen2DFingerprint(mol,sigFactory))))
    cf_arr = np.array(cf_list)
    cf_sp_mat = sp.csc_matrix(cf_arr)
    print('saving chemical features')
    sp.save_npz(f"{args.data_path}/chemical_features_1.npz", cf_sp_mat)


if __name__ == '__main__':
    args = parse_args()
    preprocess_dataset(args)