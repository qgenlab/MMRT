import pandas as pd
import numpy as np
import pickle
import torch
import esm
import os
from tqdm import tqdm
import argparse


def read_sequence_file(sequence_path):
    with open(sequence_path) as f:
        seq_dict = dict()
        seq_name = ''
        for line in f:
            if line.startswith('>'):
                seq_name = line[1:].split()[0]
                seq_dict[seq_name] = ''
                continue
            elif line.strip() != '':
                seq_dict[seq_name] += line.strip()
                
    return seq_dict


def load_esm(device):
    # initialize ESM
    print("Loading ESM model...")
    model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    model = model.to(device)
    print("ESM loading complete!")
    
    return model, alphabet


def get_esm_vectors(sequence, mutation_df, model, alphabet, has_score = True, window = 0):
    ERROR_MSG = """
    Error:
    Amino acid at position {} does not match sequence.
    Mutation file: {}
    Sequence: {}
    """

    device = next(model.parameters()).device
    batch_converter_alphabet = alphabet.get_batch_converter()

    # generate WT embedding vectors
    wt_data = [(0, sequence)]
    _, _, batch_tokens = batch_converter_alphabet(wt_data)
    
    wt_results = model(
        batch_tokens.to(device),
        repr_layers=[33]
    )['representations'][33].detach().cpu().numpy().squeeze()

    vector_outputs = []
    valid_mutations = []
    scores = []

    for row in tqdm(mutation_df.itertuples(index=False), total=len(mutation_df)):
        if has_score:
            mutation, score, mut_count = row
        else:
            mutation, score, mut_count = row[0], np.nan, row[-1]
        
        try:
            mut_list = [(m[0], int(m[1:-1]) - 1, m[-1]) for m in mutation.split(':')]
            valid_mutations.append(mutation)
            
            scores.append(score)
                
        except ValueError:
            continue

        mt_sequence = sequence
        for wt, pos, mt in mut_list:
            if wt != mt_sequence[pos]:
                raise Exception(ERROR_MSG.format(pos + 1, wt, mt_sequence[pos]))

            mt_sequence = mt_sequence[:pos] + mt + mt_sequence[pos+1:]

        # generate current MT embedding vectors
        mt_data = [(0, mt_sequence)]
        _, _, batch_tokens = batch_converter_alphabet(mt_data)
        mt_results = model(
            batch_tokens.to(device),
            repr_layers=[33]
        )['representations'][33].detach().cpu().numpy().squeeze()

        # indices for mutated positions (and adjacent, if window)
        indices = np.array([range(m[1] - window, m[1] + window + 1) for m in mut_list])

        for i in indices:
            vector_outputs.append(np.row_stack([wt_results[i + 1], mt_results[i + 1]]).squeeze())
            
    return valid_mutations, vector_outputs, scores


def parse_cl():
    parser = argparse.ArgumentParser(
        description="Generate ESM vectors from protein sequences and mutation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    );

    parser.add_argument(
        "--mutations",
        type = str,
        help = "The file containing the mutations, the position of the mutations and the output values."
    )
    
    parser.add_argument(
        "--columns",
        type = str,
        nargs = '*',
        help = "Column headers for sequence mutations and, if applicable, activity scores (separated by a space)",
        default = ['mutant', 'fitness']
    )
    
    parser.add_argument(
        "--sequences",
        type = str,
        help = "The file containing the protein sequences."
    )
    
    parser.add_argument(
        "--name",
        type = str,
        help = "The name of the protein sequence in the file.",
        default = None
    )
    
    parser.add_argument(
        "--output",
        type = str,
        help = "Path to save output vectors.",
        default = './'
    )
    
    parser.add_argument(
        "--device",
        type = str,
        default = None,
        help = "The device to use for the ESM model."
    )
    
    parser.add_argument(
        "--window",
        type = int,
        default = 0,
        help = "The size of window (default: 0)."
    )
    
    return parser.parse_args()


def main():
    """
    Usage:
    python generate_vectors.py \
    --mutations /home/bryceForrest/multi_mut/datasets/unsplit_data/parEparD_Laub2015_all.csv \
    --columns mutant fitness \
    --sequences /home/bryceForrest/multi_mut/datasets/sequence_v2.fa \
    --name 'tr|F7YBW8|F7YBW8_MESOW' \
    --output /mnt/labshare/bryceForrest/esm_vectors/no_score/ \
    --device 4 \
    --window 1
    """  
    args = parse_cl()

    input_path = args.mutations
    cols = args.columns
    sequence_path = args.sequences
    sequence_name = args.name
    output_file_path = args.output
    output_file_name = None
    gpu = args.device
    window = args.window
    has_score = True
    
    if len(cols) > 2:
        raise Exception("Number of columns (-c) should be at most 2.")
    elif len(cols) == 1:
        has_score = False
    
    df = pd.read_csv(input_path)
    df['mut_count'] = df[cols[0]].str.count(':')
    df = df[cols + ['mut_count']].dropna().sort_values('mut_count')
    num_muts = df.mut_count.unique()

    
    seq_dict = read_sequence_file(sequence_path)
    sequence = seq_dict[sequence_name]
    
    device = torch.device('cuda:' + str(gpu) if gpu is not None else 'cpu')
    model, alphabet = load_esm(device)
    
    if output_file_name is None:
        output_file_name = input_path.split('/')[-1].replace('.csv', '')
        
    if window:
        output_file_name += f'_win_{window}'
        
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    
    for n in num_muts:
        output = get_esm_vectors(sequence,
                                 df[df.mut_count == n],
                                 model,
                                 alphabet,
                                 has_score=has_score,
                                 window=window)
        
        outfile = f'{output_file_path}{output_file_name}_{n+1}.p'
        
        pickle.dump(
            (output[0], np.asarray(output[1]).reshape((-1, ((2 * window + 1) * (n + 1) * 2), 1280)), output[2]),
            open(outfile, 'wb')
        )

if __name__=='__main__': main()