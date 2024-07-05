# Multi-Mut Recursive Tree
### Instructions
Install and activate `conda` environment contained in `environment.yml`:
```
conda env create -f environment.yml
conda activate mmrt_env
```

You now need to generate ESM embedding vectors for your protein mutation fitness `csv` files.

Your protein mutation fitness file should be of the form (excerpt taken from `parEparD_Laub2015_all.csv`):

|mutant|fitness|
|--------|-------|
|L59A:W60I|0.030531 |
|L59A:W60I:K64R | 0.007202 |
|L59A:W60I:D61Q:K64L|-0.003394 |

with multiple mutations are separated by `:` character. You also need a sequence fasta file, that maps a name to its sequence.

The script `generate_vectors.py` will take these files as arguments, and output the relevant ESM vectors.

```
python generate_vectors.py \
--mutations parEparD_Laub2015_all.csv \
--columns mutant fitness \
--sequences sequences.fa \
--name 'tr|F7YBW8|F7YBW8_MESOW' \
--output esm_vectors/ \
--device 0 \
--window 1
```

Now that ESM vectors are pickled and stored in `esm_vectors/`, you can train the model and make predictions:

```
python example.py \
--train_data esm_vectors/parEparD_Laub2015_all_win_1_1.p \
esm_vectors/parEparD_Laub2015_all_win_1_2.p \
--test_data esm_vectors/parEparD_Laub2015_all_win_1_3.p \
esm_vectors/parEparD_Laub2015_all_win_1_4.p \
--batch_size 32 \
--save_log \
--save_model \
--save_path './' \
--model_name 'parEparD_Laub2015_all_win_1_tr12' \
--device 0 \
--epochs 100 \
--learning_rate 1e-5 \
--random_seed 0 \
--cadence 20 \
--save_prediction    
```

Predictions are stored in `output/parEparD_Laub2015_all_win_1_tr12.csv`.