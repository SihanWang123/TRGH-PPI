## Dependencies
TRGH-PPI runs on Python 3.10. To install all dependencies, directly run:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install rdkit
pip install scikit-learn
pip install tensorboardX
```
# Datasets
The three datasets (SHS27k, SHS148k and STRING) are stored in the dataset folder except for the last two. You can download them from https://drive.google.com/drive/folders/1PwPIakU89eFwYBF4fyKJmx916z3DgLoC?usp=drive_link
* `protein.actions.SHS27k.STRING.pro2.txt`             PPI network of SHS27k
* `protein.SHS27k.sequences.dictionary.pro3.tsv`      Protein sequences of SHS27k
* `protein.actions.SHS148k.STRING.txt`             PPI network of SHS148k
* `protein.SHS148k.sequences.dictionary.tsv`         Protein sequences of SHS148k
* `protein.STRING_all_connected.sequences.dictionary.tsv`             Protein sequences of STRING
* `edge_list_12`             Adjacency matrix for all proteins in SHS27k
* `x_list`             Feature matrix for all proteins in SHS27k
* `9606.protein.action.v11.0.txt`         PPI network of STRING

# PPI Prediction

Example: predicting unknown PPIs in SHS27k datasets with native structures:

## Using Processed Data for SHS27k Dataset

Download  `protein.actions.SHS27k.STRING.pro2.txt`, `protein.SHS27k.sequences.dictionary.pro3.tsv`, `edge_list_12`, `x_list` and `vec5_CTC.txt` to `./TRGH-PPI/protein_info/`.

## Data Processing for New Datasets (if applicable)
Prepare all related PDB files. Native protein structures can be downloaded in batches from the [RCSB PDB](https://www.rcsb.org/downloads), and predicted protein structures with errors can be downloaded from the [AlphaFold database](https://alphafold.ebi.ac.uk/). Put all of the PDB files in `./protein_info/`.

Generate adjacency matrix with native PDB files:
```
python ./protein_info/generate_adj.py --distance 12
```
Generate feature matrix:
```
python ./protein_info/generate_feat.py
```

## Training
To predict PPIs, use 'model_train.py' script to train TRGH-PPI with the following options:
* `ppi_path`             str, PPI network information
* `pseq_path`             str, Protein sequences
* `p_feat_matrix`       str, The feature matrix of all protein graphs
* `p_adj_matrix`       str, The adjacency matrix of all protein graphs
* `split`       str, Dataset split mode
* `save_path`             str, Path for saving models, configs and results
* 'epoch_num'     int, Training epochs
```
python model_train.py --ppi_path ./protein_info/protein.actions.SHS27k.STRING.pro2.txt --pseq ./protein_info/protein.SHS27k.sequences.dictionary.pro3.tsv --split random --p_feat_matrix ./protein_info/x_list.pt --p_adj_matrix ./protein_info/edge_list_12.npy --save_path ./result_save --epoch_num 500
```
## Testing
Run 'model_test.py' script to test TRGH-PPI with the following options:
* `ppi_path`             str, PPI network information
* `pseq_path`             str, Protein sequences
* `p_feat_matrix`       str, The feature matrix of all protein graphs
* `p_adj_matrix`       str, The adjacency matrix of all protein graphs
* `model_path`       str, Path for trained model
* `index_path`             str, Path for index being tested
```
python model_test.py --ppi_path ./protein.actions.SHS27k.STRING.pro2.txt --pseq ./protein.SHS27k.sequences.dictionary.pro3.tsv --p_feat_matrix ./x_list.pt --p_adj_matrix ./edge_list_12.npy --model_path ./result_save/gnn_training_seed_1/gnn_model_valid_best.ckpt --index_path ./train_val_split_data/train_val_split_1.json
```
## Output
The output after running 'model_test.py' includes:
* `valid_label_list` Real PPI labels for the test index
* `test_pre_result_list` Predicted PPI results for the test index
* `best_f1` Overall performance in terms of best-F1 score
* `aupr` Performance in terms of AUPR score for all seven PPI types (reaction, binding, ptmod, activation, inhibition, catalysis and expression)
