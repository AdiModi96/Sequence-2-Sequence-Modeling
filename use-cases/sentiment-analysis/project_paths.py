import os
import inspect

current_file_path = os.path.realpath(inspect.getfile(inspect.currentframe()))
project_folder_path = os.path.dirname(current_file_path)

datasets_folder_path = os.path.join(project_folder_path, 'datasets')
aclImdb_dataset_folder_path = os.path.join(datasets_folder_path, 'aclImdb')
if not os.path.isdir(aclImdb_dataset_folder_path):
    os.makedirs(aclImdb_dataset_folder_path)

tokenizers_folder_path = os.path.join(project_folder_path, 'tokenizers')
word_piece_tokenizer_folder_path = os.path.join(tokenizers_folder_path, 'word_piece')
if not os.path.isdir(word_piece_tokenizer_folder_path):
    os.makedirs(word_piece_tokenizer_folder_path)

models_folder_path = os.path.join(project_folder_path, 'models')
rnn_models_folder_path = os.path.join(models_folder_path, 'rnns')
if not os.path.isdir(rnn_models_folder_path):
    os.makedirs(rnn_models_folder_path)

lstm_models_folder_path = os.path.join(models_folder_path, 'lstms')
if not os.path.isdir(lstm_models_folder_path):
    os.makedirs(lstm_models_folder_path)