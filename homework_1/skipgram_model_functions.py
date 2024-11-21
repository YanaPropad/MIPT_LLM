from collections import Counter
from itertools import chain
import numpy as np
from tqdm import tqdm 
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.metrics import accuracy_score
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import umap.umap_ as umap
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def get_word_count_dict(data):

    min_count = 1

    vocabulary_with_counter = Counter(chain.from_iterable(data))

    word_count_dict = dict()
    for word, counter in vocabulary_with_counter.items():
        if counter >= min_count:
            word_count_dict[word] = counter

    return word_count_dict



def get_context_pairs(data, word_to_index, window_radius):

    context_pairs = []

    for text in data:
        for i, central_word in enumerate(text):
            context_indices = range(
                max(0, i - window_radius), min(i + window_radius, len(text))
            )
            for j in context_indices:
                if j == i:
                    continue
                context_word = text[j]
                #if central_word in vocabulary and context_word in vocabulary:
                context_pairs.append(
                    (word_to_index[central_word], word_to_index[context_word])
                    )
    
    return context_pairs


def keep_prob_to_array(keep_prob_dict, index_to_word, word_to_index):

    keep_prob_array = np.array(
        [keep_prob_dict[index_to_word[idx]] for idx in range(len(word_to_index))]
        )

    return keep_prob_array


def negative_sampling_prob_to_array(negative_sampling_prob_dict, index_to_word, word_to_index):

    negative_sampling_prob_array = np.array(
        [
            negative_sampling_prob_dict[index_to_word[idx]]
            for idx in range(len(word_to_index))
            ]
            )
    
    return negative_sampling_prob_array


def generate_batch_with_neg_samples_optimized(
    context_pairs,
    batch_size,
    keep_prob_array,
    word_to_index,
    num_negatives,
    negative_sampling_prob_array,
):  

    if isinstance(context_pairs, list):
        context_pairs = np.array(context_pairs)

    centers_negs = context_pairs[np.random.randint(len(context_pairs), size=batch_size)]
    keep_mask = np.random.random(size=batch_size) < keep_prob_array[centers_negs[:, 0]]
    valid = centers_negs[keep_mask]

    while len(valid) < batch_size:
        add_centers_negs = context_pairs[np.random.randint(len(context_pairs), size=batch_size)]
        keep_mask = np.random.random(size=batch_size) < keep_prob_array[add_centers_negs[:, 0]]
        valid = np.concatenate((valid, add_centers_negs[keep_mask]))

    batch = valid[:batch_size]    

    neg_samples = np.random.choice(
        range(len(negative_sampling_prob_array)),
        size=(batch_size, num_negatives),
        p=negative_sampling_prob_array
    )
    
    return batch, neg_samples


def train_skipgram_with_neg_sampling(
    model,
    context_pairs,
    keep_prob_array,
    word_to_index,
    batch_size,
    num_negatives,
    negative_sampling_prob_array,
    steps,
    optimizer,
    lr_scheduler,
    device,
):
    pos_labels = torch.ones(batch_size).to(device)
    neg_labels = torch.zeros(batch_size, num_negatives).to(device)
    loss_history = []
    criterion = nn.BCEWithLogitsLoss()

    ep = []
    history = []

    #for step in tqdma(range(steps)):
    for step in range(steps):
        batch, neg_samples = generate_batch_with_neg_samples_optimized(
            context_pairs,
            batch_size,
            keep_prob_array,
            word_to_index,
            num_negatives,
            negative_sampling_prob_array,
        )
        center_words = torch.tensor([pair[0] for pair in batch], dtype=torch.long).to(
            device
        )
        pos_context_words = torch.tensor(
            [pair[1] for pair in batch], dtype=torch.long
        ).to(device)
        neg_context_words = torch.tensor(neg_samples, dtype=torch.long).to(device)

        optimizer.zero_grad()
        pos_scores, neg_scores = model(
            center_words, pos_context_words, neg_context_words
        )

        loss_pos = criterion(pos_scores, pos_labels)
        loss_neg = criterion(neg_scores, neg_labels)

        loss = loss_pos + loss_neg
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        lr_scheduler.step(loss_history[-1])


        if (step)%10==0:
            ep.append(step)
            history.append(loss.cpu().data.numpy())
            clear_output(True)
            plt.plot(ep, history, color='blue')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.show()


def get_word_vector(word, embedding_matrix, word_to_index):
    return embedding_matrix[word_to_index[word]]


def find_nearest(model, word, word_to_index, index_to_word, k):

    _model_parameters = model.parameters()
    embedding_matrix_center = next(
        _model_parameters
    ).detach()  # Assuming that first matrix was for central word
    embedding_matrix_context = next(
        _model_parameters
    ).detach()  # Assuming that second matrix was for context word

    word_vector = get_word_vector(word, embedding_matrix_context, word_to_index)[None, :]
    dists = F.cosine_similarity(embedding_matrix_context, word_vector)
    index_sorted = torch.argsort(dists.cpu())
    #index_sorted = torch.argsort(dists)
    top_k = index_sorted[-k:]
    return [(index_to_word[x], dists[x].item()) for x in top_k.numpy()]


def evaluate_model(model, context_pairs, device, batch_size=64):
    """
    Evaluate the model based on the given context pairs.

    :param model: The trained model.
    :param context_pairs: A list of pairs (center_word, context_word).
    :param device: The device (CPU or GPU) on which the model is running.
    """

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for i in tqdm(range(0, len(context_pairs), batch_size), desc="Evaluating"):
            batch_pairs = context_pairs[i:i + batch_size]
            center_words = torch.tensor([pair[0] for pair in batch_pairs], dtype=torch.long).to(device)
            context_words = torch.tensor([pair[1] for pair in batch_pairs], dtype=torch.long).to(device)

            pos_scores, _ = model(center_words, context_words, torch.empty(0, dtype=torch.long).to(device))

            predicted = (pos_scores > 0).float()

            true_labels.extend([1] * len(batch_pairs))
            predicted_labels.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    
    print(f'Accuracy: {accuracy:.4f}')

    return accuracy


def get_word_embeddings(model, word_to_index, words):

    _model_parameters = model.parameters()
    embedding_matrix_center = next(
        _model_parameters
    ).detach()  # Assuming that first matrix was for central word
    embedding_matrix_context = next(
        _model_parameters
    ).detach()  # Assuming that second matrix was for context word

    word_embeddings = torch.cat(
        [embedding_matrix_context[word_to_index[x]][None, :] for x in list(words)], dim=0
    ).numpy()

    return word_embeddings




def get_3D_plot_by_structure_types(list_of_formules, embeddings_3d, accuracy, output_filename, properties_filaname):

    df_properties = pd.read_csv(properties_filename)
    df_embeddings_3d = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
    df_result = pd.concat([df_properties, df_embeddings_3d], axis=1)
    df_result['Formula'] = list_of_formules
    n_tokens = properties_filename.split('_')[-2]

    fig = go.Figure()

    for structure_type in df_result['Structure_types'].unique():
        subset = df_result[df_result['Structure_types'] == structure_type]
        
        fig.add_trace(go.Scatter3d(
            x=subset['x'],
            y=subset['y'],
            z=subset['z'],
            mode='markers',
            name=structure_type,
            marker=dict(size=5),
            text=subset['Formula'],
            customdata=subset[['ID', 'SG', 'SG_symbol', 'Crystal_system', 'Structure_types']],
            hovertemplate=(
                "<b>Formula: %{text}</b><br>" +
                "ID: %{customdata[0]}<br>" +
                "SG: %{customdata[1]}<br>" +
                "SG Symbol: %{customdata[2]}<br>" +
                "Crystal System: %{customdata[3]}<br>" +
                "Structure_types: %{customdata[4]}"
                "<extra></extra>"
                    )
    ))

        fig.update_layout(
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        title=f"Skipgram word embeddings, corpus: {n_tokens} tokens, accuracy: {accuracy.round(4)}",
        
        legend=dict(
            orientation="v", 
            title="Structure Types",
            itemclick="toggleothers",
            itemsizing="constant", 
            traceorder="normal", 
            font=dict(size=10), 
            bgcolor="rgba(255, 255, 255, 0.8)", 
            bordercolor="Black", 
            borderwidth=1     
        )
    )

    fig.write_html(output_filename)


def get_3D_plot_by_crystal_system(list_of_formules, embeddings_3d, accuracy, output_filename, properties_filename):

    df_properties = pd.read_csv(properties_filename)
    df_embeddings_3d = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
    df_result = pd.concat([df_properties, df_embeddings_3d], axis=1)
    df_result['Formula'] = list_of_formules
    n_tokens = properties_filename.split('_')[-2]

    fig = go.Figure()

    for structure_type in df_result['Crystal_system'].unique():
        subset = df_result[df_result['Crystal_system'] == structure_type]
        
        fig.add_trace(go.Scatter3d(
            x=subset['x'],
            y=subset['y'],
            z=subset['z'],
            mode='markers',
            name=structure_type,
            marker=dict(size=5),
            text=subset['Formula'],
            customdata=subset[['ID', 'SG', 'SG_symbol', 'Crystal_system', 'Structure_types']],
            hovertemplate=(
                "<b>Formula: %{text}</b><br>" +
                "ID: %{customdata[0]}<br>" +
                "SG: %{customdata[1]}<br>" +
                "SG Symbol: %{customdata[2]}<br>" +
                "Crystal System: %{customdata[3]}<br>" +
                "Structure_types: %{customdata[4]}"
                "<extra></extra>"
                    )
    ))

        fig.update_layout(
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        title=f"Skipgram word embeddings, corpus: {n_tokens} tokens, accuracy: {accuracy.round(4)}",
        
        legend=dict(
            orientation="v", 
            title="Structure Types",
#            itemclick="toggleothers",
            itemsizing="constant", 
            traceorder="normal", 
            font=dict(size=10), 
            bgcolor="rgba(255, 255, 255, 0.8)", 
            bordercolor="Black", 
            borderwidth=1     
        )
    )

    fig.write_html(output_filename)
