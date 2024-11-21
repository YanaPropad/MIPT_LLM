from random import sample

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from IPython.display import clear_output
import matplotlib.pyplot as plt

from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


def generate_chunk(text_encoded, token_to_idx, batch_size, seq_length):

    start_column = np.zeros((batch_size, 1), dtype=int) + token_to_idx['<sos>']

    start_index = np.random.randint(0, len(text_encoded) - batch_size*seq_length - 1)
    data = np.array(text_encoded[start_index:start_index + batch_size*seq_length]).reshape((batch_size, -1))
    yield np.hstack((start_column, data))


def generate_chunk_with_eos(text_encoded, token_to_idx, batch_size, seq_length):
    
    start_column = np.zeros((batch_size, 1), dtype=int) + token_to_idx['<sos>']
    end_column = np.zeros((batch_size, 1), dtype=int) + token_to_idx['<eos>']

    start_index = np.random.randint(0, len(text_encoded) - batch_size*seq_length - 1)
    end_index = start_index + batch_size * seq_length
    data = np.array(text_encoded[start_index:end_index]).reshape((batch_size, -1))

    yield np.hstack((start_column, data, end_column))


def train_basicLSTM_model(
    model,
    eos,
    text_encoded,
    token_to_idx,
    sequence_length,
    batch_size,
    num_epochs,
    num_batches,
    num_tokens,
    loss_function,
    optimizer,
    lr_scheduler,
    device):

    loss_history = []
    loss_over_epoch = []

    if eos:
        generate_chunk_function = generate_chunk_with_eos
    else:
        generate_chunk_function = generate_chunk

    for i in range(num_epochs*num_batches):

        text_batch = next(generate_chunk_function(text_encoded, token_to_idx, batch_size, sequence_length))
        text_batch = torch.from_numpy(text_batch).to(device)
        text_batch = F.one_hot(text_batch.to(torch.long), num_classes=num_tokens).to(torch.float32)

        logits_pred = model(text_batch)

        logits_next = logits_pred[:, :-1]
        actual_next_tokens = text_batch[:, 1:].cpu().numpy()
        actual_next_tokens = torch.from_numpy((actual_next_tokens * np.arange(num_tokens)).sum(axis=-1)).to(dtype=torch.int64, device=device)
                                        
        loss = loss_function(logits_next.reshape(-1, num_tokens), actual_next_tokens.reshape(-1))
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 

        if (i+1)%10==0:
            loss_over_epoch.append(i)
            loss_history.append(loss.cpu().data.numpy())
            lr_scheduler.step(loss_history[-1])
            clear_output(True)
            plt.plot(loss_over_epoch, loss_history, color='blue')
            plt.xlabel('iteration')
            plt.ylabel('loss')
            plt.show()


def generate_sample_with_eos(lstm_model, token_to_idx, idx_to_token, seed_phrase='Mg is Magnesium structured', max_length=500, temperature=1.0, mode='lstm'):

    model = lstm_model.to(device='cpu')
    device = 'cpu'
    
    num_tokens = len(token_to_idx)
    seed_phrase_idx = np.array([token_to_idx[s] for s in seed_phrase])
    x_sequence = torch.from_numpy(seed_phrase_idx).to(dtype=torch.float32, device=device)
    x_sequence = F.one_hot(x_sequence.to(torch.long), num_classes=num_tokens).to(torch.float32)

    h_t, h_next = model.lstm(x_sequence)

    for _ in range(max_length - len(seed_phrase_idx)):
        if mode=='rnn':
            h_t, h_next = model.lstm(x_sequence, h_next)
            logits = model.hid_to_logits(h_next)
        elif mode=='lstm':
            h_t, h_next = model.lstm(x_sequence, h_next)
            logits = model.hid_to_logits(h_next[0])
        elif mode=='lstm+attention':
            h_t, (h_next, c) = model.lstm(x_sequence)
            attention_output, attention_weights = model.attention(h_next, h_t, h_t)
            logits = model.hid_to_logits(attention_output)

        p_next = F.softmax(logits / temperature, dim=-1).data.numpy()[0]

        next_token_idx = np.random.choice(num_tokens, p=p_next)
        if next_token_idx == token_to_idx['<eos>']:
            break

        next_ix = torch.from_numpy(np.array([next_token_idx]))
        next_ix = (F.one_hot(next_ix.to(torch.long), num_classes=num_tokens)).to(dtype=torch.float32)
        #next_ix = torch.from_numpy(one_hot_encode(next_ix, num_tokens=num_tokens)).to(dtype=torch.float32)

        x_sequence = torch.cat([x_sequence, next_ix], dim=0)

    generated = (x_sequence*torch.arange(x_sequence.shape[-1])).sum(dim=1).numpy()
    generated = [idx_to_token[i] for i in generated]

    return ''.join(generated)


def generate_description_and_compare(model, token_to_idx, idx_to_token, query, filtered_df, max_length=500, temperature=0.8, model_type='basic_LSTM_with_eos'):

    if model_type == 'basic_LSTM_with_eos':
        description = generate_sample_with_eos(model, token_to_idx, idx_to_token, query, max_length=max_length, temperature=temperature)
    elif model_type == 'basic_LSTM_with_attention':
        description = generate_sample_with_eos(model, token_to_idx, idx_to_token, query, max_length=max_length, temperature=temperature, mode='lstm+attention')

    chemical_formula = query.split()[0]

    mp_info = filtered_df.loc[filtered_df.Formula_pretty == chemical_formula]

    mp_info = filtered_df.loc[
        (filtered_df['Formula'] == chemical_formula) | 
        (filtered_df['Formula_pretty'] == chemical_formula)
    ]

    if mp_info.empty:

        print(f'В обучающем корпусе нет кристалла с формулой {chemical_formula}\n\n'
        f'Сгенерированное описание длиной {len(description)}:\n{description}')

    else:
        robocrys_description = mp_info.Robocrys_description.values[0]
        mp_id = mp_info.ID.values[0]
        http = f'https://next-gen.materialsproject.org/materials/{mp_id}'

        print(f'MP_ID: {mp_id}, {http}\n'
        f'Оригинальное описание: {robocrys_description}\n\n'
        f'Сгенерированное описание (количество символов: {len(description)}):\n{description}')
