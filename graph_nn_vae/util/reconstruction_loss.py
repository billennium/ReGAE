from typing import Callable

import torch
from torch import Tensor
import numpy as np
import pandas as pd

g = 0

def get_reconstruction_loss(
    adjacency_matrices_batch: Tensor,
    reconstructed_graph_diagonals: Tensor,
    loss_function: Callable,
) -> Tensor:
    input_concatenated_diagonals = []
    for adjacency_matrix in adjacency_matrices_batch:
        diagonals = [
            torch.diagonal(adjacency_matrix, offset=-i).transpose(1, 0)
            for i in reversed(range(adjacency_matrix.shape[0]))
        ]
        for i in reversed(range(len(diagonals))):
            if torch.count_nonzero(diagonals[i]) == 0:
                diagonals[i] = diagonals[i].fill_(-1.0)
            else:
                break
        concatenated_diagonals = torch.cat(diagonals, dim=0)
        input_concatenated_diagonals.append(concatenated_diagonals)
    input_batch_reshaped = torch.stack(input_concatenated_diagonals)

    reconstructed_diagonals_length = reconstructed_graph_diagonals.shape[1]
    input_pad_length = reconstructed_diagonals_length - input_batch_reshaped.shape[1]
    input_batch_reshaped = torch.nn.functional.pad(
        input_batch_reshaped, (0, 0, 0, input_pad_length), value=-1.0
    )

    global g
    if g%100==0:


        tmp = np.stack(
                    [
                        reconstructed_graph_diagonals.cpu().detach().numpy().round(2),
                        input_batch_reshaped.cpu().detach().numpy(),
                    ], axis = 1
                )[:, :, :, 0]
        res = []
        if len(tmp) > 1:
            for i, row in enumerate(tmp):
                res.append(pd.DataFrame(row).transpose())
                # print(pd.DataFrame(row).transpose().shape)
        #         print('Example', i)
        #         df_tmp = pd.DataFrame(row).transpose()
        #         df_tmp['pred'] = df_tmp[0].round()
        #         df_tmp['true'] = df_tmp[1]
        #         print(df_tmp[['pred', 'true']].value_counts())
                # print(' .' * 32)
            # print('exaplmes', len(res))
            res = pd.concat(res)
            res['pred'] = res[0].apply(lambda x: round(x*2)/2)#.round()
            res['true'] = res[1]
            print(res[['pred', 'true']].value_counts())
            print(' .' * 32)

        # print('-'*500)
    g = g + 1

    return loss_function(reconstructed_graph_diagonals, input_batch_reshaped)
