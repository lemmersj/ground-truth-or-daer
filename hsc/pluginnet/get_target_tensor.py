"""Contains the function for converting from a three integer target tensor
to a one integer index.
"""
import numpy as np
import torch

#pylint: disable= invalid-name, redefined-outer-name

def get_target_tensor(target, device="cpu"):
    """Converts from the three integer target tensor to a one integer index.

    Args:
        target: a 3 binary value tensor.
        device: cuda or cpu?

    Returns:
        An index corresponding to that tensor.
    """
    target_lookup_tensor = torch.zeros((2, 2, 2)).to(device)
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if i == 0 and j == 0 and k == 0:
                    continue
                target_lookup_tensor[i, j, k] = idx
                idx += 1

    # Convert the target coarse classifier to an integer.
    batch_size = target.shape[0]

    projected_target_lookup = target_lookup_tensor.unsqueeze(0).repeat(
        batch_size, 1, 1, 1)

    target_int = projected_target_lookup[torch.arange(batch_size),
                                         target[:, 0].long(), :, :]
    target_int = target_int[torch.arange(batch_size), target[:, 1].long(), :]
    target_int = target_int[torch.arange(batch_size), target[:, 2].long()]

    # Sanity check.
    #for i in range(target.shape[0]):
    #    if target_int[i] != target_lookup_tensor[target[i, 0].long(),
    #                                             target[i, 1].long(),
    #                                             target[i, 2].long()]:
    #        print("Targets don't match!")
    #        embed()

    return target_int.long()

def reverse_target_tensor(indices, device="cpu"):
    """Converts from the three integer target tensor to a one integer index.

    Args:
        indices: a network output index.
        device: cuda or cpu?

    Returns:
        A three-bit float tensor.
    """
    target_lookup_dict = {}
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if i == 0 and j == 0 and k == 0:
                    continue
                target_lookup_dict[idx] = torch.tensor(np.array((i, j, k)))
                idx += 1

    return_tensor = torch.zeros((0, 3)).to(device)
    for i in range(indices.shape[0]):
        corresponding_3_bit = target_lookup_dict[indices[i].item()].to(
            device).unsqueeze(0)
        return_tensor = torch.cat(
            (return_tensor, corresponding_3_bit.float()), dim=0)

    return return_tensor

# Make sure it works via cycle-consistency.
if __name__ == '__main__':
    # one at a time.
    input_tensor = torch.zeros(1)
    for i in range(7):
        input_tensor[0] = i
        returned_three_binary = reverse_target_tensor(input_tensor)
        match_input_tensor = get_target_tensor(returned_three_binary)
        print(input_tensor, returned_three_binary, match_input_tensor)
