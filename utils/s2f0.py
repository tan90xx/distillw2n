import torch
from libs import JDCNet

def load_F0_models(path, device):
  F0_model = JDCNet(num_class=1, seq_len=192)
  params = torch.load(path, map_location=device, weights_only=True)['net']
  F0_model.load_state_dict(params)
  _ = F0_model.train()
  return F0_model

def wav2F0(mels, F0_model, device, norm=True):
    mels = mels.to(device)
    mels = mels.squeeze()
    F0_model = F0_model.to(device)
    with torch.no_grad():
        F0_real, _, _ = F0_model(mels.unsqueeze(1))

    # normalize f0
    # Remove unvoiced frames (replace with -1)
    gt_glob_f0s = []
    f0_targets = []
    norm_f0 = True
    if not norm_f0:
        f0_targets.append(F0_real)
    else:
        for bib in range(len(F0_real)):
            voiced_indices = F0_real[bib] > 5.0
            f0_voiced = F0_real[bib][voiced_indices]

            if len(f0_voiced) != 0:
                # Convert to log scale
                log_f0 = f0_voiced.log2()
                
                # Calculate mean and standard deviation
                mean_f0 = log_f0.mean()
                std_f0 = log_f0.std()
                if norm:
                    # Normalize the F0 sequence
                    normalized_f0 = (log_f0 - mean_f0) / std_f0
                else:
                    normalized_f0 = log_f0

                # Create the normalized F0 sequence with unvoiced frames
                normalized_sequence = torch.zeros_like(F0_real[bib])
                normalized_sequence[voiced_indices] = normalized_f0.to(normalized_sequence.dtype)
                normalized_sequence[~voiced_indices] = -10  # Assign -10 to unvoiced frames

                gt_glob_f0s.append(mean_f0)
            else:
                normalized_sequence = torch.zeros_like(F0_real[bib]) - 10.0
                gt_glob_f0s.append(torch.tensor(0.0).to(device))

            # f0_targets.append(normalized_sequence[single_side_context // 200:-single_side_context // 200])
            f0_targets.append(normalized_sequence)

    f0_targets = torch.stack(f0_targets).to(device)
    # fill nan with -10
    f0_targets[torch.isnan(f0_targets)] = -10.0
    # fill inf with -10
    f0_targets[torch.isinf(f0_targets)] = -10.0

    return f0_targets