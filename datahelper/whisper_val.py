"""TIMIT data generator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pathlib import Path
from typing import Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

class WHISPER(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
    ) -> None:
      self._parse_filesystem(root)
    
    def _parse_filesystem(self, root: str) -> None:
      data_dir = "_1_normal_trim"
      data_dir = Path(os.path.join(root, data_dir))
      self._flist = []
      for in_path in data_dir.rglob("*.wav"):
        self._flist.append(in_path)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
      input_file = self._flist[n]
      waveform, sample_rate = torchaudio.load(input_file)
      fileid_audio = Path(str(input_file).replace('_1_normal_trim', "_1_ppw_trim"))
      waveform_p, sample_rate = torchaudio.load(fileid_audio)
      return (
            waveform,
            waveform_p,
            sample_rate)
    
    def __len__(self) -> int:
      return len(self._flist)
    
if __name__ == "__main__":
  ds = WHISPER("YOURPATH")
  print(len(ds))