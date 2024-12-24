"""TIMIT data generator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os
from pathlib import Path
from typing import Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


_TIMIT_TRAIN_DATASETS = [
    ["timit/TIMIT/data/TRAIN", (".WAV", ".WRD")],
]
_TIMIT_TEST_DATASETS = [
    ["timit/TIMIT/data/TEST", (".WAV", ".WRD")],
]


def _collect_data(directory, input_ext, target_ext):
  """Traverses directory collecting input and target files."""
  # Directory from string to tuple pair of strings
  # key: the filepath to a datafile including the datafile's basename. Example,
  #   if the datafile was "/path/to/datafile.wav" then the key would be
  #   "/path/to/datafile"
  # value: a pair of strings (input_filepath, target_filepath)
  data_files = dict()
  for root, _, filenames in os.walk(directory):
    input_files = [filename for filename in filenames if input_ext in filename]
    for input_filename in input_files:
      basename = input_filename.strip(input_ext)
      input_file = os.path.join(root, input_filename)
      target_file = os.path.join(root, basename + target_ext)
      key = os.path.join(root, basename)
      assert os.path.exists(target_file)
      assert key not in data_files
      data_files[key] = (input_file, target_file)
  return data_files

class TIMIT(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        training: bool,
    ) -> None:
      self._parse_filesystem(root, training)
    
    def _parse_filesystem(self, root: str, training: bool) -> None:
      datasets = (_TIMIT_TRAIN_DATASETS if training else _TIMIT_TEST_DATASETS)
      for data_dir, (audio_ext, transcription_ext) in datasets:
        data_dir = os.path.join(root, data_dir)
        data_files = _collect_data(data_dir, audio_ext, transcription_ext)
        data_pairs = data_files.values()
        self._flist = []
        for input_file, _ in sorted(data_pairs):
          self._flist.append(input_file)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
      input_file = self._flist[n]
      # out_filepath = input_file.strip(".WAV") + ".wav"
      waveform, sample_rate = torchaudio.load(input_file)
      fileid_audio = Path(str(input_file).replace('TIMIT', "TIMIT-{}".format("vad")))
      waveform_p, sample_rate = torchaudio.load(fileid_audio)
      fileid_audio = Path(str(input_file).replace('TIMIT', "TIMIT-{}".format("vad")))
      waveform_v, sample_rate = torchaudio.load(fileid_audio)
      return (
            waveform,
            waveform_p,
            waveform_v,
            sample_rate,
            input_file)
    
    def __len__(self) -> int:
      return len(self._flist)
    
if __name__ == "__main__":
  ds = TIMIT("/data/ssd0/tianyi.tan", training=True)
  print(len(ds))