# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Common Voice Dataset"""

from datasets import AutomaticSpeechRecognition


import datasets
import os
import pandas as pd


_CITATION = """\
@inproceedings{lovenia2021ascend,
  title     = {ASCEND: A Spontaneous Chinese-English Dataset for Code-switching in Multi-turn Conversation},
  author    = {Lovenia, Holy and Cahyawijaya, Samuel and Winata, Genta Indra and Xu, Peng and Yan, Xu and Liu, Zihan and Frieske, Rita and Yu, Tiezheng and Dai, Wenliang and Barezi, Elham J and others},
  booktitle = {Proceedings of the International Conference on Language Resources and Evaluation, {LREC} 2022, 20-25 June 2022, Lu Palais du Pharo, France},
  publisher = {European Language Resources Association},
  year      = {2022},
  pages = {}
}
"""

_DESCRIPTION = """\
ASCEND (A Spontaneous Chinese-English Dataset) introduces a high-quality resource of spontaneous multi-turn conversational dialogue Chinese-English code-switching corpus collected in Hong Kong. ASCEND consists of 10.62 hours of spontaneous speech with a total of ~12.3K utterances. The corpus is split into 3 sets: training, validation, and test with a ratio of 8:1:1 while maintaining a balanced gender proportion on each set.
"""

_HOMEPAGE = "https://huggingface.co/datasets/CAiRE/ASCEND"

DEFAULT_CONFIG_NAME = "train"

_URL = "https://huggingface.co/datasets/CAiRE/ASCEND/raw/main/"
_URLS = {
    "train": _URL + "train_metadata.csv",
    "test": _URL + "test_metadata.csv",
    "validation": _URL + "validation_metadata.csv",
    "waves": _URL + "waves.zip",
}


class ASCENDConfig(datasets.BuilderConfig):
    """BuilderConfig for ASCEND."""

    def __init__(self, name, **kwargs):
        """
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ASCENDConfig, self).__init__(name, **kwargs)


class ASCEND(datasets.GeneratorBasedBuilder):
    """ASCEND: A Spontaneous Chinese-English Dataset for code-switching. Snapshot date: 5 January 2022."""

    BUILDER_CONFIGS = [
        ASCENDConfig(
            name="train",
            version=datasets.Version("1.0.0", ""),
            description=_DESCRIPTION,
        ),
        ASCENDConfig(
            name="validation",
            version=datasets.Version("1.0.0", ""),
            description=_DESCRIPTION,
        ),
        ASCENDConfig(
            name="test",
            version=datasets.Version("1.0.0", ""),
            description=_DESCRIPTION,
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "path": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000),
                "transcription": datasets.Value("string"),
                "duration": datasets.Value("float32"),
                "language": datasets.Value("string"),
                "original_speaker_id": datasets.Value("int64"),
                "session_id": datasets.Value("int64"),
                "topic": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=[AutomaticSpeechRecognition(audio_column="audio", transcription_column="transcription")],
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_path": downloaded_files["train"]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "metadata_path": downloaded_files["test"]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "metadata_path": downloaded_files["validation"]
                },
            ),
        ]

    def _generate_examples(self, metadata_path):
        print(metadata_path)
        metadata_df = pd.read_csv(metadata_path)

        for index, row in metadata_df.iterrows():
            example = {
                "id": str(index).zfill(5),
                "path": os.path.join(_WAVE_URL, row["file_name"]),
                "audio": dl_manager.download_and_extract(os.path.join(_WAVE_URL, row["file_name"])),
                "transcription": row["transcription"],
                "duration": row["duration"],
                "language": row["language"],
                "original_speaker_id": row["original_speaker_id"],
                "session_id": row["session_id"],
                "topic": row["topic"],
            }
            yield index, example