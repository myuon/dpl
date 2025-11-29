import numpy as np
from dpl import Dataset
import urllib.request
import pickle
import os


def load_data(data_type="train"):
    """
    PTB (Penn Treebank) コーパスをロードする

    Args:
        data_type: 'train', 'valid', 'test' のいずれか

    Returns:
        corpus: 単語IDのリスト
        word_to_id: 単語から単語IDへの辞書
        id_to_word: 単語IDから単語への辞書
    """
    base_url = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/"
    key_file = {
        "train": "ptb.train.txt",
        "valid": "ptb.valid.txt",
        "test": "ptb.test.txt"
    }
    save_file = {
        "train": "ptb.train.npy",
        "valid": "ptb.valid.npy",
        "test": "ptb.test.npy"
    }
    vocab_file = "ptb.vocab.pkl"

    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(dataset_dir, save_file[data_type])
    vocab_path = os.path.join(dataset_dir, vocab_file)

    if os.path.exists(save_path):
        corpus = np.load(save_path)
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        return corpus, vocab["word_to_id"], vocab["id_to_word"]

    # データのダウンロード
    file_name = key_file[data_type]
    file_path = os.path.join(dataset_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Downloading {file_name}...")
        try:
            urllib.request.urlretrieve(base_url + file_name, file_path)
        except urllib.error.URLError:
            print(f"Error downloading {file_name}")
            return None, None, None

    # 語彙の構築
    word_to_id = {}
    id_to_word = {}

    # すべてのファイルから語彙を構築（trainデータの場合のみ）
    if data_type == "train" and not os.path.exists(vocab_path):
        for dt in ["train", "valid", "test"]:
            fp = os.path.join(dataset_dir, key_file[dt])
            if not os.path.exists(fp):
                try:
                    urllib.request.urlretrieve(base_url + key_file[dt], fp)
                except:
                    continue

            with open(fp, "r") as f:
                words = f.read().replace("\n", "<eos>").strip().split()
                for word in words:
                    if word not in word_to_id:
                        word_id = len(word_to_id)
                        word_to_id[word] = word_id
                        id_to_word[word_id] = word

        # 語彙を保存
        with open(vocab_path, "wb") as f:
            pickle.dump({"word_to_id": word_to_id, "id_to_word": id_to_word}, f)

    # 既存の語彙をロード
    if os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
            word_to_id = vocab["word_to_id"]
            id_to_word = vocab["id_to_word"]

    # コーパスの作成
    with open(file_path, "r") as f:
        words = f.read().replace("\n", "<eos>").strip().split()
        corpus = np.array([word_to_id[w] for w in words if w in word_to_id])

    # 保存
    np.save(save_path, corpus)

    return corpus, word_to_id, id_to_word


class PTB(Dataset):
    """Penn Treebank コーパスのデータセット"""

    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(train=train, transform=transform, target_transform=target_transform)
        self.vocab_size = None
        self.word_to_id = None
        self.id_to_word = None

    def prepare(self):
        data_type = "train" if self.train else "test"
        corpus, word_to_id, id_to_word = load_data(data_type)

        self.data = corpus
        self.vocab_size = len(word_to_id)
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
