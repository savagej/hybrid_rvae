import torch.utils.data as data
import torch
import h5py
import numpy as np


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path, key, max_seq_len=209):
        super().__init__()
        hf = h5py.File(file_path, mode="r", swmr=True, libver='latest',)
        self.data = hf.get(key).get("table")
        # self.target = hf.get('title')
        self.max_seq_len = max_seq_len
        self.go_token = '>'
        self.pad_token = ''
        self.stop_token = '<'
        self.vocab_size = None
        self.char_to_idx = None
        self.idx_to_char = None

    def build_vocab(self, chunksize=1000):

        chars = set()
        for i in range(0, self.__len__(), chunksize):
            res = set("".join([x[1].decode() for x in self.data[i:i + chunksize]]))
            chars = set.union(*[chars, res])
            # print(vocab)
        chars = sorted(list(chars))

        chars = chars + [self.pad_token, self.go_token, self.stop_token]
        self.vocab_size = len(chars)

        # mappings itself
        self.idx_to_char = chars
        self.char_to_idx = {x: i for i, x in enumerate(self.idx_to_char)}
        return self.vocab_size, self.idx_to_char, self.char_to_idx

    def set_vocab(self, vocab_size, idx_to_char, char_to_idx):
        self.vocab_size = vocab_size
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char

    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
        We should build a custom collate_fn rather than using default collate_fn,
        because merging sequences (including padding) is not supported in default.
        Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
        https://github.com/yunjey/seq2seq-dataloader/blob/master/example.ipynb

        Args:
            data: list of tuple (src_seq, trg_seq).
                - src_seq: torch tensor of shape (?); variable length.
                - trg_seq: torch tensor of shape (?); variable length.
        Returns:
            src_seqs: torch tensor of shape (batch_size, padded_length).
            src_lengths: list of length (batch_size); valid length for each padded source sequence.
            trg_seqs: torch tensor of shape (batch_size, padded_length).
            trg_lengths: list of length (batch_size); valid length for each padded target sequence.
        """

        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = np.array([[self.char_to_idx[self.pad_token]] * (self.max_seq_len + 1)] * len(sequences))
            padded_seqs = torch.from_numpy(padded_seqs).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x[0]), reverse=True)

        # seperate source and target sequences
        enc_seqs, dec_seqs, trg_seqs = zip(*data)

        # merge sequences (from tuple of 1D tensor to 2D tensor)
        enc_seqs, enc_lengths = merge(enc_seqs)
        dec_seqs, dec_lengths = merge(dec_seqs)
        trg_seqs, trg_lengths = merge(trg_seqs)

        return enc_seqs, dec_seqs, trg_seqs

    def __getitem__(self, index):
        example = self.data[index][1].decode()[:self.max_seq_len]
        inp = [self.char_to_idx[char] for char in example]

        encoder_input = np.array([self.char_to_idx[self.go_token]] + inp)
        decoder_input = np.array([self.char_to_idx[self.go_token]] + inp)
        decoder_target = np.array(inp + [self.char_to_idx[self.stop_token]])

        return torch.from_numpy(encoder_input), torch.from_numpy(decoder_input), torch.from_numpy(decoder_target)

    def __len__(self):
        return self.data.shape[0]