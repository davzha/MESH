from pathlib import Path
import h5py
import torch
import torch.utils.data


class H5Records(torch.utils.data.Dataset):
    def __init__(
        self, path, start, end, keys, transform=None, max_objects=None, preload=False,
        ignore=None, frame_ids=None, 
    ):
        self.path = Path(path).expanduser().resolve()
        self.start = start
        self.end = end
        self.keys = keys
        self.transform = transform
        self.db = None
        self.ignore = ignore
        self.frame_ids = frame_ids

        if max_objects is None:
            self.valid_indices = list(range(start, end))
        else:
            with self.img_db() as db:
                nums = enumerate(db["n_examples"][start:end], start)
                self.valid_indices = [i for i, x in nums if x <= max_objects]
        if ignore is not None:
            for i in ignore:
                self.valid_indices.remove(i)

        if preload:
            self.db = {}
            with self.img_db() as db:
                for key in keys:
                    self.db[key] = db[key][:]
            if frame_ids is not None:
                self.db["images"] = self.db["images"][:, frame_ids]
                self.db["masks"] = self.db["masks"][:, frame_ids]

    def img_db(self):
        return h5py.File(self.path, "r")

    def __getitem__(self, item):
        idx = self.valid_indices[item]
        if self.db is None:
            self.db = self.img_db()
        result = [self.db[key][idx] for key in self.keys]
        result = [torch.from_numpy(x).float() for x in result]

        # normalize CHW image to be in [-1, 1]
        result = [x if x.dim() >= 4 else x / 255 * 2 - 1 for x in result[:2]] + result[2:]  

        if self.transform:
            result = [self.transform(x) for x in result]
        return result

    def __len__(self):
        return len(self.valid_indices)
