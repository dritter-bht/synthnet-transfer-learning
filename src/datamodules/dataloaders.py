"""Implements custom dataloaders."""


class MultiDataParallelLoader:
    """Batch Sampler for a MultiDataset.

    Iterates in parallel over different batch samplers for each dataset. Yields batches [(x_1, y_1), ..., (x_n, y_n)]
    for n datasets.
    """

    def __init__(self, dataloaders, n_batches):
        """Initialize this MultiDataLoader with a list of dataloaders and the number of batches to iterate ofer them.

        Args:
            dataloaders (list<torch.utils.data.DataLoader>): A list of pytorch dataloaders to iterate over in parallel.
            n_batches (int): Number of batches to iterate over all dataloaders.
        """
        self._dataloaders = dataloaders
        self._n_batches = n_batches
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
