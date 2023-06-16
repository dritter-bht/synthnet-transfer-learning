import os
import shutil
from types import SimpleNamespace

import click


@click.command()
@click.option(
    "--input_train_ds",
    help="Path to training data. A label-stratified validation set will be pulled out from this (according to val_size param). Directory structure must be <root/label/mesh_id/img_file.png",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    "--input_test_ds",
    help="Path to test data. Used for evaluation after training. Directory structure must be <root/label/mesh_id/img_file.png",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    "--output_dir",
    help="Output directory path for train/val/test data. Directory format according to huggingface imagefolder: root/label/imgfile.png",
    type=click.Path(),
    required=True,
)
def main(**kwargs):
    # Parse click parameters and load config
    args = SimpleNamespace(**kwargs)

    for path, dns, fns in os.walk(args.input_train_ds):
        for fn in fns:
            split_path = path.split("/")
            label = split_path[-2]
            split = "train"
            os.makedirs(f"{args.output_dir}/{split}/{label}", exist_ok=True)
            shutil.copy(f"{path}/{fn}", f"{args.output_dir}/{split}/{label}/{fn}")

    train_labels = os.listdir(f"{args.output_dir}/train")
    n_labels = len(train_labels)
    print(f"{n_labels=}")
    print(f"{train_labels=}")

    for path, dns, fns in os.walk(args.input_test_ds):
        for fn in fns:
            split_path = path.split("/")
            label = split_path[-1]
            split = "test"
            if label in train_labels:
                os.makedirs(f"{args.output_dir}/{split}/{label}", exist_ok=True)
                shutil.copy(f"{path}/{fn}", f"{args.output_dir}/{split}/{label}/{fn}")

    test_labels = os.listdir(f"{args.output_dir}/test")
    n_labels = len(test_labels)
    print(f"{n_labels=}")
    print(f"{test_labels=}")


if __name__ == "__main__":
    main()
