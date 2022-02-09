from __future__ import annotations

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

AUGMENT_ARGS = dict(
    rotation_range=10,
    shear_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
)

STANDARDIZE_ARGS = dict(
    samplewise_center=True,
    samplewise_std_normalization=True,
)

DATAFLOW_ARGS = dict(
    class_mode=None,
    color_mode="grayscale",
    shuffle=False,
)


def make_tf_dataflow(
    data_root: str,
    data_type: str,
    batch_size: int = 32,
    target_size: tuple[int, int] | None = None,
    sample_standardize: bool = True,
    validation_split: float | None = None,
    rescale: float = None,
    augment: bool = False,
    seed: int | None = None,
):

    generator_args = dict(
        validation_split=validation_split,
        rescale=rescale,
        fill_mode="reflect",
    )

    if sample_standardize is True:
        generator_args = dict(**generator_args, **STANDARDIZE_ARGS)

    if augment is True:
        generator_args = dict(**generator_args, **AUGMENT_ARGS)

    generator = ImageDataGenerator(**generator_args)

    if data_type not in ("images", "masks"):
        raise RuntimeError("`data_type` must be either 'images' or 'masks'")

    interpolation = "lanczos" if data_type == "images" else "nearest"

    subset = "training" if validation_split is not None else None

    dataflow = generator.flow_from_directory(
        target_size=target_size,
        directory=data_root,
        classes=[data_type],
        subset=subset,
        interpolation=interpolation,
        seed=seed,
        batch_size=batch_size,
        **DATAFLOW_ARGS,
    )

    val_dataflow = None
    if validation_split is not None:
        val_dataflow = generator.flow_from_directory(
            target_size=target_size,
            directory=data_root,
            classes=[data_type],
            subset="validation",
            interpolation=interpolation,
            seed=seed,
            **DATAFLOW_ARGS,
        )

    return dataflow if validation_split is None else dataflow, val_dataflow


def plot_batch(X, y=None, n_plot=10):
    plt.close('all')

    batch_std = X.std()
    vmin = -2 * batch_std
    vmax = 2 * batch_std

    X = X[:n_plot]

    if y is not None:
        y = y[:n_plot]

    n_cols = len(X)
    n_rows = 1 if y is None else 2

    fig, ax = plt.subplots(n_rows, n_cols, sharex='all', sharey='all', figsize=(25, 5))

    iterable = zip(X, y) if y is not None else X
    for idx, item in enumerate(iterable):
        if y is not None:
            img, lbl = item
            ax[0, idx].imshow(img, cmap='gray_r', vmin=vmin, vmax=vmax)
            ax[1, idx].imshow(lbl, cmap='inferno', vmin=0, vmax=1)
        else:
            ax[idx].imshow(item, cmap='gray_r', vmin=vmin, vmax=vmax)


    [this.axis('off') for this in ax.ravel()]

    return fig, ax