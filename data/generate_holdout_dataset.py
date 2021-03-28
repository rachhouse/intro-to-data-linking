import os
import pathlib
import sys

import febrl_data_transform as transform
import pandas as pd

OUTPUT_DATA_DIR = pathlib.Path(__file__).parent / "holdout"
ORIGINALS_DATA_DIR = pathlib.Path(__file__).parent / "holdout" / "originals"


def main():

    # Read in FEBRL data with dupes and separate into A/B/true links.
    dataset_A = []
    dataset_B = []
    true_links = []

    for filename in [
        "febrl_holdout_dupes_light_mod.csv",
        "febrl_holdout_dupes_medium_mod.csv",
        "febrl_holdout_dupes_heavy_mod.csv",
    ]:
        _df_A, _df_B, _df_true_links = transform.transform_febrl_dataset_with_dupes(
            ORIGINALS_DATA_DIR / filename
        )

        dataset_A.append(_df_A)
        dataset_B.append(_df_B)
        true_links.append(_df_true_links)

    df_A = pd.concat(dataset_A)
    df_B = pd.concat(dataset_B)
    df_true_links = pd.concat(true_links)

    # Read in extra, non-dupe records and split between datasets A and B.
    df_extra = transform.transform_febrl_dataset_without_dupes(
        ORIGINALS_DATA_DIR / "febrl_holdout_extras.csv"
    )
    chunk_size = int(df_extra.shape[0] / 2)

    df_A = pd.concat([df_A, df_extra.iloc[0:chunk_size]])

    df_B_extra = df_extra.iloc[chunk_size:].rename(
        columns={"person_id_A": "person_id_B"}
    )
    df_B = pd.concat([df_B, df_B_extra])

    # Shuffle rows before saving off holdout datasets.
    df_A = df_A.sample(frac=1).reset_index(drop=True)
    df_B = df_B.sample(frac=1).reset_index(drop=True)

    df_A.to_csv(OUTPUT_DATA_DIR / "febrl_holdout_a.csv", index=False)
    df_B.to_csv(OUTPUT_DATA_DIR / "febrl_holdout_b.csv", index=False)
    df_true_links.to_csv(OUTPUT_DATA_DIR / "febrl_holdout_true_links.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
