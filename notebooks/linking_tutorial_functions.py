"""Contains functions used for Introduction to Data Linking tutorial notebooks."""

import datetime
import itertools
import os
import pathlib
import re
import uuid
from typing import Optional, Tuple

import altair as alt
import jellyfish
import numpy as np
import pandas as pd
import recordlinkage as rl
import sklearn


DATA_DIR = pathlib.Path(__file__).parents[1] / "data"

TRAINING_DATASET_A = DATA_DIR / "febrl_training_a.csv"
TRAINING_DATASET_B = DATA_DIR / "febrl_training_b.csv"
TRAINING_LABELS = DATA_DIR / "febrl_training_labels.csv"


def load_febrl_training_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Load the FEBRL training data.

        Returns:
            left entity dataframe: pandas dataframe containing "left" dataset
                of FEBRL people data, indexed by person id
            right entity dataframe: pandas dataframe containing "right" dataset
                of FEBRL people data, indexed by person id
            training data labels: dataframe containing ground truth positive links,
                indexed by left person id, right person id
    """
    df_A = pd.read_csv(TRAINING_DATASET_A)
    df_A = df_A.set_index("person_id_A")

    df_B = pd.read_csv(TRAINING_DATASET_B)
    df_B = df_B.set_index("person_id_B")

    df_labels = pd.read_csv(TRAINING_LABELS)
    df_labels = df_labels.set_index(['person_id_A', 'person_id_B'])

    return df_A, df_B, df_labels


def load_febrl_evaluation_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pass


def dob_to_date(dob: str) -> Optional[pd.Timestamp]:
    """Transform string date in YYYYMMDD format to a pd.Timestamp.
    Return None if transformation is not successful.
    """
    date_pattern = r"(\d{4})(\d{2})(\d{2})"
    dob_timestamp = None

    try:
        if m := re.match(date_pattern, dob.strip()):
            dob_timestamp = pd.Timestamp(
                int(m.group(1)), int(m.group(2)), int(m.group(3))
            )
    except:
        pass

    return dob_timestamp


def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    """Augment dataframe of FEBRL person data with blocking keys and cleanup for
    comparison step.

    Args:
        df: pandas dataframe containing FEBRL-generated person data

    Returns:
        Augmented dataframe.
    """

    df["surname"] = df["surname"].fillna("")
    df["first_name"] = df["first_name"].fillna("")

    # Soundex phonetic encodings.
    df["soundex_surname"] = df["surname"].apply(lambda x: jellyfish.soundex(x))
    df["soundex_firstname"] = df["first_name"].apply(lambda x: jellyfish.soundex(x))

    # NYSIIS phonetic encodings.
    df["nysiis_surname"] = df["surname"].apply(lambda x: jellyfish.nysiis(x))
    df["nysiis_firstname"] = df["first_name"].apply(lambda x: jellyfish.nysiis(x))

    # Last 3 of SSID.
    df["ssid_last3"] = df["soc_sec_id"].apply(
        lambda x: str(x)[-3:].zfill(3) if x else None
    )
    df["soc_sec_id"] = df["soc_sec_id"].astype(str)

    # DOB to date object.
    df["dob"] = df["date_of_birth"].apply(lambda x: dob_to_date(x))


def block(df_left: pd.DataFrame, df_right: pd.DataFrame) -> pd.MultiIndex:
    """Run blocking.

    Args:
        df_left: pandas dataframe containing augmented FEBRL people data
        df_right: pandas dataframe containing augmented FEBRL people data

    Returns:
        Candidate links as a pandas MultiIndex.
    """

    indexer = rl.Index()

    indexer.add(rl.index.Block("soundex_surname"))
    indexer.add(rl.index.Block("soundex_firstname"))
    indexer.add(rl.index.Block("nysiis_surname"))
    indexer.add(rl.index.Block("nysiis_firstname"))
    indexer.add(rl.index.Block("ssid_last3"))
    indexer.add(rl.index.Block("date_of_birth"))

    return indexer.index(df_left, df_right)


def compare(
    candidate_links: pd.MultiIndex, df_left: pd.DataFrame, df_right: pd.DataFrame
) -> pd.DataFrame:
    """Run comparing.

    Args:
        candidate_links:
        df_left:
        df_right:

    Returns:
        Pandas dataframe containing candidate link feature vectors.
    """
    comparer = rl.Compare()

    # Phonetic encodings.
    comparer.add(
        rl.compare.Exact("soundex_surname", "soundex_surname", label="soundex_surname")
    )
    comparer.add(
        rl.compare.Exact(
            "soundex_firstname", "soundex_firstname", label="soundex_firstname"
        )
    )
    comparer.add(
        rl.compare.Exact("nysiis_surname", "nysiis_surname", label="nysiis_surname")
    )
    comparer.add(
        rl.compare.Exact(
            "nysiis_firstname", "nysiis_firstname", label="nysiis_firstname"
        )
    )

    # First & last name.
    comparer.add(
        rl.compare.String("surname", "surname", method="jarowinkler", label="last_name")
    )
    comparer.add(
        rl.compare.String(
            "first_name", "first_name", method="jarowinkler", label="first_name"
        )
    )

    # Address.
    comparer.add(
        rl.compare.String(
            "address_1", "address_1", method="damerau_levenshtein", label="address_1"
        )
    )
    comparer.add(
        rl.compare.String(
            "address_2", "address_2", method="damerau_levenshtein", label="address_2"
        )
    )
    comparer.add(
        rl.compare.String(
            "suburb", "suburb", method="damerau_levenshtein", label="suburb"
        )
    )
    comparer.add(
        rl.compare.String(
            "postcode", "postcode", method="damerau_levenshtein", label="postcode"
        )
    )
    comparer.add(
        rl.compare.String("state", "state", method="damerau_levenshtein", label="state")
    )

    # Other fields.
    comparer.add(rl.compare.Date("dob", "dob", label="date_of_birth"))
    comparer.add(
        rl.compare.String(
            "phone_number",
            "phone_number",
            method="damerau_levenshtein",
            label="phone_number",
        )
    )
    comparer.add(
        rl.compare.String(
            "soc_sec_id", "soc_sec_id", method="damerau_levenshtein", label="ssn"
        )
    )

    return comparer.compute(candidate_links, df_left, df_right)
