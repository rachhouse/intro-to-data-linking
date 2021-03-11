"""Contains functions used for Introduction to Data Linking tutorial notebooks."""

import pathlib
import re
from typing import Dict, Optional, Tuple, Union

import altair as alt
import jellyfish
import numpy as np
import pandas as pd
import recordlinkage as rl

DATA_DIR = pathlib.Path(__file__).parents[1] / "data"

TRAINING_DATASET_A = DATA_DIR / "febrl_training_a.csv"
TRAINING_DATASET_B = DATA_DIR / "febrl_training_b.csv"
TRAINING_LABELS = DATA_DIR / "febrl_training_labels.csv"

PathOrURL = Union[pathlib.Path, str]


def get_training_data_paths(
    colab: bool = False,
) -> Tuple[PathOrURL, PathOrURL, PathOrURL]:
    """Assemble either pathlib.Path or string URL for each of the training data files.

    Args:
        colab: bool to indicate whether the notebook is running in Google Colab

    Returns:
        * resource path to febrl_training_a.csv
        * resourcepath to febrl_training_b.csv
        * resource path to febrl_training_labels.csv
    """
    if colab:
        data_url = "https://raw.githubusercontent.com/rachhouse/intro-to-data-linking/main/data/"

        training_dataset_a = f"{data_url}febrl_training_a.csv"
        training_dataset_b = f"{data_url}febrl_training_b.csv"
        training_labels = f"{data_url}febrl_training_labels.csv"

    else:
        data_dir = pathlib.Path(__file__).parents[1] / "data"
        training_dataset_a = data_dir / "febrl_training_a.csv"
        training_dataset_b = data_dir / "febrl_training_b.csv"
        training_labels = data_dir / "febrl_training_labels.csv"

    return training_dataset_a, training_dataset_b, training_labels


def load_febrl_training_data(
    colab: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the FEBRL training data.

    Args:
        colab: bool to indicate whether the notebook is running in Google Colab

    Returns:
        left entity dataframe: pandas dataframe containing "left" dataset
            of FEBRL people data, indexed by person id
        right entity dataframe: pandas dataframe containing "right" dataset
            of FEBRL people data, indexed by person id
        training data labels: dataframe containing ground truth positive links,
            indexed by left person id, right person id
    """

    training_dataset_a, training_dataset_b, training_labels = get_training_data_paths(
        colab
    )

    df_A = pd.read_csv(training_dataset_a)
    df_A = df_A.set_index("person_id_A")

    df_B = pd.read_csv(training_dataset_b)
    df_B = df_B.set_index("person_id_B")

    df_ground_truth = pd.read_csv(training_labels)
    df_ground_truth = df_ground_truth.set_index(["person_id_A", "person_id_B"])
    df_ground_truth["ground_truth"] = df_ground_truth["ground_truth"].apply(
        lambda x: True if x == 1 else False
    )

    return df_A, df_B, df_ground_truth


def load_febrl_evaluation_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pass


def dob_to_date(dob: str) -> Optional[pd.Timestamp]:
    """Transform string date in YYYYMMDD format to a pd.Timestamp.
    Return None if transformation is not successful.
    """
    date_pattern = r"(\d{4})(\d{2})(\d{2})"
    dob_timestamp = None

    try:
        m = re.match(date_pattern, dob.strip())
        if m:
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


def evaluate_linking(
    df: pd.DataFrame,
    df_true_links: pd.DataFrame,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    score_column_name: Optional[str] = "model_score",
    ground_truth_column_name: Optional[str] = "ground_truth",
    k: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use model results to calculate precision & recall metrics, top k links,
        and bottom k links.

    Args:
        df: dataframe containing model scores, and ground truth labels
            indexed on df_left index, df_right index
        df_true_links: dataframe containing true links,
            indexed on df_left index, df_right index
        df_left: dataframe containing attributes for "left"-linked entities
        df_right: dataframe containing attributes for "right"-linked entities
        score_column_name: Optional string name of column containing model scores
        ground_truth_column_name: Optional string name of column containing ground
            truth values

    Returns:
        Tuple containing:
            pandas dataframe with precision and recall evaluation data
            pandas dataframe with top k scoring links
            pandas dataframe with bottom k scoring links
    """

    # Calculate eval data at threshold intervals from zero to max score.
    # Max score is generally 1.0 if using a ML model, but with SimSum it
    # can get much larger.
    eval_data = []
    max_score = max(1, max(df[score_column_name]))

    for threshold in np.linspace(0, max_score, 50):
        tp = df[
            (df[score_column_name] >= threshold)
            & (df[ground_truth_column_name] == True)
        ].shape[0]
        fp = df[
            (df[score_column_name] >= threshold)
            & (df[ground_truth_column_name] == False)
        ].shape[0]
        tn = df[
            (df[score_column_name] < threshold)
            & (df[ground_truth_column_name] == False)
        ].shape[0]
        fn = df[
            (df[score_column_name] < threshold) & (df[ground_truth_column_name] == True)
        ].shape[0]

        if tp + fp == 0:
            precision = None
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = None
        else:
            recall = tp / (tp + fn)

        if (precision is None) or (recall is None):
            f1 = None
        else:
            f1 = 2 * ((precision * recall) / (precision + recall))

        eval_data.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    # Assemble the top and bottom k links (sorted by model score).
    # This is done by sorting the model results frame by score, capping rows at k,
    # and then joining the original link entity data via the dataframe indices.
    # This gives us the model score as well as the actual human-readable attributes
    # for each link.
    def _join_original_entity_data_to_links(
        df_k_links: pd.DataFrame, df_left: pd.DataFrame, df_right: pd.DataFrame
    ) -> pd.DataFrame:
        """Helper function to join entity data to a dataframe of link results."""

        # Join data from left entities.
        df_k_links = pd.merge(
            df_k_links,
            df_left,
            left_on=df_left.index.name,
            right_index=True,
        )

        # Join data from right entities.
        return pd.merge(
            df_k_links,
            df_right,
            left_on=df_right.index.name,
            right_index=True,
        )

    df_top_k_links = _join_original_entity_data_to_links(
        df[[score_column_name, ground_truth_column_name]]
        .sort_values(score_column_name, ascending=False)
        .head(n=k)
        .reset_index(),
        df_left,
        df_right,
    )

    df_bottom_k_links = _join_original_entity_data_to_links(
        df[[score_column_name, ground_truth_column_name]]
        .sort_values(score_column_name)
        .head(n=k)
        .reset_index(),
        df_left,
        df_right,
    )

    return (
        pd.DataFrame(eval_data),
        df_top_k_links,
        df_bottom_k_links,
    )


def plot_model_score_distribution(
    df: pd.DataFrame,
    score_column_name: Optional[str] = "model_score",
    ground_truth_column_name: Optional[str] = "ground_truth",
) -> alt.Chart:
    """Generate an altair plot of the model score distribution, colored by ground
    truth value.

    Args:
        df: pandas dataframe containing model score and ground truth for all
            candidate links
        score_column_name: Optional string name of column containing model scores
        ground_truth_column_name: Optional string name of column containing ground
            truth values

    Returns:
        altair Chart object containing distribution
    """

    # Pre-bin model score data for plotting.
    df_score_dist = df[[score_column_name, ground_truth_column_name]].copy()
    df_score_dist[ground_truth_column_name] = df_score_dist[
        ground_truth_column_name
    ].apply(lambda x: "True Link" if x else "Not a Link")
    df_score_dist[score_column_name] = df_score_dist[score_column_name].apply(
        lambda x: round(x, 2)
    )
    df_score_dist["count"] = df_score_dist[ground_truth_column_name]
    df_score_dist = (
        df_score_dist.groupby([score_column_name, ground_truth_column_name])
        .count()
        .reset_index()
    )

    # Generate altair chart of model score distribution.
    legend_selection = alt.selection_multi(
        fields=[ground_truth_column_name], bind="legend"
    )

    color_scale = alt.Scale(
        domain=["True Link", "Not a Link"],
        scheme="tableau10",
    )

    max_score = max(1, max(df_score_dist[score_column_name]))

    model_score_distribution = (
        alt.Chart(df_score_dist, title=f"Model Score Distribution")
        .mark_bar(opacity=0.7, binSpacing=0)
        .encode(
            alt.X(
                f"{score_column_name}:Q",
                bin=alt.Bin(extent=[0, max_score], step=0.01),
                axis=alt.Axis(tickCount=5, title="Model Score (Binned)"),
            ),
            alt.Y("count", stack=None, axis=alt.Axis(title="Count of Links")),
            alt.Color(
                f"{ground_truth_column_name}",
                scale=color_scale,
                legend=alt.Legend(title="Ground Truth Label"),
            ),
            opacity=alt.condition(legend_selection, alt.value(0.7), alt.value(0.2)),
            tooltip=[
                alt.Tooltip(f"{score_column_name}", title="Model Score"),
                alt.Tooltip(f"{ground_truth_column_name}", title="Ground Truth"),
                alt.Tooltip("count", title="Count of Links"),
            ],
        )
        .properties(height=400, width=800)
        .add_selection(legend_selection)
        .interactive()
    )

    return model_score_distribution


def plot_precision_recall_vs_threshold(df: pd.DataFrame) -> alt.Chart:
    """Generate an altair plot of model precision and recall at varying thresholds.

    Args:
        df: pandas dataframe containing precision and recall values at given thresholds

    Returns:
        altair Chart object
    """
    pr_at_threshold = (
        alt.Chart(
            df[["threshold", "recall", "precision"]].melt(id_vars=["threshold"]),
            title="Precision and Recall v.s. Model Threshold",
        )
        .mark_line()
        .encode(
            alt.X("threshold:Q", axis=alt.Axis(title="Model Threshold")),
            alt.Y(
                "value:Q",
                scale=alt.Scale(domain=(0, 1)),
                axis=alt.Axis(title="Precision/Recall Value"),
            ),
            alt.Color("variable:N", legend=alt.Legend(title="Variable")),
            tooltip=alt.Tooltip(["variable", "threshold", "value"]),
        )
        .properties(height=400, width=800)
        .interactive()
    )

    return pr_at_threshold
