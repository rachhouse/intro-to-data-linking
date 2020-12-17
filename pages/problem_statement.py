import streamlit as st

def main():
    st.title("Problem Statement and Dataset Schema Investigation")

    st.markdown("""For our linking problem, we'll imagine a scenario where we have two datasets of video games - one from the French gaming site [https://www.jeuxvideo.com/](https://www.jeuxvideo.com/) and the other from the Spanish gaming site [https://vandal.elespanol.com/](https://vandal.elespanol.com/). We want to use data linking to match video games between the two sites.""")

    st.markdown("""We will use open source Kaggle datasets for this purpose:
* JVCom: [https://www.kaggle.com/floval/jvc-game-reviews](https://www.kaggle.com/floval/jvc-game-reviews)
* Vandal: [https://www.kaggle.com/floval/12-000-video-game-reviews-from-vandal](https://www.kaggle.com/floval/12-000-video-game-reviews-from-vandal)
""")

    st.markdown("We'll also use the Python `recordlinkage` library for various data linking steps.")

    st.markdown("Our first step in the data linking process is to get to know our datasets.")


    st.header("Load the datasets")

    with st.echo():
        import pandas as pd

    st.subheader("Load the JVC (JeuxVideo.Com) dataset")
    # Load and display jvc dataset.
    st.markdown("""```python
df_jvc = pd.read_csv("data/jvc_raw.csv")
print(df_jvc.shape)
df_jvc.head()
```""")

    df_jvc = pd.read_csv("data/jvc_raw.csv")
    df_jvc = df_jvc.drop(["Unnamed: 0", "game_en", "url"], axis=1)

    st.write(df_jvc.shape)
    st.write(df_jvc.head())

    # Load and display vandal dataset.
    st.subheader("Load the Vandal dataset")

    st.markdown("""```python
df_vandal = pd.read_csv("data/vandal_raw.csv")
print(df_vandal.shape)
df_vandal.head()
```""")

    df_vandal = pd.read_csv("data/vandal_raw.csv")
    df_vandal = df_vandal.drop(["id"], axis=1)
    df_vandal.head()

    st.write(df_vandal.shape)
    st.write(df_vandal.head())

    st.header("Compare the dataset schemas")

    st.markdown("""We can see that each dataset has some common columns:
* Platform
* Website rating
* User rating
* Developer
* Release date
* Age classification
* Description""")

    st.markdown("However, the data in these common columns is not formatted or captured in a consistent way. We can see that there are differences in columns like website and user rating, which use different scales for their scores:")

    col1, col2 = st.beta_columns(2)
    col1.markdown("`df_jvc[[\"website_rating\", \"public_rating\"]].head()`")
    col1.write(df_jvc[["website_rating", "public_rating"]].head())
    col2.markdown("`df_vandal[[\"website_rating\", \"user_rating\"]].head()`")
    col2.write(df_vandal[["website_rating", "user_rating"]].head())

    st.markdown("Platforms have been captured differently:")
    
    col1, col2 = st.beta_columns(2)
    col1.markdown("`df_jvc[\"platform\"].value_counts()`")
    col1.write(df_jvc["platform"].value_counts())
    col2.markdown("`df_vandal[\"platform\"].value_counts()`")
    col2.write(df_vandal["platform"].value_counts())

    st.markdown("There are differences in the French and Spanish age classifications:")
    
    col1, col2 = st.beta_columns(2)
    col1.markdown("`df_jvc[\"classification\"].value_counts()`")
    col1.write(df_jvc["classification"].value_counts())
    col2.markdown("`df_vandal[\"classification\"].value_counts()`")
    col2.write(df_vandal["classification"].value_counts())

    st.markdown("And, we're dealing with a language difference for the game name, release date, and description.")

    st.markdown("`df_jvc[[\"game_fr\", \"release\", \"description\"]].head()`")
    st.write(df_jvc[["game_fr", "release", "description"]].head())

    st.markdown("`df_jvc[[\"game_fr\", \"release\", \"description\"]].head()`")
    st.write(df_vandal[["game", "release", "preview"]].head())

    st.markdown("Lastly, we can see that both the JVC and Vandal datasets have unique columns: JVC contains a `type` column indicating the genre type of the game, and Vandal contains a URL column which supplies the game's Vandal website address.")

    col1, col2 = st.beta_columns(2)
    col1.markdown("`df_jvc[\"type\"].value_counts()[0:10]`")
    col1.write(df_jvc["type"].value_counts()[1:10])
    col2.markdown("`df_vandal[\"url\"].head()`")
    col2.write(df_vandal["url"].head())