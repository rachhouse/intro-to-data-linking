import streamlit as st

def main():
    st.title("Cleaning and Pre-Processing")

    st.markdown("Now that we've gotten acquainted with our datasets, we're ready to do some cleaning. `pandas` at the ready (along with a few other libraries)!")

    with st.echo():
        import datetime
        import re

        from typing import Optional

        import dateparser
        import pandas as pd   

    st.header("Define common columns for cleaned datasets")

    st.markdown("Here, we'll define the common columns that we want to persist once we've cleaned the datasets.")

    with st.echo():
        COMMON_COLUMNS = [
            "game_name",
            "platform",
            "developer",
            "release_date",
            "age_classification",
            "website_rating",
            "public_rating",
            "description",
        ]        

    st.header("Clean the JVC dataset")

    st.subheader("Load the raw dataset")
    st.markdown("""```python
df_jvc = pd.read_csv("data/jvc_raw.csv")
df_jvc.head()
```""")

    df_jvc = pd.read_csv("data/jvc_raw.csv")
    df_jvc = df_jvc.drop(["Unnamed: 0", "game_en", "url"], axis=1)

    st.write(df_jvc.head())

    st.subheader("JVC Age classification")
    st.markdown("We will clean up the age classification categories by stripping out the `+` and `ans`.")

    with st.echo():
        df_jvc["classification"].value_counts()
    
    st.write(df_jvc["classification"].value_counts())

    with st.echo():
        df_jvc["age_classification"] = df_jvc["classification"].apply(
            lambda x: re.sub(r"\+(\d+) ans", lambda pattern: pattern.group(1), x.strip()))

        df_jvc["age_classification"].value_counts()
    
    st.write(df_jvc["age_classification"].value_counts())

    st.subheader("JVC Release date")
    st.markdown("Next, we'll standardize the release date to a `YYYY-MM-DD` format. To do accomplish this despite the language difference, we will use a handy little library, `dateparse`, along with some custom regexs. I've combined all that we need in a new cleaning function, `convert_french_date()`.")

    st.markdown("Here's a view of our current release date data:")
    with st.echo():
        df_jvc["release"].head()
    
    st.write(df_jvc["release"].head())    

    with st.echo():
        def convert_french_date(x: str) -> Optional[str]:
            """Helper function to convert a string date in French to YYYY-MM-DD format."""

            converted_date = None

            if x in ["Date de sortie inconnue", "Unknown", "Le jeu est annulé"]:
                return converted_date    
            
            trimester_pattern = r"(\d+)\w+ trimestre (\d{4})"

            try:
                converted_date = str(dateparser.parse(x).date())
            except:
                try:
                    if re.match(trimester_pattern, x):
                        trimester = int(re.search(trimester_pattern, x).group(1))
                        year = int(re.search(trimester_pattern, x).group(2))
                        converted_date = str(datetime.datetime(year, trimester*3, 1,0,0).date())                        
                except:
                    pass
            
            return converted_date
        
        df_jvc["release_date"] = df_jvc["release"].apply(lambda x: convert_french_date(x))
        df_jvc["release_date"].head()

    st.markdown("And here is a view of our cleaned release date data:")
    st.write(df_jvc["release_date"].head())

    st.subheader("JVC Ratings")
    st.markdown("Next, we'll standardize the website and user ratings to a scale of 0-10. Once again, we can write a small helper function to do the conversion.")

    st.markdown("Here is a view of the current ratings data:")
    with st.echo():
        df_jvc[["website_rating", "public_rating"]].head()
    
    st.write(df_jvc[["website_rating", "public_rating"]].head())

    with st.echo():
        def convert_rating(x: str) -> float:
            rating_regex = r"([\d\.]+)\s?\/20"

            if x == "--/20":
                return None
            
            if re.match(rating_regex, x):
                out_of_20 = float(re.search(rating_regex, x).group(1))
                return round(out_of_20/20*10, 2)
            else:
                return None

    df_jvc["web_rating"] = df_jvc["website_rating"].apply(lambda x: convert_rating(x))
    df_jvc["user_rating"] = df_jvc["public_rating"].apply(lambda x: convert_rating(x))

    st.markdown("And here is a view of the cleaned ratings data:")
    with st.echo():
        df_jvc[["web_rating", "user_rating"]].head()
    
    st.write(df_jvc[["web_rating", "user_rating"]].head())

    st.subheader("Save off cleaned JVC dataset")
    st.markdown("Lastly for JVC, we'll save off the cleaned dataset.")

    with st.echo():
        jvc_columns = [
            "game_fr",
            "platform",
            "publishor/developer",
            "release_date",
            "age_classification",
            "web_rating",
            "user_rating", 
            "description", "type"]

        df_jvc_cleaned = df_jvc[jvc_columns].copy()
        df_jvc_cleaned.columns = COMMON_COLUMNS + ["type"]

        df_jvc_cleaned.head()

    st.write(df_jvc_cleaned.head())    

    st.markdown("""```python
df_jvc_cleaned.to_csv("jvc_cleaned.csv", index=False)
```""")

    st.header("Clean the Vandal dataset")

    st.subheader("Load the raw dataset")

    st.markdown("""```python
df_vandal = pd.read_csv("data/vandal_raw.csv")
df_vandal.head()
```""")

    df_vandal = pd.read_csv("data/vandal_raw.csv")
    df_vandal = df_vandal.drop(["id"], axis=1)
    df_vandal.head()

    st.write(df_vandal.head())

    st.subheader("Vandal Age classificaion")