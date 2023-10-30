import streamlit as st
import pandas as pd
import os
from intro import Intro  # importing function from python file from same directory


import streamlit as st

st.set_page_config(
    page_title="Automated Machine Learning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


page_bg_img_link = f"""
<style>
[data-testid="stAppViewContainer"]> .main{{

background-color: #FFDEE9;
background-image: linear-gradient(0deg,  #B5FFFC 0%, #FFDEE9 100%);




}}


[data-testid="stHeader"]{{
background-color: rgba(0,0,0,0)

}}

[data-testid="stToolbar"]{{
right : 2 rem;
}}

[data-testid="stSidebar"] > div:first-child{{


background: linear-gradient(to right bottom,
                 rgba(285,205,205,0.7),
                 rgba(285,205,205,0.3));
}}


</style>
"""
st.markdown(page_bg_img_link, unsafe_allow_html=True)


# Import profiling capability
from ydata_profiling import profile_report
from streamlit_pandas_profiling import st_profile_report

# ML Pycaret Modules
from pycaret.classification import (
    setup as classification_setup,
    compare_models as classification_compare_model,
    pull as classification_pull,
    predict_model as classification_predict_model,
    save_model as classification_save_model,
)

from pycaret.regression import (
    setup as regression_setup,
    compare_models as regression_compare_models,
    pull as regression_pull,
    predict_model as regression_predict_model,
    save_model as regression_save_model,
)

from pycaret.clustering import (
    setup as clustering_steup,
    pull as clustering_pull,
    create_model as clustering_create_model,
    predict_model as clustering_predict_model,
    plot_model as clustering_plot_model,
    save_model as clustering_save_model,
)

from pycaret.anomaly import (
    setup as anomaly_setup,
    pull as anomaly_pull,
    create_model as anomaly_create_model,
    models as anomaly_models,
    plot_model as anomaly_plot_model,
    assign_model as anomaly_assign_model,
    save_model as anomaly_save_model,
)


def main():
    st.sidebar.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.sidebar.title("AutoStreamML")
    choice = st.sidebar.radio(
        "Navigation", ["Instruction", "Upload", "Profiling", "ML", "Download"]
    )
    st.sidebar.info(
        "This Application allows us to build an automated ML Pipeline using Streamlit, Pandas and PyCaret. And it's magic!"
    )

    if os.path.exists("Sourcedata.csv"):
        df = pd.read_csv("Sourcedata.csv", index_col=None)
    else:
        df = None

    ## Section1 - Instruction ##
    if choice == "Instruction":
        Intro()

    ## Section2 - Upload ##
    elif choice == "Upload":
        upload_section(df)

    ## Section3 - Profiling dataset ##
    elif choice == "Profiling":
        profiling_section(df)

    ## Section4 - Machine Learning ##
    elif choice == "ML":
        ml_section(df)

    ## Section5 - Download pipeline ##
    elif choice == "Download":
        download_section()


## Section2 - Upload ##
def upload_section(df):
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("Sourcedata.csv", index=None)
        st.dataframe(df)


## Section3 - Profiling dataset ##
def profiling_section(df):
    st.title("Automated Exploratory Data Analysis")
    if df is not None:
        profile_report = df.profile_report()
        st_profile_report(profile_report)
    else:
        st.info("Upload a dataset for profiling.")


## Section4 - Machine Learning ##
def ml_section(df):
    st.title("Machine Learning Go 🤘🤘🤘")
    model = st.radio(
        "Select Your Model",
        [
            "Classification",
            "Regression",
            "Clustering",
            "Anomaly Detection",
        ],
    )

    if model == "Classification":
        classification_subsection(df)
    elif model == "Regression":
        regression_subsection(df)
    elif model == "Clustering":
        clustering_subsection(df)
    elif model == "Anomaly Detection":
        AnomalyDetection_subsection(df)

    # Add other model subsections here


## for CLASSIFICATION
def classification_subsection(df):
    st.subheader("Classification")
    chosen_target = st.selectbox("Select Your Target for Classification", df.columns)
    if st.button("Train Model"):
        st.info("This is dataset")
        st.dataframe(df.head())

        classification_setup(df, target=chosen_target)
        setup_df = classification_pull()
        st.info("This is ML Experiment Settings ")
        st.dataframe(setup_df)

        best_model = classification_compare_model()
        compare_df = classification_pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)

        predict = classification_predict_model(best_model)
        compare_df = classification_pull()
        st.info("This is predicted value of best Model")
        st.dataframe(predict)

        classification_save_model(best_model, "best_model")


## for Regression
def regression_subsection(df):
    st.subheader("Regression")
    chosen_target = st.selectbox("Select Your Target for Regresssion", df.columns)
    if st.button("Train Model"):
        st.info("This is dataset")
        st.dataframe(df.head())

        regression_setup(df, target=chosen_target)
        setup_df = regression_pull()
        st.info("This is ML Experiment Settings ")
        st.dataframe(setup_df)

        best_model = regression_compare_models()
        compare_df = regression_pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)

        predict = regression_predict_model(best_model)
        st.info("This is predicted value of best Model")
        st.dataframe(predict)

        regression_save_model(best_model, "best_model")


## for Clustering
def clustering_subsection(df):
    st.subheader("Clustering")
    st.info("We are using KMean Clustering Method : ")
    if st.button("Train Model"):
        st.info("This is dataset")
        st.dataframe(df.head())
        clustering_steup(df)
        setup_df = clustering_pull()
        st.info("This is ML Experiment Settings ")
        st.dataframe(setup_df)

        kmeans = clustering_create_model("kmeans")
        created_model_df = clustering_pull()
        st.info("This is the ML Kmean Clustering Model")
        st.dataframe(created_model_df)

        predict = clustering_predict_model(kmeans, data=df)
        # predict_df = clustering_pull()
        st.info(
            "generates cluster labels using a trained model to Kmean Clustering Model"
        )
        st.dataframe(predict)
        clustering_save_model(kmeans, "best_model")

        plot_mod = clustering_plot_model(kmeans)
        plot_mod


## for  Anomaly Detection
def AnomalyDetection_subsection(df):
    st.subheader("Anomaly Detection")
    method = st.selectbox(
        "Select Your method",
        [
            "iforest",
            "knn",
            "svm",
            "abod",
            "cluster",
        ],
    )
    if st.button("Train Model"):
        st.info("This is dataset")
        st.dataframe(df.head())
        anomaly_setup(df, session_id=123)
        setup_df = anomaly_pull()
        st.info("This is ML Experiment Settings ")
        st.dataframe(setup_df)

        iforest = anomaly_create_model(method)  # we use iforest model
        models = anomaly_models()
        st.info("We use these models for anomaly detection ! ")
        st.dataframe(models)

        st.info(
            "predict anomaly labels to the dataset for a given model. (1 = outlier, 0 = inlier)"
        )
        predictions = anomaly_assign_model(iforest)
        st.dataframe(predictions)

        anomaly_save_model(iforest, "best_model")

        plot_mod = anomaly_plot_model(iforest)
        plot_mod


## Section5 - Download pipeline ##


def download_section():
    st.title("Download trained pipeline!")
    with open("best_model.pkl", "rb") as file:
        st.download_button("Download", file, "trained_model.pkl")

    st.markdown("""""")
    st.markdown("""""")
    st.markdown("""""")
    st.markdown("""""")
    st.markdown("""""")
    st.markdown("""""")
    st.markdown("""""")
    st.markdown("""""")
    st.markdown("""""")

    with st.container():
        col = st.columns([1])
        # HARSH #
        with col[0]:
            st.write(
                "This app was created by [Harsh Narayan](https://portfolio-7bwl.onrender.com/)"
            )


if __name__ == "__main__":
    main()
