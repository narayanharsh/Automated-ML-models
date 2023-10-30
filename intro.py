import streamlit as st


# once we excuted this bottom code one time then it will cash it and it is available for next time again
def Intro():
    st.markdown(
        '<div style=" margin-top: 40px; color:black; font-size: 2.1rem;font-weight: 400;text-align: center; border-radius: 15px 10vw; background:yellow;">Instructions for using this web app!</div>',
        unsafe_allow_html=True,
    )
    # st.header(":blue[Instructions for using this web app!]")

    # st.subheader( " Introduction page is in process :blue[colors] and emojis :sunglasses: ")

    # st.divider()

    st.markdown(
        '<div style=" height:auto; width: 100%; padding: 1rem 8rem ; margin-top:2rem; font-size:20px; color:black; border-radius:1.5rem ;background:white;">  This Web page is about automated Machine Learninig stuff , where we create machine learning pipeline. <br> Here we use Streamlit framework to develop an interactive web application  for automated machine learning (AutoML) tasks. <br>The application consists of several sections: <br><br> &#9830 <strong>Upload</strong> :  Allows users to upload their dataset for further analysis and modeling. <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;    <mark>Note</mark> : Uploaded dataset should be in CSV format and Pre-processed. <br> <br> &#9830 <strong>Profiling</strong> : Conducts automated exploratory data analysis (EDA) using ydata Profiling, generating detailed reports on the uploaded dataset. <br> <br> &#9830 <strong>Machine Learning(ML)</strong> : Offers various AutoML capabilities for different types of tasks: <br><br> &nbsp;  &nbsp; &nbsp; &#8226; <strong>Classification</strong> : Enables users to select a target variable for classification tasks, train classification models, and displays model performance metrics. <br><br> &nbsp;  &nbsp; &nbsp; &#8226; <strong>Regression</strong> : Allows users to choose a target variable for regression tasks, trains regression models, and presents model evaluation results. <br><br> &nbsp;  &nbsp; &nbsp; &#8226; <strong>Clustering</strong> : Utilizes K-Means clustering for unsupervised clustering tasks, provides model training, and displays clustering results. <br><br> &nbsp;  &nbsp; &nbsp; &#8226; <strong>Anomaly Detection</strong> : Supports various anomaly detection methods, such as Isolation Forest and K-Nearest Neighbors, to identify outliers in the data.  <br><br> &#9830 <strong>Download</strong> : Provides the option to download the trained ML pipeline as a Pickle file.</div>',
        unsafe_allow_html=True,
    )
    # for bullet point
    # for white space &nbsp;
    # for bullet point use this diamond &#9830;

    st.markdown(
        '<div style=" margin-top: 40px; color:black; font-size: 2.1rem;font-weight: 400;text-align: center; border-radius: 15px 10vw; background:yellow;">Methods of Machine Learning! </div>',
        unsafe_allow_html=True,
    )

    # classification

    st.markdown(
        '<div style="margin-top: 40px; margin-left: 40px; color:black; font-size: 1.8rem;font-weight: 300;text-align: left;">🐥Classification : </div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style=" height:auto; width: 100%; padding: 1rem 8rem ; margin-top:2rem; font-size:20px; color:black; border-radius:1.5rem ;background:white;"> PyCaret’s Classification Module is a supervised machine learning module that is used for classifying elements into groups. <br><br> The goal is to predict the categorical class labels which are discrete and unordered. Some common use cases include predicting customer default (Yes or No), predicting customer churn (customer will leave or stay), the disease found (positive or negative). </div>',
        unsafe_allow_html=True,
    )

    # regression

    st.markdown(
        '<div style="margin-top: 40px; margin-left: 40px; color:black; font-size: 1.8rem;font-weight: 300;text-align: left;">🐥Regression : </div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style=" height:auto; width: 100%; padding: 1rem 8rem ; margin-top:2rem; font-size:20px; color:black; border-radius:1.5rem ;background:white;"> PyCaret’s Regression Module is a supervised machine learning module that is used for estimating the relationships between a dependent variable (often called the ‘outcome variable’, or ‘target’) and one or more independent variables (often called ‘features’, ‘predictors’, or ‘covariates’). <br><br> The objective of regression is to predict continuous values such as predicting sales amount, predicting quantity, predicting temperature, etc.  </div>',
        unsafe_allow_html=True,
    )

    # Clustering

    st.markdown(
        '<div style="margin-top: 40px; margin-left: 40px; color:black; font-size: 1.8rem;font-weight: 300;text-align: left;">🐥Clustering : </div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style=" height:auto; width: 100%; padding: 1rem 8rem ; margin-top:2rem; font-size:20px; color:black; border-radius:1.5rem ;background:white;"> PyCaret’s Clustering Module is an unsupervised machine learning module that performs the task of grouping a set of objects in such a way that objects in the same group (also known as a cluster) are more similar to each other than to those in other groups. <br><br> <mark>Note</mark> : Clustering is somewhere similar to the classification algorithm, but the difference is the type of dataset that we are using. In classification, we work with the labeled data set, whereas in clustering, we work with the unlabelled dataset.</div>',
        unsafe_allow_html=True,
    )

    # Anomaly Detection

    st.markdown(
        '<div style="margin-top: 40px; margin-left: 40px; color:black; font-size: 1.8rem;font-weight: 300;text-align: left;">🐥Anomaly Detection : </div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style=" height:auto; width: 100%; padding: 1rem 8rem ; margin-top:2rem; font-size:20px; color:black; border-radius:1.5rem ;background:white;"> PyCaret’s Anomaly Detection Module is an unsupervised machine learning module that is used for identifying rare items, events, or observations that raise suspicions by differing significantly from the majority of the data. <br><br>Typically, the anomalous items will translate to some kind of problems such as bank fraud, a structural defect, medical problems, or errors.<br><br> For a dataset having all the feature gaussian in nature, then the statistical approach can be generalized by defining an elliptical hypersphere that covers most of the regular data points, and the data points that lie away from the hypersphere can be considered as anomalies. </div>',
        unsafe_allow_html=True,
    )

    # Time Series

    st.markdown(
        '<div style="margin-top: 40px; margin-left: 40px; color:black; font-size: 1.8rem;font-weight: 300;text-align: left;">🐥Time Series : </div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style=" height:auto; width: 100%; padding: 1rem 8rem ; margin-top:2rem; font-size:20px; color:black; border-radius:1.5rem ;background:white;"> PyCaret Time Series module is a powerful tool for analyzing and predicting time series data using machine learning and classical statistical techniques. This module enables users to easily perform complex time series forecasting tasks by automating the entire process from data preparation to model deployment. <br><br> PyCaret Time Series Forecasting module supports a wide range of forecasting methods such as ARIMA, Prophet, and LSTM. It also provides various features to handle missing values, time series decomposition, and data visualizations.<br><br>Time series analysis is used for non-stationary data—things that are constantly fluctuating over time or are affected by time. Industries like finance, retail, and economics frequently use time series analysis because currency and sales are always changing.</div>',
        unsafe_allow_html=True,
    )

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
