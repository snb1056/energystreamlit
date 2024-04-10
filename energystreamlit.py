import streamlit as st
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib
import plotly.graph_objs as go
import sklearn.metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

df=pd.read_csv("Union.csv")
df = df.drop(["Unnamed: 0"], axis=1)
df_temp = pd.read_csv("temperature-regional.csv", sep = ";")
df_energy = pd.read_csv("energy2802.csv", sep = ";")
loaded_model = joblib.load("random_forest_model.pkl")

st.markdown("<h1 style='text-align: center;'>Energy Consumption Regression Modeling Project</h1>", unsafe_allow_html=True)

st.sidebar.title("Table of contents")
pages=["Introduction","Data Exploration", "Data Vizualisation","Modelling", "Prediction", "Conclusion"]
page=st.sidebar.radio("Go to", pages)

if page == pages[0] : 
    image = 'renewable-energy.png'
    col1, col2, col3 = st.columns([1, 2, 1])  
    with col2:
        st.image(image, caption='Energy Project', width=200)  

    st.markdown("""
    <div style='background-color: #cce5ff; padding: 10px; border-radius: 5px; text-align: center;'>
        <b>Training course: Data Analyst</b><br>
        <b>February 2024</b><br><br>
        <b>PROJECT MENTOR</b><br>
        Tarik ANOUAR<br><br>
        <b>PROJECT TEAM</b><br>
        Swetha Neelakatantan BANUVARAM <a href="https://www.linkedin.com/in/snb-5/">LinkedIn</a><br>
        Abdulla MURADOV <a href="https://www.linkedin.com/in/abumrd/#/">LinkedIn</a><br>
        Rathina Sankari NAVAJEEVAN <a href="https://www.linkedin.com/in/rathinasankari/">LinkedIn</a><br>
        Natalja SEMEROW <a href="https://www.linkedin.com/in/natascha-semerow-05610666">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)
    # Add line breaks using Markdown syntax
    st.write("")
    st.write("")
    st.markdown("<h3 style='text-align: center;'> INTRODUCTION </h1>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: blue; font-size: 18px;'>Problem Statement</h3>", unsafe_allow_html=True)
    st.write("\nUse the machine learning approach to predict energy consumption for future years and thus know the required energy production at the national level and at the departmental level. This can minimize the risk of a blackout.")
    st.markdown("""
    The main objectives of this project are:
    - Forecast of energy consumption in France at a national level & at a department level
    - Analysis of energy production by energy type:
        - Solar
        - Wind
        - Thermal
        - Hydropower
        - Nuclear
        - Bioenergy
    
    """)
    st.write("""
    <div style='background-color: #cce5ff; padding: 10px; border-radius: 5px; text-align: center;'>
        <b>Data sources:</b><br>             
        Temperature <a href="https://meteo.data.gouv.fr/">Meteo</a><br>
        Energy <a href="https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-cons-def/information/?disjunctive.libelle_region&disjunctive.nature&sort=-date_heure">ODRE</a><br>
    </div>""", unsafe_allow_html=True)

    

if page == pages[1] : 

    st.markdown("<h3 style='text-align: center;'>🔎 DATA EXPLORATION 🔍</h1>", unsafe_allow_html=True)

    ### Codes from Natascha for Exploration ###

    st.markdown("<h3 style='text-align: center;'>Presentation of data</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #cce5ff; padding: 10px; border-radius: 5px; text-align: center;'>
      <b>Energy Dataset</b><br>
    </div>
    """, unsafe_allow_html=True)
    st.write("The first lines of the Energy Dataset:")
    st.dataframe(df_energy.head())
    st.markdown("**Main Information:**")
    st.write("Number of entries: 1,980,288")
    st.write("Number of columns: 32")

  #st.dataframe(df_energy.info())

    if st.checkbox("Show NA of Energy Dataset (in %)") :
        st.dataframe(df_energy.isna().sum()/(len(df_energy))*100)
  
  #st.dataframe(df_energy.describe())

    st.markdown("""
      <div style='background-color: #cce5ff; padding: 10px; border-radius: 5px; text-align: center;'>
        <b>Temperature Dataset</b><br>
      </div>
      """, unsafe_allow_html=True)

    st.write("The first lines of the Temperature Dataset:")
    st.dataframe(df_temp.head())
    st.markdown("**Main Information:**")
    st.write("Number of entries: 36,712")
    st.write("Number of columns: 7")

    if st.button('Show information on merged dataset'):
        st.write("\nIn order to achieve our goal to predict consumption for different regions in France, we have merged these two datasets and cleaned it to apply machine learning on it:")
        st.markdown("""
    - Deleted unnecessary columns and rows
    - Translation in English for a better understanding
    - Addition of temperature dates for 2013-2016 (median of the temperature grouped by region and week)
    - Encoding and transforming all data to numerical
    """)

    st.markdown("""
        <div style='background-color: #cce5ff; padding: 10px; border-radius: 5px; text-align: center;'>
          <b>Merged Dataset</b><br>
        </div>
        """, unsafe_allow_html=True)
    st.dataframe(df.head())
    st.markdown("**Main Information:**")
    st.write("Number of entries: 751,905")
    st.write("Number of columns: 28")

    #st.dataframe(df.info())

    #if st.checkbox("Show NA of Merged Dataset (in %)") :
    st.write("Missing values of merged Dataset:")
    st.dataframe(df.isna().sum()/(len(df))*100)

    #Statistical Information added from Swetha #
    st.markdown("""
        <div style='background-color: #cce5ff; padding: 10px; border-radius: 5px; text-align: center;'>
          <b>Statistical Information of Merged Dataset</b><br>
        </div>
        """, unsafe_allow_html=True)
    st.dataframe(df.describe())


  
if page == pages[2] : 
    st.markdown("<h3 style='text-align: center;'>📊 DATA VISUALIZATION 📊</h1>", unsafe_allow_html=True)

    # Data viz swetha's part

    st.image("Yearlyconsumption.png", use_column_width=True)
    st.markdown("<div style='text-align: center;'><b>Fig 1: Regular Distribution of Energy Consumption Across Years</b></div>", unsafe_allow_html=True)
    st.markdown(""" The plot exhibits peaks and lows of energy consumption projecting a non-uniform distribution.""")
    st.write("")
    st.write("")

    st.image("Seasonsregionsyears.png", use_column_width=True)
    st.markdown("<div style='text-align: center;'><b>Fig 2: Seasonwise Energy Consumption Across Regions and Years</b></div>", unsafe_allow_html=True)
    st.markdown("""Ile-de-France and Auvergne-Rhône-Alpes regions are high energy consumers throughout the given time period yearly (2013-2022). 
                Additionally, it points out that the consumption seems to be highest in Winter followed by Spring season and then Autumn season and then least in Summer season.
             """)
    st.write("")
    st.write("")


    st.image("Corrmax.png", use_column_width=True)
    st.markdown("<div style='text-align: center;'><b>Fig 3: Correlation Matrix</b></div>", unsafe_allow_html=True)
    st.markdown("The above correlation matrix displays both positive and negative linear relationships among energy consumption and different explanatory variables presenting more than one linear relationship. This confirms from earlier analysis that energy consumption goes higher during the winter months compared to other months of the year.\n"
                "\nOverall for the seasons the coefficient 0.42 indicates seemingly strong correlation.")
    st.write("")
    st.write("")


    st.image("Hypothesis1.png", use_column_width=True)
    st.markdown("<div style='text-align: center;'><b> Fig 4: Correlation of Energy Consumption and Temperature </b></div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<u><b>Hypothesis : Temperature Vs Energy Consumption.</b></u>", unsafe_allow_html=True)
    st.write("**H0: Temperature is inversely correlated with energy consumption**\n"
                "\n**H1: Temperature is directly correlated with energy consumption.**")
    st.write("The values plotted are for temperature, an explanatory variable and energy consumption, a target variable. From the above plot, it is observed as the temperature reduces, the energy consumption increases and vice-versa. Comprehensively it can be interpretted that both the variables are inversely correlated.")


    #Rathina plots for data viz 
    st.image("Distribution of Energy Sources Year-on-Year.jpg", use_column_width=True)
    st.markdown("<div style='text-align: center;'><b><u>Fig 5: Distribution of Energy Sources Year-on-Year</b></u></div>", unsafe_allow_html=True)
    st.write("")
    st.write("The above graph shows the production of energy across different sectors in France from 2013 to 2021.\n" 
        "It is evident Nuclear energy is the highest contributor of the production of energy followed by Hydraulique_Pompage, Thermique, Eolien and Solaire.\n")
    st.write("")


    st.image ("Pie chart Renewable and Non Renewable.png", width=500 )
    st.write("")
    st.markdown("<div style='text-align: center;'><b><u>Fig 6: Distribution of Renewable and Non-Renewable Energy Sources</b></u></div>", unsafe_allow_html=True)
    st.write("")
    st.write("The above pie chart shows the distribution of energy production between renewable (Solaire, Eolien, Hydraulique_Pompage, Bioénergies) and non-renewable (Nucléaire, Thermique) sources.")
    st.write("")
    st.write("")

    st.image ("Distribution of Renewable Energy Sources.png", width=500 )
    st.markdown("<div style='text-align: center;'><b><u>Fig 7: Distribution of Renewable Energy Sources</b></u></div>", unsafe_allow_html=True)
    st.write("")
    st.write("The above pie chart shows the distribution of energy production between renewable (Solaire, Eolien, Hydraulique_Pompage, Bioénergies) and non-renewable (Nucléaire, Thermique) sources.")
    st.write("")
    st.write("")

    st.markdown("<u><b>P-value calculation using Shapiro-Wilk Test.</b></u>", unsafe_allow_html=True)
    st.write("**H0: Consumption of Energy is normally distributed across the years**\n"
         "\n**H1: Consumption of Energy is not normally distributed across the years**")
    st.write("**Shapiro-Wilk Test Result:**")
    st.write("**Statistic: 0.975**")
    st.write("**P-value: 0.0**")  
    st.image ("QQ Plot.png", width=500 )
    st.markdown("<div style='text-align: center;'><b>Fig 8: Q-Q Plot</b></div>", unsafe_allow_html=True)
    st.write("")
    st.write("")

    st.image ("KDE Plot.png", width=500 )
    st.markdown("<div style='text-align: center;'><b>Fig 9: Kernel Density Estimation</b></div>", unsafe_allow_html=True)


if page == pages[3] : 

    st.markdown("<h3 style='text-align: center;'>🧩 MODELLING 🧩</h1>", unsafe_allow_html=True)

    st.markdown("""
        <div style='background-color: #cce5ff; padding: 10px; border-radius: 5px; text-align: center;'>
          <b>Summary of Machine Learning Models Tested</b><br></div>""", unsafe_allow_html=True)
    
    st.image("SummaryMLmodels.png", use_column_width=True)

    
  # Reading dataset
  
    df = df.drop(["Code INSEE region"], axis=1)
    feats = df.drop("Consumption", axis=1)
    target = df["Consumption"]
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state=42)
  

    loaded_model = joblib.load("random_forest_model.pkl")
 
# Generate r2 scores and predictions and metrics

    y_pred_train = loaded_model.predict(X_train)
    y_pred_test = loaded_model.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = mean_squared_error(y_train,y_pred_train,squared=False)
    r2_train = r2_score(y_train, y_pred_train)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test,y_pred_test)
    rmse_test = mean_squared_error(y_test,y_pred_test,squared=False)
    r2_test = r2_score(y_test, y_pred_test)

# Creation of a dataframe to compare the metrics of the two algorithms
    model_metrics = {
    'R2': {'Train Values': r2_train, 'Test Values': r2_test},
    'MAE': {'Train Values': mae_train, 'Test Values': mae_test},
    'MSE': {'Train Values': mse_train, 'Test Values': mse_test},
    'RMSE': {'Train Values': rmse_train, 'Test Values': rmse_test}}

# Convert the dictionary to a DataFrame
    df_metrics = pd.DataFrame(model_metrics)

    # Transpose the DataFrame
    df_metrics = df_metrics.T

    # Display the metrics for the selected model vertically
    st.markdown("""
      <div style='background-color: #cce5ff; padding: 10px; border-radius: 5px; text-align: center;'>
        <b>Random Forest Regressor Model Evaluation Metrics</b><br></div>""", unsafe_allow_html=True)
    st.write(df_metrics)

    # Define the parameters
    parameters = {
    'min_samples_leaf': [1, 4, 20],
    'min_samples_split': ['None', 2, 20],
    'max_depth': ['None', 15, 30],
    'Test size': [0.02, 0.25, 0.3]}

# Convert the dictionary to a DataFrame
    df_parameters = pd.DataFrame(parameters, index=['Min', 'Best_Parameters', 'Max'])

# Display the parameters as a markdown table
    st.markdown(""" 
- **23 different versions of the Random Forest Regressor Model experiments were conducted. The above results are for the best version (V7) with the following parameters:**""")
    st.table(df_parameters)

# Display each plot for Random Forest Regressor with a caption
  
    st.image('RFRresidualplots.png')
    st.markdown(""" 
- **Fig 1: Random Forest Regressor Residual dispersion, Histogram, Actual Vs. Predicted and Q-Qplot:**
  - The Q-Q plot for test residuals suggests deviations from the normal distribution (diagonal line); at the tails points the presence of significant outliers.
                Although the model performs rather well for most data this variance across the tails indicates heteroscedasticity challenges with prediction of target variable.""")
    st.write("")
    st.write("")
    

    st.image('RFRLearningcurve.png')
    st.markdown("""
- **Fig 2: Random Forest Regressor Learning Curve:**
  - The convergence of training and validation errors suggests adding more data doesnot significantly improve the model's performance.
  - The error gap hints for an almost perfect-fit model opening up insights for tweaking the parameters to reduce bias.""") 

    st.write("")
    st.write("")

    st.image('RFRFeatureImp.png')      
    st.markdown("""
- **Fig 3: Feature Importances from Best Model of Random Forest Regressor Experiments:**
  - It has been consider to dropping low-importance features to reduce complexity and potentially improve model's predictability. """)
    st.write("")
    st.write("")



    #RFRPredictedVsActualTest

if page == pages[4] : 
#Interface utilisateur Streamlit
  st.markdown("<h3 style='text-align: center;'>💡PREDICTION💡</h1>", unsafe_allow_html=True)

  # Load the Union.csv dataset
  df = pd.read_csv("Union.csv")

  # Interface utilisateur Streamlit

  st.markdown("""
        <div style='background-color: #cce5ff; padding: 10px; border-radius: 5px; text-align: center;'>
          <b>Prediction Simulation of Random Forest Regressor</b><br></div>""", unsafe_allow_html=True)
  



  loaded_model_new = joblib.load("new_random_forest_model.pkl")


#Selection of Regions
  regions = ["Region_Auvergne-Rhône-Alpes", "Region_Centre-Val de Loire", "Region_Normandie", "Region_Nouvelle-Aquitaine", "Region_Grand Est","Region_Hauts-de-France", "Region_Occitanie"]
  region_values_list = [0,0,0,0,0,0,0]

  # Get the selected region from the user
  selected_region = st.selectbox('Select a region:', regions)

# Set the value of the selected region to 1
  region_index = regions.index(selected_region)
  region_values_list[region_index] = 1

  # Add widgets for feature entry
  year = st.slider("Year", 2013, 2025, 2016)
  month = st.slider("Month", 1,12,6)
  day = st.slider("Day", 1,31, 15)
  hour =st.slider("Hour",0,23,12)

  #Feature preprocessing with StandardScaler
  characteristics = np.array([[year, month,day,hour]+region_values_list])


  #Predict class with template
  predicted_consumption = loaded_model_new.predict(characteristics)
  predicted_consumption = np.round(predicted_consumption, 3)

   # Display the predicted consumption
  st.markdown(f"<p style='font-size:24px; font-weight:bold;'>The Predicted Consumption is : {predicted_consumption[0]}</p>", unsafe_allow_html=True)

    # Get the target consumption from the Union.csv dataset
  target_df = df[(df["Year"] == year) & 
                (df["Month"] == month) & 
                (df["Day"] == day) & 
                (df["hour"] == hour) &
                (df[selected_region] == 1)]
  if not target_df.empty:
      target_consumption = target_df["Consumption"].values[0]
    # Display the target consumption
      st.markdown(f"<p style='font-size:24px; font-weight:bold;'>The Target Consumption is : {target_consumption}</p>", unsafe_allow_html=True)

    # Calculate the absolute error between predicted and target consumption
      absolute_error = abs(predicted_consumption[0] - target_consumption)

      # Round the absolute error to three decimal points
      absolute_error_rounded = round(absolute_error, 3)

    # Display the absolute error
      st.markdown(f"<p style='font-size:24px; font-weight:bold;'>The Absolute Error is : {absolute_error_rounded}</p>", unsafe_allow_html=True)
  else:
      st.markdown("<p style='font-size:24px; font-weight:bold;'>No matching data found in the Union.csv dataset for the selected features.</p>", unsafe_allow_html=True)

if page == pages[5] :
    st.markdown("<h3 style='text-align: center;'>🏁CONCLUSION🏁</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background-color: #cce5ff; padding: 10px; border-radius: 5px; text-align: center;'>
          <b>XG Boost Model Analysis and Improvement Strategies</b><br></div>""", unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: blue; font-size: 18px;'>Model Improvement Opportunities</h3>", unsafe_allow_html=True)

# Conclusion and Model Improvement Opportunities Section from Abdulla 
# Display each image with a caption

    st.image('ResidualActualvsPredictedxgboost.png')
    st.markdown("""
- **Residuals vs. Predicted Values:**
  - Residuals clustered around the horizontal axis indicating accuracy but also suggest potential overfitting at higher values.""")
    st.write("")
    st.write("")
  
    st.image('qqplotxgboost.png')
    st.markdown(""" 
- **Normal Q-Q Plot of Residuals:**
  - Deviation from the line in tails points to the non-normality of residuals and hints at possible improvements in distribution assumptions.""")
    st.write("")
    st.write("")

    st.image('boxplotxgboost.png')
    st.markdown(""" 
- **Box Plot of Residuals:**
  - The presence of outliers and a wide interquartile range indicates a possibility of skewness in residuals, calling for refinement.""")
    
    st.write("")
    st.write("")
    

    st.image('LearningCurveXgboost.png')
    st.markdown("""
- **Learning Curve:**
  - The convergence of training and validation errors suggests adding more data may not significantly improve the model's performance.
  - The error gap hints at a well-fit model but also points to an opportunity for slight adjustments to reduce bias.""") 

    st.write("")
    st.write("")

    st.image('featureimpxgboost.png')
                
    st.markdown("""
- **Feature Importances from Best Model:**
  - We will consider dropping low-importance features to reduce complexity and potentially improve model generalizability. """)
    st.write("")
    st.write("")


# Display bullet points for the conclusion section
    st.markdown("<h3 style='color: blue; font-size: 18px;'>Conclusion and Path Forward</h3>", unsafe_allow_html=True)

    st.markdown("""
Overall, our models demonstrate a commendable fit to the dataset, marking a solid start for our initial project. 
Notwithstanding the current success, we recognize areas where further enhancements can be made:

- **Feature Analysis:**
  - A deeper dive into feature importance and engineering could unlock more predictive power and model precision.
           
- **Outlier Management:**
  - The boxplots reveal a significant presence of outliers. A thoughtful approach to outlier removal could lead to performance gains.
                
- **Model Assumptions:**
  - Initial attempts with linear models showed promise; however, residual plots revealed heteroscedasticity, guiding us towards models better suited for our data.

Continued model refinement and addressing these insights will form the cornerstone of our next phase in the project.
""")