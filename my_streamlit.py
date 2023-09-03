# python -m streamlit run my_streamlit.py
import streamlit as st
import pandas as pd
import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import hydralit_components as hc 
import streamlit as st
import base64
import matplotlib.pyplot as plt
import datetime as dt
import math
import joblib
import numpy as np
import statsmodels.api as sm
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Healthcare",
    page_icon= "health_icon.png",
    layout='wide'
)

visits = pd.read_csv("data\Visits.csv")
# diseases = pd.read_csv("data\Diseases.csv")
consultations = pd.read_csv("data\Consultations.csv")
patients = pd.read_csv("data\Patients.csv")
# appointments = pd.read_csv("data\Appointments.csv")
# diagnoses = pd.read_csv("data\Diagnoses.csv")
# LabTests = pd.read_csv("data\LabTests.csv")


#Creating Navigation bar
menu_data = [
    {'label': "Overview", 'icon': '🩺'},
    {'label': "Patients", 'icon': '🩺'},
    {'label': 'Analysis', 'icon': '👩‍⚕️'},
    {'label': 'Machine Learning', 'icon': '📈'}
]
menu_id = hc.nav_bar(menu_definition=menu_data, sticky_mode='sticky', override_theme={ 'menu_background': '#840032', 'option_active': 'white'}
)

if menu_id =="Overview":
    st.text(" ")
    st.markdown("### General Overview and Statistics about the Clinic", unsafe_allow_html=True)
    st.text(" ")

    time_period = st.sidebar.radio("Select Time Period", ['All',"Today", "Last Week", "Last Month", "2023", "2022", "2021", "2020"])
    # Filter the data based on the selected time period
    consultations["VisitDate"] = pd.to_datetime(consultations["VisitDate"])
    visits["VisitDate"] = pd.to_datetime(visits["VisitDate"])
    patients["RegistrationDate"] = pd.to_datetime(patients["RegistrationDate"])
    today = visits['VisitDate'].max().date()

    if time_period == "All":
        consultations_1 = consultations[consultations['VisitDate'].dt.year == 2022]
        consultations_2 = consultations[consultations['VisitDate'].dt.year == 2023]
        waiting_time_1 = consultations_1['WaitingTime'].mean()
        waiting_time_2 = consultations_2['WaitingTime'].mean()
    elif time_period == "Today":
        last_day_start = today.replace(day=today.day - 1)
        last_day_end = visits['VisitDate'].max().date()
        consultations_1 = consultations[consultations['VisitDate'].dt.date == (last_day_end - timedelta(days=1))]
        consultations_2 = consultations[consultations['VisitDate'].dt.date == last_day_end]
        waiting_time_1 = consultations_1['WaitingTime'].mean()
        waiting_time_2 = consultations_2['WaitingTime'].mean()
        patients = patients[patients['RegistrationDate'].dt.date == today]
        visits = visits[visits['VisitDate'].dt.date == today]
        consultations = consultations[consultations['VisitDate'].dt.date == today]
    elif time_period == "Last Week":
        last_week_start = today.replace(day=today.day - 7)
        last_week_end = today
        patients = patients[(patients['RegistrationDate'].dt.date >= last_week_start) & (patients['RegistrationDate'].dt.date <= last_week_end)]
        visits = visits[(visits['VisitDate'].dt.date >= last_week_start) & (visits['VisitDate'].dt.date <= last_week_end)]
        consultations = consultations[(consultations['VisitDate'].dt.date >= last_week_start) & (consultations['VisitDate'].dt.date <= last_week_end)]
        consultations_1 = consultations[consultations['VisitDate'].dt.date == (today - timedelta(days=7))]
        consultations_2 = consultations[consultations['VisitDate'].dt.date == today]
        waiting_time_1 = consultations_1['WaitingTime'].mean()
        waiting_time_2 = consultations_2['WaitingTime'].mean()
    elif time_period == "Last Month":
        last_month_start = today.replace(day=1, month=today.month - 1)
        last_month_end = today
        patients = patients[(patients['RegistrationDate'].dt.date >= last_month_start)  & (patients['RegistrationDate'].dt.date <= last_month_end)]
        visits = visits[(visits['VisitDate'].dt.date >= last_month_start) & (visits['VisitDate'].dt.date <= last_month_end)]
        consultations = consultations[(consultations['VisitDate'].dt.date >= last_month_start) & (consultations['VisitDate'].dt.date <= last_month_end)]
        consultations_1 = consultations[consultations['VisitDate'].dt.date == (today - timedelta(days=30))]
        consultations_2 = consultations[consultations['VisitDate'].dt.date == today]
        waiting_time_1 = consultations_1['WaitingTime'].mean()
        waiting_time_2 = consultations_2['WaitingTime'].mean()
    elif time_period in ["2023", "2022", "2021", "2020", "2019"]:
        selected_year = int(time_period)
        patients = patients[patients['RegistrationDate'].dt.year == selected_year]
        visits = visits[visits['VisitDate'].dt.year == selected_year]
        consultations = consultations[consultations['VisitDate'].dt.year == selected_year]
        if time_period == "2023":
            consultations_1 = consultations[consultations['VisitDate'].dt.year == 2022]
            consultations_2 = consultations[consultations['VisitDate'].dt.year == 2023]
            waiting_time_1 = consultations_1['WaitingTime'].mean()
            waiting_time_2 = consultations_2['WaitingTime'].mean()
        elif time_period == "2022":
            consultations_1 = consultations[consultations['VisitDate'].dt.year == 2021]
            consultations_2 = consultations[consultations['VisitDate'].dt.year == 2022]
            waiting_time_1 = consultations_1['WaitingTime'].mean()
            waiting_time_2 = consultations_2['WaitingTime'].mean()
        elif time_period == "2021":
            consultations_1 = consultations[consultations['VisitDate'].dt.year == 2020]
            consultations_2 = consultations[consultations['VisitDate'].dt.year == 2021]
            waiting_time_1 = consultations_1['WaitingTime'].mean()
            waiting_time_2 = consultations_2['WaitingTime'].mean()
        elif time_period == "2020":
            consultations_1 = consultations[consultations['VisitDate'].dt.year == 2019]
            consultations_2 = consultations[consultations['VisitDate'].dt.year == 2020]
            waiting_time_1 = consultations_1['WaitingTime'].mean()
            waiting_time_2 = consultations_2['WaitingTime'].mean()

    # Calculate the totals
    patient_visit_counts = visits["PatientID"].value_counts()
    patients_visited_more_than_once = patient_visit_counts[patient_visit_counts > 1]
    total_patients = patients.shape[0]
    total_visits = visits.shape[0]
    total_appointments = consultations.shape[0]
    average_waiting_time = round(sum(consultations['WaitingTime'])/(consultations.shape[0]),2)
    average_age_group = round(sum(patients['Age'])/(patients.shape[0]),1)
    WaitingTime_percent = ((waiting_time_1 - waiting_time_2) / waiting_time_2) * 100
    visits_frequency = 3
    # sum(patients['NumberVisits']) / patients.shape[0]
    # Style for the cards
    card_style = "box-shadow: 2px 2px 5px grey; padding: 15px; border-radius: 10px; text-align: center;"
    #  ['#840032', '#8F263B', '#9A3A45', '#A54E4E', '#B06258', '#BB7662', '#C68A6B', '#D19E75', '#DCA27E', '#E7B688', '#F2CA91', '#FFDE9B', '#b7b7b7']
    cc = st.columns(3)
    with cc[0]:
        st.write( f"<div style='{card_style} background-color: #A54E4E; color: white;'>" f"<h3>Total Patients</h3>" f"<p style='font-weight: bold; font-size: 25px;'>{total_patients}</p>" f"</div>", unsafe_allow_html=True )
    with cc[1]:
        st.write( f"<div style='{card_style} background-color: #A54E4E; color: white;'>" f"<h3>Total Visits</h3>" f"<p style='font-weight: bold; font-size: 25px;'>{total_visits}</p>" f"</div>", unsafe_allow_html=True)
    with cc[2]:
        st.write( f"<div style='{card_style} background-color: #A54E4E; color: white;'>" f"<h3>Total Appointments</h3>" f"<p style='font-weight: bold; font-size: 25px;'>{total_appointments}</p>" f"</div>", unsafe_allow_html=True )

    st.text(" ")
    st.text(" ")

    cc = st.columns(4)
    with cc[0]:
        st.write( f"<div style='{card_style} background-color: #C68A6B; color: white;'>" f"<h3>Average Waiting Time</h3>" f"<p style='font-weight: bold; font-size: 25px;'>{average_waiting_time}</p>" f"</div>", unsafe_allow_html=True )
    with cc[1]:
        st.write( f"<div style='{card_style} background-color: #C68A6B; color: white;'>" f"<h3>Average Age Group</h3>" f"<p style='font-weight: bold; font-size: 25px;'>{average_age_group}</p>" f"</div>", unsafe_allow_html=True)
    with cc[2]:
        st.write( f"<div style='{card_style} background-color: #C68A6B; color: white;'>" f"<h3>Reappointments Rate</h3>" f"<p style='font-weight: bold; font-size: 25px;'>{total_appointments}</p>" f"</div>", unsafe_allow_html=True )
    with cc[3]:
        st.write( f"<div style='{card_style} background-color: #C68A6B; color: white;'>" f"<h3>Average Visits Frequently</h3>" f"<p style='font-weight: bold; font-size: 25px;'>{visits_frequency}</p>" f"</div>", unsafe_allow_html=True )
    st.text(" ")
    st.text(" ")

    theme_bad = {'bgcolor':'#FFF0F0','content_color':'darkred','progress_color':'darkred'}
    theme_neutral = {'bgcolor': '#3A9BCD','title_color': 'darkblue','content_color': 'darkblue','icon_color': 'darkblue', 'icon': 'fa fa-question-circle'}
    theme_good = {'bgcolor': '#EFF8F7','title_color': 'green','content_color': 'green','icon_color': 'green', 'icon': 'fa fa-check-circle'}

    retention_rate = (visits.shape[0] - consultations.shape[0]) / (visits.shape[0]) * 100
    consultations['VisitDate'] = pd.to_datetime(consultations['VisitDate'])
    WaitingTime_percent = ((waiting_time_1 - waiting_time_2) / waiting_time_2) * 100

    reappointment_counts = visits["PatientID"].value_counts()
    repeated_patient_ids = reappointment_counts[reappointment_counts > 1]
    reappointment = len(repeated_patient_ids)/patients.shape[0] * 100

    refugees_count = len(patients[(patients['Nationality'] == 'Syrian') | (patients['Nationality'] == 'Palestinian')])
    total_patients_count = len(patients)
    refugees_percentage = (refugees_count / total_patients_count) * 100

    cc = st.columns(4)
    with cc[0]:
        hc.info_card(title='Retention Rate', content=f'{retention_rate:.2f}% patients who visited the clinic and left before getting treatment', bar_value=retention_rate, theme_override=theme_bad)
    with cc[1]:
        hc.info_card(title='Waiting Time', content=f'{WaitingTime_percent:.2f}% percent increase in average waiting time compared to last year',bar_value=WaitingTime_percent,theme_override=theme_bad)
    with cc[2]:
        hc.info_card(title='Reappointment',  content=f'{reappointment:.2f}% patients percent visiting the clinic more than once', sentiment='good',bar_value=reappointment)
    with cc[3]:
        hc.info_card(title='Refugees',  content=f'{refugees_percentage:.2f}% patients percent of refugees (Syrian & Palestinian)', sentiment='good',bar_value=refugees_percentage)


    # theme_bad = {'bgcolor':'#FFF0F0','content_color':'darkred','progress_color':'darkred'}
    # theme_neutral = {'bgcolor': '#3A9BCD','title_color': 'darkblue','content_color': 'darkblue','icon_color': 'darkblue', 'icon': 'fa fa-question-circle'}
    # theme_good = {'bgcolor': '#EFF8F7','title_color': 'green','content_color': 'green','icon_color': 'green', 'icon': 'fa fa-check-circle'}

    # cc = st.columns(4)
    # with cc[0]:
    #     st.markdown("<p style='font-size:14px;'>Patient Retention Rate</p>", unsafe_allow_html=True)
    #     hc.info_card(title='', content='39% With No Access', bar_value=39, theme_override=theme_bad)
    # st.empty()
    # with cc[1]:
    #     st.markdown("<p style='font-size:14px;'>Percent of Patients revisiting</p>", unsafe_allow_html=True)
    #     hc.info_card(title='', content='69% With No Access', bar_value=69, theme_override=theme_neutral)
    # st.empty()
    # with cc[2]:
    #     st.markdown("<p style='font-size:14px;'>Total Number of Appointments</p>", unsafe_allow_html=True)
    #     hc.info_card(title='', content='97.81% With Access', sentiment='good', bar_value=98, theme_override=theme_neutral)
    # st.empty()
    # with cc[3]:
    #     st.markdown("<p style='font-size:14px;'>Total Number of Clinic Types</p>", unsafe_allow_html=True)
    #     hc.info_card(title='', content='95% With Access', sentiment='good', bar_value=95)

    # # Calculate relative lengths for bars
    # max_value = max(total_patients, total_visits, total_appointments)
    # bar_length_patients = total_patients / max_value * 100
    # bar_length_visits = total_visits / max_value * 100
    # bar_length_appointments = total_appointments / max_value * 100
    # # Custom styling
    # st.markdown(
    #     """
    #     <style>
    #     .stat-bar {
    #         display: flex;
    #         align-items: center;
    #         margin: 10px;
    #     }
    #     .stat-title {
    #         width: 150px;
    #         font-weight: bold;
    #     }
    #     .stat-bar-inner {
    #         width: 300px;
    #         height: 20px;
    #         background-color: #f5f5f5;
    #         border-radius: 10px;
    #         margin: 0 10px;
    #         display: flex;
    #         align-items: center;
    #         position: relative;
    #     }
    #     .stat-fill {
    #         border-radius: 10px;
    #         height: 100%;
    #         transition: width 0.5s;
    #     }
    #     .stat-number {
    #         position: absolute;
    #         top: 50%;
    #         left: 50%;
    #         transform: translate(-50%, -50%);
    #         color: #ffffff;
    #     }
    #     .patients-fill {
    #         background-color: #009688;
    #     }
    #     .visits-fill {
    #         background-color: #F9A825;
    #     }
    #     .appointments-fill {
    #         background-color: #d32f2f;
    #     }
    #     </style>
    #     """, unsafe_allow_html=True)

    # # Display the statistics using horizontal bars with numbers inside
    # st.write( f"<div class='stat-bar'><div class='stat-title'>Total Patients:</div>" f"<div class='stat-bar-inner'><div class='stat-fill patients-fill' style='width: {bar_length_patients}%;'>" f"<div class='stat-number'>{total_patients}</div></div></div></div>", unsafe_allow_html=True)
    # st.write( f"<div class='stat-bar'><div class='stat-title'>Total Visits:</div>" f"<div class='stat-bar-inner'><div class='stat-fill visits-fill' style='width: {bar_length_visits}%;'>" f"<div class='stat-number'>{total_visits}</div></div></div></div>", unsafe_allow_html=True)
    # st.write( f"<div class='stat-bar'><div class='stat-title'>Total Appointments:</div>" f"<div class='stat-bar-inner'><div class='stat-fill appointments-fill' style='width: {bar_length_appointments}%;'>" f"<div class='stat-number'>{total_appointments}</div></div></div></div>", unsafe_allow_html=True)


if menu_id =="Patients":
    st.text(" ")
    # Define a custom gradient color scale from red to gray
    colors = ['#840032', '#840032', '#A15553', '#B7B7B7', '#b7b7b7']

    # Calculate gender distribution
    gender_counts = patients["Gender"].value_counts()

    # Calculate nationality distribution
    nationality_counts = patients["Nationality"].value_counts()

    area_counts = patients['CurrentArea'].value_counts()

    # Select the top 5 areas and group the rest as "Other"
    top_5_areas = area_counts.head(5)
    other_count = area_counts[5:].sum()
    top_5_areas['Other'] = other_count

    # Create a consistent layout for all pie charts
    layout = go.Layout(
        margin=dict(t=0, b=0, l=0, r=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#333333"),
    )

    # Create a donut chart for gender using Plotly
    fig_gender = px.pie(
        gender_counts,
        values=gender_counts.values,
        names=gender_counts.index,
        hole=0.5,
        color_discrete_sequence= ['#840032', '#b7b7b7'],  # Apply custom gradient colors
        labels={"index": "Gender"}
    )
    fig_gender.update_layout(layout, height=300, width=300)

    # Create a donut chart for nationality using Plotly
    fig_nationality = px.pie(
        nationality_counts,
        values=nationality_counts.values,
        names=nationality_counts.index,
        hole=0.5,
        color_discrete_sequence=['#840032', '#b7b7b7'],  # Apply custom gradient colors
        labels={"index": "Nationality"}
    )
    fig_nationality.update_layout(layout, height=300, width=300)

    # Create a donut chart for areas using Plotly
    fig_area = px.pie(
        top_5_areas,
        values=top_5_areas.values,
        names=top_5_areas.index,
        hole=0.5,
        color_discrete_sequence=colors,  # Apply custom gradient colors
        labels={"index": "Area"}
    )
    fig_area.update_layout(layout, height=300, width=300)

    # Display the charts side by side using Streamlit
    st.write("<h2 style='text-align: center;'>Gender and Nationality Distribution</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.plotly_chart(fig_gender, use_container_width=True)
    col2.plotly_chart(fig_nationality, use_container_width=True)
    col3.plotly_chart(fig_area, use_container_width=True)

    colors =  ['#840032', '#8F263B', '#9A3A45', '#A54E4E', '#B06258', '#BB7662', '#C68A6B', '#D19E75', '#DCA27E', '#E7B688', '#F2CA91', '#FFDE9B', '#b7b7b7']
    # Convert RegistrationDate to datetime and sort
    patients['RegistrationDate'] = pd.to_datetime(patients['RegistrationDate'])
    patients.sort_values('RegistrationDate', inplace=True)

    # Create a line chart for total number of patients over time using Plotly
    fig_total_patients = px.line(
        patients,
        x='RegistrationDate',
        y=patients.index,
        labels={'RegistrationDate': 'Date', 'index': 'Number of Patients'},
        markers=True,
        title='Total Number of Patients Over Time',
        color_discrete_sequence=colors
    )
    fig_total_patients.update_layout(
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_title_font=dict(size=12, color="#333333"),
        yaxis_title_font=dict(size=12, color="#333333"),
        font=dict(family="Arial", size=12, color="#333333"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    # Age range distribution
    age_bins = list(range(0, 101, 10))
    age_labels = [f'{age}-{age+9}' for age in age_bins[:-1]]
    patients['AgeRange'] = pd.cut(patients['Age'], bins=age_bins, labels=age_labels)
    age_range_counts = patients['AgeRange'].value_counts().sort_index()

    # Create a bar chart for age range distribution using Plotly
    fig_age_distribution = px.bar(
        x=age_range_counts.index,
        y=age_range_counts.values,
        color=age_range_counts.index,
        title='Age Range Distribution',
        labels={'x': 'Age Range', 'y': 'Count'},
        color_discrete_sequence=colors
    )
    fig_age_distribution.update_layout(
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_title_font=dict(size=12, color="#333333"),
        yaxis_title_font=dict(size=12, color="#333333"),
        font=dict(family="Arial", size=12, color="#333333"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    # Streamlit layout
    st.title("Patient Demographics")

    # Display the charts side by side using Streamlit
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_total_patients, use_container_width=True)
        st.write("<h3 style='text-align: center;'>Total Number of Patients Over Time</h3>", unsafe_allow_html=True)

    with col2:
        st.plotly_chart(fig_age_distribution, use_container_width=True)
        st.write("<h3 style='text-align: center;'>Age Range Distribution</h3>", unsafe_allow_html=True)


    # Streamlit layout
    # D54B518B-760E-E911-9C18-00155D019601
    st.title("Patient Information and Visit Records")
    patient_id = st.text_input("Enter Patient ID: Example: D54B518B-760E-E911-9C18-00155D019601", value="")

    if st.button("Submit"):
        patient_info = patients[patients["ID"] == patient_id]
        patient_visits = consultations[consultations["PatientID"] == patient_id]

        if not patient_info.empty:
            st.subheader("Patient Information")
            patient_info_table = pd.DataFrame({
                "Attribute": ["Patient ID", "Age", "Gender", "Current Area", "Nationality", "Marital Status"],
                "Value": [
                    patient_info['ID'].values[0],
                    patient_info['Age'].values[0],
                    patient_info['Gender'].values[0],
                    patient_info['CurrentArea'].values[0],
                    patient_info['Nationality'].values[0],
                    patient_info['MaritalStatus'].values[0],
                ],
            })
            st.table(patient_info_table.style.set_table_styles(
                [{'selector': 'th', 'props': [('background-color', colors[1]), ('color', 'white')]},
                {'selector': 'td', 'props': [('background-color', '#f0f2f6')]}]
            ).hide_index())

        if not patient_visits.empty:
            st.subheader("Visit Records")
            patient_visits_table = pd.DataFrame({
                "Visit Date": patient_visits['VisitDate'],
                "Clinic Type": patient_visits['ClinicType'],
                "Waiting Time": patient_visits['WaitingTime'],
                "Category": patient_visits['Category'],
                "Diseases": patient_visits['Diseases'],
            })
            st.table(patient_visits_table.style.set_table_styles(
                [{'selector': 'th', 'props': [('background-color', colors[1]), ('color', 'white')]},
                {'selector': 'td', 'props': [('background-color', '#f0f2f6')]}]
            ).hide_index())

        if patient_info.empty and patient_visits.empty:
            st.warning("Patient ID not found. Please enter a valid Patient ID.")


if menu_id =="Analysis":
    st.text(" ")
    st.title("General Analysis About Diagnoses")  

    # st.title("Clinic Analysis Dashboard")
    time_period = st.sidebar.radio("Select Time Period", ['All',"Today", "Last Week", "Last Month", "2023", "2022", "2021", "2020"])
    distinct_clinics = consultations['ClinicType'].unique()
    distinct_clinics = ['All'] + list(distinct_clinics)  # Add "All" option to the list
    selected_clinics = st.sidebar.multiselect("Select Clinic(s)", distinct_clinics, default=['All'])

    # Filter the data based on the selected time period
    consultations["VisitDate"] = pd.to_datetime(consultations["VisitDate"])
    visits["VisitDate"] = pd.to_datetime(visits["VisitDate"])
    patients["RegistrationDate"] = pd.to_datetime(patients["RegistrationDate"])
    today = visits['VisitDate'].max().date()

    if time_period == "Today":
        visits = visits[visits['VisitDate'].dt.date == today]
        consultations = consultations[(consultations['VisitDate'].dt.date == today) & (consultations['ClinicType'].isin(selected_clinics))]
    elif time_period == "Last Week":
        last_week_start = today - dt.timedelta(days=today.weekday() + 7)
        last_week_end = today - dt.timedelta(days=today.weekday() + 1)
        patients = patients[(patients['RegistrationDate'].dt.date >= last_week_start) & (patients['RegistrationDate'].dt.date <= last_week_end)]
        visits = visits[(visits['VisitDate'].dt.date >= last_week_start) & (visits['VisitDate'].dt.date <= last_week_end)]
        consultations = consultations[(consultations['VisitDate'].dt.date >= last_week_start) & (consultations['VisitDate'].dt.date <= last_week_end) & (consultations['ClinicType'].isin(selected_clinics))]
    elif time_period == "Last Month":
        last_month_start = datetime.date(today.year, today.month - 1, 1)
        last_month_end = datetime.date(today.year, today.month, 1) - datetime.timedelta(days=1)
        patients = patients[(patients['RegistrationDate'].dt.date >= last_month_start) & (patients['RegistrationDate'].dt.date <= last_month_end)]
        visits = visits[(visits['VisitDate'].dt.date >= last_month_start) & (visits['VisitDate'].dt.date <= last_month_end)]
        consultations = consultations[(consultations['VisitDate'].dt.date >= last_month_start) & (consultations['VisitDate'].dt.date <= last_month_end) & (consultations['ClinicType'].isin(selected_clinics))]
    elif time_period in ["2023", "2022", "2021", "2020"]:
        selected_year = int(time_period)
        patients = patients[patients['RegistrationDate'].dt.year == selected_year]
        visits = visits[visits['VisitDate'].dt.year == selected_year]
        consultations = consultations[(consultations['VisitDate'].dt.year == selected_year) & (consultations['ClinicType'].isin(selected_clinics))]

    # Calculate the total number of visits for each age range
    age_bins = list(range(0, 101, 10))
    age_labels = [f'{age}-{age+9}' for age in age_bins[:-1]]
    patients['AgeRange'] = pd.cut(patients['Age'], bins=age_bins, labels=age_labels)
    age_range_visits = patients.groupby('AgeRange')['NumberVisits'].mean().sort_index().round().astype(int)

    # Clinic Distribution
    area_counts = consultations['ClinicType'].value_counts().head(10)

    colors = ['#840032', '#8F263B', '#9A3A45', '#A54E4E', '#B06258', '#BB7662', '#C68A6B', '#D19E75', '#DCA27E', '#E7B688', '#F2CA91', '#FFDE9B', '#b7b7b7']
    cc = st.columns(2)
    with cc[0]:
        # Age Range Visits Plotly Bar Chart
        st.subheader("Average Number of Visits by Age Range")
        # Define your custom color mapping
        color_mapping = {
            '0-9': '#840032',
            '10-19': '#8F263B',
            '20-29': '#9A3A45',
            '30-39': '#A54E4E',
            '40-49': '#B06258',
            '50-59': '#BB7662',
            '60-69': '#C68A6B',
            '70-79': '#D19E75',
            '80-89': '#DCA27E',
            '90-99': '#E7B688',
            '100+': '#F2CA91',
        }

        # Create a bar chart for age range distribution using Plotly
        fig_age_distribution = px.bar(
            x=age_range_visits.index,
            y=age_range_visits.values,
            color=age_range_visits.index,
            title='Age Range Distribution',
            labels={'x': 'Age Range', 'y': 'Count'},
            color_discrete_map=color_mapping
        )
        fig_age_distribution.update_layout(
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis_title_font=dict(size=12, color="#333333"),
            yaxis_title_font=dict(size=12, color="#333333"),
            font=dict(family="Arial", size=12, color="#333333"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        # age_range_fig = px.bar(x=age_range_visits.index, y=age_range_visits.values, color_discrete_sequence=colors,
        #                     labels={'x': 'Age Range', 'y': 'Average Number of Visits'})
        # age_range_fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_age_distribution)
    with cc[1]:
        # Clinic Distribution Plotly Bar Chart
        st.subheader("Top Clinic Distribution")
        clinic_counts_fig = px.bar(x=area_counts.index, y=area_counts.values, color_discrete_sequence=colors,
                                labels={'x': 'Clinic', 'y': 'Count'})
        clinic_counts_fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='rgba(0,0,0,0)')  # Remove background grid
        st.plotly_chart(clinic_counts_fig)


    # Convert 'VisitDate' columns to datetime
    visits['VisitDate'] = pd.to_datetime(visits['VisitDate'])
    consultations['VisitDate'] = pd.to_datetime(consultations['VisitDate'])

    # Count the number of visits and consultations per week
    visits_count = visits.resample('W', on='VisitDate').size().reset_index(name='Visits')
    consultations_count = consultations.resample('W', on='VisitDate').size().reset_index(name='Consultations')

    # Merge the two counts based on 'VisitDate'
    data = visits_count.merge(consultations_count, on='VisitDate', how='outer').fillna(0)
    st.subheader("Total Patient Visits vs Consultations Per Week")
    # Create a line chart
    fig = px.line(data, x='VisitDate', y=['Visits', 'Consultations'], labels={'value': 'Count'})
    # Set line colors for visits and consultations
    fig.update_traces(line=dict(color='#840032'), selector=dict(name='Visits'))
    fig.update_traces(line=dict(color='#D19E75'), selector=dict(name='Consultations'))

    fig.update_layout(
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_title_font=dict(size=12, color="#333333"),
        yaxis_title_font=dict(size=12, color="#333333"),
        font=dict(family="Arial", size=12, color="#333333"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    # Display the chart using Streamlit
    st.plotly_chart(fig, use_container_width=True)


    # # Convert 'VisitDate' column to datetime format
    # consultations['VisitDate'] = pd.to_datetime(consultations['VisitDate'])

    # # Filter distinct clinic types
    # distinct_clinics = consultations['ClinicType'].unique()

    # # Streamlit layout
    # st.title("Clinic Analysis Dashboard")

    # # Clinic type selection
    # selected_clinics = st.multiselect("Select Clinic(s)", distinct_clinics)

    # # Time range selection
    # time_range_options = {
    #     "Last Day": '6/21/2023',
    #     "Last Week": '6/14/2023',
    #     "Last Month": '5/21/2023',
    #     "Last Year": '6/21/2022',
    #     "All Years": None  # Set to None to include all data
    # }
    # selected_time_range = st.selectbox("Select Time Range", list(time_range_options.keys()))

    # # Calculate start and end dates based on selected time range
    # if selected_time_range == "Last Day":
    #     end_date = pd.to_datetime(time_range_options[selected_time_range])
    #     start_date = end_date
    # elif selected_time_range == "Last Week":
    #     end_date = pd.to_datetime(time_range_options[selected_time_range])
    #     start_date = end_date - pd.DateOffset(weeks=1)
    # elif selected_time_range == "Last Month":
    #     end_date = pd.to_datetime(time_range_options[selected_time_range])
    #     start_date = end_date - pd.DateOffset(months=1)
    # elif selected_time_range == "Last Year":
    #     end_date = pd.to_datetime(time_range_options[selected_time_range])
    #     start_date = end_date - pd.DateOffset(years=1)
    # elif selected_time_range == "All Years":
    #     start_date = consultations['VisitDate'].min()  # Include all data

    # # Filter data based on selected clinic(s) and time range
    # filtered_consultations = consultations[(consultations['ClinicType'].isin(selected_clinics)) &
    #                                     (consultations['VisitDate'] >= start_date) &
    #                                     (consultations['VisitDate'] <= end_date)]


    # # Charts
    # if not filtered_consultations.empty:
    #     # Total number of patients visiting selected clinics
    #     total_patients_chart = px.bar(x=selected_clinics, y=filtered_consultations.groupby('ClinicType')['PatientID'].nunique(),
    #                                 labels={'x': 'Clinic', 'y': 'Total Number of Patients'},
    #                                 title='Total Number of Patients Visiting Selected Clinics')
    #     st.plotly_chart(total_patients_chart)

    #     # Average waiting time for patients visiting selected clinics
    #     avg_waiting_time_chart = px.bar(x=selected_clinics, y=filtered_consultations.groupby('ClinicType')['WaitingTime'].mean(),
    #                                     labels={'x': 'Clinic', 'y': 'Average Waiting Time'},
    #                                     title='Average Waiting Time for Patients Visiting Selected Clinics')
    #     st.plotly_chart(avg_waiting_time_chart)

    #     # Clinic visit frequency
    #     visit_frequency_chart = px.bar(x=selected_clinics, y=filtered_consultations.groupby('ClinicType')['PatientID'].value_counts().mean(level=0),
    #                                 labels={'x': 'Clinic', 'y': 'Average Visit Frequency'},
    #                                 title='Average Clinic Visit Frequency for Patients Visiting Selected Clinics')
    #     st.plotly_chart(visit_frequency_chart)

    #     # Line chart with patients visits and consultations over time
    #     combined_data = pd.concat([patients.set_index('ID')['Age'], consultations.set_index('PatientID')['VisitDate']], axis=1)
    #     combined_data.columns = ['Age', 'VisitDate']
    #     combined_data['Consultation'] = True
    #     combined_data['Visit'] = False
    #     combined_data.loc[combined_data.index.isin(filtered_consultations['PatientID']), 'Consultation'] = False
    #     combined_data.loc[combined_data.index.isin(filtered_consultations['PatientID']), 'Visit'] = True

    #     visits_consultations_chart = px.line(combined_data, x='VisitDate', y=['Age'],
    #                                         labels={'x': 'Date', 'y': 'Count'},
    #                                         title='Patients Visits and Consultations Over Time')
    #     visits_consultations_chart.update_traces(line=dict(dash='dot'), selector=dict(name='Age'))
    #     st.plotly_chart(visits_consultations_chart)
    # else:
    #     st.warning("No data available for the selected filters.")

if menu_id =="Machine Learning":
    # Function to train or load the ARIMA model
    @st.cache(allow_output_mutation=True)
    def train_or_load_arima_model():
        try:
            # Load the saved model if it exists
            model = joblib.load("arima_model.pkl")
        except FileNotFoundError:
            # Train the model if it doesn't exist
            waiting_time_data = pd.read_csv("data/WaitingTime.csv")
            time_series = waiting_time_data["time"].astype(float)
            model = auto_arima(time_series, seasonal=False, stepwise=True, suppress_warnings=True)
            joblib.dump(model, "arima_model.pkl")
        
        return model

    # Streamlit UI
    st.title("ARIMA Forecasting")

    # Load the data and define time_series outside the button click event
    waiting_time_data = pd.read_csv("data/WaitingTime.csv")
    time_series = waiting_time_data["time"].astype(float)

    # Button to trigger forecasting
    if st.button("Run ARIMA Forecast"):
        # Train or load the ARIMA model
        model = train_or_load_arima_model()

        # ARIMA forecasting code using the loaded model
        forecasted, conf_int = model.predict(n_periods=14, return_conf_int=True)
        actual = time_series[-14:].values
        mae = np.mean(np.abs(actual - forecasted))
        rmse = np.sqrt(np.mean((actual - forecasted) ** 2))

        # Create a DataFrame to hold actual and forecasted values without the index column
        results = pd.DataFrame({
            "Actual": time_series[-14:].values,
            "Forecasted": forecasted
        })

        # Display the table without the index column
        st.table(results.style.set_table_styles(
            [{'selector': 'th', 'props': [('background-color', '#8F263B'), ('color', 'white')]},
            {'selector': 'td', 'props': [('background-color', '#f0f2f6')]}]).hide_index())

        st.text(" ")
        st.text(" ")

        card_style = "box-shadow: 2px 2px 5px grey; padding: 15px; border-radius: 10px; text-align: center;"
        cc = st.columns(2)
        with cc[0]:
            st.write(f"<div style='{card_style} background-color: #A54E4E; color: white;'>Mean Absolute Error (MAE): <span style='font-size: 24px; font-weight: bold;'>{mae:.2f}</span></div>", unsafe_allow_html=True)
        with cc[1]:
            st.write(f"<div style='{card_style} background-color: #A54E4E; color: white;'>Root Mean Squared Error (RMSE): <span style='font-size: 24px; font-weight: bold;'>{rmse:.2f}</span></div>", unsafe_allow_html=True)
        
        st.text(" ")
        st.text(" ")
        # Create a Matplotlib figure and axis for the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the forecasted values with custom color
        ax.plot(waiting_time_data.index, time_series, label="Actual", color='#840032')
        ax.plot(range(len(time_series), len(time_series) + 14), forecasted, label="Forecast", linestyle='dashed', color='#840032')
        ax.fill_between(range(len(time_series), len(time_series) + 14), conf_int[:, 0], conf_int[:, 1], alpha=0.2)
        ax.set_xlabel("Day")
        ax.set_ylabel("Average Waiting Time")
        ax.set_title("ARIMA Forecast")
        ax.legend()

        # Display the Matplotlib plot using Streamlit
        st.pyplot(fig)

    # # Create a line chart to plot actual and forecasted values
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(len(time_series) - 14, len(time_series)), actual, label="Actual", color='#840032', marker='o')
    # plt.plot(range(len(time_series) - 14, len(time_series)), forecasted, label="Forecast", linestyle='dashed', color='#FFDE9B', marker='o')
    # plt.xlabel("Day")
    # plt.ylabel("Average Waiting Time")
    # plt.title("ARIMA Forecast")
    # plt.legend()
    # st.pyplot()

