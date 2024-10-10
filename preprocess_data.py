from openxai.experiment_utils import split_to_train_test_files
from sklearn.preprocessing import LabelEncoder
import pandas as pd

import warnings; warnings.filterwarnings("ignore")

def get_preprocessed_data(data_name):
    if data_name == 'heart_failure_clinical_records_dataset':
        # split data
        split_to_train_test_files(data_name)

        path = f"data/{data_name}/{data_name}.csv"
        data = pd.read_csv(path)

        return data, path

    elif data_name == 'MBA':
        file_path = f'data/{data_name}/{data_name}.csv'
        data = pd.read_csv(file_path)

        # irrlevant columns
        data.drop('application_id', axis=1, inplace=True)

        # fill empty values
        data['race'] = data['race'].fillna('Unknown')

        # convert categorical data
        data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
        data['admission'] = data['admission'].map({'Admit': 1, 'Waitlist': 0}).fillna(0)

        columns = ['major', 'race', 'work_industry']
        le = LabelEncoder()
        for cols in columns:
            data[cols] = le.fit_transform(data[cols])

    elif data_name == "student_performance_factors":
        file_path = f'data/{data_name}/{data_name}.csv'
        data = pd.read_csv(file_path)

        # fill missing values with the first (most frequent) mode
        data.Teacher_Quality.fillna(data['Teacher_Quality'].mode()[0], inplace=True)
        data.Parental_Education_Level.fillna(data['Parental_Education_Level'].mode()[0], inplace=True)
        data.Distance_from_Home.fillna(data['Distance_from_Home'].mode()[0], inplace=True)

        # convert categorical data
        categorical_columns = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                               'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                               'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                               'Parental_Education_Level', 'Distance_from_Home', 'Gender']

        label_encoders = {}
        for column in categorical_columns:
            lb = LabelEncoder()
            data[column] = lb.fit_transform(data[column])
            label_encoders[column] = lb

    elif data_name == "undergraduate_admission_test_survey_in_bangladesh":
        file_path = f'data/{data_name}/{data_name}.csv'
        data = pd.read_csv(file_path)

        # fill empty values
        data['HSC_GPA'].fillna(data['HSC_GPA'].mean(), inplace=True)

    elif data_name == "car_price_prediction":
        file_path = f'data/{data_name}/{data_name}.csv'
        data = pd.read_csv(file_path)

        # delete irrlevant column
        data = data.drop('Car ID', axis=1)

        # update columns
        data['age'] = 2024 - data['Year']
        data = data.drop('Year', axis=1)

        data['Condition'] = data['Condition'].replace('Like New', 'New')

        # categorical columns
        categorical_columns = ['Brand', 'Fuel Type', 'Transmission', 'Condition']
        data = pd.get_dummies(data, columns=categorical_columns)

        for i in data.columns:
            if data[i].dtype == bool:
                data[i] = data[i].astype(int)

        label_encoders = {}
        for column in ["Model"]:
            lb = LabelEncoder()
            data[column] = lb.fit_transform(data[column])
            label_encoders[column] = lb

    elif data_name == "2017_2020_bmi":
        # split data
        split_to_train_test_files(data_name)

        path = f"data/{data_name}/{data_name}.csv"
        data = pd.read_csv(path)

        # rename categorical column specific labels
        data['yr'] = data['yr'].replace({'19-44': 32, '45-64': 55, '65+': 65})

    elif data_name == "healthcare_noshows_appointments":
        file_path = f'data/{data_name}/{data_name}.csv'
        data = pd.read_csv(file_path)

        data = data.drop(['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood'], axis=1)
        # data = pd.concat([data.drop('Neighbourhood', axis=1), pd.get_dummies(data['Neighbourhood'])], axis=1)
        gender_map = {'M': 0, 'F': 1}
        data['Gender'] = data['Gender'].map(gender_map)

    else: #todo
            print()

    # create new csv file
    new_file = f"{data_name}-new"
    data.to_csv(f'data/{data_name}/{new_file}.csv', index=False)

    split_to_train_test_files(new_file)

    path = f"data/{data_name}/{new_file}.csv"
    data = pd.read_csv(path)

    return data, path