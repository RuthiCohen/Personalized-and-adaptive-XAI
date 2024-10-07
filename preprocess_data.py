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

        # create new csv file
        new_file = f"{data_name}-new"
        data.to_csv(f'data/{data_name}/{new_file}.csv', index=False)

        split_to_train_test_files(new_file)

        path = f"data/{data_name}/{new_file}.csv"
        data = pd.read_csv(path)

    # elif data_name == "student_performance_factors":
    #     file_path = f'data/{data_name}/{data_name}.csv'
    #     data = pd.read_csv(file_path)
    #
    #     # fill missing values with the first (most frequent) mode
    #     data.Teacher_Quality.fillna(data['Teacher_Quality'].mode()[0], inplace=True)
    #     data.Parental_Education_Level.fillna(data['Parental_Education_Level'].mode()[0], inplace=True)
    #     data.Distance_from_Home.fillna(data['Distance_from_Home'].mode()[0], inplace=True)
    #
    #     # convert categorical data
    #     categorical_columns = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
    #                            'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
    #                            'School_Type', 'Peer_Influence', 'Learning_Disabilities',
    #                            'Parental_Education_Level', 'Distance_from_Home', 'Gender']
    #
    #     label_encoders = {}
    #     for column in categorical_columns:
    #         lb = LabelEncoder()
    #         data[column] = lb.fit_transform(data[column])
    #         label_encoders[column] = lb
    #
    #     # create new csv file
    #     new_file = f"{data_name}-new"
    #     data.to_csv(f'data/{data_name}/{new_file}.csv', index=False)
    #
    #     split_to_train_test_files(new_file)
    #
    #     path = f"data/{data_name}/{new_file}.csv"
    #     data = pd.read_csv(path)

    else: #todo
            print()

    return data, path