from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def countries_pred_batch():


    # Read the immigration data
    immigration_df = pd.read_csv(r"chosen_data/regression/total imgration to oced country.csv", encoding = "ISO-8859-1")

    # Filter the immigration data to include relevant columns
    immigration_df = immigration_df[['Country', 'Year', 'Value']]

    # Create an empty dictionary to store predicted immigration values for each country
    predictions = {}

    # Iterate over each country
    for country in immigration_df['Country'].unique():
        # Filter the data for the current country
        country_data = immigration_df[immigration_df['Country'] == country]

        # Extract the immigration values
        immigration_values = country_data['Value'].values.reshape(-1, 1)

        # Split the data into training and testing sets
        X_train = immigration_values[:-1]
        y_train = immigration_values[1:]

        # Create and fit a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the next 20 years of immigration values
        last_year_value = immigration_values[-1]
        future_years = [last_year_value]
        predicted_values = []
        for _ in range(20):
            future_values = model.predict(np.array(future_years[-1]).reshape(-1, 1))
            predicted_values.append(future_values[0])
            future_years.append(future_values[0])

        # Store the predicted values for the current country
        predictions[country] = predicted_values

    # Sort the countries based on the final predicted immigration values
    sorted_countries = sorted(predictions.items(), key=lambda x: x[1][-1], reverse=True)

    # Display seven countries at a time
    num_countries_per_group = 7
    num_groups = (len(sorted_countries) + num_countries_per_group - 1) // num_countries_per_group


    for group in range(num_groups):
        start_index = group * num_countries_per_group
        end_index = (group + 1) * num_countries_per_group
        current_countries = sorted_countries[start_index:end_index]

        # Plot the predicted immigration values for each country in the current group
        for country, values in current_countries:
            years = range(immigration_df['Year'].max() + 1, immigration_df['Year'].max() + 21)
            plt.plot(years, values, label=country)

        plt.xlabel('Year')
        plt.ylabel('Predicted Immigration Value')
        plt.title('Predicted Immigration Values for the Next 20 Years')
        plt.legend()
        plt.show()

def countries_pred_solo():


    # Read the immigration data
    immigration_df = pd.read_csv(r"chosen_data/regression/total imgration to oced country.csv", encoding = "ISO-8859-1")

    # Filter the immigration data to include relevant columns
    immigration_df = immigration_df[['Country', 'Year', 'Value']]

    # Create an empty dictionary to store predicted immigration values for each country
    predictions = {}

    # Iterate over each country
    for country in immigration_df['Country'].unique():
        # Filter the data for the current country
        country_data = immigration_df[immigration_df['Country'] == country]

        # Extract the immigration values
        immigration_values = country_data['Value'].values.reshape(-1, 1)

        # Split the data into training and testing sets
        X_train = immigration_values[:-1]
        y_train = immigration_values[1:]

        # Create and fit a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the next 20 years of immigration values
        last_year_value = immigration_values[-1]
        future_years = [last_year_value]
        predicted_values = []
        for _ in range(20):
            future_values = model.predict(np.array(future_years[-1]).reshape(-1, 1))
            predicted_values.append(future_values[0])
            future_years.append(future_values[0])

        # Store the predicted values for the current country
        predictions[country] = predicted_values

    # Plot the predicted immigration values for each country

    for country, values in predictions.items():
        years = range(immigration_df['Year'].max() + 1, immigration_df['Year'].max() + 21)

        plt.plot(years, values, label=country)

        plt.xlabel('Year')
        plt.ylabel('Predicted Immigration Value')
        plt.title('Predicted Immigration Values for the Next 20 Years')
        plt.legend()
        plt.show()


def countries_pred_solo_with_before_info():



    immigration_df = pd.read_csv(r"chosen_data/regression/total imgration to oced country.csv", encoding = "ISO-8859-1")
    # Filter the immigration data to include relevant columns
    immigration_df = immigration_df[['Country', 'Year', 'Value']]

    # Create an empty dictionary to store predicted immigration values for each country
    predictions = {}

    # Iterate over each country
    for country in immigration_df['Country'].unique():
        # Filter the data for the current country
        country_data = immigration_df[immigration_df['Country'] == country]

        # Extract the immigration values
        immigration_values = country_data['Value'].values.reshape(-1, 1)

        # Split the data into training and testing sets
        X_train = immigration_values[:-1]
        y_train = immigration_values[1:]

        # Create and fit a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the next 20 years of immigration values
        last_year_value = immigration_values[-1]
        future_years = [last_year_value]
        predicted_values = []
        for _ in range(20):
            future_values = model.predict(np.array(future_years[-1]).reshape(-1, 1))
            predicted_values.append(future_values[0])
            future_years.append(future_values[0])

        # Store the predicted values for the current country
        predictions[country] = predicted_values

    # Plot the original and predicted immigration values for each country
    for country, values in predictions.items():
        original_years = immigration_df.loc[immigration_df['Country'] == country, 'Year'].values
        original_values = immigration_df.loc[immigration_df['Country'] == country, 'Value'].values

        plt.plot(original_years, original_values, label='Original Values for ' + country)

        future_years = range(original_years[-1] + 1, original_years[-1] + 21)
        plt.plot(future_years, values, label='Predicted Values for ' + country)

        plt.xlabel('Year')
        plt.ylabel('Immigration Value')
        plt.title('Original and Predicted Immigration Values for the Next 20 Years')
        plt.legend()
        plt.show()


def predict_with_r2_score():
    # Read the immigration data
    immigration_df = pd.read_csv(r"chosen_data/regression/total imgration to oced country.csv", encoding = "ISO-8859-1")

    # Filter the immigration data to include relevant columns
    immigration_df = immigration_df[['Country', 'Year', 'Value']]

    # Create an empty dictionary to store predicted immigration values and R-squared for each country
    predictions = {}

    # Iterate over each country
    for country in immigration_df['Country'].unique():
        # Filter the data for the current country
        country_data = immigration_df[immigration_df['Country'] == country]

        # Extract the immigration values
        immigration_values = country_data['Value'].values.reshape(-1, 1)

        # Split the data into training and testing sets
        X_train = immigration_values[:-1]
        y_train = immigration_values[1:]

        # Create and fit a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the next 20 years of immigration values
        last_year_value = immigration_values[-1]
        future_years = [last_year_value]
        predicted_values = []
        r2_scores=[]
        for _ in range(20):
            future_values = model.predict(np.array(future_years[-1]).reshape(-1, 1))
            predicted_values.append(future_values[0])
            future_years.append(future_values[0])

        # Calculate the R-squared value
        r2 = r2_score(y_train, model.predict(X_train))
        r2_scores.append(r2)

        # Store the predicted values and R-squared for the current country
        predictions[country] = {'predicted_values': predicted_values, 'r2': r2}

    # Plot the original and predicted immigration values for each country
    for country, values in predictions.items():
        original_years = immigration_df.loc[immigration_df['Country'] == country, 'Year'].values
        original_values = immigration_df.loc[immigration_df['Country'] == country, 'Value'].values

        plt.plot(original_years, original_values, label='Original Values for ' + country)

        future_years = range(original_years[-1] + 1, original_years[-1] + 21)
        plt.plot(future_years, values['predicted_values'], label='Predicted Values for ' + country)

        print(f"R-squared for {country}: {values['r2']:.4f}")

    plt.xlabel('Year')
    plt.ylabel('Immigration Value')
    plt.title('Original and Predicted Immigration Values for the Next 20 Years')
    plt.legend()
    plt.show()

    average_r2 = np.mean(r2_scores)
    print(f"\nAverage R-squared score: {average_r2:.4f}")

