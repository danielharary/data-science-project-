import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def cleaning_working_pop():

        ######### add 0 to nan in working population

        # Read the CSV file into a DataFrame
        df = pd.read_csv('chosen_data/working_population_by_sex_age_place_of_birth.csv', encoding = "ISO-8859-1")

        # Replace missing values with 0
        df.fillna(0, inplace=True)

        # Save the updated DataFrame to a new CSV file
        df.to_csv('working_population_by_sex_age_place_of_birth_clean.csv', index=False)


        ##### check if any nan left?

        df = pd.read_csv('working_population_by_sex_age_place_of_birth_clean.csv', encoding = "ISO-8859-1")
        nan_counts = df.isna().sum()

        # Get the total number of NaN values in the DataFrame
        total_nan_count = nan_counts.sum()

        # Print the results
        print("NaN counts per column:")
        print(nan_counts)
        print("\nTotal number of NaN values:", total_nan_count)


        ##### clean unnessecary col

        df = pd.read_csv('working_population_by_sex_age_place_of_birth_clean.csv', encoding = "ISO-8859-1")
        df = df.drop('Unnamed: 10', axis=1)
        df.to_csv('working_population_by_sex_age_place_of_birth_clean.csv', index=False)

        #### take only place of birth and sex - total

        df = pd.read_csv('working_population_by_sex_age_place_of_birth_clean.csv', encoding = "ISO-8859-1")
        df_total_sex = df[df['Sex'] == 'Total'].copy()
        df_total_sex_and_total_birthplace = df_total_sex[df_total_sex['Place of birth'] == 'Total'].copy()
        df_total_sex_and_total_birthplace.to_csv('working_population_by_sex_age_place_of_birth_clean_total.csv', index=False)


        ###### choose biggest year

        df = pd.read_csv('working_population_by_sex_age_place_of_birth_clean_total.csv', encoding = "ISO-8859-1")

        # Group the DataFrame by "Reference area" and select the row with the maximum value in "Time"
        new_df = df.loc[df.groupby('Reference area')['Time'].idxmax()].copy()

        # Save the new DataFrame to a new CSV file
        new_df.to_csv('working_population_by_sex_age_place_of_birth_clean_total_biggest_year.csv', index=False)


        ######clean the bad countries

        df = pd.read_csv('working_population_by_sex_age_place_of_birth_clean_total_biggest_year.csv', encoding = "ISO-8859-1")

        countries_to_drop = ['Bahrain', 'Kuwait']

        # Drop rows containing the specified countries
        df = df[~df['Reference area'].isin(countries_to_drop)]

        # Save the updated DataFrame to a new CSV file
        df.to_csv('working_population_by_sex_age_place_of_birth_clean_total_biggest_year_without_bad_countries.csv', index=False)



def average_age_dist_oced_non_oced():


        cleaning_working_pop()

        ####### create average for oecd countries and not oced countries



        ###Read the CSV file into a DataFrame

        df = pd.read_csv('working_population_by_sex_age_place_of_birth_clean_total_biggest_year_without_bad_countries.csv')

        # List of OECD countries
        oecd_countries = [
            'Australia',
            'Austria',
            'Belgium',
            'Canada',
            'Chile',
            'Czech Republic',
            'Denmark',
            'Estonia',
            'Finland',
            'France',
            'Germany',
            'Greece',
            'Hungary',
            'Iceland',
            'Ireland',
            'Israel',
            'Italy',
            'Japan',
            'South Korea',
            'Latvia',
            'Lithuania',
            'Luxembourg',
            'Mexico',
            'Netherlands',
            'New Zealand',
            'Norway',
            'Poland',
            'Portugal',
            'Slovakia',
            'Slovenia',
            'Spain',
            'Sweden',
            'Switzerland',
            'Turkey',
            'United Kingdom',
            'United States'
        ]

        # Calculate averages for OECD countries
        oecd_avg = df[df['Reference area'].isin(oecd_countries)][['15-24', '25-54', '55-64', '65+']].mean()

        # Calculate averages for non-OECD countries
        non_oecd_avg = df[~df['Reference area'].isin(oecd_countries)][['15-24', '25-54', '55-64', '65+']].mean()

        # Create a new DataFrame with the averages
        new_df = pd.DataFrame({'OECD Average': oecd_avg, 'Non-OECD Average': non_oecd_avg})

        # Swap the rows and columns
        new_df = new_df.transpose()

        # Create a single plot for both pie charts
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the pie chart for "OECD Average"
        axes[0].pie(new_df.loc['OECD Average'], labels=new_df.columns, autopct='%1.1f%%')
        axes[0].set_title('Age Distribution - OECD Average')

        # Plot the pie chart for "Non-OECD Average"
        axes[1].pie(new_df.loc['Non-OECD Average'], labels=new_df.columns, autopct='%1.1f%%')
        axes[1].set_title('Age Distribution - Non-OECD Average')

        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the combined plot
        plt.show()


def pie_for_working_pop_by_sex_age_place_of_birth():

    cleaning_working_pop()



    ######## create pie chart for each country

    df = pd.read_csv('working_population_by_sex_age_place_of_birth_clean_total_biggest_year_without_bad_countries.csv', encoding = "ISO-8859-1")

    for index, row in df.iterrows():
        # Extract the country name from the "Reference area" column
        country_name = row['Reference area']
        try:

            # Extract the age range values for the current row


            age_ranges = row[['15-24', '25-54', '55-64', '65+']]

            # Create a pie chart
            plt.figure()
            plt.pie(age_ranges, labels=age_ranges.index, autopct='%1.1f%%')
            plt.title(f'Age Distribution in {country_name}')

            # Display the pie chart
            #plt.show()
        except:
            print(country_name)




def bar_graph_oecd_non_oecd():

        df = pd.read_csv('working_population_by_sex_age_place_of_birth_clean.csv', encoding="ISO-8859-1")
        df_total_sex = df[df['Sex'] == 'Total'].copy()
        df_total_sex_and_diff_birth = df_total_sex[df_total_sex['Place of birth'].isin(['Native-born', 'Foreign-born'])]

        # print(df_total_sex_and_diff_birth)

        new_df = df_total_sex_and_diff_birth.loc[
                df_total_sex_and_diff_birth.groupby(['Reference area', 'Place of birth'])['Time'].idxmax()].copy()

        # df_total_sex_and_diff_birth.to_csv('test.csv', index=False)

        # print(df_total_sex_and_diff_birth[df_total_sex_and_diff_birth['Reference area']=='Angola']['Time'])

        # Read the CSV file into a DataFrame

        df = new_df

        # List of OECD countries
        oecd_countries = [
                'Australia',
                'Austria',
                'Belgium',
                'Canada',
                'Chile',
                'Czech Republic',
                'Denmark',
                'Estonia',
                'Finland',
                'France',
                'Germany',
                'Greece',
                'Hungary',
                'Iceland',
                'Ireland',
                'Israel',
                'Italy',
                'Japan',
                'South Korea',
                'Latvia',
                'Lithuania',
                'Luxembourg',
                'Mexico',
                'Netherlands',
                'New Zealand',
                'Norway',
                'Poland',
                'Portugal',
                'Slovakia',
                'Slovenia',
                'Spain',
                'Sweden',
                'Switzerland',
                'Turkey',
                'United Kingdom',
                'United States'
        ]

        # Filter rows for OECD countries
        oecd_df = df[df['Reference area'].isin(oecd_countries)]

        # Filter rows for non-OECD countries
        non_oecd_df = df[~df['Reference area'].isin(oecd_countries)]

        # Count the number of native-born and foreign-born workers in OECD countries
        non_oecd_native_avg = non_oecd_df[non_oecd_df['Place of birth'] == 'Native-born'][
                ['15-24', '25-54', '55-64', '65+']].mean()
        non_oecd_foreign_avg = non_oecd_df[non_oecd_df['Place of birth'] == 'Foreign-born'][
                ['15-24', '25-54', '55-64', '65+']].mean()

        print(oecd_df[oecd_df['Place of birth'] == 'Native-born'])
        print(oecd_df[oecd_df['Place of birth'] == 'Foreign-born'])

        # Create labels for the age ranges
        age_ranges = ['15-24', '25-54', '55-64', '65+']

        # Create an array for the x-axis positions of the bars
        x = np.arange(len(age_ranges))

        # Set the width of the bars
        bar_width = 0.35

        # Create the bar graph
        plt.bar(x - bar_width / 2, non_oecd_native_avg, width=bar_width, label='Native-born')
        plt.bar(x + bar_width / 2, non_oecd_foreign_avg, width=bar_width, label='Foreign-born')

        # Add labels, title, and legend
        plt.xlabel('Age Range')
        plt.ylabel('Average Number of Workers')
        plt.title('Age Distribution of Native-born vs Foreign-born Workers in Non-OECD Countries')
        plt.xticks(x, age_ranges)
        plt.legend()

        # Show the plot
        plt.show()


def male_vs_female_dist_imig():



        # Read the CSV file
        df = pd.read_csv('chosen_data/outflow_imagrants_clean.csv')

        # Filter the data by gender (Women)
        women_df = df[df['Sex'] == 'Female']

        # Filter the data by gender (Men)
        men_df = df[df['Sex'] == 'Male']

        # Calculate the total number of immigrants by gender
        total_women = women_df['Outflow of nationals (thousands)'].sum()
        total_men = men_df['Outflow of nationals (thousands)'].sum()

        # Create a pie chart
        labels = ['Female', 'Male']
        sizes = [total_women, total_men]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.1, 0)  # Explode the first slice (Women)

        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Distribution of Immigrants by Gender')

        plt.show()


def dist_imig_oecd_vs_non_oecd():


        # Read the CSV file
        df = pd.read_csv('chosen_data/outflow_imagrants_clean.csv')

        # Define the list of OECD countries
        oecd_countries = ['Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Colombia', 'Czech Republic',
                          'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland',
                          'Ireland', 'Israel', 'Italy', 'Japan', 'Korea', 'Latvia', 'Lithuania', 'Luxembourg',
                          'Mexico', 'Netherlands', 'New Zealand', 'Norway', 'Poland', 'Portugal', 'Slovak Republic',
                          'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom', 'United States']

        # Filter the data by OECD countries
        oecd_df = df[df['Country of destination'].isin(oecd_countries)]

        # Filter the data by non-OECD countries
        non_oecd_df = df[~df['Country of destination'].isin(oecd_countries)]

        # Calculate the total number of immigrants from OECD and non-OECD countries
        total_oecd = oecd_df['Outflow of nationals (thousands)'].sum()
        total_non_oecd = non_oecd_df['Outflow of nationals (thousands)'].sum()

        # Create a pie chart
        labels = ['OECD Countries', 'Non-OECD Countries']
        sizes = [total_oecd, total_non_oecd]
        colors = ['#ffcc99', '#b3b3cc']
        explode = (0.1, 0)  # Explode the first slice (OECD Countries)

        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Distribution of Immigrants from OECD vs Non-OECD Countries')

        plt.show()

def bar_graph_students_oecd_vs_non_oecd():


        df = pd.read_csv("chosen_data/foreign_students.csv")

        # Define the list of OECD countries
        oecd_countries = ['Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Colombia', 'Czech Republic',
                          'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland',
                          'Ireland', 'Israel', 'Italy', 'Japan', 'Korea', 'Latvia', 'Lithuania', 'Luxembourg',
                          'Mexico', 'Netherlands', 'New Zealand', 'Norway', 'Poland', 'Portugal', 'Slovak Republic',
                          'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom', 'United States']

        # Filter the data by OECD countries
        oecd_df = df[df['Reference Area'].isin(oecd_countries)]

        # Filter the data by non-OECD countries
        non_oecd_df = df[~df['Reference Area'].isin(oecd_countries)]


        oecd_outflow = oecd_df['Observation Value'].mean()
        non_oecd_outflow = non_oecd_df['Observation Value'].mean()

        # Create a bar graph
        data = [oecd_outflow, non_oecd_outflow]
        labels = ['OECD', 'None OECD']

        colors = ['blue', 'orange']  # Specify the colors for each bar

        plt.bar(labels, data, color=colors)  # Set the colors for the bars
        plt.xlabel('Country')
        plt.ylabel('Outflow of Students')
        plt.title('Outflow of Students: OECD vs None OECD')

        # Display the bar graph
        plt.show()


def Countries_with_the_highest_and_least_amount_of_students_immigrating_from_them():


        # Load the CSV file
        df = pd.read_csv('chosen_data/foreign_students.csv')

        # Group the data by country and calculate the total immigration count
        country_immigration = df.groupby('Reference Area')['Observation Value'].sum()

        # Sort the countries by immigration count in descending order
        sorted_countries = country_immigration.sort_values(ascending=False)

        # Get the top 5 countries with the highest immigration count
        top_countries = sorted_countries.head(5)

        # Get the 5 countries with the lowest immigration count
        bottom_countries = sorted_countries.tail(5)

        # Display the names of the top 5 countries with the highest immigration count
        print("Countries with the highest amount of students immigrating from them:")
        for country, count in top_countries.items():
            print(country)

        # Display the names of the 5 countries with the lowest immigration count
        print("\nCountries with the lowest amount of students immigrating from them:")
        for country, count in bottom_countries.items():
            print(country)


def Students_imig_by_year():



        # Load the CSV file
        df = pd.read_csv('chosen_data/foreign_students.csv')

        # Group the data by year and calculate the total immigration count
        year_immigration = df.groupby('Time Period')['Observation Value'].sum()

        # Create a line graph
        year_immigration.plot(kind='line')

        # Set the axis labels and title
        plt.xlabel('Year')
        plt.ylabel('Immigration Count')
        plt.title('Overall Student Immigration Count by Year')

        # Display the line graph
        plt.show()


def display_most_imig():


        # Read the CSV file
        data = pd.read_csv(r"chosen_data/regression/total imgration to oced country.csv", encoding = "ISO-8859-1")

        # Group the data by 'Country' and calculate the immigration difference between the earliest and latest years
        grouped_data = data.groupby('Country')['Value'].agg(lambda x: x.iloc[-1] - x.iloc[0]).reset_index()

        # Sort the countries based on immigration difference in descending order and get the top 5 countries
        top_growth_countries = grouped_data.sort_values('Value', ascending=False).head(5)

        # Filter the data for the top 5 growth countries
        filtered_data = data[data['Country'].isin(top_growth_countries['Country'])]

        # Plotting the growth of top 5 countries over the years
        fig, ax = plt.subplots(figsize=(12, 6))
        for country in top_growth_countries['Country']:
            country_data = filtered_data[filtered_data['Country'] == country]
            country_data.plot(x='Year', y='Value', ax=ax, label=country)

        plt.xlabel('Year')
        plt.ylabel('Total Immigration')
        plt.title('Growth of Total Immigration for Top 5 Countries')
        plt.legend(loc='upper left')
        plt.show()


def display_all_imig_by_growth():

        # Read the CSV file
        data = pd.read_csv(r"chosen_data/regression/total imgration to oced country.csv", encoding = "ISO-8859-1")

        # Group the data by 'Country' and calculate the immigration difference between the earliest and latest years
        grouped_data = data.groupby('Country')['Value'].agg(lambda x: x.iloc[-1] - x.iloc[0]).reset_index()

        # Sort the countries based on immigration difference in descending order
        sorted_countries = grouped_data.sort_values('Value', ascending=False)['Country'].tolist()

        # Determine the number of graphs needed
        num_graphs = len(sorted_countries) // 7 + 1

        # Plotting separate graphs for each group of 7 countries, sorted by growth
        for i in range(num_graphs):
            start_idx = i * 7
            end_idx = (i + 1) * 7

            # Extract the countries for the current graph
            current_countries = sorted_countries[start_idx:end_idx]

            # Filter the data for the current countries
            filtered_data = data[data['Country'].isin(current_countries)]

            # Plotting the graph for the current countries
            fig, ax = plt.subplots(figsize=(12, 6))
            for country in current_countries:
                country_data = filtered_data[filtered_data['Country'] == country]
                country_data.plot(x='Year', y='Value', ax=ax, label=country)

            plt.xlabel('Year')
            plt.ylabel('Total Immigration')
            plt.title(f'Growth of Total Immigration by Year (Countries {start_idx + 1} to {end_idx})')
            plt.legend(loc='upper left')
            plt.show()


def display_all_imig_by_amount():

        # Read the CSV file
        data = pd.read_csv(r"chosen_data/regression/total imgration to oced country.csv", encoding = "ISO-8859-1")

        # Group the data by 'Country' and calculate the sum of 'Value' for each country
        grouped_data = data.groupby('Country')['Value'].sum().reset_index()

        # Sort the countries based on total immigration in descending order
        sorted_countries = grouped_data.sort_values('Value', ascending=False)['Country'].tolist()

        # Determine the number of graphs needed
        num_graphs = len(sorted_countries) // 7 + 1

        # Plotting separate graphs for each group of 7 countries, sorted by total immigration
        for i in range(num_graphs):
            start_idx = i * 7
            end_idx = (i + 1) * 7

            # Extract the countries for the current graph
            current_countries = sorted_countries[start_idx:end_idx]

            # Filter the data for the current countries
            filtered_data = data[data['Country'].isin(current_countries)]

            # Plotting the graph for the current countries
            fig, ax = plt.subplots(figsize=(12, 6))
            for country in current_countries:
                country_data = filtered_data[filtered_data['Country'] == country]
                country_data.plot(x='Year', y='Value', ax=ax, label=country)

            plt.xlabel('Year')
            plt.ylabel('Total Immigration')
            plt.title(f'Total Immigration by Year (Countries {start_idx + 1} to {end_idx})')
            plt.legend(loc='upper left')
            plt.show()



def corr_gdp_imig():

    # Read the GDP data
    gdp_df = pd.read_csv("chosen_data/df_total_do_not_touch.csv")

    # Filter the GDP data to include relevant columns and years leading up to 2050
    gdp_df = gdp_df[gdp_df['Year'] <= 2050][['Country', 'GDP Total']]

    # Read the immigration data
    immigration_df = pd.read_csv(r"chosen_data/regression/total imgration to oced country.csv", encoding = "ISO-8859-1")

    # Filter the immigration data to include relevant columns and years leading up to 2050
    immigration_df = immigration_df[immigration_df['Year'] <= 2050][['Country', 'Value']]

    # Calculate average GDP for each country
    avg_gdp_df = gdp_df.groupby('Country')['GDP Total'].mean().reset_index()

    # Merge the two dataframes on 'Country' column
    merged_df = pd.merge(immigration_df, avg_gdp_df, on='Country')

    merged_df= merged_df.groupby(['Country','GDP Total'])['Value'].mean().reset_index()


    #Perform correlation analysis
    correlation = merged_df['Value'].corr(merged_df['GDP Total'])
    print("Correlation between GDP and immigration:", correlation)
    # Scatter plot of GDP vs. immigration
    plt.scatter(merged_df['GDP Total'], merged_df['Value'])
    plt.xlabel('GDP Total')
    plt.ylabel('Immigration')
    plt.title('GDP vs. Immigration')
    plt.show()




def where_isralies_imig_most():



        ##A to p

        df_A_to_P=pd.read_csv("chosen_data/scraping/AtoP_Foriegn_Born_Population_scrap_clean.csv", encoding = "ISO-8859-1")

        new_df_A_to_P=df_A_to_P[df_A_to_P["Country of birth"]=='Israel']



        # Remove commas (',') from the values in column A

        new_df_A_to_P["Value"]=new_df_A_to_P["Value"].str.replace(',', '')
        new_df_A_to_P['Value'] = new_df_A_to_P['Value'].astype(int)
        #print(new_df)

        grouped = new_df_A_to_P.groupby('Country or Area')
        max_rows = grouped.apply(lambda x: x.loc[x['Value'].idxmax()])

        # Reset the index of the new DataFrame
        new_df_A_to_P = max_rows.reset_index(drop=True)

        print(new_df_A_to_P)


        ## p to z

        df_P_to_Z = pd.read_csv("chosen_data/scraping/PtoZ_Foriegn_Born_Population_scrap_clean.csv", encoding="ISO-8859-1")

        new_df_P_to_Z = df_P_to_Z[df_P_to_Z["Country of birth"] == 'Israel']

        # Remove commas (',') from the values in column A

        new_df_P_to_Z["Value"] = new_df_P_to_Z["Value"].str.replace(',', '')
        new_df_P_to_Z['Value'] = new_df_P_to_Z['Value'].astype(int)
        # print(new_df)

        grouped = new_df_P_to_Z.groupby('Country or Area')
        max_rows = grouped.apply(lambda x: x.loc[x['Value'].idxmax()])

        # Reset the index of the new DataFrame
        new_df_P_to_Z = max_rows.reset_index(drop=True)

        print(new_df_P_to_Z)



        ##unite the two dfs

        united_df=pd.concat([new_df_A_to_P,new_df_P_to_Z])
        united_df.loc[len(united_df.index)] = ['United States Of America', 2000, 'Total','Both Sexes','Israel',2000,140000]
        print(united_df)

        united_df['Country or Area'].mask(united_df['Country or Area'] == 'TÃÂÃÂ¼rkiye', 'Turkey', inplace=True)

        united_df=united_df[united_df['Country or Area']!="State of Palestine"]



        ##create pie chart



        # Calculate the sum of values for countries below 1%

        sum_total=united_df['Value'].sum()

        threshold = sum_total*0.01
        small_countries = united_df[united_df['Value'] < threshold]
        small_countries_sum = small_countries['Value'].sum()

        # Filter out the small countries
        filtered_df = united_df[united_df['Value'] >= threshold]

        # Append 'Others' category to the filtered DataFrame
        filtered_df.loc[len(filtered_df)] = ['Others', 2000, 'Total','Both Sexes','Israel',2000,small_countries_sum]

        # Create a pie chart
        plt.pie(filtered_df['Value'], labels=filtered_df['Country or Area'], autopct='%1.1f%%')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Distribution of Values by Country')

        # Display the pie chart
        plt.show()





def where_do_continents_imig():




        ### save only the Country of birth which is in the list of continents
        df_A_to_P=pd.read_csv("chosen_data/scraping/AtoP_Foriegn_Born_Population_scrap_clean.csv", encoding = "ISO-8859-1")
        df_P_to_Z = pd.read_csv("chosen_data/scraping/PtoZ_Foriegn_Born_Population_scrap_clean.csv",
                                encoding="ISO-8859-1")




        lst_continents=["America, South",'America, North','Asia', 'Europe','AFRICA','OCEANIA','Australia','New Zealand']


        df_A_to_P_new=df_A_to_P[df_A_to_P["Country of birth"].isin(lst_continents)]
        df_P_to_Z_new=df_P_to_Z[df_P_to_Z["Country of birth"].isin(lst_continents)]


        #A to P

        ## save only the max rows
        df_A_to_P_new["Value"]=df_A_to_P_new["Value"].str.replace(',', '')
        df_A_to_P_new['Value'] = df_A_to_P_new['Value'].astype(int)
        #print(new_df)

        grouped = df_A_to_P_new.groupby('Country or Area')
        max_rows = grouped.apply(lambda x: x.loc[x['Value'].idxmax()])

        # Reset the index of the new DataFrame
        df_A_to_P_new = max_rows.reset_index(drop=True)

        print(df_A_to_P_new)


        ## p to z


        # Remove commas (',') from the values in column A

        df_P_to_Z_new["Value"] = df_P_to_Z_new["Value"].str.replace(',', '')
        df_P_to_Z_new['Value'] = df_P_to_Z_new['Value'].astype(int)
        # print(new_df)

        grouped = df_P_to_Z_new.groupby('Country or Area')
        max_rows = grouped.apply(lambda x: x.loc[x['Value'].idxmax()])

        # Reset the index of the new DataFrame
        df_P_to_Z_new = max_rows.reset_index(drop=True)

        print(df_P_to_Z_new)

        united_df = pd.concat([df_A_to_P_new, df_P_to_Z_new])

        ### for each continent , display a pie chart of the countries that are relevant to it (with others)


        ##replace neccesary rows

        united_df['Country or Area'].mask(united_df['Country or Area'] == 'TÃÂÃÂ¼rkiye', 'Turkey', inplace=True)
        united_df['Country of birth'].mask(united_df['Country of birth'] == 'Australia', 'OCEANIA', inplace=True)
        united_df['Country of birth'].mask(united_df['Country of birth'] == 'New Zealand', 'OCEANIA', inplace=True)

        lst_continents = ["America, South", 'America, North', 'Asia', 'Europe', 'AFRICA', 'OCEANIA']

        for continent in lst_continents:
                df_continent=united_df[united_df["Country of birth"]==continent]

                # Calculate the sum of values for countries below 1%

                sum_total=df_continent['Value'].sum()
                print(sum_total)

                threshold = sum_total*0.01
                small_countries = df_continent[df_continent['Value'] < threshold]
                small_countries_sum = small_countries['Value'].sum()

                # Filter out the small countries
                filtered_df = df_continent[df_continent['Value'] >= threshold]

                # Append 'Others' category to the filtered DataFrame
                filtered_df.loc[len(filtered_df)] = ['Others', 2000, 'Total','Both Sexes',continent,2000,small_countries_sum]

                # Create a pie chart
                plt.pie(filtered_df['Value'], labels=filtered_df['Country or Area'], autopct='%1.1f%%')
                plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                plt.title('Where does '+continent+' imigrate to')

                # Display the pie chart
                plt.show()

