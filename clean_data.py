import pandas as pd


## open csv and delete unnessecary column and save it

# df=pd.read_csv("inflow_imagrants_back_to_country.csv", encoding = "ISO-8859-1")
# df = df.drop('Unnamed: 6', axis=1)
# print(df.columns)
# df.to_csv("inflow_imagrants_clean.csv",index=False)





#df=pd.read_csv("working_population_by_sex_age_place_of_birth.csv", encoding = "ISO-8859-1")
#df = df.drop('Unnamed: 6', axis=1)
#print(df.columns)
#df.to_csv("outflow_imagrants_clean.csv",index=False)







# df=pd.read_csv("chosen_data/df_total_do_not_touch.csv", encoding = "ISO-8859-1")
#
# selected_col=["Country","Year","GDP Total"]
# df=df[selected_col]
#
# df.to_csv("GDP_by_year_per_country.csv",index=False)




##################################cleaning the scraping data##########################################



# df=pd.read_csv("chosen_data/scraping/AtoP_Foriegn_Born_Population_scrap.csv", encoding = "ISO-8859-1")
# print(df.columns)
# # df = df.drop('ÃÂ ', axis=1)
# # df.to_csv("chosen_data/scraping/AtoP_Foriegn_Born_Population_scrap_clean.csv",index=False)
#
#
# df=pd.read_csv("chosen_data/scraping/PtoZ_Foriegn_Born_Population_scrap_clean.csv", encoding = "ISO-8859-1")
#
#
#
# #print(df.columns)
# df = df.drop('ÃÂÃÂ ', axis=1)
# df.to_csv("chosen_data/scraping/PtoZ_Foriegn_Born_Population_scrap_clean.csv",index=False)


# df=pd.read_csv("chosen_data/scraping/AtoP_Foriegn_Born_Population_scrap.csv", encoding = "ISO-8859-1")
# print(df.columns)
#
# df = df.rename(columns={'ï»¿Country or Area': 'Country or Area'})
# df = df.drop('ÃÂ ', axis=1)
# df.to_csv("chosen_data/scraping/AtoP_Foriegn_Born_Population_scrap_clean.csv",index=False)


# df_old=pd.read_csv("chosen_data/scraping/AtoP_Foriegn_Born_Population_scrap_clean.csv", encoding = "ISO-8859-1")
# df=pd.read_csv("chosen_data/scraping/PtoZ_Foriegn_Born_Population_scrap.csv", encoding = "ISO-8859-1")
# print(df.columns)
# df = df.drop('H', axis=1)
# df.columns=df_old.columns
# # df = df.rename(columns={'ÃÂ¯ÃÂ»ÃÂ¿Country or Area': 'Country or Area'})
# df.to_csv("chosen_data/scraping/PtoZ_Foriegn_Born_Population_scrap_clean.csv",index=False)