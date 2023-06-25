from bs4 import BeautifulSoup
import pandas as pd
import requests



end_range=26129
lst_col_names=[]
lst_data=[]
indexes_didnt_work=[]
flag_col=0
for index in range(1,end_range+1):

    print("index number : ",index)

    url = "https://data.un.org/Data.aspx?q=foreign&d=POP&f=tableCode%3a44&c=2,3,6,8,12,17,18&s=_countryEnglishNameOrderBy:asc,refYear:desc,areaCode:asc&v="
    url+=str(index)
    response = requests.get(url)
    html_content = response.content


    soup = BeautifulSoup(html_content, "html.parser")


    #Find the table element with the specified class
    table = soup.find('div', {'class': 'DataContainer'})


    try:
        for t in table:
                for i,tr in enumerate(t):

                    lst_temp = []
                    for td in tr:

                        ##columns
                        if i==0 and flag_col==0:
                            lst_col_names.append(td.text)

                        #data
                        else:
                            flag_col=1
                            lst_temp.append(td.text)

                    if i!=0:
                        lst_data.append(lst_temp)
    except:
        indexes_didnt_work.append(index)





#Print the table content

print(lst_col_names)
print(lst_data)
print("indexes that didnt work:")
print(indexes_didnt_work)

df=pd.DataFrame(data=lst_data,columns=lst_col_names)
df.to_csv("Foreign_born_population_by_country_area_of_birth_age_and_sex.csv",index=False)





