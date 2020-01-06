# # Deleted items

# title = st.text_input('Enter Job Title')
# t = 
# def location(city, state):
#     if city:
#         city = str(city.replace(' ', '+'))
#     if state:
#         state = str(state.replace(' ','+'))
#     return(city, state)

# city, state = location(city, state)       

# ind_location = city+"%2C+"+state      
# li_location = city+"%2C%20"+state

# ind_url = "https://www.indeed.com/jobs?q=title%3A(%22data+scientist%22+OR+%22data+science%22+OR+%22data+analyst%22)&l="+ind_location+"&radius=50&sort=date" 
# li_url = "https://www.linkedin.com/jobs/search?keywords=Data%20Scientist&location="+li_location+"&trk=guest_job_search_jobs-search-bar_search-submit&redirect=false&position=1&pageNum=0&f_TP=1%2C2"

# indeed = WebScrape_Indeed.extract_title(ind_url)
# linkedin = WebScrape_LinkedIn.extract_title(li_url)

# # df3 = df1.join(df2, how='inner', lsuffix='LI', rsuffix='Ind')

# # data_li = pd.read_csv('LinkedIn.csv', usecols=['description'])

# # data_ind = pd.read_csv('Indeed.csv', usecols=['description'])

# # df3 = df1.join(df2, how='inner', lsuffix='LI', rsuffix='Ind')

# # data_li = pd.read_csv('LinkedIn.csv', usecols=['description'])

#  # title = st.text_input('Enter Job Title')

#  @st.cache
# def clean(raw):
#     raw = ' '.join(raw.tolist())
#     for char in '-.,\n':
#         raw = raw.replace(char,' ')
    
#     pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
#     raw = nltk.regexp_tokenize(raw, pattern)
#     return raw


# @st.cache
# def word_count(string):
#     freq = defaultdict(int)
#     List1 = [x.lower() for x in search_terms]
#     List2 = [x.lower() for x in string]

#     for item in List1:
#         if item in List2:
#             freq[item] = List2.count(item)
#         else:
#             freq[item] = 0 
#     return freq

# Cdict = Counter(test) + Counter(test2) 

# result_chart = pd.DataFrame.from_dict(Cdict, orient='index', columns=['Count'])

# ind_url = []
# li_url = []
# for i in ind_cities:
#     ind = "https://www.indeed.com/jobs?q=title%3A(%22"+title_ind+"%22)&l="+i+"&radius=50&sort=date" 
#     ind_url.append(ind)       
   
# for i in li_cities:
#     li = "https://www.linkedin.com/jobs/search?keywords="+title_li+"&location="+i+"&trk=guest_job_search_jobs-search-bar_search-submit&redirect=false&position=1&pageNum=0&f_TP=1%2C2"
#     li_url.append(li)

# for x in ind_url:
#     indeed = WebScrape_Indeed.extract_title(x)
    
# for x in li_url:
#     linkedin = WebScrape_Indeed.extract_title(x)

# app_mode = st.button("Begin", key=10)
# if app_mode == True:
# result_chart = st.progress(0)
# for percent_complete in range(100):
#     result_chart.progress(percent_complete + 1)
# option = st.selectbox('Search for:', ('...', 'Data Scientist', 'Data Analyst'))
# if option is not '...':
#     st.write('Searching for: ', option)

# historic_data = pd.read_csv('historic_data.csv', index_col='Tech')
# historic_data = historic.set_index('Tech', drop=True)

# st.button('Search', key=6)


# check = st.checkbox('All Skills', value=False, key=4)
# if check is not True:
#     try: 
#         search_terms = st.multiselect('Choose Skills', options=search_terms, key=3)
#     except:
#         pass
# else:
#     search_terms = search_terms 

# ind_cities = Cities.ind_cities
# li_cities = Cities.li_cities

# total_results= total_result.nlargest(10, 'Percentage')
# tot_date = pd.to_datetime(dtotal['date'])
# tot_date = tot_date.values.astype(int)
# tot_date = pd.to_datetime(tot_date.mean())
# tot_date = tot_date.dt.strftime('%m/%d/%Y')
# low = pd.to_datetime(min(dtotal.date),format='%m-%d-%Y')
# high = pd.to_datetime(max(dtotal.date))
# st.header("Database Results: ")
# st.bar_chart(total_result_chart)

# st.write(alt.Chart(total_result_chart).mark_bar().encode(x='Tech', y='Percentage', color='Percentage', order=alt.Order('Percentage', sort='ascending')))

# results = results.Percentage.astype(float)
# results = result.sort_values('Percentage', ascending=False)
# results= results.nlargest(n=10, columns='Percentage')

# result_chart = results.nlargest(10, 'Percentage')
# merged = results.merge(total_result, left_index=True, right_index=True)
# merged_chart = merged.sort_values('Percentage_x', ascending=False).head(10)

# co_date = pd.to_datetime(indeed['date'])
# co_date = pd.to_datetime(co_date).values.astype(int)
# co_date = pd.to_datetime(co_date.mean())
# co_date = co_date.dt.strftime('%m/%d/%Y')

# st.header("Web Results: ")

# st.bar_chart(result_chart)

# st.write(alt.Chart(result_chart).mark_bar().encode(x='Tech', y='Percentage', color='Percentage', order=alt.Order('Percentage', sort='ascending')))

# st.write(merged_chart)
# st.pyplot(result_chart)
# st.bar_chart(merged_chart)

# search_terms = st.multiselect('Choose Skills', options=search_terms, key=3)

# date = dtotal['date']
# date = pd.to_datetime(date).values.astype(int)
# d = pd.to_datetime(date.mean())
# total_resulty['Date'] = d

# ldate = linkedin['date']
# ldate = pd.to_datetime(ldate).values.astype(int)
# ld = pd.to_datetime(ldate.mean())

# plz1 = functions.compare(skills, resulty)
# plz2 = functions.compare(skills, total_resulty)
# df_test1 = pd.DataFrame(plz1)
# df_test2 = pd.DataFrame(plz2)
# df_test3 = df_test1.subtract(df_test2, fill_value=0)
# df_test3 = df_test3.reset_index()
# df_test3['Date'] = ld
# df_test3 = df_test3.rename(columns={'index':'Tech'})

# if st.button('Search: ', key=2):
#     st.dataframe(df_test3.style.highlight_max(axis=0))
#     st.write(alt.Chart(df_test3).mark_bar(size=15).encode(x='Date', y='Percentage', color='Percentage').configure_view(strokeWidth=0,height=400,width=300))

# st.line_chart(df_test3)

# skills = st.multiselect('Skill Comparison: ', options=search_terms)


# plz1 = functions.compare(skills, resulty)
# plz2 = functions.compare(skills, total_resulty)
# df_test1 = pd.DataFrame(plz1)
# df_test2 = pd.DataFrame(plz2)
# df_test3 = df_test1.subtract(df_test2, fill_value=0)
# df_test3 = df_test3.reset_index()
# df_test3 = df_test3.rename(columns={'index':'Tech'})

# if st.button('Search: ', key=2):
#     st.dataframe(df_test3.style.highlight_max(axis=0))
    # st.write(alt.Chart(df_test3).mark_bar(size=15).encode(x='Tech', y='Percentage', color='Percentage').configure_view(strokeWidth=0,height=400,width=300))

# cit = input('Please, enter a city:\n')
# stat = input('Please, enter a state:\n')
# city = str(cit.replace(' ', '+'))
# state = str(stat.replace(' ','+'))
# location = city+"%2C%20"+state

# print(f'Searching {city},{state}. Please wait...')

# search_url = "https://www.linkedin.com/jobs/search?keywords=Data%20Scientist&location="+location+"&trk=guest_job_search_jobs-search-bar_search-submit&redirect=false&position=1&pageNum=0&f_TP=1%2C2"

# st.deck_gl_chart(
#             viewport={
#                 'latitude': map_graph['lat'].median(),
#                 'longitude':  map_graph['lon'].median(),
#                 'zoom': 2.5
#             },
#             layers=[{
#                 'type': 'ScatterplotLayer',
#                 'data': map_graph,
#                 'radiusScale': 250,
#    'radiusMinPixels': 5,
#                 'getFillColor': [26,118,255],
#                 'pickable': True},
#             {'type': 'ScatterplotLayer',
#             'data': db_graph,
#             'radiusScale': 250,
#             'radiusMinPixels': 5,
#             'getFillColor': [55,83,109],
#             'pickable': True}]
#         )

# st.sidebar.markdown('Database Results: ', total_length)
# st.sidebar.markdown('Web Results: ', web_len)

# view_state = pdk.ViewState(latitude=db_graph['lat'].median(), longitude=db_graph['lon'].median(), zoom=2.5)
# db_layer = pdk.Layer('ScatterplotLayer', db_graph, get_position='[lon, lat]', radiusScale=10,
#                   radiusMinPixels=2, get_fill_color=[55, 83, 109, 140], pickable=True, auto_highlight=True)
# web_layer = pdk.Layer('ScatterplotLayer', map_graph, get_position='[lon, lat]', radiusScale=10,
#                   radiusMinPixels=2, get_fill_color=[180, 0, 200, 140], pickable=True, auto_highlight=True)

# r = pdk.Deck(layers=[db_layer, web_layer],initial_view_state=view_state, mapbox_key='pk.eyJ1IjoiZG1hdWdlciIsImEiOiJjazE4OXhuYmgxZ3d2M21uaXV5eWt6Nm5iIn0.8mXDhVPh7XaS9VSr1Uj7cw')
# st.deck_gl_chart(r)

# fig = go.Figure()
# fig.add_trace(go.Bar(
#     x=output_result_chart['Tech'],
#     y=output_result_chart['Percentage'],
#     name='Tech Prediction',
#     marker_color='rgb(55, 83, 109)'
# ))
# fig.update_layout(xaxis_tickfont_size=14, yaxis=dict(title='Skill Distribution (%)', titlefont_size=16, tickfont_size=14,), barmode='group', xaxis_tickangle=-45)
# st.plotly_chart(fig)

# final_chart = st.bar_chart(out_chart)

# final_chart.add_rows(look_chart)

# c = alt.Chart(out_chart).mark_bar().encode(x='a', y='b', color='Tech')
# st.altair_chart(c, width=-1)