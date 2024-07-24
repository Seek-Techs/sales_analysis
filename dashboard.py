import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from plotly.express import bar, pie, line
import plotly.express as px
# this is for jupyter notebook to show the plot in the notebook itself instead of opening a new window
# %matplotlib inline

st.set_page_config(page_title='Sales Data!!!', page_icon=':bar_chart:', layout='wide')
st.title(' :bar_chart: Sales Data')

# adding a padding to the top of the title
st.markdown('<style>div.block-container {padding-top:irem;}</style>',unsafe_allow_html=True)

#read in data
@st.cache_data
def get_data():
    df = pd.read_csv('./data/cars.csv',index_col=0)
    return df





f1 = st.file_uploader(':file_folder: Upload a file', type=(['csv', 'txt','xlsx','xls']))
if f1 is not None:
    filename = f1.name
    st.write(filename)

    df = pd.read_csv(filename, encoding = 'unicode_escape')
    # st.write(df)
else:
    os.chdir(r'C:\Users\USER\Documents\HOLLERTECH FILES\DASHBOARD')
    @st.cache_data
    def get_data():
        df = pd.read_csv("sales_data_sample.csv", encoding = 'unicode_escape') 
        return df
    df = get_data()
    # st.write(df)

column1, column2 = st.columns((2))
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

#Getting the min and max date
# startdate = pd.to_datetime(df['ORDERDATE']).min()
# enddate = pd.to_datetime(df['ORDERDATE']).max()

# with column1:
#     date1 = pd.to_datetime(st.date_input('Start Date', startdate))

# with column2:
#     date2 = pd.to_datetime(st.date_input('End Date', startdate))

# df = df[(df['ORDERDATE'] >= date1) & (df['ORDERDATE']<= date2)].copy()


st.sidebar.header('Choose your filter: ')
region = st.sidebar.multiselect('Pick your Region', df['COUNTRY'].unique())
if not region:
    df2 = df.copy()
else:
    df2 = df[df['COUNTRY'].isin(region)]

# Create for state
state = st.sidebar.multiselect('Pick your State', options=df['STATE'].unique())
if not state:
    df3 = df2.copy()
else:
    df3 = df2[df2['STATE'].isin(state)]


# Create for city also
city = st.sidebar.multiselect('Pick your City', options=df['CITY'].unique())
# Filter the data based on region, state and city
if not region and not state and not city:
    filterd_df = df
elif not state and not city:
    filterd_df = df[df['COUNTRY'].isin(region)]
elif not region and not city:
    filterd_df = df[df['STATE'].isin(state)]
elif not region and not state:
    filterd_df = df[df['CITY'].isin(city)]
elif state and city:
    filterd_df = df3[df['STATE'].isin(state) & df3['CITY'].isin(city)]
elif region and city:
    filterd_df = df3[df['COUNTRY'].isin(region) & df3['CITY'].isin(city)]
elif city:
    filterd_df = df3[df3['CITY'].isin(city)]
else:
    filterd_df = df3[df3['COUNTRY'].isin(region) & df3['STATE'].isin(state) & df3['CITY'].isin(city)]

# Product categories
category_df = filterd_df.groupby(by=['PRODUCTCODE'], as_index=False)['SALES'].sum()

# Top 5 Product by sales
top_5 = filterd_df.groupby(by=['PRODUCTCODE'])['SALES'].sum().sort_values(ascending=False).head(5)
product_code = top_5.index.to_numpy()
sales = top_5.to_numpy()

# calculate KPI's
average_sales = int(filterd_df['SALES'].mean())
transaction_count = filterd_df.shape[0]
total_sales = filterd_df['SALES'].sum()
# print(f"Total Sales: ${total_sales:,.2f}")
# earliest_make_year = df_select['make-year'].min()
#popular_automation = df_select['Automation'].mode()

st.divider()
first_column, second_column, third_column = st.columns((3))

with first_column:
    st.subheader("Average Sales:")
    st.subheader(f"US $ {average_sales:,}")
with second_column:
    st.subheader("Number of Transaction:")
    st.subheader(f"{transaction_count:,} Products")
with third_column:
    st.subheader("Total Sales:")
    st.subheader(f"{total_sales:,}")
    
st.divider()

with column1:
    st.subheader("Category wise Sales")
    fig = px.bar(category_df, x = "PRODUCTCODE", y = "SALES", text = ['${:,.2f}'.format(x) for x in category_df["SALES"]],
                 template = "seaborn")
    st.plotly_chart(fig,use_container_width=True, height = 200)

with column2:
    st.subheader('Regional Wise Sales')
    fig = px.pie(filterd_df, names = 'COUNTRY', values='SALES', hole=0.5)
    fig.update_traces(textposition='outside',text=filterd_df['COUNTRY'], textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with column1: 
    st.subheader('Top 5 Product Category Wise Sales')
    fig = px.bar(top_5, x = product_code, y=sales, template='seaborn', text_auto=True, 
                 labels={'x': 'Product Category', 'y': 'Sales'},)
    st.plotly_chart(fig, use_container_width=True, height=200)


with column2: 
    st.subheader('Deal Size Count')
    deal_size_counts = filterd_df['DEALSIZE'].value_counts()
    fig = px.bar(deal_size_counts, x=deal_size_counts.index,  y=deal_size_counts.values,  
        text_auto=True,
        labels={'x': 'Deal Size Category', 'y': 'Number of Customers'},  
        template='seaborn')
    st.plotly_chart(fig, use_container_width=True, height=200)


st.subheader(":point_right: NUMBER OF CUSTOMERS DEAL SIZE BY SALES")

customer_dealsize = pd.pivot_table(data = filterd_df, values = "SALES", index = ["COUNTRY"], aggfunc={'SALES': sum, }, columns = "DEALSIZE")
st.write(customer_dealsize.style.background_gradient(cmap="Blues"))

cl1, cl2 = st.columns((2))
with cl1:
    with st.expander("Category_View Data"):
        st.write(category_df.style.background_gradient(cmap="Blues"))
        csv = category_df.to_csv(index = False).encode('utf-8')
        st.download_button("Download Data", data = csv, file_name = "Product_code.csv", mime = "text/csv",
                            help = 'Click here to download the data as a CSV file')

with cl2:
    with st.expander("Region_ViewData"):
        region = filterd_df.groupby(by = "COUNTRY", as_index = False)["SALES"].sum()
        st.write(region.style.background_gradient(cmap="Oranges"))
        csv = region.to_csv(index = False).encode('utf-8')
        st.download_button("Download Data", data = csv, file_name = "Region.csv", mime = "text/csv",
                        help = 'Click here to download the data as a CSV file')
        
filterd_df["month_year"] = filterd_df["ORDERDATE"].dt.to_period("M")
st.subheader('Time Series Analysis')

linechart = pd.DataFrame(filterd_df.groupby(filterd_df["month_year"].dt.strftime("%Y : %b"))["SALES"].sum()).reset_index()
fig2 = px.line(linechart, x = "month_year", y="SALES", labels = {"Sales": "Amount"},height=500, width = 1000,template="gridon")
st.plotly_chart(fig2,use_container_width=True)

with st.expander("View Data of TimeSeries:"):
    st.write(linechart.T.style.background_gradient(cmap="Blues"))
    csv = linechart.to_csv(index=False).encode("utf-8")
    st.download_button('Download Data', data = csv, file_name = "TimeSeries.csv", mime ='text/csv')

# Create a treem based on Region, category, sub-Category
st.subheader("Hierarchical view of Sales using TreeMap")
fig3 = px.treemap(filterd_df, path = ["COUNTRY","PRODUCTCODE"], values = "SALES",hover_data = ["SALES"],
                  color = "PRODUCTCODE")
fig3.update_layout(width = 800, height = 650)
st.plotly_chart(fig3, use_container_width=True)

# chart1, chart2 = st.columns((2))
# with chart1:
#     st.subheader('Segment wise Sales')
#     fig = px.pie(filterd_df, values = "SALES", names = "Segment", template = "plotly_dark")
#     fig.update_traces(text = filtered_df["Segment"], textposition = "inside")
#     st.plotly_chart(fig,use_container_width=True)

# with chart2:
#     st.subheader('Category wise Sales')
#     fig = px.pie(filtered_df, values = "Sales", names = "Category", template = "gridon")
#     fig.update_traces(text = filtered_df["Category"], textposition = "inside")
#     st.plotly_chart(fig,use_container_width=True)

import plotly.figure_factory as ff
st.subheader(":point_right: Month wise Sub-Category Sales Summary")
with st.expander("Summary_Table"):
    df_sample = df[0:5][['QUANTITYORDERED', 
       'SALES', 'ORDERDATE', 
        'PRODUCTCODE',
       'CITY', 'STATE', 
       'COUNTRY',
       'DEALSIZE']]
    fig = ff.create_table(df_sample, colorscale = "Cividis")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Month wise sub-Category Table")
    filterd_df["month"] = filterd_df["ORDERDATE"].dt.month_name()
    sub_category_Year = pd.pivot_table(data = filterd_df, values = "SALES", index = ["DEALSIZE"],columns = "month")
    st.write(sub_category_Year.style.background_gradient(cmap="Blues"))

# Create a scatter plot
data1 = px.scatter(filterd_df, x = "SALES", y = "MSRP", size = "QUANTITYORDERED")
data1['layout'].update(title="Relationship between Sales and Profits using Scatter Plot.",
                       titlefont = dict(size=20),xaxis = dict(title="Sales",titlefont=dict(size=19)),
                       yaxis = dict(title = "MSRP", titlefont = dict(size=19)))
st.plotly_chart(data1,use_container_width=True)

with st.expander("View Data"):
    st.write(filterd_df.iloc[:500,1:20:2].style.background_gradient(cmap="Oranges"))

# Download orginal DataSet
csv = df.to_csv(index = False).encode('utf-8')
st.download_button('Download Data', data = csv, file_name = "Data.csv",mime = "text/csv")

# ---- HIDE STREAMLIT STYLE ----
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    footer:after{
        content: 'Created by Samson Afolabi';
        visibility: visible;
        position: relative;
        right: 115px;
    }
    {
        background: LightBlue;
    }
    </style>
    """,
    unsafe_allow_html=True,
)