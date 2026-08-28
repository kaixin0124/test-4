from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title='Malaysia Condominium Price Prediction', page_icon='🏢', layout='wide')
NUMERIC_FEATURES=['Bedroom','Bathroom','Property Size','# of Floors','Total Units','Parking Lot','Ad List']
CATEGORICAL_FEATURES=['Address','Property Type','Land Title','Tenure Type','Floor Range','Category']
ALL_FEATURES=NUMERIC_FEATURES+CATEGORICAL_FEATURES

@st.cache_data
def load_data():
    p=Path(__file__).resolve().parent/'houses.csv'
    if not p.exists(): raise FileNotFoundError(f'houses.csv not found: {p}')
    df=pd.read_csv(p); clean=df.copy()
    required=['price']+ALL_FEATURES
    missing=[c for c in required if c not in clean.columns]
    if missing: raise ValueError(f'Missing required columns: {missing}')
    clean['price']=(clean['price'].astype(str).str.replace('RM','',regex=False).str.replace(',','',regex=False).str.replace(' ','',regex=False))
    clean['price']=pd.to_numeric(clean['price'],errors='coerce')
    for c in ['Bedroom','Bathroom']:
        clean[c]=pd.to_numeric(clean[c].astype(str).str.extract(r'(\d+(?:\.\d+)?)')[0],errors='coerce')
    for c in ['Property Size','Ad List','# of Floors','Total Units','Parking Lot']:
        clean[c]=clean[c].astype(str).str.replace(',','',regex=False).str.replace('RM','',regex=False).str.strip().replace(['-','–','—','', 'nan','None'],np.nan)
        clean[c]=pd.to_numeric(clean[c],errors='coerce')
    for c in CATEGORICAL_FEATURES:
        clean[c]=clean[c].astype(str).str.strip().replace(['-','–','—','', 'nan','None'],np.nan).fillna('Unknown')
    clean=clean.drop_duplicates().dropna(subset=['price','Property Size'])
    clean=clean[(clean.price>0)&(clean['Property Size']>0)].copy()
    for c in NUMERIC_FEATURES: clean[c]=clean[c].fillna(clean[c].median() if clean[c].notna().any() else 0)
    return df,clean

@st.cache_resource
def train_models(clean):
    X=pd.get_dummies(clean[ALL_FEATURES],columns=CATEGORICAL_FEATURES,drop_first=False).astype(float); y=clean.price
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    scaler=StandardScaler(); Xtrs=scaler.fit_transform(Xtr); Xtes=scaler.transform(Xte)
    models={'Linear Regression':LinearRegression(),'KNN':KNeighborsRegressor(n_neighbors=5),'Random Forest':RandomForestRegressor(n_estimators=200,max_depth=10,min_samples_leaf=1,min_samples_split=2,random_state=42,n_jobs=-1),'SVM':SVR(kernel='rbf',C=100000,gamma='scale',epsilon=.1)}
    preds={}; metrics={}; fitted={}
    for name,m in models.items():
        if name in ['KNN','SVM']: m.fit(Xtrs,ytr); pred=m.predict(Xtes)
        else: m.fit(Xtr,ytr); pred=m.predict(Xte)
        fitted[name]=m; preds[name]=pred; metrics[name]={'MAE':mean_absolute_error(yte,pred),'MSE':mean_squared_error(yte,pred),'RMSE':np.sqrt(mean_squared_error(yte,pred)),'R²':r2_score(yte,pred)}
    return Xtr,Xte,ytr,yte,scaler,preds,metrics,fitted

df,clean=load_data()
st.sidebar.title('🏢 Condo Price'); st.sidebar.caption('BMDS2003 Data Science'); st.sidebar.divider()
page=st.sidebar.radio('Go to',['Overview','Exploratory Analysis','Model Performance','Price Prediction','Dataset'])
st.sidebar.divider(); st.sidebar.subheader('Property Information')

def opts(c): return sorted(clean[c].dropna().astype(str).unique().tolist())
bedroom=st.sidebar.slider('Bedroom',1,max(1,int(clean.Bedroom.max())),min(3,max(1,int(clean.Bedroom.max()))))
bathroom=st.sidebar.slider('Bathroom',1,max(1,int(clean.Bathroom.max())),min(2,max(1,int(clean.Bathroom.max()))))
property_size=st.sidebar.number_input('Property Size (sq.ft.)',min_value=1.0,value=float(clean['Property Size'].median()),step=50.0)
ad_list=st.sidebar.number_input('Ad List',min_value=0.0,value=float(clean['Ad List'].median()),step=1000.0)
floors=st.sidebar.number_input('# of Floors',min_value=0.0,value=float(clean['# of Floors'].median()),step=1.0)
total_units=st.sidebar.number_input('Total Units',min_value=0.0,value=float(clean['Total Units'].median()),step=10.0)
parking=st.sidebar.number_input('Parking Lot',min_value=0.0,value=float(clean['Parking Lot'].median()),step=1.0)
address=st.sidebar.selectbox('Address',opts('Address')); ptype=st.sidebar.selectbox('Property Type',opts('Property Type')); land=st.sidebar.selectbox('Land Title',opts('Land Title')); tenure=st.sidebar.selectbox('Tenure Type',opts('Tenure Type')); floor_range=st.sidebar.selectbox('Floor Range',opts('Floor Range')); category=st.sidebar.selectbox('Category',opts('Category'))

st.title('🏠 Malaysia Condominium Price Prediction'); st.write('Estimate a condominium price using property and building information.')

if page=='Overview':
    st.header('Overview'); a,b,c,d=st.columns(4); a.metric('Original Records',f'{len(df):,}'); b.metric('Cleaned Properties',f'{len(clean):,}'); c.metric('Predictor Variables',len(ALL_FEATURES)); d.metric('Target Variable','House Price')
    st.divider(); st.subheader('Project Objective'); st.write('This application uses regression machine learning models to predict Malaysian condominium prices based on property and building characteristics.'); st.subheader('Models Used'); st.write('Linear Regression, K-Nearest Neighbours (KNN), Random Forest Regression, and Support Vector Machines (SVM).')

elif page=='Exploratory Analysis':
    st.header('Exploratory Data Analysis')
    t1,t2,t3=st.tabs(['Price Distribution','Price Boxplot','Correlation Heatmap'])
    with t1:
        fig,ax=plt.subplots(figsize=(9,5)); ax.hist(clean.price,bins=40); ax.set_title('Distribution of House Prices'); ax.set_xlabel('House Price (RM)'); ax.set_ylabel('Frequency'); plt.tight_layout(); st.pyplot(fig)
    with t2:
        fig,ax=plt.subplots(figsize=(9,5)); sns.boxplot(x=clean.price,ax=ax); ax.set_title('House Price Boxplot'); ax.set_xlabel('House Price (RM)'); plt.tight_layout(); st.pyplot(fig)
    with t3:
        corr=clean[['Bedroom','Bathroom','Property Size','# of Floors','Total Units','Parking Lot','Ad List','price']].corr(); fig,ax=plt.subplots(figsize=(10,7)); sns.heatmap(corr,annot=True,fmt='.2f',cmap='Greens',vmin=-1,vmax=1,linewidths=.5,ax=ax); ax.set_title('Correlation Heatmap'); plt.tight_layout(); st.pyplot(fig)
    st.divider(); st.subheader('Distribution of Property Types'); counts=clean['Property Type'].value_counts().sort_values(); fig,ax=plt.subplots(figsize=(9,5)); counts.plot(kind='barh',ax=ax); ax.set_title('Distribution of Property Types'); ax.set_xlabel('Number of Properties'); ax.set_ylabel('Property Type'); plt.tight_layout(); st.pyplot(fig)

elif page=='Model Performance':
    st.header('Model Performance'); Xtr,Xte,ytr,yte,scaler,preds,metrics,fitted=train_models(clean); mdf=pd.DataFrame(metrics).T; st.subheader('Performance Comparison'); st.dataframe(mdf.style.format({'MAE':'RM {:,.2f}','MSE':'RM {:,.2f}','RMSE':'RM {:,.2f}','R²':'{:.4f}'}),use_container_width=True)
    selected=st.selectbox('Select Model',list(preds)); pred=preds[selected]; fig,ax=plt.subplots(figsize=(9,6)); ax.scatter(yte,pred,alpha=.6,label='Predicted Price'); lo=min(float(yte.min()),float(np.min(pred))); hi=max(float(yte.max()),float(np.max(pred))); ax.plot([lo,hi],[lo,hi],linestyle='--',label='Perfect Prediction'); ax.set_title(f'{selected}: Actual vs Predicted House Prices'); ax.set_xlabel('Actual House Price (RM)'); ax.set_ylabel('Predicted House Price (RM)'); ax.legend(); plt.tight_layout(); st.pyplot(fig)

elif page=='Price Prediction':
    st.header('Price Prediction'); Xtr,Xte,ytr,yte,scaler,preds,metrics,fitted=train_models(clean); selected=st.selectbox('Select Regression Model',['Linear Regression','KNN','Random Forest','SVM']); inp=pd.DataFrame([{'Bedroom':bedroom,'Bathroom':bathroom,'Property Size':property_size,'# of Floors':floors,'Total Units':total_units,'Parking Lot':parking,'Ad List':ad_list,'Address':address,'Property Type':ptype,'Land Title':land,'Tenure Type':tenure,'Floor Range':floor_range,'Category':category}]); Xall=pd.get_dummies(clean[ALL_FEATURES],columns=CATEGORICAL_FEATURES,drop_first=False); Xin=pd.get_dummies(inp,columns=CATEGORICAL_FEATURES,drop_first=False).reindex(columns=Xall.columns,fill_value=0).astype(float); pred=fitted[selected].predict(scaler.transform(Xin) if selected in ['KNN','SVM'] else Xin)[0]; st.subheader('Prediction Result');
    if st.button('Predict Price',type='primary'): st.success(f'Estimated Property Price: RM {pred:,.2f}')
    st.divider(); st.subheader('Selected Property Information'); st.dataframe(inp,use_container_width=True)

else:
    st.header('Dataset'); st.write(f'Original records: **{len(df):,}**'); st.write(f'Cleaned records: **{len(clean):,}**'); st.dataframe(clean,use_container_width=True,height=600); st.download_button('Download Cleaned Dataset',clean.to_csv(index=False).encode(), 'cleaned_houses.csv','text/csv')
