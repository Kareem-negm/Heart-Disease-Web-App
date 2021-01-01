import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st



st.write("""
# Heart Disease
 Detect if someone has Heart Disease using machine learning and python !
""")


#Get the data
df = pd.read_csv("C:/Users/negmk/Desktop/streamlit/Heart Disease/heart.csv")
st.image("https://img.cruisecritic.net/img-cc/image/15996/image_x_21.jpg?auto=format&fit=crop&crop=focalpoint&ar=2%3A1&ixlib=react-9.0.2&w=900&dpr=1")

st.subheader('Data Information:')
#Show the data as a table (you can also use st.write(df))
st.dataframe(df)
#Get statistics on the data
st.write(df.describe())
# Show the data as a chart.
chart = st.line_chart(df)


#Split the data into independent 'X' and dependent 'Y' variables
X=df.drop('target',axis=1)
y=df['target']
# Split the dataset into 75% Training set and 25% Testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Get the feature input from the user
def get_user_input():
    age = st.sidebar.slider('age', 29, 77, 29)
    sex  = st.sidebar.selectbox("what is your sex? (male=1 ,femal=0)",(0,1))
    cp  = st.sidebar.selectbox('chest pain type',(0,1,2,3))
    trestbps  = st.sidebar.slider(' resting blood pressure ', 90, 200, 117)
    chol = st.sidebar.slider('serum cholestoral in mg/dl',126, 564,300)
    fbs  = st.sidebar.selectbox("fasting blood sugar 1 = true; 0 = false ?",(0,1))
    restecg  = st.sidebar.selectbox('resting electrocardiographic results',(0,1,2))
    thalach  = st.sidebar.selectbox('maximum heart rate achieved', (0, 1))
    exang  = st.sidebar.selectbox('exercise induced angina (1 = yes; 0 = no)',(0,1))
    oldpeak   = st.sidebar.selectbox('ST depression induced by exercise relative to rest',(0,1,2))
    slope   = st.sidebar.selectbox('the slope of the peak exercise ST segment',(0,1,2))
    ca   = st.sidebar.selectbox('number of major vessels (0-3) colored by flourosopy',(0,1,2))
    thal   = st.sidebar.selectbox('3 = normal; 6 = fixed defect; 7 = reversable defect',(0,1,2))
        

    
    user_data = {'age ': age ,
              'sex ': sex ,
                 'cp ': cp ,
                 'trestbps ': trestbps ,
                 'chol ': chol ,
              'fbs ': fbs ,
              'restecg ': restecg ,
                 'thalach ': thalach ,
                 'exang  ': exang  ,
                 'oldpeak  ': oldpeak  ,
                 'slope  ': slope  ,
                 'ca  ': ca  ,
                 'thal  ': thal  ,
                 }
    features = pd.DataFrame(user_data, index=[0])
    return features
user_input = get_user_input()
st.subheader('User Input :')
st.write(user_input)


RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)


#Show the models metrics
st.subheader('Model Test Accuracy Score')
st.write( str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%' )
prediction = RandomForestClassifier.predict(user_input)
st.subheader('Classification: ')
st.write(prediction)


st.subheader('predicted probabilities: ')
predicted_probabilities=RandomForestClassifier.predict_proba(user_input)
st.write(predicted_probabilities)
 
if prediction==0:
    
    st.subheader('you dont have heart disease')
else:
    st.subheader('you have heart disease, please Go to the doctor')

    