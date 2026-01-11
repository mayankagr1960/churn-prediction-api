##Real-time Prediction via API

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

#Load Model
model = joblib.load('churn_prediction/model/final_churn_model.pkl')

#Define Basemodel
class Table_Schema(BaseModel):
    gender:                                     int
    SeniorCitizen:                              int
    Partner:                                    int
    Dependents:                                 int
    tenure:                                   float
    PhoneService:                               int
    MultipleLines:                              int
    OnlineSecurity:                             int
    OnlineBackup:                               int
    DeviceProtection:                           int
    TechSupport:                                int
    StreamingTV:                                int
    StreamingMovies:                            int
    PaperlessBilling:                           int
    MonthlyCharges:                           float
    TotalCharges:                             float
    InternetService_Fiber_optic:                int = Field(..., alias='InternetService_Fiber optic')
    InternetService_No:                         int
    Contract_One_year:                          int = Field(..., alias='Contract_One year')
    Contract_Two_year:                          int = Field(..., alias='Contract_Two year')
    PaymentMethod_Credit_card:                  int = Field(..., alias='PaymentMethod_Credit card (automatic)')
    PaymentMethod_Electronic_check:             int = Field(..., alias='PaymentMethod_Electronic check')
    PaymentMethod_Mailed_check:                 int = Field(..., alias='PaymentMethod_Mailed check')
    tenure_group:                               int

#Allow using Python names internally
#class Config:
 #   allow_population_by_field_name = True
  #  allow_population_by_alias = True
       
#Initialize API
app = FastAPI()

#Get Endpoint
@app.get("/")
def read_root():
    return('Churn model is ready for prediction. Please provide customer details..!!')

#Predict Endpoint
@app.post('/predict')
def predict_post(data: Table_Schema):
    #Convert input data to dataframe
    indata = pd.DataFrame([data.dict(by_alias=True)])

    #Prediction using ML Model
    #[:,1]Predict probablity for +ve class 1(Churn = True)
    #[0]Extract Scalar value
    churn_proba = model.predict_proba(indata)[:,1][0]
    churn_proba_01 = churn_proba * 100

    #Convert probablity to prediction value, with 0.5 threshold
    #int to convert True/False to 1/0
    churn_pred = int(churn_proba > 0.5)
    return {f"Churn probability of a customer: {churn_proba_01:.2f}%", 
            f"Churn prediction by Model: {churn_pred}",
            f"{'Customer will likely churn.' if churn_pred == 1 else 'Customer will likely stay.'}"
            }

# Load batch data (For batch processing)
#indata = pd.read_csv("batch_customers.csv")
