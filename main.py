from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

# Load the trained House Price model
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "house_price_model.joblib"
)
model = joblib.load(MODEL_PATH)

app = FastAPI(title="House Price Prediction API")


# Define input features
class HouseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    MSSubClass: int
    MSZoning: str
    LotFrontage: float | None = None
    LotArea: int
    Street: str
    Alley: str | None = None
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: str | None = None
    MasVnrArea: float | None = None
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: str | None = None
    BsmtCond: str | None = None
    BsmtExposure: str | None = None
    BsmtFinType1: str | None = None
    BsmtFinSF1: int
    BsmtFinType2: str | None = None
    BsmtFinSF2: int
    BsmtUnfSF: int
    TotalBsmtSF: int
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: str | None = None

    first_floor_sf: int = Field(alias="1stFlrSF")
    second_floor_sf: int = Field(alias="2ndFlrSF")

    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: int
    BsmtHalfBath: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: str | None = None
    GarageType: str | None = None
    GarageYrBlt: float | None = None
    GarageFinish: str | None = None
    GarageCars: int
    GarageArea: int
    GarageQual: str | None = None
    GarageCond: str | None = None
    PavedDrive: str
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int

    three_ssn_porch: int = Field(alias="3SsnPorch")

    ScreenPorch: int
    PoolArea: int
    PoolQC: str | None = None
    Fence: str | None = None
    MiscFeature: str | None = None
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: str
    SaleCondition: str


# Health endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Root endpoint
@app.get("/")
def root():
    return {"status": "Ready to serve"}


# Prediction endpoint
@app.post("/predict")
def predict(data: HouseFeatures):
    features = pd.DataFrame(
        [data.model_dump(by_alias=True)]
    )

    prediction = float(model.predict(features)[0])

    return {
        "predicted_price": round(prediction, 2),
        "currency": "USD"
    }